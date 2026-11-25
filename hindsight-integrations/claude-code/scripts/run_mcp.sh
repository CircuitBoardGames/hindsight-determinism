#!/usr/bin/env bash
# Launch the Hindsight MCP server inside the plugin's persistent venv.
# Bootstraps the venv if missing; otherwise just execs.
set -e

# omp substitutes ${CLAUDE_PLUGIN_ROOT} into this launcher's command line but does not export it, so under omp
# REQ_SRC was /requirements.txt and pip ended the server before it answered (0.7.5-airoboros.6). The root is
# the directory above this script's own scripts/; exported, so lib/config.py's settings lookup sees it too.
if [ -z "${CLAUDE_PLUGIN_ROOT:-}" ]; then
  CLAUDE_PLUGIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export CLAUDE_PLUGIN_ROOT

# Claude Code sets CLAUDE_PLUGIN_DATA. omp, which loads this plugin's MCP server too, expands only
# ${CLAUDE_PLUGIN_ROOT}, so under omp it was empty and `mkdir -p ""` ended the server (exit 1) before
# it answered. Fall back to the directory Claude Code itself gives this plugin (data/<plugin>-<marketplace>),
# so both harnesses share one venv and one state dir.
# scripts/lib/state.py falls back to the same dir.
if [ -z "${CLAUDE_PLUGIN_DATA:-}" ]; then
  _cfg="${CLAUDE_CONFIG_DIR:-${HOME:?run_mcp.sh: CLAUDE_PLUGIN_DATA, CLAUDE_CONFIG_DIR and HOME are all unset}/.claude}"
  CLAUDE_PLUGIN_DATA="${_cfg%/}/plugins/data/hindsight-memory-hindsight"
fi
export CLAUDE_PLUGIN_DATA

VENV="${CLAUDE_PLUGIN_DATA}/venv"
REQ_SRC="${CLAUDE_PLUGIN_ROOT}/requirements.txt"
REQ_CACHED="${CLAUDE_PLUGIN_DATA}/requirements.txt"

# Resolve the venv interpreter. On Windows-built venvs `bin/python` is
# `python.exe`; bash's `[ -x ]` does not honor PATHEXT, so probe both forms.
# Standard Windows CPython (python.org installer, Windows Store, `py -m venv`)
# puts the interpreter under `Scripts/` instead of `bin/` — probe that too.
resolve_py() {
  if [ -x "${VENV}/bin/python" ]; then
    PY="${VENV}/bin/python"
    PIP="${VENV}/bin/pip"
  elif [ -x "${VENV}/bin/python.exe" ]; then
    PY="${VENV}/bin/python.exe"
    PIP="${VENV}/bin/pip.exe"
  elif [ -x "${VENV}/Scripts/python.exe" ]; then
    PY="${VENV}/Scripts/python.exe"
    PIP="${VENV}/Scripts/pip.exe"
  else
    PY=""
    PIP=""
  fi
}

resolve_py
if [ -z "${PY}" ]; then
  mkdir -p "${CLAUDE_PLUGIN_DATA}"
  if ! python3 -m venv "${VENV}" 2>/dev/null; then
    python -m venv "${VENV}"
  fi
  resolve_py
  if [ -z "${PY}" ]; then
    echo "[Hindsight MCP] venv create failed: no python interpreter at ${VENV}/bin/ or ${VENV}/Scripts/" >&2
    exit 1
  fi
fi

# Re-pip only when the requirements cache is missing, requirements drifted, or
# `mcp` is not importable from the venv. Splitting this from venv creation
# keeps warm starts cheap and avoids re-running pip over a venv that's already
# in use (which fails with ERROR_SHARING_VIOLATION on Windows).
if [ ! -f "${REQ_CACHED}" ] \
   || ! diff -q "${REQ_SRC}" "${REQ_CACHED}" >/dev/null 2>&1 \
   || ! "${PY}" -c "import mcp" >/dev/null 2>&1; then
  "${PIP}" install --quiet -r "${REQ_SRC}"
  cp "${REQ_SRC}" "${REQ_CACHED}"
fi

# Claude Code launches MCP servers from the session project directory. If that
# directory contains an unreadable .env, FastMCP's default settings loader can
# fail before our config loads. Run from plugin-owned data instead, while
# preserving the session project cwd for bank derivation.
export HINDSIGHT_MCP_PROJECT_CWD="${PWD}"
mkdir -p "${CLAUDE_PLUGIN_DATA}"
cd "${CLAUDE_PLUGIN_DATA}"

exec "${PY}" "${CLAUDE_PLUGIN_ROOT}/scripts/mcp_server.py"
