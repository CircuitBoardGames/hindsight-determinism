"""Regression tests for scripts/run_mcp.sh venv interpreter resolution.

`run_mcp.sh` resolves the plugin venv's Python interpreter with a bash
`resolve_py()` helper before either exec-ing it or (if missing) re-creating
the venv. The probe must understand every venv layout the plugin can run on:

  - POSIX:                       ``<venv>/bin/python``
  - msys2 / mingw (git-bash):    ``<venv>/bin/python.exe``  (added in #1564)
  - standard Windows CPython:    ``<venv>/Scripts/python.exe``

The third layout is what the python.org installer, the Windows Store Python,
and ``py -m venv`` all produce — the *common* Windows case. Before the
``Scripts/`` branch existed, ``resolve_py`` returned an empty ``$PY`` on those
venvs, so the launcher fell through to venv re-creation, which then failed
whenever ``python``/``python3`` were not on the spawning process's PATH (issue
#1758, sub-item 3a). These tests invoke the real bash function against a
fabricated venv tree and assert on the interpreter it selects.
"""

import os
import stat
import subprocess
import textwrap

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
RUN_MCP_SH = os.path.abspath(os.path.join(SCRIPTS_DIR, "run_mcp.sh"))


def _resolve_py(venv_dir: str) -> str:
    """Run run_mcp.sh's ``resolve_py`` against ``venv_dir`` and return ``$PY``.

    The script's top-level body (which would actually create a venv and exec a
    server) is never run: a tiny driver sources only the ``resolve_py``
    function definition out of the file, sets ``VENV``, calls it, and echoes
    ``$PY``. This keeps the test hermetic and exercises the exact shell logic
    that ships.
    """
    driver = textwrap.dedent(
        """
        set -e
        # Extract just the resolve_py function definition from the script and
        # evaluate it, so none of the script's top-level side effects run.
        func="$(sed -n '/^resolve_py() {/,/^}/p' "$1")"
        eval "$func"
        VENV="$2"
        resolve_py
        printf '%s' "$PY"
        """
    )
    result = subprocess.run(
        ["bash", "-c", driver, "bash", RUN_MCP_SH, venv_dir],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_executable(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("#!/usr/bin/env python\n")
    os.chmod(path, 0o755)


def _run_tail_with_fake_exec(plugin_data: str, plugin_root: str) -> str:
    """Run the launcher tail with a fake exec and return project/server cwd."""
    driver = textwrap.dedent(
        """
        set -e
        export CLAUDE_PLUGIN_DATA="$1"
        export CLAUDE_PLUGIN_ROOT="$2"
        PY=python
        exec() { printf '%s\n%s' "$HINDSIGHT_MCP_PROJECT_CWD" "$(pwd)"; }
        sed -n '/^export HINDSIGHT_MCP_PROJECT_CWD=/,$p' "$3" | source /dev/stdin
        """
    )
    result = subprocess.run(
        ["bash", "-c", driver, "bash", plugin_data, plugin_root, RUN_MCP_SH],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class TestResolvePyVenvLayouts:
    """`resolve_py` must find the interpreter in every supported venv layout."""

    def test_windows_scripts_layout_is_resolved(self, tmp_path):
        """Standard Windows CPython venv: interpreter lives in ``Scripts/``.

        RED before the fix: ``resolve_py`` only probed ``bin/`` and returned an
        empty ``$PY`` for this (common) layout, triggering a doomed venv
        re-creation (issue #1758/3a). GREEN after the ``Scripts/`` branch.
        """
        venv = tmp_path / "venv"
        scripts_python = venv / "Scripts" / "python.exe"
        _make_executable(str(scripts_python))
        # Deliberately no bin/ — this is the genuine Windows layout.
        assert not (venv / "bin").exists()

        resolved = _resolve_py(str(venv))

        assert resolved == str(scripts_python), (
            "resolve_py must select <venv>/Scripts/python.exe for a standard "
            "Windows CPython venv; got: " + repr(resolved)
        )

    def test_posix_bin_layout_is_resolved(self, tmp_path):
        """Regression guard: the POSIX ``bin/python`` path still wins."""
        venv = tmp_path / "venv"
        bin_python = venv / "bin" / "python"
        _make_executable(str(bin_python))

        resolved = _resolve_py(str(venv))

        assert resolved == str(bin_python), (
            "resolve_py must still select <venv>/bin/python on POSIX; got: "
            + repr(resolved)
        )


def _run_launcher(tmp_path, env_extra, venv_dir):
    """Run the whole launcher against a fabricated venv whose python only records how it was exec'd.

    `-c "import mcp"` succeeds and the requirements cache matches, so nothing is installed or started.
    """
    record = tmp_path / "record"
    fake = venv_dir / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.write_text(
        "#!/bin/sh\n"
        '[ "$1" = -c ] && exit 0\n'
        'printf "data=%s\\ncwd=%s\\nargv=%s\\nroot=%s\\n" "$CLAUDE_PLUGIN_DATA" "$(pwd)" "$*" "$CLAUDE_PLUGIN_ROOT" > "$RECORD"\n'
    )
    fake.chmod(0o755)
    plugin_root = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
    with open(os.path.join(plugin_root, "requirements.txt"), encoding="utf-8") as fh:
        (venv_dir.parent / "requirements.txt").write_text(fh.read())
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"), "RECORD": str(record),
           "CLAUDE_PLUGIN_ROOT": plugin_root, **env_extra}
    env = {k: v for k, v in env.items() if v is not None}  # None: leave the variable unset, as omp does
    result = subprocess.run(["bash", RUN_MCP_SH], env=env, cwd=str(tmp_path), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in record.read_text().splitlines())


class TestOmpLaunch:
    """omp substitutes ${CLAUDE_PLUGIN_ROOT} into the command line but exports neither variable."""

    def test_neither_variable_set_still_starts_the_server_from_its_own_root(self, tmp_path):
        """RED on 0.7.5-airoboros.6: REQ_SRC was /requirements.txt; on a real venv pip failed, here there is no pip."""
        data = tmp_path / "home" / ".claude" / "plugins" / "data" / "hindsight-memory-hindsight"
        got = _run_launcher(tmp_path, {"CLAUDE_PLUGIN_ROOT": None}, data / "venv")
        root = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
        assert got["root"] == root, "exported, so lib/config.py finds the plugin's settings.json"
        assert got["argv"].endswith(os.path.join(root, "scripts", "mcp_server.py"))
        assert got["data"] == str(data)


class TestPluginDataDefault:
    """Claude Code sets both variables; omp sets neither in the environment."""

    def test_unset_plugin_data_falls_back_to_claude_codes_own_dir(self, tmp_path):
        """RED before the fix: `mkdir -p ""` failed and the launcher exited 1 before the server ran."""
        data = tmp_path / "home" / ".claude" / "plugins" / "data" / "hindsight-memory-hindsight"
        got = _run_launcher(tmp_path, {}, data / "venv")
        assert got["data"] == str(data), "the server must see the defaulted dir, for state.py"
        assert got["cwd"] == str(data)
        assert got["argv"].endswith("scripts/mcp_server.py")

    def test_claude_config_dir_is_honoured(self, tmp_path):
        data = tmp_path / "cfg" / "plugins" / "data" / "hindsight-memory-hindsight"
        got = _run_launcher(tmp_path, {"CLAUDE_CONFIG_DIR": str(tmp_path / "cfg")}, data / "venv")
        assert got["data"] == str(data)

    def test_an_empty_plugin_data_is_treated_as_unset(self, tmp_path):
        data = tmp_path / "home" / ".claude" / "plugins" / "data" / "hindsight-memory-hindsight"
        got = _run_launcher(tmp_path, {"CLAUDE_PLUGIN_DATA": ""}, data / "venv")
        assert got["data"] == str(data)

    def test_a_trailing_slash_on_claude_config_dir_adds_no_double_slash(self, tmp_path):
        data = tmp_path / "cfg" / "plugins" / "data" / "hindsight-memory-hindsight"
        got = _run_launcher(tmp_path, {"CLAUDE_CONFIG_DIR": str(tmp_path / "cfg") + "/"}, data / "venv")
        assert got["data"] == str(data)

    def test_no_home_and_no_config_dir_fails_with_a_message(self, tmp_path):
        r = subprocess.run(["bash", RUN_MCP_SH], env={"PATH": os.environ["PATH"]}, cwd=str(tmp_path),
                           capture_output=True, text=True, check=False)
        assert r.returncode != 0
        assert "CLAUDE_PLUGIN_DATA, CLAUDE_CONFIG_DIR and HOME are all unset" in r.stderr, r.stderr

    def test_a_set_plugin_data_still_wins(self, tmp_path):
        data = tmp_path / "given"
        got = _run_launcher(tmp_path, {"CLAUDE_PLUGIN_DATA": str(data)}, data / "venv")
        assert got["data"] == str(data) and got["cwd"] == str(data)


class TestMcpServerWorkingDirectory:
    """The launcher must not run the server from Claude Code's project cwd."""

    def test_exec_runs_from_plugin_data_dir_when_project_env_is_unreadable(self, tmp_path):
        """FastMCP probes `.env` in cwd, so use the plugin-owned data dir."""
        project = tmp_path / "project"
        project.mkdir()
        env_file = project / ".env"
        env_file.write_text("SECRET=value\n")
        env_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        plugin_data = tmp_path / "plugin-data"
        plugin_root = tmp_path / "plugin-root"
        plugin_root.mkdir()

        previous = os.getcwd()
        try:
            os.chdir(project)
            project_cwd, server_cwd = _run_tail_with_fake_exec(str(plugin_data), str(plugin_root)).splitlines()
        finally:
            os.chdir(previous)

        assert project_cwd == str(project)
        assert server_cwd == str(plugin_data)
