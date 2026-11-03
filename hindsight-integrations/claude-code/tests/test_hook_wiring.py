"""Every hook command, from hooks.json and from setup_hooks.py, run as Claude Code runs it.

A missing script must exit 1: python3 exits 2 on one, and Claude Code reads 2 from UserPromptSubmit as a
BLOCK, so a deleted plugin dir refuses every prompt (hindsight-determinism#22).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import setup_hooks

INTEGRATION_ROOT = Path(__file__).resolve().parent.parent


def _wirings(root: Path):
    """(label, command, env) for every hook command, each pointed at `root`."""
    hooks = json.loads((INTEGRATION_ROOT / "hooks" / "hooks.json").read_text())["hooks"]
    for event, groups in hooks.items():
        for h in (h for g in groups for h in g["hooks"]):
            yield f"hooks.json {event}", h["command"], {"CLAUDE_PLUGIN_ROOT": str(root)}
    for event, groups in setup_hooks.build_hooks(str(root)).items():
        for h in (h for g in groups for h in g["hooks"]):
            yield f"setup_hooks {event}", h["command"], {}


def _run(command, env):
    return subprocess.run(["sh", "-c", command], input="the hook input", capture_output=True, text=True,
                          timeout=30, env={"PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}", **env})


def test_every_event_is_wired_from_both_sources__control(tmp_path):
    """The tests below loop over _wirings: this pins that the loop sees all eight commands."""
    labels = sorted(label for label, _, _ in _wirings(tmp_path))
    events = ["SessionEnd", "SessionStart", "Stop", "UserPromptSubmit"]
    assert labels == sorted([f"hooks.json {e}" for e in events] + [f"setup_hooks {e}" for e in events])


def test_a_missing_script_exits_1_and_names_itself(tmp_path):
    root = tmp_path / "gone"
    for label, command, env in _wirings(root):
        r = _run(command, env)
        assert r.returncode == 1, f"{label}: exit {r.returncode}, stderr {r.stderr!r}"
        assert f"hindsight: hook script missing or unreadable: {root}/scripts/" in r.stderr, label


def test_a_present_script_keeps_its_exit_code_and_gets_stdin__control(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ["session_start.py", "recall.py", "retain.py", "session_end.py"]:
        (scripts / name).write_text("import sys\nsys.stdout.write(sys.stdin.read())\nsys.exit(3)\n")
    for label, command, env in _wirings(tmp_path):
        r = _run(command, env)
        assert (r.returncode, r.stdout) == (3, "the hook input"), f"{label}: {r.returncode} {r.stdout!r} {r.stderr!r}"
