"""dsh turns retain from dsh's own session record, tagged `source:dsh` (hindsight-determinism#6).

dsh's `dsh-hooks-claude-code` bridge sends `transcript_path: ""`; the hub's dsh-adapter adds
`"harness": "dsh"` to the payload. retain.py then reads
`$DSH_HOME/sessions/<cwd slug>/<session id>/session.v4.jsonl.zstd` (the shape Session-Succession's
dsh trigger reads, dsh 0.2.1-rc.5). The bank is an in-memory fake: nothing here reaches a server.
"""

import json
import subprocess

import pytest
from conftest import FakeHTTPResponse, make_transcript_file
from test_hooks import _run_hook

pytestmark = pytest.mark.skipif(subprocess.run(["which", "zstd"], capture_output=True).returncode, reason="no zstd")

SID = "019a8d7e-dsh-sess"
FACT = "the staging database listens on port 6543"


def _record(dsh_home, rows, sid=SID):
    """A dsh session record, written as two zstd frames as a live file grows."""
    lines = [json.dumps(r, separators=(",", ":")) + "\n" for r in rows]
    half = len(lines) // 2
    frames = b"".join(
        subprocess.run(["zstd", "-cq"], input="".join(part).encode(), capture_output=True, check=True).stdout
        for part in (lines[:half], lines[half:])
    )
    d = dsh_home / "sessions" / "--var-lib-dsh-w--" / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.v4.jsonl.zstd").write_bytes(frames)


def _user(text, kind="user"):
    return {
        "type": "user/message",
        "seq": 3,
        "data": {"id": "u1", "role": "user", "content": [{"type": "text", "text": text}], "source": {"kind": kind}},
    }


def _assistant(text):
    return {
        "type": "assistant/message",
        "seq": 10,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "role": "assistant",
                "content": [{"type": "reasoning", "text": "thinking it over"}, {"type": "text", "text": text}],
            },
        },
    }


HEADER = {"type": "session", "version": 4, "id": SID, "cwd": "/var/lib/dsh/w", "delegationDepth": 0}


class FakeBank:
    """POST .../memories stores the item; POST .../memories/recall returns every stored item."""

    def __init__(self):
        self.items = []

    def urlopen(self, req, timeout=None):
        if req.full_url.endswith("/memories/recall"):
            return FakeHTTPResponse(
                {
                    "results": [
                        {"text": i["content"], "type": "experience", "tags": i.get("tags", [])} for i in self.items
                    ]
                }
            )
        if req.full_url.endswith("/memories") and req.get_method() == "POST":
            self.items += json.loads(req.data.decode())["items"]
            return FakeHTTPResponse({"status": "accepted"})
        return FakeHTTPResponse({})


def _dsh_stop(sid=SID):
    return {
        "session_id": sid,
        "transcript_path": "",
        "cwd": "/var/lib/dsh/w",
        "hook_event_name": "Stop",
        "stop_hook_active": False,
        "harness": "dsh",
    }


def test_a_dsh_turn_recalls_a_fact_retained_from_a_dsh_turn(monkeypatch, tmp_path):
    _record(tmp_path / "dsh", [HEADER, _user("Where does staging's database listen?"), _assistant(FACT)])
    bank = FakeBank()
    env = {"DSH_HOME": str(tmp_path / "dsh")}

    _run_hook("retain", _dsh_stop(), monkeypatch, tmp_path, urlopen_side_effect=bank.urlopen, extra_env=env)

    assert len(bank.items) == 1, "the dsh turn was not retained"
    item = bank.items[0]
    assert "source:dsh" in item["tags"]
    assert FACT in item["content"] and "Where does staging" in item["content"]

    # A later dsh turn (another session) asks; the bridge's UserPromptSubmit payload, through the adapter.
    prompt = {
        "session_id": "019a8d7f-next",
        "transcript_path": "",
        "cwd": "/var/lib/dsh/w",
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Which port does the staging database use?",
        "harness": "dsh",
    }
    out = _run_hook("recall", prompt, monkeypatch, tmp_path, urlopen_side_effect=bank.urlopen, extra_env=env)
    assert FACT in json.loads(out)["hookSpecificOutput"]["additionalContext"]


def test_a_claude_code_turn_is_untagged_and_never_reads_a_dsh_record__control(monkeypatch, tmp_path):
    """No `harness` field: the transcript is read as before, with no `source:dsh` tag, even when a dsh
    record exists under the same session id."""
    _record(tmp_path / "dsh", [HEADER, _user("dsh question"), _assistant("dsh answer")])
    transcript = make_transcript_file(
        tmp_path, [{"role": "user", "content": "cc question"}, {"role": "assistant", "content": "cc answer"}]
    )
    bank = FakeBank()
    hook = {"session_id": SID, "transcript_path": transcript, "cwd": "/home/user/p", "hook_event_name": "Stop"}

    _run_hook(
        "retain",
        hook,
        monkeypatch,
        tmp_path,
        urlopen_side_effect=bank.urlopen,
        extra_env={"DSH_HOME": str(tmp_path / "dsh")},
        extra_settings={"retainTags": ["cc"]},
    )

    assert len(bank.items) == 1
    assert bank.items[0]["tags"] == ["cc"]
    assert "cc answer" in bank.items[0]["content"] and "dsh answer" not in bank.items[0]["content"]


def test_a_notice_that_is_not_the_operator_is_not_retained_as_the_user(monkeypatch, tmp_path):
    """A `user/message` of another source kind (a plugin notice, a peer, a kickoff) is not the operator."""
    _record(
        tmp_path / "dsh",
        [
            HEADER,
            _user("OPERATOR ASKS", "user"),
            _assistant("first answer"),
            _user("PLUGIN NOTICE", "session-succession"),
            _assistant("second answer"),
        ],
    )
    bank = FakeBank()

    _run_hook(
        "retain",
        _dsh_stop(),
        monkeypatch,
        tmp_path,
        urlopen_side_effect=bank.urlopen,
        extra_env={"DSH_HOME": str(tmp_path / "dsh")},
    )

    assert len(bank.items) == 1
    assert "OPERATOR ASKS" in bank.items[0]["content"]
    assert "PLUGIN NOTICE" not in bank.items[0]["content"]


def test_an_id_that_is_not_one_path_segment_finds_nothing(monkeypatch, tmp_path):
    """The session id comes from the payload: a glob or a path in it must not reach another record."""
    _record(tmp_path / "dsh", [HEADER, _user("q"), _assistant("a")])
    bank = FakeBank()

    _run_hook(
        "retain",
        _dsh_stop("*"),
        monkeypatch,
        tmp_path,
        urlopen_side_effect=bank.urlopen,
        extra_env={"DSH_HOME": str(tmp_path / "dsh")},
    )

    assert bank.items == []
