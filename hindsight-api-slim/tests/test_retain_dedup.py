"""Write-time dedup (engine/retain/dedup.py): parsing, invalidate-not-delete, fallback, scope.

No database and no real LLM: the engine, the store and the judge are fakes, so each test pins
one behaviour of the dedup module itself.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.retain import dedup
from hindsight_api.engine.retain.dedup import Decision, Fact


def _d(day: int) -> datetime:
    return datetime(2026, 9, day, tzinfo=UTC)


NEW = [
    Fact("n-a", "Hindsight listens on port 9999", "world", occurred_start=_d(20)),
    Fact("n-b", "The operator prefers short PR bodies", "world", occurred_start=_d(20)),
    Fact("n-c", "The deploy ran on Tuesday", "experience", occurred_start=_d(20), occurred_end=_d(20)),
    Fact("n-d", "Nothing like this was ever stored", "world"),
]
CANDS = {
    "n-a": [Fact("old-port", "Hindsight listens on port 8888", "world", occurred_start=_d(1))],
    "n-b": [Fact("old-pref", "The operator likes PR bodies to be brief", "world")],
    "n-c": [
        Fact("old-dep", "The deploy started Monday", "experience", occurred_start=_d(14), occurred_end=_d(14)),
        Fact("unrelated", "Lunch was pizza", "experience"),
    ],
}


def _keys() -> dict[str, str]:
    return dedup.build_prompt(NEW, CANDS)[1]


def _key(uid: str) -> str:
    return next(k for k, v in _keys().items() if v == uid)


def _answer(*items) -> str:
    return json.dumps(list(items))


# ---------------------------------------------------------------- parsing


def test_parse_maps_every_action_to_real_ids():
    raw = (
        "<think>C9 [looks] like</think>```json\n"
        + _answer(
            {"id": _key("n-a"), "action": "update", "target_ids": [_key("old-port")]},
            {"id": _key("n-b"), "action": "skip", "target_ids": [_key("old-pref")]},
            {
                "id": _key("n-c"),
                "action": "merge",
                "target_ids": [_key("old-dep")],
                "merged_content": "The deploy started Monday and ran into Tuesday",
                "merged_type": "experience",
            },
            {"id": _key("n-d"), "action": "store"},
        )
        + "\n```"
    )
    got = dedup.parse_decisions(raw, _keys(), CANDS)
    assert got["n-a"] == Decision("update", ["old-port"], None, None)
    assert got["n-b"] == Decision("skip", ["old-pref"], None, None)
    assert got["n-c"] == Decision("merge", ["old-dep"], "The deploy started Monday and ran into Tuesday", "experience")
    assert got["n-d"] == Decision()


def test_parse_never_retires_a_fact_the_model_was_not_shown():
    raw = _answer(
        # old-dep is a real candidate, but of n-c, not n-a; C99 does not exist at all
        {"id": _key("n-a"), "action": "update", "target_ids": [_key("old-dep"), "C99"]},
        {"id": _key("n-d"), "action": "skip"},  # n-d had no candidates to be a duplicate of
        {"id": _key("n-c"), "action": "merge", "target_ids": [_key("old-dep")]},  # merge without text
        {"id": "N42", "action": "skip"},  # not a new fact
        {"id": _key("n-b"), "action": "delete", "target_ids": [_key("old-pref")]},  # not an action
    )
    got = dedup.parse_decisions(raw, _keys(), CANDS)
    assert all(d == Decision() for d in got.values()), got


@pytest.mark.parametrize("raw", ["", "not json at all", "[{", '{"id": "N1"}', "[1, 2, 3]", None])
def test_malformed_output_stores_everything(raw):
    got = dedup.parse_decisions(raw, _keys(), CANDS)
    assert set(got) == {f.id for f in NEW}
    assert all(d == Decision() for d in got.values()), got


@pytest.mark.asyncio
async def test_llm_failure_stores_everything():
    llm = SimpleNamespace(call=AsyncMock(side_effect=RuntimeError("quota exhausted")))
    got = await dedup.judge(llm, NEW, CANDS)
    assert all(d == Decision() for d in got.values())
    llm.call.assert_awaited_once()  # one batched call, not one per fact


# ---------------------------------------------------------------- applying


class FakeEngine:
    """Records curation calls; deleting is a test failure, not a code path."""

    def __init__(self):
        self.update_memory_unit = AsyncMock(return_value={})

    async def delete_memory_unit(self, *a, **k):  # pragma: no cover - must never run
        raise AssertionError("dedup deleted a memory")

    def calls(self):
        return [(c.args[1], c.kwargs) for c in self.update_memory_unit.await_args_list]


@pytest.mark.asyncio
async def test_update_and_merge_invalidate_targets_never_delete():
    eng = FakeEngine()
    decisions = {
        "n-a": Decision("update", ["old-port"]),
        "n-c": Decision("merge", ["old-dep"], "The deploy started Monday and ran into Tuesday"),
    }
    counts = await dedup.apply_decisions(eng, "bank", NEW, CANDS, decisions, request_context="rc")
    calls = eng.calls()
    invalidated = {uid: kw["reason"] for uid, kw in calls if kw.get("state") == "invalidated"}
    assert invalidated == {
        "old-port": "retain dedup: update, superseded by n-a",
        "old-dep": "retain dedup: merge, superseded by n-c",
    }
    # the survivor of a merge carries the merged text and the union of the occurrences
    (merge_edit,) = [kw for uid, kw in calls if uid == "n-c" and "text" in kw]
    assert merge_edit["text"] == "The deploy started Monday and ran into Tuesday"
    assert merge_edit["occurred_start"] == _d(14).isoformat()
    assert "occurred_end" not in merge_edit  # n-c already ends last
    # the survivor is written before anything it supersedes is retired
    assert [uid for uid, _ in calls].index("n-c") < [uid for uid, _ in calls].index("old-dep")
    assert not any(uid == "n-a" for uid, _ in calls)  # an update with no new text leaves the new fact alone
    assert counts["update"] == 1 and counts["merge"] == 1


@pytest.mark.asyncio
async def test_skip_invalidates_the_new_fact_as_a_duplicate():
    eng = FakeEngine()
    await dedup.apply_decisions(eng, "bank", NEW, CANDS, {"n-b": Decision("skip", ["old-pref"])}, "rc")
    assert eng.calls() == [
        (
            "n-b",
            {"state": "invalidated", "reason": "retain dedup: skip, duplicate of old-pref", "request_context": "rc"},
        )
    ]


@pytest.mark.asyncio
async def test_a_failed_curation_leaves_the_fact_stored():
    eng = FakeEngine()
    eng.update_memory_unit.side_effect = RuntimeError("row moved")
    counts = await dedup.apply_decisions(eng, "bank", NEW, CANDS, {"n-a": Decision("update", ["old-port"])}, "rc")
    assert counts["update"] == 0 and counts["store"] == len(NEW)


# ---------------------------------------------------------------- candidate scope


class TagScopedStore:
    """recall_unified that honours a tag filter the way the real store does."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def recall_unified(self, *, tags=None, **kw):
        self.calls.append({"tags": tags, **kw})
        hits = [r for r in self.rows if not tags or set(tags) & set(r.tags)]
        return {"world": SimpleNamespace(semantic=hits), "experience": SimpleNamespace(semantic=[])}


@pytest.mark.asyncio
async def test_candidates_come_from_the_whole_bank_not_the_session():
    # The stored twin was retained by ANOTHER session; the new fact is tagged with this one.
    twin = SimpleNamespace(
        id="old-port",
        text="Hindsight listens on port 8888",
        fact_type="world",
        similarity=0.93,
        tags=["session-1"],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=_d(1),
    )
    far = SimpleNamespace(
        id="far",
        text="Unrelated",
        fact_type="world",
        similarity=0.40,
        tags=["session-1"],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
    )
    self_hit = SimpleNamespace(
        id="n-a",
        text="Hindsight listens on port 9999",
        fact_type="world",
        similarity=1.0,
        tags=["session-2"],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
    )
    store = TagScopedStore([twin, far, self_hit])
    eng = SimpleNamespace(_get_backend=AsyncMock(return_value="pool"), embeddings=None)
    new = [Fact("n-a", "Hindsight listens on port 9999", "world")]
    new[0].tags = ["session-2"]
    with (
        patch("hindsight_api.engine.memories.get_memories", return_value=store),
        patch("hindsight_api.engine.retain.embedding_utils.generate_embeddings_batch", AsyncMock(return_value=[[0.1]])),
    ):
        got = await dedup.find_candidates(eng, "bank", new, min_similarity=0.85)
    assert [c.id for c in got["n-a"]] == ["old-port"]  # cross-session twin kept; far and self dropped
    assert store.calls[0]["fact_types"] == ["world", "experience"]


# ---------------------------------------------------------------- concurrency
#
# Measured 2026-10-06 on the omp bank: dedup was 49% of retain wall time (median 17 s), spent in two
# serial loops -- one recall per new fact (~0.19 s each) and one judge call per 30 facts (~10 s
# each). Both loops' inputs are fixed before either runs, so they can overlap; applying stays serial.


class _InFlight:
    """Counts how many awaits of one kind overlap."""

    def __init__(self):
        self.now = 0
        self.peak = 0

    async def hold(self):
        self.now += 1
        self.peak = max(self.peak, self.now)
        await asyncio.sleep(0.01)
        self.now -= 1


class OverlapStore:
    """recall_unified that answers each fact with its own twin, and records the overlap."""

    def __init__(self):
        self.flight = _InFlight()

    async def recall_unified(self, *, query_text, **kw):
        await self.flight.hold()
        twin = SimpleNamespace(
            id="old-" + query_text,
            text=query_text,
            fact_type="world",
            similarity=0.95,
            occurred_start=None,
            occurred_end=None,
            mentioned_at=None,
        )
        return {"world": SimpleNamespace(semantic=[twin]), "experience": SimpleNamespace(semantic=[])}


@pytest.mark.asyncio
async def test_candidate_searches_overlap_and_stay_bounded():
    store = OverlapStore()
    eng = SimpleNamespace(_get_backend=AsyncMock(return_value="pool"), embeddings=None)
    new = [Fact(f"n{i}", f"fact {i}", "world") for i in range(3 * dedup.SEARCH_CONCURRENCY)]
    with (
        patch("hindsight_api.engine.memories.get_memories", return_value=store),
        patch(
            "hindsight_api.engine.retain.embedding_utils.generate_embeddings_batch",
            AsyncMock(return_value=[[0.1]] * len(new)),
        ),
    ):
        got = await dedup.find_candidates(eng, "bank", new, min_similarity=0.85)
    assert store.flight.peak > 1, "candidate searches ran one at a time"
    assert store.flight.peak <= dedup.SEARCH_CONCURRENCY, "searches outran the bound on pool connections"
    # each fact keeps its OWN candidates: no cross-wiring between overlapping searches
    assert {fid: [c.id for c in cs] for fid, cs in got.items()} == {f.id: ["old-" + f.text] for f in new}


@pytest.mark.asyncio
async def test_judge_calls_overlap_and_decisions_apply_in_fact_order():
    new = [Fact(f"n{i:03}", f"fact {i}", "world") for i in range(2 * dedup.MAX_FACTS_PER_CALL + 5)]
    cands = {f.id: [Fact("old-" + f.id, "twin", "world")] for f in new}
    flight = _InFlight()

    async def fake_judge(llm, part, candidates):
        await flight.hold()
        return {f.id: Decision("skip", ["old-" + f.id]) for f in part}

    applied = []

    async def fake_apply(engine, bank_id, part, candidates, decisions, rc):
        applied.extend(f.id for f in part)
        assert all(decisions[f.id].action == "skip" for f in part)
        return {**dict.fromkeys(dedup.ACTIONS, 0), "skip": len(part)}

    @asynccontextmanager
    async def fake_acquire(backend):
        yield None

    stored = [SimpleNamespace(unit_id=f.id, text=f.text, fact_type="world") for f in new]
    memories = SimpleNamespace(get_memories=AsyncMock(return_value=stored))
    eng = SimpleNamespace(_get_backend=AsyncMock(return_value="pool"))
    with (
        patch("hindsight_api.engine.memories.get_memories", return_value=memories),
        patch("hindsight_api.engine.db_utils.acquire_with_retry", fake_acquire),
        patch.object(dedup, "find_candidates", AsyncMock(return_value=cands)),
        patch.object(dedup, "judge", fake_judge),
        patch.object(dedup, "apply_decisions", fake_apply),
    ):
        counts = await dedup.dedup_retained(eng, "bank", [f.id for f in new], None, 0.85, "rc")
    assert flight.peak == 3, "the three judge slices did not run side by side (peak %d)" % flight.peak
    assert applied == [f.id for f in new], "decisions were not applied in fact order"
    assert counts["skip"] == len(new)


# ---------------------------------------------------------------- setting


def test_setting_is_off_by_default_and_per_bank(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.delenv("HINDSIGHT_API_RETAIN_DEDUP", raising=False)
    assert HindsightConfig.from_env().retain_dedup is False
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_DEDUP", "true")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_DEDUP_MIN_SIMILARITY", "0.9")
    cfg = HindsightConfig.from_env()
    assert cfg.retain_dedup is True and cfg.retain_dedup_min_similarity == 0.9
    assert {"retain_dedup", "retain_dedup_min_similarity"} <= HindsightConfig.get_configurable_fields()
