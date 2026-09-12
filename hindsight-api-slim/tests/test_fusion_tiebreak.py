"""RRF fusion must impose a TOTAL order, not a stable-by-arrival one.

`reciprocal_rank_fusion` scores a document at rank r in any list as 1/(k + r), so when the
same rank appears in several lists the scores are EXACTLY equal — not nearly equal. The vector
arm runs once per fact_type, so in normal operation every rank is an n-way exact tie; measured
on a 339-row bank, 120 candidates carried only 40 distinct scores and all 120 sat in tie groups
of three.

Sorting those by score alone leaves Python's stable sort to break ties by insertion order, which
is the order the retrieval arms returned rows — the database's scan order. That is not stable
across two copies of the same data: a dump/restore rewrites physical order, so a restored copy
orders tied documents differently from its source while every row, score and version is
identical.

What that does to a recall depends on the budget filter behind it. Up to v0.9.1 the filter
stopped at the first fact that did not fit, so tie order changed HOW MANY facts came back. From
v0.9.2 `select_facts_within_budget` skips an oversized fact and keeps going, which mostly steadies
the count but not the contents: different facts still come back. The end-to-end test at the
bottom compares the facts themselves, so it keeps meaning something across that change.
"""

import itertools

from hindsight_api.engine.fact_budget import select_facts_within_budget
from hindsight_api.engine.search.fusion import reciprocal_rank_fusion
from hindsight_api.engine.search.types import RetrievalResult
from hindsight_api.engine.token_encoding import count_tokens


def _arm(ids: list[str]) -> list[RetrievalResult]:
    return [RetrievalResult(id=i, text=f"text-{i}", fact_type="world") for i in ids]


def _fuse_ids(arms: list[list[RetrievalResult]]) -> list[str]:
    return [c.retrieval.id for c in reciprocal_rank_fusion(arms)]


def test_tied_documents_get_a_total_order_independent_of_arrival():
    """Three arms, disjoint ids, identical ranks — every rank is a 3-way exact tie.

    Presenting the same arms in a different order is what a differently-ordered scan does. The
    fused order must not move. Without the id tiebreaker this fails: the fused order simply
    follows whichever arm was seen first.
    """
    a, b, c = _arm(["a1", "a2", "a3"]), _arm(["b1", "b2", "b3"]), _arm(["c1", "c2", "c3"])

    assert _fuse_ids([a, b, c]) == _fuse_ids([c, b, a]) == _fuse_ids([b, a, c])


def test_ties_really_are_exact():
    """Guards the premise: if these scores ever stop being equal the test above proves nothing."""
    merged = reciprocal_rank_fusion([_arm(["x"]), _arm(["y"]), _arm(["z"])])

    scores = {m.rrf_score for m in merged}
    assert len(scores) == 1, f"expected one shared score, got {scores}"


def test_tiebreak_is_by_id_ascending_and_score_still_dominates():
    """The tiebreaker must not disturb ranking: score first, id only within a tie."""
    # "low" appears in one arm at rank 1; "zzz" and "aaa" appear at rank 2 in two arms each.
    merged = reciprocal_rank_fusion([_arm(["low", "zzz"]), _arm(["low", "aaa"])])
    order = [m.retrieval.id for m in merged]

    assert order[0] == "low", f"a higher RRF score must outrank the tiebreaker: {order}"
    assert order[1:] == ["aaa", "zzz"], f"ties must order by id ascending: {order}"


def test_order_is_stable_across_repeated_calls():
    arms = [_arm(["q", "r"]), _arm(["r", "s"]), _arm(["s", "q"])]
    assert len({tuple(_fuse_ids(arms)) for _ in range(25)}) == 1


_RANKS = 30
_BUDGET_FRACTION = 0.4


def _words(arm: str, rank: int) -> str:
    # Deterministic, uneven lengths (1-23 words): tie order only changes what fits when the tied
    # facts differ in size.
    return " ".join(["memory"] * ((rank * 7919 + ord(arm) * 104729) % 23 + 1))


def test_budgeted_recall_returns_the_same_facts_whatever_order_the_arms_arrive():
    """End to end through the real fusion AND the real recall budget filter.

    This is what a caller sees, so it is the property that has to survive an upgrade: the same
    candidates must yield the same facts no matter which arm the database returned first. Every
    arrival order of three arms is tried. Compares the selected facts, not their count, because
    the v0.9.2 filter keeps counts steady while score-only fusion still changes the contents.
    """
    arms = {
        arm: [
            RetrievalResult(id=f"{arm}{rank:02d}", text=_words(arm, rank), fact_type="world")
            for rank in range(1, _RANKS + 1)
        ]
        for arm in "abc"
    }
    total = sum(count_tokens(r.text) for results in arms.values() for r in results)
    budget = int(total * _BUDGET_FRACTION)

    selections = {}
    for order in itertools.permutations("abc"):
        fused = reciprocal_rank_fusion([arms[a] for a in order])
        selections[order] = select_facts_within_budget(
            fact_ids_ordered=[c.retrieval.id for c in fused],
            text_by_id={c.retrieval.id: c.retrieval.text for c in fused},
            max_tokens=budget,
            count_tokens=count_tokens,
        ).ids

    sizes = sorted({len(ids) for ids in selections.values()})
    assert all(0 < n < _RANKS * 3 for n in sizes), (
        f"the budget must truncate for tie order to matter; selected {sizes} of {_RANKS * 3}"
    )
    distinct = {tuple(ids) for ids in selections.values()}
    assert len(distinct) == 1, (
        f"{len(distinct)} different fact selections across the 6 arm orders (sizes {sizes})"
    )
