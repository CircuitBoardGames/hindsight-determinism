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

It surfaces as a changed RESULT COUNT rather than a reordering, because a caller downstream
truncates by token budget and stops at the first fact that does not fit. Different tie order,
different texts near the cut, different number of facts returned.
"""

from hindsight_api.engine.search.fusion import reciprocal_rank_fusion
from hindsight_api.engine.search.types import RetrievalResult


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
