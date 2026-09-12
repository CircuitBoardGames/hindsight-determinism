"""Each retrieval arm must reach fusion in a TOTAL order: score descending, then id (hub#1029).

`reciprocal_rank_fusion` scores by rank, i.e. by list position. Retrieval scores tie exactly and often
-- BM25 `ts_rank_cd` is coarse (0.2, 0.3, 0.6), graph activations saturate (tanh 1), and near-duplicate
facts share a cosine -- so an arm sorted by score alone keeps the database's scan order among equals.
Two copies of the same data then hand fusion different ranks, the fused order differs, and the
token-budget cut returns different facts: measured on a physically reordered restore, 20 of 24
queries.
"""

import itertools

from hindsight_api.engine.memory_engine import _order_arm
from hindsight_api.engine.search.types import RetrievalResult

SCORE_ATTRS = ("similarity", "bm25_score", "activation", "combined_score")


def _r(rid: str, score_attr: str, score):
    r = RetrievalResult(id=rid, text=f"text-{rid}", fact_type="world")
    setattr(r, score_attr, score)
    return r


def test_tied_scores_order_by_id_whatever_the_arrival_order():
    """Every arrival order of the same results -- what a differently ordered scan produces -- must
    come out identical. Score-only sorting fails this: ties keep the arrival order."""
    base = [("d", 0.2), ("a", 0.6), ("c", 0.2), ("b", 0.6), ("e", 0.2)]
    for attr in SCORE_ATTRS:
        seen = set()
        for perm in itertools.permutations(base):
            rs = [_r(i, attr, s) for i, s in perm]
            _order_arm(rs, attr)
            seen.add(tuple(r.id for r in rs))
        assert seen == {("a", "b", "c", "d", "e")}, (attr, seen)


def test_score_still_dominates_the_id():
    rs = [_r("a", "bm25_score", 0.2), _r("z", "bm25_score", 0.9)]
    _order_arm(rs, "bm25_score")
    assert [r.id for r in rs] == ["z", "a"]


def test_missing_or_none_score_counts_as_zero():
    rs = [_r("b", "activation", None), _r("a", "activation", 0.1)]
    _order_arm(rs, "activation")
    assert [r.id for r in rs] == ["a", "b"]
