"""The BM25 arm's LIMIT must cut a TOTAL order: score, then id (hub#1029).

The arm is `ORDER BY <score> LIMIT <n>` in SQL. BM25 scores tie often and exactly -- ts_rank_cd is
coarse -- so when the LIMIT falls inside a block of equal scores, WHICH tied rows survive is the
database's scan order. A dump/restore rewrites that order, and a restored copy then hands fusion a
different candidate SET. `_order_arm` cannot repair it: it re-sorts in Python after retrieval, and a
row the SQL already cut is gone.

Measured on 0.10.2, 2026-10-01: two restores of one production dump, the second CLUSTERed, gave
23/24 identical answers; the diverging query's bm25/world and bm25/observation arms held different
sets of 300 at a tie of 0.9 and 0.5. These are SQL-shape assertions, like
test_bm25_min_score_pushdown.py, so they run with no text-search extension installed.
"""

import re

import pytest

from hindsight_api.config import VALID_TEXT_SEARCH_EXTENSIONS
from hindsight_api.engine.sql.postgresql import PostgreSQLDialect

ARM_KWARGS = dict(
    table="memory_units",
    cols="id, text",
    fact_type="world",
    bank_id_param="$2",
    limit_param="$3",
    text_param="$4",
)


@pytest.mark.parametrize("extension", VALID_TEXT_SEARCH_EXTENSIONS)
@pytest.mark.parametrize("floor", [0.0, 0.3])
def test_bm25_limit_cuts_a_total_order(extension, floor):
    sql = PostgreSQLDialect().build_bm25_arm(**ARM_KWARGS, text_search_extension=extension, bm25_min_score=floor)
    order_by = re.search(r"ORDER BY (.*?) LIMIT \$3", sql, re.S)
    assert order_by, sql
    assert order_by.group(1).rstrip().endswith(", id"), order_by.group(1)
