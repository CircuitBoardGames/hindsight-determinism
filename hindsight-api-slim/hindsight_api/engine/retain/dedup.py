"""Write-time dedup of world/experience facts: store / skip / update / merge.

Ported from TencentDB-Agent-Memory (MIT), ``MemoryCore/src/core/record/l1-dedup.ts`` and
``prompts/l1-dedup.ts``. Consolidation only deduplicates observations, so raw facts used to
pile up as paraphrases, and nothing let a newer fact retire the one it contradicts (hub#1734).

Runs after a retain batch has stored its facts (``HINDSIGHT_API_RETAIN_DEDUP``, off by default):

1. For each new world/experience fact, the nearest stored world/experience facts in the WHOLE
   bank are found through the store's own recall (``recall_unified``, semantic arm, no tag
   filter). Tencent filters candidates by session; that would miss exactly the cross-session
   duplicates this exists for. Facts from this same batch are never candidates.
2. ONE LLM call judges every new fact that has a candidate, over one deduplicated candidate
   pool. A batch with no candidate above ``retain_dedup_min_similarity`` makes no call.
3. Decisions are applied through :meth:`MemoryEngine.update_memory_unit` — the curation path —
   so a retired fact is INVALIDATED (moved to ``invalidated_memory_units``, revertible), never
   deleted, and its ``invalidation_reason`` names the fact that superseded it:

   - store:  nothing.
   - skip:   the NEW fact is invalidated as a duplicate of the candidate(s).
   - update: the targets are invalidated; the new fact stays (text replaced by
             ``merged_content`` when the model carried a still-true detail over).
   - merge:  the new fact becomes ``merged_content``, its occurred range widened to the union
             of the targets', and the targets are invalidated.

Any failure — embedding, recall, the LLM, unparseable output — leaves every fact stored, as if
dedup were off. Only a target the model was actually shown for that fact can be retired.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..llm_wrapper import parse_llm_json, sanitize_llm_output

if TYPE_CHECKING:
    from ..memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

DEDUP_FACT_TYPES = ("world", "experience")
ACTIONS = ("store", "skip", "update", "merge")
TOP_K = 5
# ponytail: a huge document retain is judged in slices of this many facts (one call each)
# rather than one prompt that outgrows the model; raise it if calls per retain matter more.
MAX_FACTS_PER_CALL = 30
# ponytail: candidate searches in flight at once, each holding a pool connection for one recall. A
# retain's own pipeline shares the pool (100 by default), so this stays small; raise it if dedup's
# search phase still shows in [retain-timing] and the pool has headroom.
SEARCH_CONCURRENCY = 8

SYSTEM_PROMPT = """You deduplicate a long-term memory bank. Each NEW fact was just extracted. Compare it with the EXISTING facts listed as its candidates and choose exactly one action:

- "store": it is new information, or you are not sure. It is kept as is.
- "skip": a candidate already says everything the new fact says (a paraphrase, or a vaguer version). The new fact is dropped.
- "update": the new fact and the target(s) describe the same fact or state, and the new fact supersedes them: it is newer, corrects them, or is strictly more specific. The targets are retired and the new fact stays. Give merged_content only to carry over a detail from the targets that is still true.
- "merge": the new fact and the target(s) are complementary, non-contradictory pieces of the same fact or event. Write ONE fact (merged_content) that holds all of their information without repetition. The targets are retired.

Rules:
- target_ids may only name candidates listed for that new fact. Name every candidate the decision covers.
- Different facts about the same subject are not duplicates: another value, another time, another event, another person -> "store".
- A changed state (a new value replacing an old one) is "update", never "merge": the result must not present the old value as current.
- When in doubt, "store". A wrong skip, update or merge loses information; a duplicate only costs space.
- merged_content: one self-contained statement in the same language and style as the facts, keeping their dates.
- merged_type: "world" (facts about the world or other people) or "experience" (the narrator's own actions and experiences). Omit it to keep the new fact's type.

Output ONLY a JSON array with one object per new fact, nothing else:
[{"id": "N1", "action": "store|skip|update|merge", "target_ids": ["C1"], "merged_content": "...", "merged_type": "world"}]"""


@dataclass
class Fact:
    """A fact as the judge sees it: a new one, or a stored candidate."""

    id: str
    text: str
    fact_type: str
    date: str | None = None  # ISO date of occurrence, else of mention
    occurred_start: Any = None
    occurred_end: Any = None


@dataclass
class Decision:
    action: str = "store"
    target_ids: list[str] = field(default_factory=list)  # real unit ids, never the prompt keys
    merged_text: str | None = None
    merged_type: str | None = None


def _fact_date(m: Any) -> str | None:
    d = getattr(m, "occurred_start", None) or getattr(m, "mentioned_at", None)
    return d.date().isoformat() if d is not None else None


def to_fact(m: Any) -> Fact:
    """Build a Fact from a StoredMemory (``unit_id``) or a RetrievalResult (``id``)."""
    return Fact(
        id=str(getattr(m, "unit_id", None) or m.id),
        text=m.text,
        fact_type=m.fact_type,
        date=_fact_date(m),
        occurred_start=getattr(m, "occurred_start", None),
        occurred_end=getattr(m, "occurred_end", None),
    )


def build_prompt(new_facts: list[Fact], candidates: dict[str, list[Fact]]) -> tuple[str, dict[str, str]]:
    """Render the user prompt over one deduplicated pool. Returns it and {prompt key: unit id}.

    Facts are addressed by short keys (N1, C1) rather than UUIDs: fewer tokens, and a key the
    model invents maps to nothing instead of to some other fact.
    """
    keys: dict[str, str] = {}
    key_of: dict[str, str] = {}
    pool: list[str] = []
    for f in new_facts:
        for c in candidates.get(f.id, []):
            if c.id not in key_of:
                key_of[c.id] = f"C{len(key_of) + 1}"
                keys[key_of[c.id]] = c.id
                pool.append(f"{key_of[c.id]} [{c.fact_type}, {c.date or 'undated'}] {c.text}")
    news: list[str] = []
    for i, f in enumerate(new_facts, 1):
        keys[f"N{i}"] = f.id
        cands = ", ".join(key_of[c.id] for c in candidates.get(f.id, []))
        news.append(f"N{i} [{f.fact_type}, {f.date or 'undated'}] {f.text}\n   candidates: {cands or 'none'}")
    prompt = "EXISTING facts:\n" + "\n".join(pool) + "\n\nNEW facts:\n" + "\n".join(news)
    return prompt, keys


def parse_decisions(raw: Any, keys: dict[str, str], candidates: dict[str, list[Fact]]) -> dict[str, Decision]:
    """Parse the judge's answer into {new unit id: Decision}; anything unusable means store.

    A missing, unknown or malformed entry is "store". Targets outside the candidates that fact
    was shown are dropped, and an update/merge left with no target becomes "store", so the model
    can never retire a fact it did not see. A skip needs at least one candidate to be a
    duplicate of.
    """
    new_ids = [uid for k, uid in keys.items() if k.startswith("N")]
    decisions = {uid: Decision() for uid in new_ids}
    try:
        text = raw if isinstance(raw, str) else str(raw)
        items = parse_llm_json(re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S))
    except Exception as e:  # unparseable: everything is stored
        logger.warning(f"[RETAIN_DEDUP] unparseable judge output, storing all: {e}")
        return decisions
    if isinstance(items, dict):  # tolerate {"decisions": [...]}
        items = next((v for v in items.values() if isinstance(v, list)), [])
    if not isinstance(items, list):
        return decisions
    for item in items:
        if not isinstance(item, dict):
            continue
        uid = keys.get(str(item.get("id", "")))
        action = item.get("action")
        if uid not in decisions or action not in ACTIONS:
            continue
        allowed = {c.id for c in candidates.get(uid, [])}
        raw_targets = item.get("target_ids") or []
        targets = [keys.get(str(t)) for t in raw_targets] if isinstance(raw_targets, list) else []
        targets = list(dict.fromkeys(t for t in targets if t in allowed))
        merged = item.get("merged_content")
        merged = (sanitize_llm_output(merged) or "").strip() if isinstance(merged, str) else ""
        mtype = item.get("merged_type") if item.get("merged_type") in DEDUP_FACT_TYPES else None
        if action in ("update", "merge") and not targets:
            continue
        if action == "skip" and not allowed:
            continue
        if action == "merge" and not merged:
            continue
        decisions[uid] = Decision(action, targets, merged or None, mtype)
    return decisions


async def find_candidates(
    engine: MemoryEngine, bank_id: str, new_facts: list[Fact], min_similarity: float
) -> dict[str, list[Fact]]:
    """Nearest stored world/experience facts per new fact, bank-wide (no tag/session filter)."""
    from ..memories import get_memories
    from . import embedding_utils

    store = get_memories()
    pool = await engine._get_backend()
    batch_ids = {f.id for f in new_facts}
    embs = await embedding_utils.generate_embeddings_batch(engine.embeddings, [f.text for f in new_facts])
    gate = asyncio.Semaphore(SEARCH_CONCURRENCY)

    async def search(f: Fact, emb: Any) -> list[Fact]:
        async with gate:
            arms = await store.recall_unified(
                conn=pool,
                bank_id=bank_id,
                fact_types=list(DEDUP_FACT_TYPES),
                query_embedding=str(emb),
                query_text=f.text,
                limit=TOP_K + len(batch_ids),
                tags=None,  # the whole bank: cross-session duplicates are the point
                enable_text_search=False,
                enable_graph=False,
            )
        hits = [r for a in arms.values() for r in a.semantic]
        hits = [r for r in hits if str(r.id) not in batch_ids and (r.similarity or 0.0) >= min_similarity]
        hits.sort(key=lambda r: -(r.similarity or 0.0))
        return [to_fact(r) for r in hits[:TOP_K]]

    # Independent reads, so they overlap: one at a time was ~0.19 s per fact on the omp bank.
    found = await asyncio.gather(*(search(f, emb) for f, emb in zip(new_facts, embs, strict=True)))
    return {f.id: hits for f, hits in zip(new_facts, found, strict=True) if hits}


async def judge(llm_config: Any, new_facts: list[Fact], candidates: dict[str, list[Fact]]) -> dict[str, Decision]:
    """One LLM call over the facts that have candidates. Any failure means store."""
    prompt, keys = build_prompt(new_facts, candidates)
    try:
        result = await llm_config.call(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            scope="retain_dedup",
            max_retries=2,
        )
    except Exception as e:
        logger.warning(f"[RETAIN_DEDUP] judge call failed, storing all: {e}")
        return {f.id: Decision() for f in new_facts}
    return parse_decisions(result.content, keys, candidates)


def _iso(d: Any) -> str | None:
    return d.isoformat() if d is not None else None


async def apply_decisions(
    engine: MemoryEngine,
    bank_id: str,
    new_facts: list[Fact],
    candidates: dict[str, list[Fact]],
    decisions: dict[str, Decision],
    request_context: Any,
) -> dict[str, int]:
    """Apply decisions through the curation path. Returns a count per action applied."""
    counts = dict.fromkeys(ACTIONS, 0)
    retired: set[str] = set()
    for f in new_facts:
        d = decisions.get(f.id) or Decision()
        targets = [t for t in d.target_ids if t not in retired]
        try:
            if d.action == "skip":
                of = targets or [c.id for c in candidates.get(f.id, [])][:1]
                await engine.update_memory_unit(
                    bank_id,
                    f.id,
                    state="invalidated",
                    reason=f"retain dedup: skip, duplicate of {', '.join(of)}",
                    request_context=request_context,
                )
                retired.add(f.id)
            elif d.action in ("update", "merge") and targets:
                edit: dict[str, Any] = {}
                if d.merged_text and d.merged_text != f.text:
                    edit["text"] = d.merged_text
                if d.merged_type and d.merged_type != f.fact_type:
                    edit["new_fact_type"] = d.merged_type
                if d.action == "merge":  # the merged fact spans every piece's occurrence
                    by_id = {c.id: c for c in candidates.get(f.id, [])}
                    parts = [f, *(by_id[t] for t in targets)]
                    starts = [p.occurred_start for p in parts if p.occurred_start is not None]
                    ends = [p.occurred_end for p in parts if p.occurred_end is not None]
                    if starts and min(starts) != f.occurred_start:
                        edit["occurred_start"] = _iso(min(starts))
                    if ends and max(ends) != f.occurred_end:
                        edit["occurred_end"] = _iso(max(ends))
                # Survivor first: if this edit fails, nothing has been retired for it.
                if edit:
                    await engine.update_memory_unit(bank_id, f.id, request_context=request_context, **edit)
                for t in targets:
                    await engine.update_memory_unit(
                        bank_id,
                        t,
                        state="invalidated",
                        reason=f"retain dedup: {d.action}, superseded by {f.id}",
                        request_context=request_context,
                    )
                    retired.add(t)
            else:
                d = Decision()
            counts[d.action] += 1
        except Exception as e:
            logger.warning(f"[RETAIN_DEDUP] applying {d.action} to {f.id} failed, left as stored: {e}")
            counts["store"] += 1
    return counts


async def dedup_retained(
    engine: MemoryEngine,
    bank_id: str,
    unit_ids: list[str],
    llm_config: Any,
    min_similarity: float,
    request_context: Any,
) -> dict[str, int]:
    """Dedup the facts one retain batch just stored. Never raises: on failure, all are stored."""
    from ..db_utils import acquire_with_retry
    from ..memories import get_memories
    from ..memory_engine import fq_table

    counts = dict.fromkeys(ACTIONS, 0)
    try:
        backend = await engine._get_backend()
        async with acquire_with_retry(backend) as conn:
            stored = await get_memories().get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)
        new_facts = [to_fact(m) for m in stored if m.fact_type in DEDUP_FACT_TYPES]
        if not new_facts:
            return counts
        candidates = await find_candidates(engine, bank_id, new_facts, min_similarity)
        judged = [f for f in new_facts if f.id in candidates]
        counts["store"] = len(new_facts) - len(judged)
        parts = [judged[i : i + MAX_FACTS_PER_CALL] for i in range(0, len(judged), MAX_FACTS_PER_CALL)]
        # Every slice's inputs (its facts and their candidates) are fixed before any judging, so the
        # calls run side by side. Applying stays in fact order: a later decision may name a target an
        # earlier one already retired.
        verdicts = await asyncio.gather(*(judge(llm_config, part, candidates) for part in parts))
        for part, decisions in zip(parts, verdicts, strict=True):
            for action, n in (
                await apply_decisions(engine, bank_id, part, candidates, decisions, request_context)
            ).items():
                counts[action] += n
    except Exception as e:
        logger.warning(f"[RETAIN_DEDUP] bank={bank_id}: dedup failed, all facts stored: {e}")
        return counts
    if any(counts[a] for a in ("skip", "update", "merge")):
        logger.info(f"[RETAIN_DEDUP] bank={bank_id}: {counts}")
    return counts
