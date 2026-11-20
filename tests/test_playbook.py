"""TDD for the client-side ACE playbook store (SQLite, mcp_server/playbook.py).

Hermetic: a temp SQLite file + a deterministic async fake embedder (bag-of-words,
cosine ~ word overlap) so near-dup merge + ranking are exercised without the
backend or BGE-M3. Async driven via asyncio.run() (no pytest-asyncio needed).
"""
from __future__ import annotations

import asyncio
import hashlib
import math

from mcp_server.playbook import PlaybookStore

_DIM = 64


async def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * _DIM
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % _DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v))
        out.append([x / n for x in v] if n else v)
    return out


def _store(tmp_path):
    return PlaybookStore(db_path=str(tmp_path / "pb.db"), embed_fn=fake_embed,
                         dedup_threshold=0.9)


def test_add_new_bullet(tmp_path):
    s = _store(tmp_path)
    b = asyncio.run(s.add_bullet(
        "always use wait True on upsert to avoid dropped batches",
        section="failure_mode"))
    assert b.deduped is False
    assert b.helpful == 0
    assert b.section == "failure_mode"
    assert s.count() == 1


def test_near_duplicate_merges_and_reinforces(tmp_path):
    s = _store(tmp_path)
    b1 = asyncio.run(s.add_bullet("always use wait True on upsert to avoid dropped batches"))
    b2 = asyncio.run(s.add_bullet("always use wait True on upsert calls to avoid dropped batches"))
    assert b2.id == b1.id
    assert b2.deduped is True
    assert b2.helpful == 1
    assert s.count() == 1


def test_distinct_bullet_not_merged(tmp_path):
    s = _store(tmp_path)
    asyncio.run(s.add_bullet("always use wait True on upsert to avoid dropped batches"))
    asyncio.run(s.add_bullet("the frontend uses react and vite for rendering"))
    assert s.count() == 2


def test_search_ranks_relevant_first(tmp_path):
    s = _store(tmp_path)
    b1 = asyncio.run(s.add_bullet("always use wait True on upsert to avoid dropped batches"))
    asyncio.run(s.add_bullet("the frontend uses react and vite for rendering"))
    hits = asyncio.run(s.search("how should I call upsert", top_k=5))
    assert hits
    assert hits[0].id == b1.id
    assert hits[0].score is not None


def test_mark_helpful_persists(tmp_path):
    s = _store(tmp_path)
    b = asyncio.run(s.add_bullet("prefer hybrid retrieval for technical queries"))
    s.mark(b.id, helpful=True)
    again = asyncio.run(s.search("hybrid retrieval", top_k=1))
    assert again and again[0].helpful >= 1


def test_persists_across_instances(tmp_path):
    db = str(tmp_path / "pb.db")
    s1 = PlaybookStore(db_path=db, embed_fn=fake_embed)
    asyncio.run(s1.add_bullet("paper_id is the canonical key across the stack"))
    s2 = PlaybookStore(db_path=db, embed_fn=fake_embed)
    assert s2.count() == 1
