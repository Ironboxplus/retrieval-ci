"""TDD for the server-side ACE playbook service (grow-and-refine, scope, search)."""
from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.playbook.schemas import (
    PlaybookRecordRequest,
    PlaybookSearchRequest,
)
from app.services.playbook.service import PlaybookService
from core.db.database import Base

_DIM = 64


def _bow_embed(texts):
    """Deterministic bag-of-words embedding: shared tokens -> high cosine."""
    out = []
    for t in texts:
        v = [0.0] * _DIM
        for w in t.lower().replace("=", " ").replace("_", " ").split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % _DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v))
        out.append([x / n for x in v] if n else v)
    return out


def _service(embed_fn=_bow_embed, dedup_threshold=0.92):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = session_local()
        try:
            yield session
        finally:
            session.close()

    return PlaybookService(embed_fn=embed_fn, dedup_threshold=dedup_threshold,
                           session_factory=session_factory)


def _rec(project="retrival", content="lesson", section="insights", domain=None):
    return PlaybookRecordRequest(project_slug=project, content=content,
                                 section=section, domain=domain)


def test_record_persists_bullet():
    svc = _service()
    b = svc.record(_rec(content="Always wait=True on Qdrant upsert"), owner_user_id="u1")
    assert b.id > 0
    assert b.helpful == 0
    assert not b.deduped
    listed = svc.list_bullets(owner_user_id="u1", project_slug="retrival")
    assert listed.total == 1
    assert listed.bullets[0].content == "Always wait=True on Qdrant upsert"


def test_record_identical_content_dedupes_and_reinforces():
    svc = _service()
    svc.record(_rec(content="Always wait=True on Qdrant upsert"), owner_user_id="u1")
    again = svc.record(_rec(content="Always wait=True on Qdrant upsert"), owner_user_id="u1")
    assert again.deduped
    assert again.helpful == 1
    assert svc.list_bullets(owner_user_id="u1", project_slug="retrival").total == 1


def test_record_near_duplicate_merges_grow_and_refine():
    # constant-vector embed => any two contents are cosine 1.0 => always merge
    svc = _service(embed_fn=lambda texts: [[1.0, 0.0]] * len(texts))
    svc.record(_rec(content="upsert with wait true"), owner_user_id="u1")
    b2 = svc.record(_rec(content="totally different words here"), owner_user_id="u1")
    assert b2.deduped
    assert svc.list_bullets(owner_user_id="u1", project_slug="retrival").total == 1


def test_scope_isolated_by_owner_and_project():
    svc = _service()
    svc.record(_rec(project="a", content="lesson one"), owner_user_id="u1")
    svc.record(_rec(project="b", content="lesson two"), owner_user_id="u1")
    svc.record(_rec(project="a", content="lesson three"), owner_user_id="u2")
    assert svc.list_bullets(owner_user_id="u1", project_slug="a").total == 1
    assert svc.list_bullets(owner_user_id="u1").total == 2   # both projects
    assert svc.list_bullets(owner_user_id="u2", project_slug="a").total == 1


def test_search_ranks_by_similarity_within_scope():
    svc = _service()
    svc.record(_rec(content="qdrant upsert wait flag reliability"), owner_user_id="u1")
    svc.record(_rec(content="react frontend antd table layout"), owner_user_id="u1")
    res = svc.search(PlaybookSearchRequest(project_slug="retrival",
                                           query="qdrant upsert reliability", top_k=5),
                     owner_user_id="u1")
    assert res.total >= 1
    assert "qdrant" in res.bullets[0].content
    assert res.bullets[0].score is not None


def test_search_does_not_leak_across_owner():
    svc = _service()
    svc.record(_rec(content="secret lesson for u1"), owner_user_id="u1")
    res = svc.search(PlaybookSearchRequest(project_slug="retrival",
                                           query="secret lesson", top_k=5),
                     owner_user_id="u2")
    assert res.total == 0


def test_mark_helpful_and_harmful():
    svc = _service()
    b = svc.record(_rec(content="a lesson"), owner_user_id="u1")
    svc.mark(b.id, helpful=True, owner_user_id="u1")
    marked = svc.mark(b.id, helpful=False, owner_user_id="u1")
    assert marked.helpful == 1
    assert marked.harmful == 1
    # cross-owner mark is a no-op (returns None)
    assert svc.mark(b.id, helpful=True, owner_user_id="intruder") is None


def test_delete_soft_removes_then_resurrects_on_rerecord():
    svc = _service()
    b = svc.record(_rec(content="ephemeral lesson"), owner_user_id="u1")
    assert svc.delete(b.id, owner_user_id="u1").deleted
    assert svc.list_bullets(owner_user_id="u1").total == 0
    # same content again -> resurrect the soft-deleted row, not a duplicate
    again = svc.record(_rec(content="ephemeral lesson"), owner_user_id="u1")
    assert again.deduped
    assert svc.list_bullets(owner_user_id="u1").total == 1
    # cross-owner delete is a no-op
    assert not svc.delete(again.id, owner_user_id="intruder").deleted


def test_admin_list_spans_users():
    svc = _service()
    svc.record(_rec(content="l1"), owner_user_id="u1")
    svc.record(_rec(content="l2"), owner_user_id="u2")
    assert svc.admin_list().total == 2
    assert svc.admin_list(owner_user_id="u1").total == 1
