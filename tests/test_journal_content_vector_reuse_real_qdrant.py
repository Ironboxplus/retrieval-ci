"""A re-indexed journal doc copies the stored vector of every chunk whose text is unchanged.

Real SQLite (file under ``tmp_path``) and a real Qdrant scratch collection. The embed function
is deterministic and records every text it is asked to embed; its vector carries the number of
the call that produced it, so a re-embedded chunk is told apart from a copied one by value.

Invariants under test (PLAN-2026-11-15-content-vector-reuse.md, section 1): no headless chunk
(every live SQL chunk has its point, no point survives for a superseded entry), no duplicate
chunk vector (exactly one point per live chunk), and a vector is copied only from a point with
the same content and the current ``embed_fp``.
"""
from __future__ import annotations

import hashlib
import time
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.services.journal.chunker import JournalTextChunk  # noqa: E402
from app.services.journal.repository import JournalRepository  # noqa: E402
from app.services.journal.schemas import JournalBulkDocItem, JournalRecordRequest  # noqa: E402
from app.services.journal.service import JournalService  # noqa: E402
from app.services.journal.vector_store import JournalVectorStore  # noqa: E402
from core.db.database import Base  # noqa: E402
from core.indexing.embedding_profile import current_embed_fp  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
OWNER = "journal-reuse-owner"
DIM = 8
TITLE = "notes/guide.md"


class RecordingEmbedder:
    """Deterministic embed function; ``calls`` holds the texts of every call, in order."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False
        self._number = 0  # survives ``calls.clear()``: a vector names the call that made it

    def __call__(self, texts, include_sparse):
        batch = list(texts)
        self.calls.append(batch)
        self._number += 1
        if self.fail:
            raise RuntimeError("embed endpoint down")
        number = self._number
        dense = []
        for text in batch:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            dense.append([digest[i] / 255.0 + 0.1 for i in range(DIM - 1)] + [float(number)])
        sparse = None
        if include_sparse:
            sparse = [{str(digest_index(text)): float(number)} for text in batch]
        return dense, sparse

    @property
    def asked(self) -> list[str]:
        return [text for batch in self.calls for text in batch]


def digest_index(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:6], 16)


@pytest.fixture()
def env(tmp_path: Path):
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=30)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    from config import settings

    name = f"journal_content_reuse_{uuid.uuid4().hex[:8]}_test"
    assert name not in {"journal_chunks", "code_chunks", "papers"}
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    if settings.embedding.supports_sparse:
        client.create_collection(
            collection_name=name, vectors_config={"dense": params},
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()})
    else:
        client.create_collection(collection_name=name, vectors_config=params)
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = name
    inner.client = client

    engine = create_engine(f"sqlite:///{(tmp_path / 'journal.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    embedder = RecordingEmbedder()
    service = JournalService(
        repository=JournalRepository(session_factory=session_factory),
        vector_store=JournalVectorStore(collection_name=name, store=inner),
        embed_fn=embedder)

    class Env:
        pass

    state = Env()
    state.service, state.embedder, state.client, state.name = service, embedder, client, name
    state.scope = str(tmp_path / "ws")
    try:
        yield state
    finally:
        for attempt in range(3):
            try:
                client.delete_collection(name)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    warnings.warn(f"could not drop scratch collection {name}: {exc}")
                time.sleep(0.5 * (attempt + 1))


def _body(sections: int, *, edit: dict[int, str] | None = None) -> str:
    edit = edit or {}
    parts = ["# Guide"]
    for i in range(sections):
        parts.append(f"## Section {i}\n\n{edit.get(i, f'original paragraph number {i} with its own words')}")
    return "\n\n".join(parts)


def _bulk(env, body: str, *, scope: str | None = None, title: str = TITLE):
    return env.service.index_docs_bulk(
        project_slug=scope or env.scope, owner_user_id=OWNER, client="mcp",
        files=[JournalBulkDocItem(
            title=title, body=body, content_hash=hashlib.sha256(body.encode()).hexdigest(),
            size_bytes=len(body.encode()))])


def _live(env, scope: str | None = None):
    return env.service.list_entries(
        project_slug=scope or env.scope, source_kind="doc", owner_user_id=OWNER).entries


def _live_one(env, scope: str | None = None) -> str:
    entries = _live(env, scope)
    assert len(entries) == 1, f"exactly one live version expected, got {len(entries)}"
    return entries[0].entry_id


def _sql_chunks(env, entry_id: str):
    return sorted(env.service.get_entry(entry_id, OWNER).chunks, key=lambda c: c.chunk_index)


def _points(env, entry_ids):
    """``{chunk_id: (payload, dense, sparse dict | None)}`` of every point of the entries."""
    found = {}
    offset = None
    flt = qmodels.Filter(must=[qmodels.FieldCondition(
        key="entry_id", match=qmodels.MatchAny(any=list(entry_ids)))])
    while True:
        points, offset = env.client.scroll(
            env.name, scroll_filter=flt, limit=256, offset=offset,
            with_payload=True, with_vectors=True)
        for point in points:
            vector = point.vector
            dense = vector["dense"] if isinstance(vector, dict) else vector
            sparse = vector.get("sparse") if isinstance(vector, dict) else None
            lexical = ({str(i): v for i, v in zip(sparse.indices, sparse.values)}
                       if sparse is not None else None)
            found[point.payload["chunk_id"]] = (point.payload, list(dense), lexical)
        if offset is None:
            return found


def _by_content(points):
    return {payload["content"]: (dense, sparse) for payload, dense, sparse in points.values()}


def _assert_complete(env, entry_id: str):
    """Every SQL chunk of the live entry has exactly one point and nothing else does."""
    sql = _sql_chunks(env, entry_id)
    points = _points(env, [entry_id])
    assert sorted(points) == sorted(c.chunk_id for c in sql), "SQL chunks and points differ"
    assert len(points) == len(sql)
    for payload, _dense, _sparse in points.values():
        assert payload["entry_id"] == entry_id


def _assert_retired(env, entry_id: str):
    entry = env.service.repository.get_entry(entry_id, include_deleted=True)
    assert entry is not None and entry.deleted_at is not None, "old revision must be superseded"
    assert _points(env, [entry_id]) == {}, "a superseded entry must keep no point"


def _assert_vectors_close(left, right):
    assert left == pytest.approx(right, abs=1e-5)


def test_bulk_append_embeds_only_the_new_chunk_and_copies_the_rest(env):
    _bulk(env, _body(4))
    old_id = _live_one(env)
    old_by_content = _by_content(_points(env, [old_id]))
    first_calls = len(env.embedder.calls)
    env.embedder.calls.clear()

    _bulk(env, _body(5))

    new_id = _live_one(env)
    assert new_id != old_id
    new_chunks = _sql_chunks(env, new_id)
    new_texts = [c.content for c in new_chunks]
    expected_misses = [t for t in new_texts if t not in old_by_content]
    assert len(expected_misses) == 1 and len(new_texts) > 1, first_calls
    assert env.embedder.asked == expected_misses, "only the new chunk text may be embedded"

    new_points = _points(env, [new_id])
    for payload, dense, sparse in new_points.values():
        if payload["content"] in old_by_content:
            old_dense, old_sparse = old_by_content[payload["content"]]
            _assert_vectors_close(dense, old_dense)
            assert sparse == old_sparse
    _assert_complete(env, new_id)
    _assert_retired(env, old_id)
    assert len(_points(env, [old_id, new_id])) == len(new_chunks)


def test_a_point_with_another_embed_fp_is_not_copied(env):
    _bulk(env, _body(4))
    old_id = _live_one(env)
    old_points = _points(env, [old_id])
    stale_chunk = next(c for c in _sql_chunks(env, old_id) if "number 1 " in c.content)
    stale_point = [p for p in old_points.values() if p[0]["chunk_id"] == stale_chunk.chunk_id][0]
    old_vector = stale_point[1]
    assert stale_point[0]["embed_fp"] == current_embed_fp()
    env.client.set_payload(
        env.name, payload={"embed_fp": "some-other-model"}, wait=True,
        points=qmodels.Filter(must=[qmodels.FieldCondition(
            key="chunk_id", match=qmodels.MatchValue(value=stale_chunk.chunk_id))]))
    env.embedder.calls.clear()

    _bulk(env, _body(5))

    new_id = _live_one(env)
    new_texts = [c.content for c in _sql_chunks(env, new_id)]
    assert sorted(env.embedder.asked) == sorted(
        [stale_chunk.content, next(t for t in new_texts if "number 4 " in t)]), (
        "the stale-model chunk must be embedded again, the others copied")
    new_by_content = _by_content(_points(env, [new_id]))
    stale_new_dense = new_by_content[stale_chunk.content][0]
    assert stale_new_dense != pytest.approx(old_vector, abs=1e-3), "stale vector was copied forward"
    for payload, _dense, _sparse in _points(env, [new_id]).values():
        assert payload["embed_fp"] == current_embed_fp()
    _assert_complete(env, new_id)
    _assert_retired(env, old_id)


def test_unchanged_reindex_embeds_nothing_and_keeps_one_live_version(env):
    _bulk(env, _body(3))
    old_id = _live_one(env)
    before = _points(env, [old_id])
    env.embedder.calls.clear()

    result = _bulk(env, _body(3))

    assert result["deduped"] == 1 and result["indexed"] == 0
    assert env.embedder.calls == []
    assert _live_one(env) == old_id
    assert _points(env, [old_id]).keys() == before.keys()
    _assert_complete(env, old_id)


def test_embed_failure_on_the_misses_keeps_the_old_revision_complete(env):
    _bulk(env, _body(4))
    old_id = _live_one(env)
    old_chunk_ids = sorted(_points(env, [old_id]))
    env.embedder.calls.clear()
    env.embedder.fail = True

    with pytest.raises(RuntimeError, match="embed endpoint down"):
        _bulk(env, _body(5))

    assert [len(batch) for batch in env.embedder.calls] == [1], (
        "only the one new chunk text may be sent to the embedder")
    pending = [e for e in env.service.pending_entries().entries]
    assert len(pending) == 1 and pending[0].entry_id != old_id
    # The failed entry stays pending in SQL (recoverable); the old revision is untouched.
    assert {e.entry_id for e in _live(env)} == {old_id, pending[0].entry_id}
    assert env.service.repository.get_entry(old_id).indexed_at is not None
    assert sorted(_points(env, [old_id])) == old_chunk_ids
    _assert_complete(env, old_id)
    assert _points(env, [pending[0].entry_id]) == {}, "the failed entry must leave no point"


def _record_doc(env, body: str, *, scope: str | None = None, title: str = TITLE):
    return env.service.record(JournalRecordRequest(
        project_slug=scope or env.scope, client="mcp", source_kind="doc", title=title,
        body=body, owner_user_id=OWNER, dedupe_policy="return_existing"))


def test_record_doc_append_embeds_only_the_new_chunk(env):
    first = _record_doc(env, _body(3))
    old_id = first.entry.entry_id
    old_by_content = _by_content(_points(env, [old_id]))
    env.embedder.calls.clear()

    second = _record_doc(env, _body(4))

    new_id = second.entry.entry_id
    assert second.indexed and new_id != old_id
    new_texts = [c.content for c in _sql_chunks(env, new_id)]
    assert env.embedder.asked == [t for t in new_texts if t not in old_by_content]
    assert len(env.embedder.asked) == 1
    for payload, dense, _sparse in _points(env, [new_id]).values():
        if payload["content"] in old_by_content:
            _assert_vectors_close(dense, old_by_content[payload["content"]][0])
    assert _live_one(env) == new_id
    _assert_complete(env, new_id)
    _assert_retired(env, old_id)


def test_record_memory_entry_never_borrows_vectors(env):
    request = dict(project_slug=env.scope, client="codex", source_kind="memory",
                   body="the same lesson text", owner_user_id=OWNER)
    first = env.service.record(JournalRecordRequest(**request))
    env.embedder.calls.clear()
    second = env.service.record(JournalRecordRequest(**{**request, "body": "the same lesson text!"}))
    assert second.entry.entry_id != first.entry.entry_id
    assert len(env.embedder.asked) == second.chunk_count, "memory entries embed every chunk"


def test_fan_out_to_a_child_scope_reuses_for_both_scopes(env):
    parent = env.scope
    child = str(Path(parent) / "docs")
    _record_doc(env, _body(3), scope=child, title="guide.md")
    _bulk(env, _body(3), scope=parent, title="docs/guide.md")
    parent_old, child_old = _live_one(env, parent), _live_one(env, child)
    old_by_content = _by_content(_points(env, [parent_old]))
    # Either scope's previous copy may lend the vector (they were embedded by different calls).
    old_vectors: dict[str, list] = {}
    for entry_id in (parent_old, child_old):
        for text, (dense, _sparse) in _by_content(_points(env, [entry_id])).items():
            old_vectors.setdefault(text, []).append(dense)
    env.embedder.calls.clear()

    result = _bulk(env, _body(4), scope=parent, title="docs/guide.md")

    assert result["files"] == 2
    parent_new, child_new = _live_one(env, parent), _live_one(env, child)
    assert len({parent_old, child_old, parent_new, child_new}) == 4
    new_texts = [c.content for c in _sql_chunks(env, parent_new)]
    assert env.embedder.asked == [t for t in new_texts if t not in old_by_content]
    assert len(env.embedder.asked) == 1, "one physical file is embedded once for both scopes"
    for entry_id in (parent_new, child_new):
        _assert_complete(env, entry_id)
        for payload, dense, _sparse in _points(env, [entry_id]).values():
            if payload["content"] in old_by_content:
                assert any(dense == pytest.approx(old, abs=1e-5)
                           for old in old_vectors[payload["content"]]), "vector was not copied"
    for entry_id in (parent_old, child_old):
        _assert_retired(env, entry_id)
    every = _points(env, [parent_old, child_old, parent_new, child_new])
    assert len(every) == len(_sql_chunks(env, parent_new)) + len(_sql_chunks(env, child_new))


def _chunk(entry_id: str, index: int, content: str) -> JournalTextChunk:
    return JournalTextChunk(
        chunk_id=f"{entry_id}_c{index}", entry_id=entry_id, chunk_index=index, level=2,
        content=content, heading=None, token_count=1)


def _seed_points(env, entry_id: str, texts, *, with_sparse: bool):
    chunks = [_chunk(entry_id, i, text) for i, text in enumerate(texts)]
    dense = [[float(i + 1)] + [0.5] * (DIM - 1) for i in range(len(texts))]
    sparse = [{str(i + 1): 1.0} for i in range(len(texts))] if with_sparse else None
    env.service.vector_store.upsert_chunks(
        chunks, project_id=1, project_slug="s", entry_id=entry_id, owner_user_id=OWNER,
        client="mcp", source_kind="doc", tags=[], dense_vectors=dense, sparse_vectors=sparse)
    return dense


def test_store_lookup_matches_content_pages_and_batches(env):
    store = env.service.vector_store
    texts = [f"chunk text {i}" for i in range(300)]
    dense = _seed_points(env, "e-many", texts, with_sparse=True)
    _seed_points(env, "e-other", ["unrelated text"], with_sparse=True)

    found = store.reusable_vectors_by_content(
        ["e-many"], texts + ["never stored"], require_sparse=True)

    assert sorted(found) == sorted(texts), "all 300 matches (past one scroll page and one retrieve batch)"
    first_dense, first_sparse = found["chunk text 0"]
    _assert_vectors_close(first_dense, [x / (sum(v * v for v in dense[0]) ** 0.5) for x in dense[0]])
    assert first_sparse == {"1": 1.0}
    assert store.reusable_vectors_by_content(["e-other"], ["chunk text 1"], require_sparse=False) == {}


def test_store_lookup_skips_points_without_sparse_when_required(env):
    store = env.service.vector_store
    if not store.store.supports_sparse:
        pytest.skip("embedding profile has no sparse vectors")
    _seed_points(env, "e-dense-only", ["dense only text"], with_sparse=False)
    assert store.reusable_vectors_by_content(
        ["e-dense-only"], ["dense only text"], require_sparse=True) == {}
    found = store.reusable_vectors_by_content(
        ["e-dense-only"], ["dense only text"], require_sparse=False)
    assert list(found) == ["dense only text"] and found["dense only text"][1] is None
