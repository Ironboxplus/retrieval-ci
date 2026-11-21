"""Re-indexing must not re-embed chunks whose stored point already matches.

2026-10-13: the chunker fix (chonkie byte offsets, whole-file chunks for large sources)
means every project is re-indexed once, and the server re-embedded every chunk of every
submitted file. Most chunks come back byte-identical (all of a pure-ASCII file), so the
stored vectors of a point whose id and canonical payload both match are reused, and only
chunks whose text or position changed reach the embedding model.

Real Qdrant on loopback, private scratch collection, real (injected) deterministic embed
functions that record what they were asked to embed. Skipped unless Qdrant is reachable.
"""
from __future__ import annotations

import time
import warnings
from contextlib import contextmanager

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.services.index_catalog.repository import IndexCatalogRepository  # noqa: E402
from app.services.index_catalog.schemas import IndexFileRecord, IndexReportRequest  # noqa: E402
from app.services.index_catalog.service import IndexCatalogService  # noqa: E402
from core.db.database import Base  # noqa: E402
from app.services.code_kb.schemas import CodeChunkIn, CodeFileIn, CodeIndexRequest  # noqa: E402
from app.services.code_kb.service import CodeKBService  # noqa: E402
from app.services.code_kb.vector_store import CodeVectorStore, point_id  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402

# Loopback only and hardcoded, never from settings/QDRANT_URL.
LOCAL_URL = "http://127.0.0.1:6333"
COLLECTION = "code_chunks_vector_reuse_test"
OWNER = "reuse-owner"
PROJECT = "/repo"
DIM = 16


def _drop_collection(client: QdrantClient, attempts: int = 4):
    failure = None
    for attempt in range(attempts):
        try:
            client.delete_collection(COLLECTION)
            return None
        except Exception as exc:  # noqa: BLE001
            failure = exc
            if attempt + 1 < attempts:
                time.sleep(0.4 * (attempt + 1))
    return failure


def _scratch_catalog() -> IndexCatalogRepository:
    """The SQL catalog the store reads its file inventory from, on a private in-memory SQLite
    (never the global database)."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return IndexCatalogRepository(session_factory=session_factory)


def _report_to_catalog(store: CodeVectorStore, project: str, files: list[CodeFileIn]) -> None:
    """What the jobs layer does after ``CodeKBService.index`` (jobs.py ``_write_catalog``):
    report the scope's files to the catalog. Called directly, ``index`` never writes it, and
    the cross-scope fan-out finds the other scopes only through it."""
    IndexCatalogService(repository=store.catalog).report(
        IndexReportRequest(
            host="box", project=project, replace=True,
            files=[IndexFileRecord(rel_path=f.rel_path, kind="code", collection=COLLECTION,
                                   language=f.language, chunk_count=len(f.chunks))
                   for f in files]),
        owner_user_id=OWNER)


@pytest.fixture()
def store():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=30)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    _drop_collection(client)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)},
        sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
    )
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = COLLECTION
    inner.client = client
    if not inner.supports_sparse:
        pytest.skip("production stores dense+sparse; this configuration is dense-only")
    cvs = CodeVectorStore(collection_name=COLLECTION, store=inner, catalog=_scratch_catalog())
    cvs.ensure_search_payload_indexes()
    try:
        yield cvs
    finally:
        failure = _drop_collection(client, attempts=3)
        if failure is not None:
            warnings.warn(f"could not drop `{COLLECTION}` in teardown ({failure})", stacklevel=1)


def _service(store: CodeVectorStore, embedded: list[str]) -> CodeKBService:
    def embed(texts):
        embedded.extend(texts)
        rows = []
        for text in texts:
            row = [0.0] * DIM
            row[sum(text.encode("utf-8")) % DIM] = 1.0
            row[len(text) % DIM] += 0.5
            rows.append(row)
        return rows

    def sparse_embed(texts):
        return [{str(sum(text.encode("utf-8")) % 997): 1.0, str(len(text)): 0.25}
                for text in texts]

    return CodeKBService(embed_fn=embed, sparse_embed_fn=sparse_embed, vector_store=store)


def _file(path: str, pieces: list[tuple[int, str]]) -> CodeFileIn:
    return CodeFileIn(rel_path=path, language="python", chunks=[
        CodeChunkIn(start_line=start, end_line=start + code.count("\n"),
                    label=code.split("(")[0].replace("def ", "").strip(),
                    content=f"# path: {path}\n{code}", code=code)
        for start, code in pieces])


ASCII_FILE = _file("pkg/plain.py", [(1, "def alpha(x):\n    return x + 1\n"),
                                    (4, "def beta(y):\n    return y * 2\n")])
OLD_SHIFTED = _file("pkg/notes.py", [(1, "def gamma(z):\n    # copy → blit\n    return z\n"),
                                     (5, "lta(w):\n    return w - 1\n")])
FIXED = _file("pkg/notes.py", [(1, "def gamma(z):\n    # copy → blit\n    return z\n"),
                               (5, "def delta(w):\n    return w - 1\n")])


def _vectors(store: CodeVectorStore, file: CodeFileIn) -> dict:
    points = store.store.client.scroll(COLLECTION, limit=100, with_payload=True,
                                       with_vectors=True)[0]
    return {point.id: (point.vector["dense"], point.vector["sparse"]) for point in points
            if point.payload["path"] == file.rel_path and point.payload["project"] == PROJECT
            and point.payload["owner_user_id"] == OWNER}


def test_full_reindex_embeds_only_the_chunks_that_changed(store):
    embedded: list[str] = []
    service = _service(store, embedded)
    service.index(CodeIndexRequest(project=PROJECT, host="box", files=[ASCII_FILE, OLD_SHIFTED]),
                  owner_user_id=OWNER)
    assert len(embedded) == 4
    before = _vectors(store, ASCII_FILE)

    embedded.clear()
    response = service.index(
        CodeIndexRequest(project=PROJECT, host="box", files=[ASCII_FILE, FIXED]),
        owner_user_id=OWNER)

    assert embedded == [FIXED.chunks[1].content]
    assert response.indexed == 4
    assert _vectors(store, ASCII_FILE) == before
    assert store.list_paths(owner_user_id=OWNER, project=PROJECT) == {"pkg/plain.py", "pkg/notes.py"}
    stored_code = {point.payload["code"] for point in store.store.client.scroll(
        COLLECTION, limit=100, with_payload=True)[0]}
    assert "lta(w):\n    return w - 1\n" not in stored_code  # the mis-sliced chunk was pruned
    assert "def delta(w):\n    return w - 1\n" in stored_code


def test_incremental_resubmission_of_an_unchanged_file_embeds_nothing(store):
    embedded: list[str] = []
    service = _service(store, embedded)
    service.index(CodeIndexRequest(project=PROJECT, host="box", files=[ASCII_FILE]),
                  owner_user_id=OWNER)
    embedded.clear()

    service.index(CodeIndexRequest(project=PROJECT, host="box", files=[ASCII_FILE],
                                   incremental=True, manifest=["pkg/plain.py"]),
                  owner_user_id=OWNER)

    assert embedded == []
    assert len(_vectors(store, ASCII_FILE)) == 2


def test_a_point_from_another_project_or_owner_is_never_reused(store):
    embedded: list[str] = []
    service = _service(store, embedded)
    service.index(CodeIndexRequest(project=PROJECT, host="box", files=[ASCII_FILE]),
                  owner_user_id=OWNER)
    embedded.clear()

    service.index(CodeIndexRequest(project="/other", host="box", files=[ASCII_FILE]),
                  owner_user_id=OWNER)
    service.index(CodeIndexRequest(project=PROJECT, host="box", files=[ASCII_FILE]),
                  owner_user_id="someone-else")

    assert len(embedded) == 4


# ---------------------------------------------------------------------------------------
# Content reuse (2026-11-15): a line shift changes the point id and the line payload but not
# the embedded text, so the stored vector of a shifted chunk must be copied into the NEW
# point instead of being re-embedded. Reuse only ever copies; the old points are still
# pruned after the upsert, so there is never a headless or a duplicate chunk vector.
# ---------------------------------------------------------------------------------------

PIECE_A = "def alpha(x):\n    return x + 1\n"
PIECE_B = "def beta(y):\n    return y * 2\n"
PIECE_C = "def gamma(z):\n    return z - 3\n"
PIECE_NEW = "def zero():\n    return 0\n"


def _shift_file(path: str = "pkg/shift.py") -> tuple[CodeFileIn, CodeFileIn]:
    """(before, after): after has one new chunk on top, so every old chunk moves 3 lines."""
    before = _file(path, [(1, PIECE_A), (4, PIECE_B), (7, PIECE_C)])
    after = _file(path, [(1, PIECE_NEW), (4, PIECE_A), (7, PIECE_B), (10, PIECE_C)])
    return before, after


def _expected_ids(file: CodeFileIn, project: str = PROJECT, rel_path: str | None = None) -> set[int]:
    return {point_id(OWNER, project, rel_path or file.rel_path, chunk.start_line, chunk.label, ordinal)
            for ordinal, chunk in enumerate(file.chunks)}


def _points(store: CodeVectorStore, path: str, project: str = PROJECT) -> dict:
    """id -> point (payload + dense/sparse) of one file of one scope, read straight from Qdrant."""
    points = store.store.client.scroll(COLLECTION, limit=1000, with_payload=True,
                                       with_vectors=True)[0]
    return {point.id: point for point in points
            if point.payload["path"] == path and point.payload["project"] == project
            and point.payload["owner_user_id"] == OWNER}


def _vector_by_content(points: dict) -> dict:
    return {point.payload["content"]: (point.vector["dense"], point.vector["sparse"])
            for point in points.values()}


def _incremental(file: CodeFileIn, project: str = PROJECT) -> CodeIndexRequest:
    return CodeIndexRequest(project=project, host="box", files=[file], incremental=True,
                            manifest=[file.rel_path])


def test_incremental_line_shift_embeds_only_the_inserted_chunk(store):
    embedded: list[str] = []
    service = _service(store, embedded)
    before, after = _shift_file()
    service.index(CodeIndexRequest(project=PROJECT, host="box", files=[before]),
                  owner_user_id=OWNER)
    old_points = _points(store, before.rel_path)
    old_vectors = _vector_by_content(old_points)
    assert len(old_points) == 3
    embedded.clear()

    response = service.index(_incremental(after), owner_user_id=OWNER)

    assert embedded == [after.chunks[0].content]
    assert response.indexed == 4
    new_points = _points(store, after.rel_path)
    assert set(new_points) == _expected_ids(after)               # exactly the new ids ...
    assert not set(new_points) & set(old_points)                 # ... every old id pruned
    assert len(new_points) == len(after.chunks)                  # one point per chunk, none headless
    new_vectors = _vector_by_content(new_points)
    for chunk in before.chunks:                                  # the copy equals the old vector
        assert new_vectors[chunk.content] == old_vectors[chunk.content]
    assert {point.payload["start_line"] for point in new_points.values()} == {1, 4, 7, 10}


def test_full_index_line_shift_embeds_only_the_inserted_chunk(store):
    embedded: list[str] = []
    service = _service(store, embedded)
    before, after = _shift_file()
    steady = ASCII_FILE
    service.index(CodeIndexRequest(project=PROJECT, host="box", files=[before, steady]),
                  owner_user_id=OWNER)
    old_points = _points(store, before.rel_path)
    old_vectors = _vector_by_content(old_points)
    steady_before = _vectors(store, steady)
    embedded.clear()

    response = service.index(
        CodeIndexRequest(project=PROJECT, host="box", files=[after, steady]), owner_user_id=OWNER)

    assert embedded == [after.chunks[0].content]
    assert response.indexed == 6
    new_points = _points(store, after.rel_path)
    assert set(new_points) == _expected_ids(after)
    assert not set(new_points) & set(old_points)
    new_vectors = _vector_by_content(new_points)
    for chunk in before.chunks:
        assert new_vectors[chunk.content] == old_vectors[chunk.content]
    assert _vectors(store, steady) == steady_before
    everything = store.list_point_ids_for_paths(owner_user_id=OWNER, project=PROJECT)
    assert everything == _expected_ids(after) | _expected_ids(steady)   # no stale id anywhere


def test_a_stored_vector_under_another_embed_fp_is_never_reused_by_content(store):
    embedded: list[str] = []
    service = _service(store, embedded)
    before, after = _shift_file()
    service.index(CodeIndexRequest(project=PROJECT, host="box", files=[before]),
                  owner_user_id=OWNER)
    old_points = _points(store, before.rel_path)
    foreign_id = next(identity for identity, point in old_points.items()
                      if point.payload["content"] == before.chunks[1].content)
    client = store.store.client
    client.set_payload(COLLECTION, payload={"embed_fp": "some-older-model"},
                       points=[foreign_id], wait=True)
    poison = [0.0] * DIM
    poison[DIM - 1] = 9.0
    client.update_vectors(COLLECTION, points=[qmodels.PointVectors(
        id=foreign_id, vector={"dense": poison})], wait=True)
    embedded.clear()

    service.index(_incremental(after), owner_user_id=OWNER)

    # New chunk + the one whose stored vector belongs to another model; the other two copy.
    assert sorted(embedded) == sorted([after.chunks[0].content, before.chunks[1].content])
    new_vectors = _vector_by_content(_points(store, after.rel_path))
    assert new_vectors[before.chunks[1].content][0] != poison
    for point in _points(store, after.rel_path).values():
        assert point.payload["embed_fp"] != "some-older-model"


def test_a_stored_point_without_a_sparse_vector_is_not_reused_by_content(store):
    embedded: list[str] = []
    service = _service(store, embedded)
    before, after = _shift_file()
    service.index(CodeIndexRequest(project=PROJECT, host="box", files=[before]),
                  owner_user_id=OWNER)
    old_points = _points(store, before.rel_path)
    lexical_less = next(identity for identity, point in old_points.items()
                        if point.payload["content"] == before.chunks[2].content)
    store.store.client.delete_vectors(COLLECTION, vectors=["sparse"], points=[lexical_less],
                                      wait=True)
    embedded.clear()

    service.index(_incremental(after), owner_user_id=OWNER)

    assert sorted(embedded) == sorted([after.chunks[0].content, before.chunks[2].content])
    for point in _points(store, after.rel_path).values():
        assert point.vector["sparse"] is not None


def test_identical_chunks_in_one_file_each_get_a_vector_and_a_point(store):
    embedded: list[str] = []
    service = _service(store, embedded)
    twin = "def twin(x):\n    return x\n"
    before = _file("pkg/twins.py", [(1, twin), (4, twin), (7, PIECE_A)])
    after = _file("pkg/twins.py", [(1, PIECE_NEW), (5, twin), (8, twin), (11, PIECE_A)])

    service.index(CodeIndexRequest(project=PROJECT, host="box", files=[before]),
                  owner_user_id=OWNER)

    assert sorted(embedded) == sorted([before.chunks[0].content, before.chunks[2].content])
    first = _points(store, before.rel_path)
    assert len(first) == 3 and set(first) == _expected_ids(before)   # two distinct twin points
    twin_points = [point for point in first.values() if point.payload["content"] == before.chunks[0].content]
    assert len(twin_points) == 2
    assert twin_points[0].vector["dense"] == twin_points[1].vector["dense"]
    embedded.clear()

    service.index(_incremental(after), owner_user_id=OWNER)

    assert embedded == [after.chunks[0].content]
    second = _points(store, after.rel_path)
    assert set(second) == _expected_ids(after) and len(second) == 4
    assert not set(second) & set(first)
    twins_after = [point for point in second.values() if point.payload["content"] == before.chunks[0].content]
    assert len(twins_after) == 2                                   # both twins kept a point
    for point in twins_after:
        assert point.vector["dense"] == twin_points[0].vector["dense"]
        assert point.vector["sparse"] == twin_points[0].vector["sparse"]


def test_fan_out_target_scope_gets_exactly_the_shifted_ids_without_re_embedding(store):
    embedded: list[str] = []
    service = _service(store, embedded)
    child = "/repo/pkg"                       # the same physical file as /repo + pkg/shift.py
    before, after = _shift_file("pkg/shift.py")
    child_rel = "shift.py"
    child_file = _file(child_rel, [(1, "def stale():\n    pass\n")])
    service.index(CodeIndexRequest(project=child, host="box", files=[child_file]),
                  owner_user_id=OWNER)
    _report_to_catalog(store, child, [child_file])
    service.index(CodeIndexRequest(project=PROJECT, host="box", files=[before]),
                  owner_user_id=OWNER)
    _report_to_catalog(store, PROJECT, [before])
    assert set(_points(store, child_rel, child)) == _expected_ids(before, child, child_rel)
    old_child = _points(store, child_rel, child)
    embedded.clear()

    service.index(_incremental(after), owner_user_id=OWNER)

    assert embedded == [after.chunks[0].content]
    target = _points(store, child_rel, child)
    assert set(target) == _expected_ids(after, child, child_rel)   # exactly the new ids
    assert not set(target) & set(old_child)                        # nothing stale left behind
    assert len(target) == len(after.chunks)
    assert store.list_point_ids_for_paths(owner_user_id=OWNER, project=child) == set(target)
    assert set(_points(store, after.rel_path)) == _expected_ids(after)
