"""Code chunks: split to the token budget on the server, stamp ``embed_fp``, reuse by fingerprint.

Real tokenizer; the vector store is the real ``CodeVectorStore`` on qdrant-client's local
engine (no server needed), so reuse is decided by what a stored point really carries.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tests.embedding_model_support import bge_m3_snapshot

REPO = Path(__file__).resolve().parents[1]
OWNER = "owner-1"
PROJECT = "/work/repo"


@pytest.fixture(scope="module", autouse=True)
def _snapshot():
    path = bge_m3_snapshot()
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", path)
    mp.setenv("HF_HUB_OFFLINE", "1")
    yield
    mp.undo()


def _count(text: str) -> int:
    from core.indexing.embedding_profile import count_tokens, get_profile
    return count_tokens([text], get_profile())[0]


def _big_code(min_tokens: int = 3000) -> str:
    source = (REPO / "app" / "services" / "code_kb" / "service.py").read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    code = ""
    for line in lines:
        code += line
        if _count(code) >= min_tokens:
            return code
    raise AssertionError("fixture source is too short")


def _record(code: str, *, path: str = "pkg/mod.py", start_line: int = 10, ordinal: int = 0, label: str = "big"):
    from core.code_kb.vector_records import CodeChunkVectorRecord
    return CodeChunkVectorRecord(
        rel_path=path, language="python", start_line=start_line,
        end_line=start_line + code.rstrip("\n").count("\n"), label=label,
        content=f"# path: {path}\n{code}", code=code, chunk_ordinal=ordinal)


def test_an_oversized_chunk_is_split_into_exact_line_pieces():
    from core.code_kb.vector_records import fit_records_to_budget
    from core.indexing.embedding_profile import get_profile

    code = _big_code()
    original = _record(code)
    assert _count(original.content) > get_profile().chunk_max_tokens

    pieces = fit_records_to_budget([original])

    assert len(pieces) >= 2
    assert "".join(p.code for p in pieces) == code                         # the lines partition the chunk
    assert all(p.content == f"# path: pkg/mod.py\n{p.code}" for p in pieces)
    assert all(_count(p.content) <= get_profile().chunk_max_tokens for p in pieces)
    assert all(p.rel_path == "pkg/mod.py" and p.language == "python" and p.label == "big" for p in pieces)
    # 1-based, inclusive line ranges that follow each other exactly
    assert pieces[0].start_line == 10
    for before, after in zip(pieces, pieces[1:]):
        assert after.start_line == before.end_line + 1
    assert pieces[-1].end_line == original.end_line
    source_lines = (original.start_line - 1) * ["\n"] + code.splitlines(keepends=True)
    for piece in pieces:
        assert "".join(source_lines[piece.start_line - 1:piece.end_line]) == piece.code


def test_pieces_get_distinct_point_ids_and_the_chunk_ordinal_counts_final_pieces():
    from core.code_kb.vector_records import code_chunk_point_id, fit_records_to_budget

    small = _record("def f():\n    return 1\n", path="a.py", ordinal=0, label="f", start_line=1)
    big = _record(_big_code(), path="a.py", ordinal=1, label="big", start_line=20)
    tail = _record("def g():\n    return 2\n", path="a.py", ordinal=2, label="g", start_line=500)

    out = fit_records_to_budget([small, big, tail])

    assert out[0] == small and out[-1].label == "g"
    assert [r.chunk_ordinal for r in out] == list(range(len(out)))        # per-file enumeration of the result
    assert len(out) > 3
    assert len({code_chunk_point_id("ns", r) for r in out}) == len(out)


def test_ordinals_restart_for_every_file_and_in_budget_chunks_are_untouched():
    from core.code_kb.vector_records import fit_records_to_budget

    a = [_record("x = 1\n", path="a.py", ordinal=0), _record("y = 2\n", path="a.py", ordinal=1, start_line=3)]
    assert fit_records_to_budget(a) == a


def test_the_payload_carries_the_embedding_fingerprint():
    from core.code_kb.vector_records import build_code_chunk_payload
    from core.indexing.embedding_profile import get_profile

    payload = build_code_chunk_payload(_record("x = 1\n"), scope={"owner_user_id": OWNER})
    assert payload["embed_fp"] == get_profile().fingerprint
    with pytest.raises(ValueError, match="embed_fp"):
        build_code_chunk_payload(_record("x = 1\n"), scope={"embed_fp": "forged"})


def test_the_service_splits_chunks_before_they_become_points():
    from app.services.code_kb.schemas import CodeChunkIn, CodeFileIn
    from app.services.code_kb.service import _chunk_points
    from core.indexing.embedding_profile import get_profile

    code = _big_code()
    files = [
        CodeFileIn(rel_path="pkg/mod.py", language="python", chunks=[
            CodeChunkIn(start_line=1, end_line=1, label="a", content="# path: pkg/mod.py\nx = 1\n", code="x = 1\n"),
            CodeChunkIn(start_line=10, end_line=10 + code.rstrip("\n").count("\n"), label="big",
                        content=f"# path: pkg/mod.py\n{code}", code=code),
        ]),
        CodeFileIn(rel_path="other.py", language="python", chunks=[
            CodeChunkIn(start_line=1, end_line=1, label="z", content="# path: other.py\nz = 1\n", code="z = 1\n")]),
    ]

    points = _chunk_points(files)

    assert [p.rel_path for p in points].count("other.py") == 1
    mod = [p for p in points if p.rel_path == "pkg/mod.py"]
    assert len(mod) >= 3 and [p.chunk_ordinal for p in mod] == list(range(len(mod)))
    assert all(_count(p.content) <= get_profile().chunk_max_tokens for p in points)
    assert [p.chunk_ordinal for p in points if p.rel_path == "other.py"] == [0]


def test_fan_out_hands_each_target_the_pieces_of_its_own_file():
    """A file that grew pieces must not shift the vectors of the files after it."""
    from app.services.code_kb.schemas import CodeChunkIn, CodeFileIn
    from app.services.code_kb.service import CodeKBService, _chunk_points

    code = _big_code()
    files = [
        CodeFileIn(rel_path="a.py", language="python", chunks=[
            CodeChunkIn(start_line=1, end_line=1 + code.rstrip("\n").count("\n"), label="big",
                        content=f"# path: a.py\n{code}", code=code)]),
        CodeFileIn(rel_path="b.py", language="python", chunks=[
            CodeChunkIn(start_line=1, end_line=1, label="b", content="# path: b.py\nb = 1\n", code="b = 1\n")]),
    ]
    items = _chunk_points(files)
    dense = [[float(i)] for i in range(len(items))]

    class Store:
        supports_sparse = False

        def __init__(self):
            self.upserts = []

        def list_owner_files(self, **_):
            return {("/proj-b", "a.py"), ("/proj-b", "b.py")}

        def list_point_ids(self, **_):
            return {1}

        def upsert(self, *, items, dense_vectors, **kw):
            self.upserts.append(([i.rel_path for i in items], [v[0] for v in dense_vectors]))
            return len(items)

        def delete_point_ids(self, ids):
            return 0

    store = Store()
    service = CodeKBService(embed_fn=lambda texts: [[0.0]] * len(texts), vector_store=store)
    service._fan_out_sync(
        owner_user_id=OWNER, project="/proj-a", host="h", files=files, items=items, dense=dense, sparse=None,
        owned_index={"/proj-a/a.py": [("/proj-b", "a.py")], "/proj-a/b.py": [("/proj-b", "b.py")]})

    by_path = {paths[0]: (paths, vectors) for paths, vectors in store.upserts}
    assert set(by_path) == {"a.py", "b.py"}
    a_paths, a_vectors = by_path["a.py"]
    assert set(a_paths) == {"a.py"} and len(a_vectors) == len(items) - 1
    assert by_path["b.py"] == (["b.py"], [dense[-1][0]])


# ------------------------------------------------------------------ reuse by fingerprint
@pytest.fixture()
def store():
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    from app.services.code_kb.vector_store import CodeVectorStore
    from core.storage.qdrant_store import QdrantStore

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="code_reuse_test",
        vectors_config={"dense": qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE)},
        sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
    )
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = "code_reuse_test"
    inner.client = client
    return CodeVectorStore(collection_name="code_reuse_test", store=inner)


def _seed(store, record):
    store.upsert(owner_user_id=OWNER, project=PROJECT, host="h", items=[record],
                 dense_vectors=[[1.0, 0.0, 0.0, 0.0]], sparse_vectors=[{"7": 0.5}])


def _reusable(store, record):
    return store.reusable_vectors(owner_user_id=OWNER, project=PROJECT, items=[record], require_sparse=True)[0]


def _set_fp(store, record, value):
    from qdrant_client.http import models as qmodels

    from app.services.code_kb.vector_store import point_id
    pid = point_id(OWNER, PROJECT, record.rel_path, record.start_line, record.label, record.chunk_ordinal)
    if value is None:
        store.store.client.delete_payload("code_reuse_test", keys=["embed_fp"], points=[pid])
    else:
        store.store.client.set_payload("code_reuse_test", payload={"embed_fp": value}, points=[pid])


def test_a_point_with_the_current_fingerprint_and_same_content_is_reused(store):
    record = _record("def f():\n    return 1\n")
    _seed(store, record)
    dense, sparse = _reusable(store, record)
    assert dense == [1.0, 0.0, 0.0, 0.0] and sparse == {"7": 0.5}


@pytest.mark.parametrize("stored", ["0123456789abcdef", None])
def test_a_point_with_another_or_no_fingerprint_is_not_reused(store, stored):
    record = _record("def f():\n    return 1\n")
    _seed(store, record)
    _set_fp(store, record, stored)                  # red on old code: the old point is reused
    assert _reusable(store, record) is None


def test_changed_content_is_still_not_reused(store):
    record = _record("def f():\n    return 1\n")
    _seed(store, record)
    edited = dataclasses.replace(record, code="def f():\n    return 2\n",
                                 content="# path: pkg/mod.py\ndef f():\n    return 2\n")
    assert _reusable(store, edited) is None
