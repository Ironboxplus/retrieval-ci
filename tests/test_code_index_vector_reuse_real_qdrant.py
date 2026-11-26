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

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.services.code_kb.schemas import CodeChunkIn, CodeFileIn, CodeIndexRequest  # noqa: E402
from app.services.code_kb.service import CodeKBService  # noqa: E402
from app.services.code_kb.vector_store import CodeVectorStore  # noqa: E402
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
    cvs = CodeVectorStore(collection_name=COLLECTION, store=inner)
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
