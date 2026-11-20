"""Public imports reuse only the caller's exact published chunk vectors."""

from dataclasses import replace

import pytest
from qdrant_client import QdrantClient, models

from app.services.code_kb.vector_store import CodeVectorStore
from app.services.public_code.index_service import PublicCodeIndexService
from app.services.public_code.vector_store import PublicCodeVectorStore
from core.storage.qdrant_store import QdrantStore
from tests.test_public_code_index_service import (
    Reader, Repository, catalog, chunk_source, manifest,
)
from core.code_kb.vector_records import CodeChunkVectorRecord


OWNER = "import-admin"
PROJECT = "c:/repos/ffmpeg"


class SparseTestStore(QdrantStore):
    @property
    def supports_sparse(self):
        return True


def record():
    chunk = chunk_source(manifest().files[0].path, "", language="c")[0]
    return CodeChunkVectorRecord(chunk.path, "c", chunk.start_line, chunk.end_line,
                                 chunk.label, chunk.content, chunk.code)


@pytest.fixture
def stores():
    # Real Qdrant engine, test-private collections; no production connection or GPU.
    client = QdrantClient(":memory:")
    wrappers = []
    for name in ("private_import_reuse_test", "public_import_reuse_test"):
        client.delete_collection(name)
        client.create_collection(
            name, vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.DOT)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        inner = SparseTestStore.__new__(SparseTestStore)
        inner.client, inner.collection_name = client, name
        wrappers.append(inner)
    yield (CodeVectorStore(collection_name=wrappers[0].collection_name, store=wrappers[0]),
           PublicCodeVectorStore(collection_name=wrappers[1].collection_name, store=wrappers[1]))
    client.close()


def seed(private, item=None, *, owner=OWNER, project=PROJECT, sparse=True):
    private.upsert(owner_user_id=owner, project=project, host="operator",
                   items=[item or record()], dense_vectors=[[0.25, 0.75]],
                   sparse_vectors=[{"7": 0.6}] if sparse else None)


def index(private, public, embed):
    return PublicCodeIndexService(
        source_registry=catalog(), repository=Repository(), source_reader=Reader(),
        vector_store=public, chunk_fn=chunk_source, embed_fn=embed,
        private_vector_store_factory=lambda: private,
    ).index_source("ffmpeg", refresh_mirror=False,
                   private_owner_user_id=OWNER, private_project=PROJECT)


def test_matching_private_index_avoids_embedding_and_keeps_private_data(stores):
    private, public = stores
    seed(private)

    def forbidden_embed(texts):
        raise AssertionError("matching private chunks must not be embedded again")

    result = index(private, public, forbidden_embed)
    assert result.chunk_count == result.reused_chunk_count == 1
    assert private.count(owner_user_id=OWNER, project=PROJECT) == 1
    points, _ = public.store.client.scroll(public.collection_name, with_vectors=True)
    assert points[0].vector["dense"] == [0.25, 0.75]
    assert points[0].vector["sparse"].indices == [7]
    assert points[0].payload["content"] == record().content
    assert not {"owner_user_id", "project", "host"}.intersection(points[0].payload)


@pytest.mark.parametrize("reason", ["absent", "owner", "project", "changed", "sparse"])
def test_unusable_private_index_embeds_normally(stores, reason):
    private, public = stores
    if reason != "absent":
        seed(private, replace(record(), content="unpublished change") if reason == "changed" else None,
             owner="another-user" if reason == "owner" else OWNER,
             project="c:/another/repo" if reason == "project" else PROJECT,
             sparse=reason != "sparse")
    calls = []

    def embed(texts):
        calls.extend(texts)
        return [[0.9, 0.1] for _ in texts], [{"8": 0.8} for _ in texts]

    result = index(private, public, embed)
    assert result.reused_chunk_count == 0
    assert calls == [record().content]
    assert public.count_revision("ffmpeg", "commit-a") == 1


def test_private_lookup_failure_is_not_reported_as_missing_index(stores, monkeypatch):
    private, public = stores
    seed(private)

    def failed_lookup(**kwargs):
        raise RuntimeError("private Qdrant unavailable")

    monkeypatch.setattr(private, "reusable_vectors", failed_lookup)
    with pytest.raises(RuntimeError, match="private Qdrant unavailable"):
        index(private, public, lambda _: (_ for _ in ()).throw(AssertionError("no fallback")))
    assert public.count_revision("ffmpeg", "commit-a") == 0


def test_mixed_batch_reuses_only_matching_ordinals_and_embeds_the_rest(stores):
    private, public = stores
    seed(private)
    first = chunk_source(record().rel_path, "", language="c")[0]
    second = replace(first, content="another chunk on the same line", code="second")
    calls = []

    def embed(texts):
        calls.extend(texts)
        return [[0.9, 0.1] for _ in texts], [{"8": 0.8} for _ in texts]

    result = PublicCodeIndexService(
        source_registry=catalog(), repository=Repository(), source_reader=Reader(),
        vector_store=public, chunk_fn=lambda *args, **kwargs: [first, second],
        embed_fn=embed, private_vector_store_factory=lambda: private,
    ).index_source("ffmpeg", refresh_mirror=False,
                   private_owner_user_id=OWNER, private_project=PROJECT)
    assert result.chunk_count == 2
    assert result.reused_chunk_count == 1
    assert calls == [second.content]
    assert public.count_revision("ffmpeg", "commit-a") == 2
