"""Public source indexing promotes only a complete validated revision."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.public_code.index_service import PublicCodeIndexService
from app.services.public_code.source_catalog import PublicCodeSource, PublicCodeSourceCatalog
from core.code_kb.chunker import CodeChunk
from core.indexing.revision_manifest import RevisionFile, RevisionManifest


def catalog():
    return PublicCodeSourceCatalog(
        catalog_id="fixture",
        sources=(
            PublicCodeSource(
                source_id="ffmpeg",
                domain="multimedia",
                relative_path="multimedia/ffmpeg",
                upstream_url="https://example.invalid/ffmpeg.git",
            ),
        ),
    )


def manifest():
    return RevisionManifest(
        revision_id="commit-a",
        files=(
            RevisionFile(
                path="libavcodec/sample.c",
                content_hash="blob-a",
                language="c",
                content_role="source",
                size_bytes=32,
            ),
        ),
    )


class Reader:
    def __init__(self, *, fail_read=False):
        self.fail_read = fail_read
        self.manifest_calls = 0
        self.calls = []

    def refresh_source(self, source):
        self.calls.append(("refresh", source.source_id))

    def build_manifest(self, _source, *, revision=None):
        assert revision is None
        self.manifest_calls += 1
        self.calls.append(("manifest", _source.source_id))
        return manifest()

    def read_source(self, _source, _file):
        if self.fail_read:
            raise RuntimeError("read failed")
        return "int answer(void) { return 42; }\n"


@dataclass
class SourceState:
    active_revision: str | None = None


class Repository:
    def __init__(self):
        self.active_revision = None
        self.recorded = []
        self.begun = []
        self.validated = []
        self.promoted = []
        self.failed = []
        self.cleaned = []

    def reconcile_catalog(self, _catalog):
        raise AssertionError("index_source must not reconcile the bootstrap catalog")

    def get_source(self, _source_id):
        return SourceState(self.active_revision)

    def record_candidate_revision(self, source_id, revision_manifest, *, chunk_count):
        self.recorded.append((source_id, revision_manifest.revision_id, chunk_count))

    def begin_candidate_revision(self, source_id, revision_manifest):
        self.begun.append((source_id, revision_manifest.revision_id))

    def validate_candidate_revision(self, source_id, revision, *, vector_count):
        self.validated.append((source_id, revision, vector_count))

    def promote_revision(self, source_id, revision):
        previous_revision = self.active_revision
        self.promoted.append((source_id, revision))
        self.active_revision = revision
        return previous_revision

    def mark_revision_failed(self, source_id, revision, *, error_message, cleanup_pending=True):
        self.failed.append((source_id, revision, error_message, cleanup_pending))

    def mark_revision_cleaned(self, source_id, revision):
        self.cleaned.append((source_id, revision))


class VectorStore:
    def __init__(self, *, count_offset=0, cleanup_failure_revision=None):
        self.count_offset = count_offset
        self.cleanup_failure_revision = cleanup_failure_revision
        self.items = []
        self.deleted = []

    def delete_revision(self, source_id, revision):
        if revision == self.cleanup_failure_revision:
            raise RuntimeError("cleanup unavailable")
        self.deleted.append((source_id, revision))

    def upsert(self, *, source_id, revision, domain, items, dense_vectors, sparse_vectors):
        self.items.extend(items)
        return len(items)

    def count_revision(self, source_id, revision):
        return len(self.items) + self.count_offset


def chunk_source(path, _source, *, language):
    return [CodeChunk(
        path=path,
        start_line=1,
        end_line=1,
        content=f"# path: {path}\nint answer(void) {{ return 42; }}",
        code="int answer(void) { return 42; }",
        label="answer",
    )]


def embed(texts):
    return [[1.0, float(index)] for index, _ in enumerate(texts)], [{"1": 1.0} for _ in texts]


def service(*, reader=None, vector_store=None, repository=None):
    return PublicCodeIndexService(
        source_registry=catalog(),
        repository=repository or Repository(),
        source_reader=reader or Reader(),
        vector_store=vector_store or VectorStore(),
        chunk_fn=chunk_source,
        embed_fn=embed,
        batch_size=2,
    )


def test_complete_revision_is_counted_validated_then_promoted():
    repository = Repository()
    vector_store = VectorStore()
    reader = Reader()

    progress = []
    result = service(
        reader=reader,
        repository=repository,
        vector_store=vector_store,
    ).index_source("ffmpeg", progress_cb=progress.append)

    assert result.status == "completed"
    assert result.file_count == 1
    assert result.chunk_count == 1
    assert repository.recorded == [("ffmpeg", "commit-a", 1)]
    assert repository.begun == [("ffmpeg", "commit-a")]
    assert repository.validated == [("ffmpeg", "commit-a", 1)]
    assert repository.promoted == [("ffmpeg", "commit-a")]
    assert vector_store.deleted[0] == ("ffmpeg", "commit-a")
    assert reader.calls[:2] == [("refresh", "ffmpeg"), ("manifest", "ffmpeg")]
    assert [event.phase for event in progress] == [
        "refreshing_mirror",
        "building_manifest",
        "preparing_candidate",
        "chunking",
        "chunking",
        "embedding",
        "validating",
        "promoting",
        "completed",
    ]
    assert progress[-1].files_done == 1
    assert progress[-1].chunks_done == 1


def test_successful_promotion_retires_the_previous_vector_revision():
    repository = Repository()
    repository.active_revision = "commit-old"
    vector_store = VectorStore()

    result = service(repository=repository, vector_store=vector_store).index_source("ffmpeg")

    assert result.status == "completed"
    assert vector_store.deleted == [
        ("ffmpeg", "commit-a"),
        ("ffmpeg", "commit-old"),
    ]
    assert repository.cleaned == [("ffmpeg", "commit-old")]


def test_previous_revision_cleanup_failure_does_not_fail_the_promoted_revision():
    repository = Repository()
    repository.active_revision = "commit-old"
    vector_store = VectorStore(cleanup_failure_revision="commit-old")

    result = service(repository=repository, vector_store=vector_store).index_source("ffmpeg")

    assert result.status == "completed_with_cleanup_pending"
    assert repository.active_revision == "commit-a"
    assert repository.failed == []
    assert repository.cleaned == []
    assert vector_store.deleted == [("ffmpeg", "commit-a")]


def test_chunks_that_share_a_source_line_receive_distinct_file_ordinals():
    vector_store = VectorStore()

    def same_line_chunks(path, _source, *, language):
        return [
            CodeChunk(path, 1, 1, f"# path: {path}\nfirst", "first", "symbol"),
            CodeChunk(path, 1, 1, f"# path: {path}\nsecond", "second", "symbol"),
        ]

    result = PublicCodeIndexService(
        source_registry=catalog(),
        repository=Repository(),
        source_reader=Reader(),
        vector_store=vector_store,
        chunk_fn=same_line_chunks,
        embed_fn=embed,
    ).index_source("ffmpeg")

    assert result.chunk_count == 2
    assert [item.chunk_ordinal for item in vector_store.items] == [0, 1]


def test_vector_count_mismatch_fails_candidate_and_preserves_active_revision():
    repository = Repository()
    repository.active_revision = "commit-old"
    vector_store = VectorStore(count_offset=-1)

    with pytest.raises(RuntimeError, match="vector count mismatch"):
        service(repository=repository, vector_store=vector_store).index_source("ffmpeg")

    assert repository.active_revision == "commit-old"
    assert repository.promoted == []
    assert repository.failed and repository.failed[0][0:2] == ("ffmpeg", "commit-a")
    assert vector_store.deleted[-1] == ("ffmpeg", "commit-a")


def test_read_failure_never_records_or_promotes_candidate():
    repository = Repository()
    repository.active_revision = "commit-old"
    vector_store = VectorStore()

    with pytest.raises(RuntimeError, match="read failed"):
        service(
            reader=Reader(fail_read=True),
            repository=repository,
            vector_store=vector_store,
        ).index_source("ffmpeg")

    assert repository.active_revision == "commit-old"
    assert repository.recorded == []
    assert repository.begun == [("ffmpeg", "commit-a")]
    assert repository.promoted == []
    assert repository.failed and repository.failed[0][0:2] == ("ffmpeg", "commit-a")


def test_current_revision_is_a_noop_without_qdrant_mutation():
    repository = Repository()
    repository.active_revision = "commit-a"
    vector_store = VectorStore()

    result = service(repository=repository, vector_store=vector_store).index_source("ffmpeg")

    assert result.status == "current"
    assert vector_store.deleted == []
    assert vector_store.items == []
    assert repository.recorded == []
    assert repository.begun == []


def test_empty_chunk_result_fails_without_replacing_active_revision():
    repository = Repository()
    repository.active_revision = "commit-old"
    vector_store = VectorStore()
    index_service = PublicCodeIndexService(
        source_registry=catalog(),
        repository=repository,
        source_reader=Reader(),
        vector_store=vector_store,
        chunk_fn=lambda *_args, **_kwargs: [],
        embed_fn=embed,
    )

    with pytest.raises(RuntimeError, match="produced no chunks"):
        index_service.index_source("ffmpeg")

    assert repository.active_revision == "commit-old"
    assert repository.promoted == []
    assert repository.failed and repository.failed[0][0:2] == ("ffmpeg", "commit-a")


def test_embedding_count_mismatch_cleans_candidate_revision():
    repository = Repository()
    repository.active_revision = "commit-old"
    vector_store = VectorStore()
    index_service = PublicCodeIndexService(
        source_registry=catalog(),
        repository=repository,
        source_reader=Reader(),
        vector_store=vector_store,
        chunk_fn=chunk_source,
        embed_fn=lambda _texts: ([], []),
    )

    with pytest.raises(RuntimeError, match="embedding count mismatch"):
        index_service.index_source("ffmpeg")

    assert repository.active_revision == "commit-old"
    assert repository.promoted == []
    assert vector_store.deleted[-1] == ("ffmpeg", "commit-a")


def test_imported_mirror_is_indexed_without_fetching_upstream():
    reader = Reader()
    progress = []
    result = service(reader=reader).index_source(
        "ffmpeg", progress_cb=progress.append, refresh_mirror=False)

    assert result.status == "completed"
    assert ("refresh", "ffmpeg") not in reader.calls
    assert reader.calls[0] == ("manifest", "ffmpeg")
    assert "refreshing_mirror" not in [item.phase for item in progress]
