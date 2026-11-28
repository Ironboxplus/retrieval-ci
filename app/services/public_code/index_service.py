"""Latest-wins indexing workflow for curated public source revisions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import logging
from typing import Protocol

from app.services.public_code.source_catalog import PublicCodeSource, PublicCodeSourceSelector
from core.code_kb.models import CodeChunk
from core.code_kb.vector_records import CodeChunkVectorRecord
from core.indexing.embedders import embed_texts
from core.indexing.revision_manifest import RevisionFile, RevisionManifest


ChunkFn = Callable[[str, str], list[CodeChunk]]
EmbedFn = Callable[
    [Sequence[str]],
    tuple[list[list[float]], list[dict[str, float]] | None],
]
ProgressFn = Callable[["PublicCodeIndexProgress"], None]


logger = logging.getLogger(__name__)


class PublicCodeSourceReader(Protocol):
    def refresh_source(self, source: PublicCodeSource) -> None: ...

    def build_manifest(
        self,
        source: PublicCodeSource,
        *,
        revision: str | None = None,
    ) -> RevisionManifest: ...

    def read_source(self, source: PublicCodeSource, file: RevisionFile) -> str: ...


class PublicCodeRevisionRepository(Protocol):
    def get_source(self, source_id: str): ...

    def record_candidate_revision(
        self,
        source_id: str,
        manifest: RevisionManifest,
        *,
        chunk_count: int,
    ) -> None: ...

    def validate_candidate_revision(
        self,
        source_id: str,
        revision: str,
        *,
        vector_count: int,
    ) -> None: ...

    def promote_revision(self, source_id: str, revision: str) -> str | None: ...

    def mark_revision_failed(
        self,
        source_id: str,
        revision: str,
        *,
        error_message: str,
    ) -> None: ...


class PublicCodeRevisionVectorStore(Protocol):
    @property
    def supports_sparse(self) -> bool: ...

    def delete_revision(self, source_id: str, revision: str) -> int: ...

    def upsert(
        self,
        *,
        source_id: str,
        revision: str,
        domain: str,
        items: Sequence[CodeChunkVectorRecord],
        dense_vectors: Sequence[list[float]],
        sparse_vectors: Sequence[dict[str, float]] | None,
    ) -> int: ...

    def count_revision(self, source_id: str, revision: str) -> int: ...


class PrivateCodeVectorReader(Protocol):
    def reusable_vectors(
        self, *, owner_user_id: str, project: str,
        items: Sequence[CodeChunkVectorRecord], require_sparse: bool,
    ) -> list[tuple[list[float], dict[str, float] | None] | None]: ...


@dataclass(frozen=True, slots=True)
class PublicCodeIndexResult:
    source_id: str
    revision: str
    status: str
    file_count: int
    chunk_count: int
    reused_chunk_count: int = 0


@dataclass(frozen=True, slots=True)
class PublicCodeIndexProgress:
    source_id: str
    revision: str | None
    phase: str
    files_done: int = 0
    files_total: int = 0
    chunks_done: int = 0
    current_path: str | None = None
    message: str | None = None


class PublicCodeIndexService:
    """Stream one Git revision into an isolated candidate, then promote exactly once."""

    def __init__(
        self,
        *,
        source_registry: PublicCodeSourceSelector,
        repository: PublicCodeRevisionRepository,
        source_reader: PublicCodeSourceReader,
        vector_store: PublicCodeRevisionVectorStore,
        chunk_fn: Callable[..., list[CodeChunk]],
        embed_fn: EmbedFn | None = None,
        batch_size: int = 128,
        private_vector_store_factory: Callable[[], PrivateCodeVectorReader] | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._source_registry = source_registry
        self._repository = repository
        self._source_reader = source_reader
        self._vector_store = vector_store
        self._chunk_fn = chunk_fn
        self._embed_fn = embed_fn or _embed_with_shared_model
        self._batch_size = batch_size
        self._private_vector_store_factory = private_vector_store_factory

    def index_source(
        self,
        source_id: str,
        progress_cb: ProgressFn | None = None,
        *,
        refresh_mirror: bool = True,
        private_owner_user_id: str | None = None,
        private_project: str | None = None,
    ) -> PublicCodeIndexResult:
        """Index the current mirror HEAD while preserving the previous active revision.

        refresh_mirror=False indexes the mirror as it stands: a bundle import has just
        written the revision to index, and fetching upstream is what it replaces.
        A private scope supplies vectors only for exact chunks derived from this mirror;
        it cannot introduce untracked files or private edits into the public corpus."""

        source = self._source_registry.select(source_ids=(source_id,))[0]
        if refresh_mirror:
            _emit_progress(progress_cb, PublicCodeIndexProgress(
                source_id=source_id,
                revision=None,
                phase="refreshing_mirror",
                message="fetching the configured upstream into the bare mirror",
            ))
            self._source_reader.refresh_source(source)
        _emit_progress(progress_cb, PublicCodeIndexProgress(
            source_id=source_id,
            revision=None,
            phase="building_manifest",
            message="reading tracked source files from the mirror",
        ))
        manifest = self._source_reader.build_manifest(source)
        current = self._repository.get_source(source_id)
        if current is not None and current.active_revision == manifest.revision_id:
            _emit_progress(progress_cb, PublicCodeIndexProgress(
                source_id=source_id,
                revision=manifest.revision_id,
                phase="completed",
                files_done=len(manifest.files),
                files_total=len(manifest.files),
                message="mirror revision is already active",
            ))
            return PublicCodeIndexResult(
                source_id=source_id,
                revision=manifest.revision_id,
                status="current",
                file_count=len(manifest.files),
                chunk_count=0,
            )

        revision = manifest.revision_id
        chunk_count = 0
        reused_chunk_count = 0
        files_done = 0
        pending_records: list[CodeChunkVectorRecord] = []
        try:
            private_store = None
            if private_project:
                if not private_owner_user_id:
                    raise ValueError("private vector reuse requires an authenticated owner")
                if self._private_vector_store_factory is None:
                    raise RuntimeError("private vector reuse is not configured")
                private_store = self._private_vector_store_factory()
            _emit_progress(progress_cb, PublicCodeIndexProgress(
                source_id=source_id,
                revision=revision,
                phase="preparing_candidate",
                files_total=len(manifest.files),
                message="removing a prior partial candidate for this revision",
            ))
            self._repository.begin_candidate_revision(source_id, manifest)
            self._vector_store.delete_revision(source_id, revision)
            for file in manifest.files:
                _emit_progress(progress_cb, PublicCodeIndexProgress(
                    source_id=source_id,
                    revision=revision,
                    phase="chunking",
                    files_done=files_done,
                    files_total=len(manifest.files),
                    chunks_done=chunk_count,
                    current_path=file.path,
                    message="reading and AST chunking source file",
                ))
                source_text = self._source_reader.read_source(source, file)
                for chunk_ordinal, chunk in enumerate(
                    self._chunk_fn(file.path, source_text, language=file.language)
                ):
                    pending_records.append(_vector_record(
                        chunk,
                        language=file.language,
                        chunk_ordinal=chunk_ordinal,
                    ))
                    if len(pending_records) >= self._batch_size:
                        _emit_progress(progress_cb, PublicCodeIndexProgress(
                            source_id=source_id,
                            revision=revision,
                            phase="embedding",
                            files_done=files_done,
                            files_total=len(manifest.files),
                            chunks_done=chunk_count,
                            current_path=file.path,
                            message=f"embedding and upserting {len(pending_records)} chunks",
                        ))
                        inserted, reused = self._flush_batch(
                            source, revision, pending_records, private_store=private_store,
                            private_owner_user_id=private_owner_user_id, private_project=private_project)
                        chunk_count += inserted
                        reused_chunk_count += reused
                        pending_records.clear()
                files_done += 1
                _emit_progress(progress_cb, PublicCodeIndexProgress(
                    source_id=source_id,
                    revision=revision,
                    phase="chunking",
                    files_done=files_done,
                    files_total=len(manifest.files),
                    chunks_done=chunk_count + len(pending_records),
                    current_path=file.path,
                    message="source file chunked",
                ))
            if pending_records:
                _emit_progress(progress_cb, PublicCodeIndexProgress(
                    source_id=source_id,
                    revision=revision,
                    phase="embedding",
                    files_done=files_done,
                    files_total=len(manifest.files),
                    chunks_done=chunk_count,
                    message=f"embedding and upserting final {len(pending_records)} chunks",
                ))
                inserted, reused = self._flush_batch(
                    source, revision, pending_records, private_store=private_store,
                    private_owner_user_id=private_owner_user_id, private_project=private_project)
                chunk_count += inserted
                reused_chunk_count += reused
            if chunk_count == 0:
                raise RuntimeError(f"public-code revision produced no chunks: {source_id}:{revision}")

            self._repository.record_candidate_revision(
                source_id,
                manifest,
                chunk_count=chunk_count,
            )
            _emit_progress(progress_cb, PublicCodeIndexProgress(
                source_id=source_id,
                revision=revision,
                phase="validating",
                files_done=files_done,
                files_total=len(manifest.files),
                chunks_done=chunk_count,
                message="comparing the exact Qdrant count with the candidate manifest",
            ))
            vector_count = self._vector_store.count_revision(source_id, revision)
            if vector_count != chunk_count:
                raise RuntimeError(
                    "vector count mismatch for "
                    f"{source_id}:{revision}: expected {chunk_count}, got {vector_count}"
                )
            self._repository.validate_candidate_revision(
                source_id,
                revision,
                vector_count=vector_count,
            )
            _emit_progress(progress_cb, PublicCodeIndexProgress(
                source_id=source_id,
                revision=revision,
                phase="promoting",
                files_done=files_done,
                files_total=len(manifest.files),
                chunks_done=chunk_count,
                message="atomically activating the validated revision",
            ))
            previous_revision = self._repository.promote_revision(source_id, revision)
        except Exception as exc:
            error_message = str(exc)
            cleanup_pending = True
            try:
                self._vector_store.delete_revision(source_id, revision)
                cleanup_pending = False
            except Exception as cleanup_error:
                error_message = f"{error_message}; candidate cleanup failed: {cleanup_error}"
            self._repository.mark_revision_failed(
                source_id,
                revision,
                error_message=error_message,
                cleanup_pending=cleanup_pending,
            )
            _emit_progress(progress_cb, PublicCodeIndexProgress(
                source_id=source_id,
                revision=revision,
                phase="failed",
                files_done=files_done,
                files_total=len(manifest.files),
                chunks_done=chunk_count,
                message=error_message,
            ))
            raise

        result_status = "completed"
        completion_message = "revision indexed and activated"
        if private_project:
            completion_message += f"; reused {reused_chunk_count}/{chunk_count} private chunk vectors"
        if previous_revision:
            try:
                self._vector_store.delete_revision(source_id, previous_revision)
                self._repository.mark_revision_cleaned(source_id, previous_revision)
            except Exception:
                result_status = "completed_with_cleanup_pending"
                completion_message = (
                    "revision activated; superseded vectors remain pending background cleanup"
                )
                logger.exception(
                    "public-code superseded revision cleanup failed source=%s revision=%s",
                    source_id,
                    previous_revision,
                )
        _emit_progress(progress_cb, PublicCodeIndexProgress(
            source_id=source_id,
            revision=revision,
            phase="completed",
            files_done=files_done,
            files_total=len(manifest.files),
            chunks_done=chunk_count,
            message=completion_message,
        ))
        return PublicCodeIndexResult(
            source_id=source_id,
            revision=revision,
            status=result_status,
            file_count=len(manifest.files),
            chunk_count=chunk_count,
            reused_chunk_count=reused_chunk_count,
        )

    def _flush_batch(
        self,
        source: PublicCodeSource,
        revision: str,
        records: Sequence[CodeChunkVectorRecord],
        *, private_store: PrivateCodeVectorReader | None = None,
        private_owner_user_id: str | None = None, private_project: str | None = None,
    ) -> tuple[int, int]:
        cached = (private_store.reusable_vectors(
            owner_user_id=private_owner_user_id, project=private_project, items=records,
            require_sparse=self._vector_store.supports_sparse,
        ) if private_store is not None else [None] * len(records))
        if len(cached) != len(records):
            raise RuntimeError("private vector lookup count mismatch")
        missing = [index for index, vectors in enumerate(cached) if vectors is None]
        dense_vectors, sparse_vectors = (self._embed_fn([records[index].content for index in missing])
                                         if missing else ([], None))
        if len(dense_vectors) != len(missing):
            raise RuntimeError(
                f"embedding count mismatch: expected {len(missing)}, got {len(dense_vectors)}"
            )
        if sparse_vectors is not None and len(sparse_vectors) != len(missing):
            raise RuntimeError(
                f"sparse embedding count mismatch: expected {len(missing)}, got {len(sparse_vectors)}"
            )
        for ordinal, index in enumerate(missing):
            cached[index] = (dense_vectors[ordinal],
                             sparse_vectors[ordinal] if sparse_vectors is not None else None)
        combined_dense = [vectors[0] for vectors in cached]
        combined_sparse = [vectors[1] or {} for vectors in cached]
        inserted = self._vector_store.upsert(
            source_id=source.source_id,
            revision=revision,
            domain=source.domain,
            items=records,
            dense_vectors=combined_dense,
            sparse_vectors=combined_sparse if any(combined_sparse) else None,
        )
        return inserted, len(records) - len(missing)


def _vector_record(
    chunk: CodeChunk,
    *,
    language: str,
    chunk_ordinal: int,
) -> CodeChunkVectorRecord:
    return CodeChunkVectorRecord(
        rel_path=chunk.path,
        language=language,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        label=chunk.label,
        content=chunk.content,
        code=chunk.code,
        chunk_ordinal=chunk_ordinal,
    )


def _embed_with_shared_model(
    texts: Sequence[str],
) -> tuple[list[list[float]], list[dict[str, float]] | None]:
    result = embed_texts(list(texts), priority="background")
    dense_vectors = [vector.tolist() for vector in result.dense]
    return dense_vectors, result.sparse


def _emit_progress(callback: ProgressFn | None, progress: PublicCodeIndexProgress) -> None:
    if callback is None:
        return
    try:
        callback(progress)
    except Exception:
        logger.exception(
            "public-code progress callback failed source=%s phase=%s",
            progress.source_id,
            progress.phase,
        )
