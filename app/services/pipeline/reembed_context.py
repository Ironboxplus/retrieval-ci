"""The StageContext of a re-embed task: which scope on which collection, and its dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from qdrant_client import models

from app.services.pipeline.context import StageContext
from app.services.pipeline.reembed_scopes import ReembedScope, collection_name, get_scope
from app.services.pipeline.stages import _embed_chunks_dir, _embed_task_dir, _embed_vectors_dir, load_embed_host
from core.indexing.embedding_profile import current_embed_fp

# Pause after each Qdrant write batch. The same 250 ms scripts/remote_embed/local_ingest.py
# defaults to (--pause-ms): Qdrant, swap and SQLite share one SATA SSD on .88, and
# back-to-back wait=True writes starve live search reads until gRPC hits DEADLINE_EXCEEDED.
WRITE_PAUSE_S = 0.25
# Points per write batch (a 1024-d dense vector as JSON is ~10 KB, so a batch stays far below
# Qdrant's 32 MB request limit), the same size every Qdrant writer here uses.
WRITE_BATCH = 256
# Chunks per shard when the embed stage uses this backend's embedding service (no registered host to
# say). A shard is written whole or not at all, so it is also what a restart can lose: at the
# ~25 chunks/s of long chunks, 10,000 is about seven minutes.
LOCAL_SHARD_SIZE = 10_000
# papers:rebuild select: chunk rows per SQLite range scan, and the rows scanned between two
# checkpoints (a restart redoes at most that much of the scan).
PAPER_SCAN_BATCH = 200_000
PAPER_CHECKPOINT_ROWS = 500_000
# papers:rebuild select pauses while the host's MemAvailable is under the first bound and resumes
# only above the second: a scan of the paper tables swapped .88 once (the journal select did,
# 2026-11-14), and a single threshold would flap around it.
MEM_PAUSE_BELOW_BYTES = 1 << 30
MEM_RESUME_ABOVE_BYTES = 3 << 29
MEM_POLL_S = 5.0


@dataclass
class ReembedContext(StageContext):
    """``venue`` carries the scope name and ``year`` is 0; ``ssh_target`` names the embed host
    (``"local"``: this backend's vLLM embedding service, ``execution_mode`` ``"local"``).

    ``client_factory``, ``embed_fn`` and ``journal_repository`` are the seams the stages reach
    the outside through. Production leaves them None: the shared Qdrant client of the backend's
    settings, the backend's resident BGE-M3 (background priority) and the SQL-backed journal
    repository. Tests pass real ones for scratch collections and a scratch database.
    """

    scope_name: str = ""
    collection: str = ""
    client_factory: Optional[Callable[[], Any]] = None
    embed_fn: Optional[Callable[[list], tuple]] = None
    journal_repository: Any = None
    # papers:rebuild: the SQLite file of the paper tables (None: the backend's own database), the
    # roots Markdown is looked up under (None: settings.paper_markdown_roots) and the scan/guard sizes.
    papers_db: Optional[Path] = None
    markdown_roots: Optional[Sequence[str]] = None
    scan_batch: int = PAPER_SCAN_BATCH
    checkpoint_rows: int = PAPER_CHECKPOINT_ROWS
    mem_pause_below: int = MEM_PAUSE_BELOW_BYTES
    mem_resume_above: int = MEM_RESUME_ABOVE_BYTES
    mem_poll_s: float = MEM_POLL_S
    pause_s: float = WRITE_PAUSE_S
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.scope_name = self.scope_name or self.venue
        scope = get_scope(self.scope_name)             # an unknown scope is refused when the context is built
        self.collection = self.collection or collection_name(scope)

    def shard_size(self) -> int:
        """Chunks per select shard: the embed host's registered size, or ``LOCAL_SHARD_SIZE`` on this GPU."""
        if self.execution_mode == "local":
            return LOCAL_SHARD_SIZE
        return load_embed_host(self.ssh_target).shard_size

    @property
    def scope(self) -> ReembedScope:
        return get_scope(self.scope_name)

    @property
    def fingerprint(self) -> str:
        return current_embed_fp()

    @property
    def client(self) -> Any:
        if self._client is None:
            if self.client_factory is not None:
                self._client = self.client_factory()
            else:
                from core.storage.qdrant_store import QdrantStore
                self._client = QdrantStore(collection_name=self.collection, create_if_missing=False).client
        return self._client

    def encode(self, texts: list) -> tuple:
        """``(dense [n, dim], sparse [n] of token id -> weight)`` of ``texts`` in this process."""
        if self.embed_fn is not None:
            return self.embed_fn(texts)
        from app.services.pipeline.stages import _shared_encode
        return _shared_encode(texts)

    def papers_db_path(self) -> Path:
        if self.papers_db is not None:
            return Path(self.papers_db)
        from core.db.database import DB_PATH
        return Path(DB_PATH)

    def repository(self) -> Any:
        if self.journal_repository is None:
            from app.services.journal.repository import JournalRepository
            self.journal_repository = JournalRepository()
        return self.journal_repository

    # ------------------------------------------------------------------ paths
    @property
    def chunks_dir(self) -> Path:
        return _embed_chunks_dir(self)

    @property
    def vectors_dir(self) -> Path:
        return _embed_vectors_dir(self)

    @property
    def select_dir(self) -> Path:
        """The select stage's side files (stamps, oversized points, journal sidecars)."""
        return _embed_task_dir(self) / "select"

    def abort_if_cancelled(self) -> None:
        if self.abort.is_set():
            raise RuntimeError("cancelled")

    # --------------------------------------------------------------- filters
    def must_not_current(self) -> models.FieldCondition:
        return models.FieldCondition(key="embed_fp", match=models.MatchValue(value=self.fingerprint))
