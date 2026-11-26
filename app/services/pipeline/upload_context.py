"""UploadStageContext — the opaque carrier for the single-document upload pipeline.

Doc-keyed analog of StageContext (context.py). Where the venue pipeline is keyed
on (venue, year) + a manifest + remote SMB paths, an uploaded document is keyed
on a single `doc_id` + the locally-saved `file_path`. The orchestrator forwards
this object to stage callbacks opaquely, so it is just a plain data carrier:
resolved md path + artifact probes (md exists / SQL chunk count) + the same
TaskStore state proxy used by StageContext.

LOCAL ONLY by design: per W62 the HTTP upload path must not default an
ssh_target, so extract/preindex/index all run on Arc — there are no kss_/smb_
remote paths here.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger
from sqlalchemy import func

from app.services.pipeline.store import TaskStore

# Repo root = 4 parents up (app/services/pipeline/upload_context.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_UPLOAD_ROOT = _PROJECT_ROOT / "data" / "uploads"


def upload_md_path(file_path: str, upload_root: Path = _DEFAULT_UPLOAD_ROOT) -> Path:
    """Single source of truth for an uploaded doc's extracted-markdown path.

    Shared by UploadStageContext.md_path AND the local workers — index_one.py
    resolves it from doc_id alone (load file_path from the DB), so the convention
    must live in exactly one place or the two sides drift apart."""
    return Path(upload_root) / "md" / f"{Path(file_path).stem}.md"


@dataclass
class UploadStageContext:
    """Opaque per-run context for one uploaded document. Construct one per run."""

    doc_id: int
    file_path: str
    store: TaskStore
    task_id: int

    project_root: Path = _PROJECT_ROOT
    upload_root: Path = _DEFAULT_UPLOAD_ROOT

    # Cancellation flag; the orchestrator swaps in its own Event so run_local()
    # kills the stage's subprocess when the runner cancels (see StageContext.abort).
    abort: threading.Event = field(default_factory=threading.Event)

    # ----------------------------------------------------------- computed paths
    @property
    def _src(self) -> Path:
        return Path(self.file_path)

    @property
    def ext(self) -> str:
        """Lower-cased file extension incl. dot (e.g. '.pdf')."""
        return self._src.suffix.lower()

    @property
    def file_stem(self) -> str:
        return self._src.stem

    @property
    def md_dir(self) -> Path:
        """Local dir holding extracted markdown for uploads."""
        return Path(self.upload_root) / "md"

    @property
    def md_path(self) -> Path:
        """Where the extract stage writes this document's markdown."""
        return self.md_dir / f"{self.file_stem}.md"

    # --------------------------------------------------------------- fs probes
    def md_exists(self) -> bool:
        """True iff the extracted markdown exists AND is non-empty."""
        try:
            return self.md_path.is_file() and self.md_path.stat().st_size > 0
        except OSError as exc:  # best-effort probe
            logger.warning(f"upload_context.md_exists.error path={self.md_path}: {exc}")
            return False

    # -------------------------------------------------------------- sql chunks
    def chunk_count(self, *, session_factory: Optional[Callable[[], Any]] = None) -> int:
        """Count SQL chunks already written for THIS doc_id (preindex artifact)."""
        from core.db import get_db_session
        from core.db.models import Chunk as DBChunk

        sf = session_factory or get_db_session
        with sf() as db:
            return int(
                db.query(func.count(DBChunk.id))
                .filter(DBChunk.doc_id == self.doc_id)
                .scalar()
                or 0
            )

    # ----------------------------------------------------------- state proxy
    def state_get(self, key: str, default: Any = None) -> Any:
        return self.store.get_state(self.task_id).get(key, default)

    def state_set(self, **kv: Any) -> None:
        self.store.update_state(self.task_id, **kv)
        logger.debug(f"upload_context.state.set task={self.task_id} keys={list(kv)}")
