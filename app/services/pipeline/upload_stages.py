"""The 2 upload-ingest stages: extract -> index.

The single-document analog of stages.py. Same decomposition (pure builders +
thin run()), same StageSpec contract, driven by the SAME generic orchestrator.

  - extract  scripts/upload/extract_one.py — marker PDF→MD (the SAME MarkerExtractor
             the venue pipeline wraps, pymupdf fallback; NOT pypdf) + persist PDF
             and MD to the SMB corpus share (papers/{pdf,markdown}/uploads/),
             mirroring the paper layout. Probe = local md exists.
  - index    scripts/upload/index_one.py — IndexService.index_document: chunk +
             embed (BGE-M3 dense+sparse) + Qdrant upsert + SQL chunk write, atomic.
             Probe = Qdrant points for this doc_id vs SQL chunk count.

Differences from the venue pipeline (all forced by "one uploaded file"):
  - No `download` stage — the file is already saved under data/uploads/.
  - Marker runs LOCALLY on Arc (marker-pdf is installed here), so there is no
    ssh_target and W62 does not apply. The venue pipeline ships PDFs to a remote
    GPU box only because it extracts 64K papers in batch; one upload doesn't need
    the cluster. The venue preindex/embed/ingest split likewise exists only for
    that remote transport — locally index_document does it all in one call.
  - Artifacts are doc-scoped (md file for THIS doc, Qdrant points for THIS
    doc_id), so satisfied-skip is SAFE here — unlike the venue embed/ingest whose
    artifacts live in shared cross-venue dirs (W72).

Backend only polls (each probe counts artifacts); the worker does the work.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional

from loguru import logger

from app.services.pipeline.spec import GateResult, Progress, StageSpec
from app.services.pipeline.runners import EventSink, run_local
from app.services.pipeline.upload_context import UploadStageContext

PointCounter = Callable[[int], int]


# ----------------------------------------------------- default Qdrant counter
def _qdrant_point_count(doc_id: int) -> int:
    """Count Qdrant points in the `papers` collection whose payload doc_id matches.

    Dependency-light HTTP count (same localhost+NO_PROXY approach as
    stages._qdrant_reachable). index_document writes the payload doc_id as a
    STRING (index_service.py: `str_doc_id = str(doc_id)`, payload
    `"doc_id": str_doc_id`), and the doc_id payload index is keyword(str) — so we
    MUST match the string form. Matching int(doc_id) never hits (it counted 0,
    making index_gate falsely fail with all points actually present — task #95).
    Best-effort: any failure → 0 (the gate then reports incomplete rather than
    crashing the poll)."""
    import json
    import urllib.request

    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    body = json.dumps(
        {"filter": {"must": [{"key": "doc_id", "match": {"value": str(doc_id)}}]}, "exact": True}
    ).encode()
    try:
        req = urllib.request.Request(
            "http://localhost:6333/collections/papers/points/count",
            data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 localhost only
            payload = json.loads(resp.read())
        return int(payload.get("result", {}).get("count", 0) or 0)
    except Exception as exc:  # best-effort probe — never crash the stage poll
        logger.warning(f"upload.qdrant.count.error doc_id={doc_id}: {exc}")
        return 0


# ===================================================================== extract
def build_extract_cmd(ctx: UploadStageContext) -> list[str]:
    return [
        sys.executable, "scripts/upload/extract_one.py",
        "--doc-id", str(ctx.doc_id),
        "--input", str(ctx.file_path),
        "--output", str(ctx.md_path),
    ]


def extract_probe(ctx: UploadStageContext) -> Progress:
    return Progress(done=1 if ctx.md_exists() else 0, total=1)


def extract_gate(ctx: UploadStageContext) -> GateResult:
    if not ctx.md_exists():
        return GateResult(False, "no markdown produced (extraction yielded empty output)")
    return GateResult(True)


def make_upload_extract_stage(ctx: UploadStageContext) -> StageSpec:
    def run(c: UploadStageContext, sink: EventSink) -> None:
        sink(f"extract doc {c.doc_id}: marker {c.ext} -> {c.md_path.name} (+persist to SMB)")
        run_local(build_extract_cmd(c), c.project_root, sink)

    return StageSpec(
        key="extract", title=f"Extract upload doc {ctx.doc_id} (marker)",
        run=run, probe=extract_probe, gate=extract_gate,
        poll_interval_s=5.0,
        # local single-file extract — no shared resource to serialize on.
        resource_key="",
    )


# ======================================================================= index
# Local index = chunk + embed (BGE-M3 dense+sparse) + Qdrant upsert + SQL chunk
# write, all via the proven IndexService.index_document (index_one.py). The venue
# pipeline splits preindex/embed/ingest only because embed runs on a remote box
# (dump SQL -> ship -> GPU -> parquet -> ingest); locally there is no ship, so
# splitting would either re-chunk twice or re-implement the 8-field Qdrant payload
# + point-id scheme by hand (W-landmines). One honest local stage instead.
def build_index_cmd(ctx: UploadStageContext) -> list[str]:
    return [
        sys.executable, "scripts/upload/index_one.py",
        "--doc-id", str(ctx.doc_id),
    ]


def index_probe(
    ctx: UploadStageContext, *,
    point_counter: Optional[PointCounter] = None,
    session_factory=None,
) -> Progress:
    counter = point_counter or _qdrant_point_count
    total = ctx.chunk_count(session_factory=session_factory)  # 1 point per chunk
    done = counter(ctx.doc_id)
    return Progress(done=done, total=total)


def index_gate(
    ctx: UploadStageContext, *,
    point_counter: Optional[PointCounter] = None,
    session_factory=None,
) -> GateResult:
    pr = index_probe(ctx, point_counter=point_counter, session_factory=session_factory)
    if pr.total > 0 and pr.done >= pr.total:
        return GateResult(True)
    return GateResult(False, f"only {pr.done}/{pr.total} chunks upserted to Qdrant")


def make_upload_index_stage(ctx: UploadStageContext, *, session_factory=None) -> StageSpec:
    def run(c: UploadStageContext, sink: EventSink) -> None:
        sink(f"index doc {c.doc_id}: embed chunks + upsert to Qdrant")
        run_local(build_index_cmd(c), c.project_root, sink)

    return StageSpec(
        key="index", title=f"Index upload doc {ctx.doc_id} -> Qdrant",
        run=run,
        probe=lambda c: index_probe(c, session_factory=session_factory),
        gate=lambda c: index_gate(c, session_factory=session_factory),
        poll_interval_s=5.0,
        # Single global Qdrant slot (shared with venue ingest): commit-charge
        # headroom is tight (W58/W60), two concurrent upserts race for mmap.
        resource_key="qdrant",
    )
