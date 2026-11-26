"""Pipeline registry — assemble the ordered ingest pipeline for a StageContext.

This is the single place that declares stage ORDER. The orchestrator drives
whatever list it gets back; everything else (which conference, which year) is
carried by `ctx`, so this is fully generic.
"""
from __future__ import annotations

from typing import List

from loguru import logger

from app.services.pipeline.arxiv_stages import ArxivStageContext, build_arxiv_pipeline
from app.services.pipeline.context import StageContext
from app.services.pipeline.download_batch import DownloadBatchContext, build_download_batch_pipeline
from app.services.pipeline.spec import StageSpec
from app.services.pipeline.stages import (
    make_cleanup_stage,
    make_download_stage,
    make_embed_stage_for,
    make_extract_stage_for,
    make_ingest_stage,
    make_preindex_stage,
)
from app.services.pipeline.upload_stages import (
    make_upload_extract_stage,
    make_upload_index_stage,
)

# Canonical stage order. Exposed for callers that need the keys without building.
STAGE_KEYS = ["download", "extract", "preindex", "embed", "ingest", "cleanup"]

# Upload pipeline: a single local file already on disk, so no `download`, and
# preindex+embed+ingest collapse into one local `index` stage — IndexService.
# index_document does chunk+embed+Qdrant-upsert+SQL-write atomically, so splitting
# would re-chunk or hand-roll the 8-field payload (W-landmines). W62: local only
# (the HTTP upload path has no ssh_target).
UPLOAD_STAGE_KEYS = ["extract", "index"]


def build_ingest_pipeline(ctx: StageContext) -> List[StageSpec]:
    """Build the full [download, extract, preindex, embed, ingest, cleanup] pipeline.

    extract + embed branch on ctx.execution_mode (make_*_stage_for): "local"
    (default — self-hosted, GPU work via run_local on this box, W106), "host" (an
    ssh host from the registry, deploy/compute88/embed_hosts.json) or "remote"
    (retired KSS-style ssh + SMB + ship.sh, only when named). download / preindex /
    ingest run local in every mode."""
    if isinstance(ctx, ArxivStageContext):
        return build_arxiv_pipeline(ctx)
    if isinstance(ctx, DownloadBatchContext):
        return build_download_batch_pipeline(ctx)
    specs = [
        make_download_stage(ctx),
        make_extract_stage_for(ctx),
        make_preindex_stage(ctx),
        make_embed_stage_for(ctx),
        make_ingest_stage(ctx),
        make_cleanup_stage(ctx),
    ]
    logger.info(
        f"registry.build_ingest_pipeline venue={ctx.venue} year={ctx.year} "
        f"execution_mode={ctx.execution_mode} stages={[s.key for s in specs]}"
    )
    return specs


def build_upload_pipeline(ctx) -> List[StageSpec]:
    """Build the [extract, index] pipeline for one uploaded document."""
    specs = [
        make_upload_extract_stage(ctx),
        make_upload_index_stage(ctx),
    ]
    logger.info(
        f"registry.build_upload_pipeline doc_id={ctx.doc_id} "
        f"stages={[s.key for s in specs]}"
    )
    return specs
