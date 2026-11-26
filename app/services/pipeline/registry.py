"""Pipeline registry — assemble the ordered ingest pipeline for a StageContext.

This is the single place that declares stage ORDER. The orchestrator drives
whatever list it gets back; everything else (which conference, which year) is
carried by `ctx`, so this is fully generic.
"""
from __future__ import annotations

from typing import List

from loguru import logger

from app.services.pipeline.context import StageContext
from app.services.pipeline.spec import StageSpec
from app.services.pipeline.stages import (
    make_download_stage,
    make_embed_stage,
    make_embed_stage_local,
    make_extract_stage,
    make_extract_stage_local,
    make_ingest_stage,
    make_preindex_stage,
)
from app.services.pipeline.upload_stages import (
    make_upload_extract_stage,
    make_upload_index_stage,
)

# Canonical stage order. Exposed for callers that need the keys without building.
STAGE_KEYS = ["download", "extract", "preindex", "embed", "ingest"]

# Upload pipeline: a single local file already on disk, so no `download`, and
# preindex+embed+ingest collapse into one local `index` stage — IndexService.
# index_document does chunk+embed+Qdrant-upsert+SQL-write atomically, so splitting
# would re-chunk or hand-roll the 8-field payload (W-landmines). W62: local only
# (the HTTP upload path has no ssh_target).
UPLOAD_STAGE_KEYS = ["extract", "index"]


def build_ingest_pipeline(ctx: StageContext) -> List[StageSpec]:
    """Build the full [download, extract, preindex, embed, ingest] pipeline.

    extract + embed branch on ctx.execution_mode: "local" (self-hosted, GPU work
    via run_local on this box, W106) vs "remote" (default — KSS-style ssh + SMB +
    ship.sh). download / preindex / ingest run local in BOTH modes."""
    local = ctx.execution_mode == "local"
    extract = make_extract_stage_local(ctx) if local else make_extract_stage(ctx)
    embed = make_embed_stage_local(ctx) if local else make_embed_stage(ctx)
    specs = [
        make_download_stage(ctx),
        extract,
        make_preindex_stage(ctx),
        embed,
        make_ingest_stage(ctx),
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
