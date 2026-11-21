"""The 5 ingest stages, each expressed as pure builders + a thin run().

Decomposition (so the logic is unit-testable offline):
  build_*_cmd(ctx)      — exact command (list/str). Pure. Asserted in tests.
  *_probe(ctx)          — count artifacts → Progress. Pure read.
  *_gate / *_preflight  — GateResult verdict. Pure (modulo gpu_health / qdrant ping).
  make_*_stage(ctx)     — assemble a StageSpec wiring builder → runner.

Stages (LOCAL = Arc, KSS = remote GPU box over ssh):
  1 download  LOCAL   scripts/download_subset.py
  2 extract   KSS     run_marker_kss.sh (tmux), then POLL md count on SMB
  3 preindex  LOCAL   preindex_md_to_sql_only.py (--conf/--year)
  4 embed     LOCAL+KSS  dump → ship up → KSS embed (poll) → ship down
  5 ingest    LOCAL   local_ingest.py → Qdrant

Everything is generic for ANY (venue, year): no conference is special-cased and
GPU ids are always discovered, never hardcoded.
"""
from __future__ import annotations

import re
import shlex
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from loguru import logger
from sqlalchemy import func

from core.db import get_db_session
from core.db.models import Chunk as DBChunk, Paper as DBPaper
from app.services.pipeline.context import StageContext
from app.services.pipeline.spec import GateResult, Progress, StageSpec
from app.services.pipeline.runners import (
    EventSink,
    gpu_health,
    run_local,
    run_remote,
)

# Remote script paths are derived from ctx.remote_repo so the same code drives
# KSS, the 4090 box, or any future ssh target. Defaults in StageContext keep
# back-compat with existing KSS layout. No host name appears in this module.
def _marker_sh(ctx: StageContext) -> str:
    return f"{ctx.remote_repo}/scripts/remote_marker/run_marker_kss.sh"


def _embed_sh(ctx: StageContext) -> str:
    return f"{ctx.remote_repo}/scripts/remote_embed/run_embed_kss.sh"

# Local remote_embed shard dirs (relative to project_root).
_CHUNKS_GLOB = "chunks_*.jsonl.gz"
_VECTORS_GLOB = "vectors_*.parquet"
# local_ingest.py records the parquet FILENAME per line in ingest_done.txt
# (e.g. "vectors_0102.parquet"), so the shard index is parsed out of the name.
_SHARD_RE = re.compile(r"vectors_(\d+)")
_CHUNK_SHARD_RE = re.compile(r"chunks_(\d+)")

PollFn = Callable[[StageContext], Progress]
SleepFn = Callable[[float], None]


# --------------------------------------------------------------------- helpers
def _count_files(directory: Path, pattern: str) -> int:
    """Count files matching a glob in directory. Missing dir → 0."""
    try:
        if not directory.exists():
            return 0
        return sum(1 for _ in directory.glob(pattern))
    except OSError as exc:
        logger.warning(f"stages.count_files.error dir={directory} pat={pattern}: {exc}")
        return 0


def _count_shards_in_range(
    directory: Path, pattern: str, regex: "re.Pattern[str]",
    start: int, end: Optional[int] = None,
) -> int:
    """Count shard files whose NUMBER (parsed via `regex`) is in [start, end).
    `end=None` → no upper bound. Missing dir → 0.

    W81: per-task shard boundaries must be NUMBER-based, never a raw file count
    of the SHARED chunks/ + vectors/ dirs. Concurrent pipelines interleave
    shards from different venues into one dir, so a count mixes venues and the
    [start, end) slice drifts onto another task's data.
    """
    try:
        if not directory.exists():
            return 0
        n = 0
        for p in directory.glob(pattern):
            m = regex.search(p.name)
            if not m:
                continue
            idx = int(m.group(1))
            if idx >= start and (end is None or idx < end):
                n += 1
        return n
    except OSError as exc:
        logger.warning(f"stages.count_shards.error dir={directory} pat={pattern}: {exc}")
        return 0


def _embed_chunks_dir(ctx: StageContext) -> Path:
    return ctx.project_root / "data" / "remote_embed" / "chunks"


def _embed_vectors_dir(ctx: StageContext) -> Path:
    return ctx.project_root / "data" / "remote_embed" / "vectors"


def _embed_smb_vectors_dir(ctx: StageContext) -> Path:
    """The locally-mounted SMB share where the KSS embed worker writes parquets.
    Same NAS the worker writes to (zero-copy, cross-host same-source), readable
    from Arc with a plain glob (no ssh) — the SAME mechanism extract uses via
    smb_md_dir. Polled by embed_smb_probe so step 4 tracks live progress instead
    of the C-drive copy, which stays empty until ship-down (W82)."""
    return Path(ctx.smb_vectors_dir)


def _max_chunk_id(session_factory=get_db_session) -> int:
    with session_factory() as db:
        val = db.query(func.max(DBChunk.id)).scalar()
        return int(val or 0)


def _count_papers(ctx: StageContext, session_factory=get_db_session) -> int:
    with session_factory() as db:
        return int(
            db.query(func.count(DBPaper.paper_id))
            .filter(DBPaper.venue == ctx.venue, DBPaper.year == ctx.year)
            .scalar()
            or 0
        )


def _poll_until_complete(
    ctx: StageContext,
    sink: EventSink,
    probe_fn: PollFn,
    *,
    sleep_fn: SleepFn,
    poll_s: float,
    stall_limit: int = 40,
) -> None:
    """Poll probe_fn until complete, or until progress stalls for `stall_limit`
    consecutive iterations. Generic + injectable sleep so tests run instantly.

    The remote launcher returns immediately (work continues in tmux), so the
    stage stays 'running' by polling artifact counts on the SMB share.
    """
    last_done = -1
    stalls = 0
    while True:
        pr = probe_fn(ctx)
        sink(f"poll {ctx.venue} {ctx.year}: {pr.done}/{pr.total} ({pr.percent}%)")
        if pr.is_complete:
            logger.info(f"stage.poll.complete venue={ctx.venue} year={ctx.year} {pr.done}/{pr.total}")
            return
        if pr.done <= last_done:
            stalls += 1
            if stalls >= stall_limit:
                logger.warning(
                    f"stage.poll.stall venue={ctx.venue} year={ctx.year} "
                    f"done={pr.done} stalls={stalls}"
                )
                sink(f"progress stalled at {pr.done}/{pr.total}, stopping poll", level="warning")
                return
        else:
            stalls = 0
            last_done = pr.done
        sleep_fn(poll_s)


# ===================================================================== STAGE 1
def download_min_interval_seconds(ctx: StageContext) -> float:
    if ctx.venue.upper() == "ICML" and ctx.year == 2026:
        return 5.0
    return 0.0


def download_effective_concurrency(ctx: StageContext) -> int:
    if download_min_interval_seconds(ctx) > 0:
        return 1
    return ctx.concurrency


def build_download_cmd(ctx: StageContext) -> list[str]:
    cmd = [
        sys.executable, "scripts/download_subset.py",
        "--conf", ctx.venue, "--year", str(ctx.year),
        "--concurrency", str(download_effective_concurrency(ctx)),
    ]
    min_interval = download_min_interval_seconds(ctx)
    if min_interval > 0:
        cmd.extend(["--min-interval-seconds", f"{min_interval:g}"])
    return cmd


def download_probe(ctx: StageContext) -> Progress:
    return Progress(done=ctx.pdf_count(), total=ctx.manifest_slice_total())


def download_gate(ctx: StageContext) -> GateResult:
    done = ctx.pdf_count()
    total = ctx.manifest_slice_total()
    if total == 0 or done / total >= 0.9:
        return GateResult(True)
    return GateResult(False, f"only {done}/{total} PDFs downloaded")


def make_download_stage(ctx: StageContext) -> StageSpec:
    def run(c: StageContext, sink: EventSink) -> None:
        cmd = build_download_cmd(c)
        sink(
            f"download {c.venue} {c.year} "
            f"(concurrency={download_effective_concurrency(c)}, "
            f"min_interval={download_min_interval_seconds(c):g}s)"
        )
        run_local(cmd, c.project_root, sink)

    return StageSpec(
        key="download", title=f"Download {ctx.venue} {ctx.year} PDFs",
        run=run, probe=download_probe, gate=download_gate,
        # Single global slot for HTTP downloads (be a good citizen to remote
        # servers; one venue's manifest pull at a time).
        resource_key="net",
    )


# ===================================================================== STAGE 2
def _effective_world_size(ctx: StageContext) -> int:
    """ranks_per_gpu × #preflight-passed GPUs, or fallback ctx.world_size when
    preflight hasn't run yet (e.g. unit tests that build the cmd without state).
    Stacking is decided here so kss old-style callers (ranks_per_gpu=1) keep
    1 rank/GPU and 4090 callers (ranks_per_gpu=2) get 2 ranks stacked per card.
    """
    gpu_ids = ctx.state_get("gpu_ids", "")
    if gpu_ids and ctx.ranks_per_gpu > 0:
        n_gpus = len([g for g in gpu_ids.split(",") if g.strip()])
        return n_gpus * ctx.ranks_per_gpu
    return ctx.world_size


def build_extract_remote_cmd(ctx: StageContext) -> str:
    # shlex.quote handles venues whose lower-cased name contains a space
    # ("USENIX Security"). Without quoting, the launcher receives 4 args
    # instead of 3 and run_marker_kss.sh blows up with `((: ...: syntax error`.
    return (
        f"bash {shlex.quote(_marker_sh(ctx))} "
        f"{shlex.quote(ctx.kss_pdf_dir)} "
        f"{shlex.quote(ctx.kss_md_dir)} "
        f"{_effective_world_size(ctx)}"
    )


def extract_env_prefix(ctx: StageContext) -> str:
    gpu_ids = ctx.state_get("gpu_ids", "")
    if not gpu_ids:
        return "STAGGER=3 "
    return f"GPU_IDS={gpu_ids} STAGGER=3 "


def extract_preflight(ctx: StageContext) -> GateResult:
    # Marker (Surya layout + OCR + table + recognition) typically needs ~10-12GB
    # per rank. We require ≥16GB FREE on each candidate card so a co-tenant
    # job (e.g. someone's vLLM holding 23GB) doesn't get our marker stacked on
    # top → OOM cascade. Threshold trips out cards that look "alive" via
    # nvidia-smi but are de-facto unusable.
    # Idle = ≥85% memory free AND ≥16GB absolute. Ratio guards against
    # mixed-size lab boxes where 18GB free on a 32GB card still means another
    # job is parked there; absolute MB guards against tiny cards.
    gpu = gpu_health(ctx.ssh_target, min_free_mb=16384, min_free_ratio=0.85)
    if not gpu:
        logger.error(
            f"stage.extract.preflight.no_gpu target={ctx.ssh_target} "
            f"(no GPU ≥85% idle with ≥16GB free)"
        )
        return GateResult(
            False,
            f"no GPU on {ctx.ssh_target} is ≥85% idle with ≥16GB free; "
            f"retry once a card is idle or pick another ssh_target",
        )
    ctx.state_set(gpu_ids=",".join(map(str, gpu)))
    logger.info(f"stage.extract.preflight.ok gpus={gpu}")
    return GateResult(True)


def extract_probe(ctx: StageContext) -> Progress:
    return Progress(done=ctx.md_count(), total=ctx.pdf_count())


def extract_gate(ctx: StageContext, *, sample: int = 5) -> GateResult:
    """Sample up to `sample` md files: each must be > 200 bytes AND have a
    markdown header line. Also require done/total >= 0.95."""
    md_dir = Path(ctx.smb_md_dir)
    files = sorted(md_dir.glob("*.md"))[:sample] if md_dir.exists() else []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return GateResult(False, f"MD sample unreadable ({f.name}): {exc}")
        if len(text.encode("utf-8")) <= 200 or not any(
            line.lstrip().startswith("#") for line in text.splitlines()
        ):
            logger.error(f"stage.extract.gate.bad_md file={f.name} bytes={len(text)}")
            return GateResult(False, "MD sample looks empty/garbage")

    done = ctx.md_count()
    total = ctx.pdf_count()
    if total > 0 and done / total < 0.95:
        return GateResult(False, f"only {done}/{total} MD extracted")
    return GateResult(True)


def make_extract_stage(
    ctx: StageContext,
    *,
    probe_fn: Optional[PollFn] = None,
    sleep_fn: SleepFn = time.sleep,
    poll_s: float = 30.0,
) -> StageSpec:
    probe = probe_fn or extract_probe

    def run(c: StageContext, sink: EventSink) -> None:
        # W32 warmup: a `find` over the venue's PDF tree on the SMB share BEFORE
        # 6 marker ranks each rglob it. Cold CIFS readdir under N concurrent
        # workers goes into D+ uninterruptible wait — workers stall for minutes.
        # The pre-warm cost is ~one `ls`; afterwards rglob hits the cache.
        sink("extract step 0/3: SMB cache pre-warm (find)")
        _warmup_smb_pdfs(c, sink)

        cmd = build_extract_remote_cmd(c)
        prefix = extract_env_prefix(c)
        sink(f"extract step 1/3: launch marker on {c.ssh_target}: {prefix}{cmd}")
        run_remote(c.ssh_target, cmd, sink, env_prefix=prefix)
        sink("extract step 2/3: marker launched in tmux; polling MD output on SMB")
        _poll_until_complete(c, sink, probe, sleep_fn=sleep_fn, poll_s=poll_s)

        # Without this, manifest.markdown_path stays empty after extract → the
        # downstream preindex stage reads manifest and finds 0 papers → gate
        # fails ("only 0/N papers in SQL"). Previously this was a manual step
        # between extract and preindex (W71). Now it's inside extract so the
        # pipeline is fully automated.
        sink("extract step 3/3: sync MD paths into manifest")
        _sync_manifest_md_paths(c, sink)

    return StageSpec(
        key="extract", title=f"Extract {ctx.venue} {ctx.year} (KSS marker)",
        run=run, probe=probe, gate=extract_gate, preflight=extract_preflight,
        poll_interval_s=poll_s,
        # extract and embed share this key on the same box, so two pipelines
        # can't double-book the GPUs (one in extract, one in embed → OOM).
        resource_key=f"gpu-{ctx.ssh_target}",
    )


def _warmup_smb_pdfs(ctx: StageContext, sink: EventSink) -> None:
    """SSH a single recursive `find` over the venue's SMB PDF tree to warm the
    CIFS cache before N marker ranks race to rglob it. Best-effort; failure
    here just means the workers eat a slower cold rglob (W32).
    """
    pdf_root = f"{ctx.kss_smb_root}/papers/pdf/{ctx.conf_lower}"
    cmd = f'find {shlex.quote(pdf_root)} -name "*.pdf" -type f > /dev/null'
    try:
        run_remote(ctx.ssh_target, cmd, sink)
    except Exception as exc:  # noqa: BLE001 — log and continue
        logger.warning(f"stage.extract.warmup.failed target={ctx.ssh_target}: {exc}")
        sink(f"warmup failed (non-fatal): {exc}", level="warning")


def _sync_manifest_md_paths(ctx: StageContext, sink: EventSink) -> None:
    """Run scripts/sync/sync_new_conf_md_smb.py --conf <venue> --year <year>
    to populate manifest.markdown_path from the SMB markdown tree, so preindex
    can find the freshly-extracted MDs.
    """
    cmd = [
        sys.executable, "scripts/sync/sync_new_conf_md_smb.py",
        "--conf", ctx.conf_lower, "--year", str(ctx.year),
        "--execute",
    ]
    run_local(cmd, ctx.project_root, sink)


# ================================================= STAGE 2 (LOCAL / self-hosted)
# Default modelscope BGE-M3 cache on the 4090 box (W106; env.md). Overridable
# via BGE_M3_PATH — same env the embedder + chunker honor.
_BGE_M3_FALLBACK = "/data/arc/.cache/modelscope/BAAI/bge-m3"


def _local_gpu_allowlist() -> Optional[set[int]]:
    """Optional physical-GPU allowlist for self-hosted local stages, from env
    PIPELINE_LOCAL_GPU_IDS (e.g. "2,3" or "3"). Restricts extract/embed to these
    cards so a SHARED box never lands marker/embed on a reserved GPU — e.g. GPU0
    is a co-tenant's card on the 4090 host (W70 "don't touch"). Unset → no
    restriction (dedicated box). Set it in the backend launch env."""
    import os
    raw = os.environ.get("PIPELINE_LOCAL_GPU_IDS", "").strip()
    if not raw:
        return None
    ids = {int(x) for x in raw.replace(" ", ",").split(",") if x.strip().isdigit()}
    return ids or None


def _local_idle_gpus(*, min_free_mb: int, min_free_ratio: float) -> list[int]:
    """Idle local GPUs (no ssh) ∩ the PIPELINE_LOCAL_GPU_IDS allowlist (if set).
    The intersection is the W106 shared-box safety gate: a card must be both free
    AND permitted before marker/embed may use it.

    The two thresholds are env-overridable for a shared box where the strict
    "≥85% idle" politeness ratio is too conservative (every card carries a
    co-tenant's 10-15GB but still has plenty of room):

    * ``PIPELINE_LOCAL_MIN_FREE_RATIO`` — set 0 to drop the ratio gate and rely on
      the absolute MB floor alone ("use any card with enough room").
    * ``PIPELINE_LOCAL_MIN_FREE_MB`` — the absolute free-memory floor. This is the
      "don't crowd out the existing jobs" guarantee: a card qualifies only if it
      has at least this much FREE, so our ~8GB marker / ~6GB embed lands with
      headroom and never OOMs the co-tenant. Keep it ≥ our job's footprint.
    """
    import os
    ratio_env = os.environ.get("PIPELINE_LOCAL_MIN_FREE_RATIO", "").strip()
    mb_env = os.environ.get("PIPELINE_LOCAL_MIN_FREE_MB", "").strip()
    if ratio_env:
        try:
            min_free_ratio = float(ratio_env)
        except ValueError:
            logger.warning(f"stage.local.bad_ratio_env={ratio_env!r} — using default {min_free_ratio}")
    if mb_env:
        try:
            min_free_mb = int(mb_env)
        except ValueError:
            logger.warning(f"stage.local.bad_mb_env={mb_env!r} — using default {min_free_mb}")
    gpus = gpu_health(None, min_free_mb=min_free_mb, min_free_ratio=min_free_ratio)
    allow = _local_gpu_allowlist()
    if allow is not None:
        kept = [g for g in gpus if g in allow]
        logger.info(f"stage.local.gpu_allowlist allow={sorted(allow)} idle={gpus} kept={kept}")
        return kept
    return gpus


def _local_gpu_env(ctx: StageContext) -> Optional[dict]:
    """Env pinning a local GPU subprocess to the FIRST preflight-selected idle
    GPU. embed_worker.py hardcodes cuda:0, so CUDA_VISIBLE_DEVICES must scope it
    to ONE physical card — else it inherits the backend's CVD and embeds on the
    live-search GPU. CUDA_DEVICE_ORDER=PCI_BUS_ID makes the index physical (W70).
    None when state has no gpu_ids (inherit env)."""
    import os
    gpu_ids = ctx.state_get("gpu_ids", "")
    first = next((g for g in gpu_ids.split(",") if g.strip()), None)
    if not first:
        return None
    return {**os.environ, "CUDA_VISIBLE_DEVICES": first, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def build_extract_local_cmd(ctx: StageContext, *, gpu: int, rank: int, world_size: int) -> list[str]:
    """ONE marker rank, bound to a physical GPU, writing to the LOCAL md dir.

    No run_marker_kss.sh / tmux / ssh — a plain local subprocess per rank.
    Mirrors marker_worker.py's exact arg names (--input/--output/--gpu/--rank/
    --world-size/--recursive)."""
    return [
        sys.executable, "scripts/remote_marker/marker_worker.py",
        "--input", str(ctx.pdf_dir),
        "--output", str(ctx.local_md_dir),
        "--gpu", str(gpu),
        "--rank", str(rank),
        "--world-size", str(world_size),
        "--recursive",
    ]


def extract_local_preflight(ctx: StageContext) -> GateResult:
    """LOCAL gpu_health (no ssh): same thresholds as remote extract (≥16GB free,
    ≥85% idle), intersected with the PIPELINE_LOCAL_GPU_IDS allowlist (W106/W70).
    Records the idle GPU ids in state for run() to distribute ranks."""
    gpu = _local_idle_gpus(min_free_mb=16384, min_free_ratio=0.85)
    if not gpu:
        logger.error(
            "stage.extract.local.preflight.no_gpu "
            "(no allowed local GPU ≥85% idle with ≥16GB free)"
        )
        return GateResult(
            False,
            "no allowed local GPU is ≥85% idle with ≥16GB free; retry once a card "
            "is idle or widen PIPELINE_LOCAL_GPU_IDS",
        )
    ctx.state_set(gpu_ids=",".join(map(str, gpu)))
    logger.info(f"stage.extract.local.preflight.ok gpus={gpu}")
    return GateResult(True)


def extract_local_probe(ctx: StageContext) -> Progress:
    return Progress(done=ctx.md_count_local(), total=ctx.pdf_count())


def extract_local_gate(ctx: StageContext, *, sample: int = 5) -> GateResult:
    """Same acceptance as remote extract_gate, but sampled from the LOCAL md dir:
    each of up to `sample` md files > 200 bytes AND has a markdown header; also
    require done/total >= 0.95."""
    md_dir = ctx.local_md_dir
    files = sorted(md_dir.glob("*.md"))[:sample] if md_dir.exists() else []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return GateResult(False, f"MD sample unreadable ({f.name}): {exc}")
        if len(text.encode("utf-8")) <= 200 or not any(
            line.lstrip().startswith("#") for line in text.splitlines()
        ):
            logger.error(f"stage.extract.local.gate.bad_md file={f.name} bytes={len(text)}")
            return GateResult(False, "MD sample looks empty/garbage")

    done = ctx.md_count_local()
    total = ctx.pdf_count()
    if total > 0 and done / total < 0.95:
        return GateResult(False, f"only {done}/{total} MD extracted")
    return GateResult(True)


def _plan_local_ranks(ctx: StageContext) -> list[tuple[int, int, int]]:
    """Distribute ranks across idle GPUs with ranks_per_gpu stacking.

    Returns [(rank, gpu, world_size), ...]. world_size = len(gpus) * ranks_per_gpu
    (the SAME stacking logic as the remote extract's _effective_world_size). Rank
    r lands on gpus[r % len(gpus)] so ranks_per_gpu procs stack on each card."""
    gpu_ids = ctx.state_get("gpu_ids", "")
    gpus = [int(g) for g in gpu_ids.split(",") if g.strip()]
    if not gpus:
        # Preflight (which gates this stage) must have populated gpu_ids. Never
        # silently fall back to GPU 0 — on a shared box that's a co-tenant's card
        # (W70). Fail loud instead.
        raise RuntimeError(
            "extract.local: no gpu_ids in state — extract_local_preflight must "
            "run and select an allowed idle GPU before run()"
        )
    per = ctx.ranks_per_gpu if ctx.ranks_per_gpu > 0 else 1
    world_size = len(gpus) * per
    return [(r, gpus[r % len(gpus)], world_size) for r in range(world_size)]


def _sync_manifest_md_paths_local(ctx: StageContext, sink: EventSink) -> None:
    """Scoped manifest backfill for THIS (conf, year) from the LOCAL filesystem.
    NOT sync_new_conf_md_smb.py (SMB). Writes only the slice (W106)."""
    cmd = [
        sys.executable, "scripts/sync/sync_local_conf_md.py",
        "--conf", ctx.conf_lower, "--year", str(ctx.year),
    ]
    run_local(cmd, ctx.project_root, sink)


def make_extract_stage_local(
    ctx: StageContext,
    *,
    probe_fn: Optional[PollFn] = None,
    sleep_fn: SleepFn = time.sleep,
    poll_s: float = 30.0,
) -> StageSpec:
    """LOCAL extract: launch N marker ranks via run_local (one process per rank,
    distributed across idle GPUs), then backfill manifest md paths from local FS.
    No run_marker_kss.sh / ssh / SMB (W106)."""
    probe = probe_fn or extract_local_probe

    def run(c: StageContext, sink: EventSink) -> None:
        plan = _plan_local_ranks(c)
        sink(f"extract.local: launch {len(plan)} marker rank(s) on local GPUs")
        for rank, gpu, world_size in plan:
            cmd = build_extract_local_cmd(c, gpu=gpu, rank=rank, world_size=world_size)
            sink(f"extract.local rank {rank}/{world_size} gpu={gpu}: {' '.join(map(str, cmd))}")
            run_local(cmd, c.project_root, sink)
        sink("extract.local: sync MD paths into manifest (local slice)")
        _sync_manifest_md_paths_local(c, sink)

    return StageSpec(
        key="extract", title=f"Extract {ctx.venue} {ctx.year} (local marker)",
        run=run, probe=probe, gate=extract_local_gate, preflight=extract_local_preflight,
        poll_interval_s=poll_s,
        # Local GPU box mutex — distinct from any ssh target's "gpu-{target}".
        resource_key="gpu-local",
    )


# ===================================================================== STAGE 3
def build_preindex_cmd(ctx: StageContext) -> list[str]:
    return [
        sys.executable, "scripts/preindex_md_to_sql_only.py",
        "--conf", ctx.venue, "--year", str(ctx.year),
        "--chunker-type", "section_l1",
    ]


def preindex_probe(ctx: StageContext, *, session_factory=get_db_session) -> Progress:
    done = _count_papers(ctx, session_factory)
    total = max(ctx.manifest_slice_extracted(), ctx.pdf_count())
    return Progress(done=done, total=total)


def preindex_gate(ctx: StageContext, *, session_factory=get_db_session) -> GateResult:
    pr = preindex_probe(ctx, session_factory=session_factory)
    if pr.total > 0 and pr.done < 0.95 * pr.total:
        return GateResult(False, f"only {pr.done}/{pr.total} papers in SQL")
    return GateResult(True)


def make_preindex_stage(ctx: StageContext, *, session_factory=get_db_session) -> StageSpec:
    def run(c: StageContext, sink: EventSink) -> None:
        # First-write-wins: a crashed-mid-preindex resume re-enters this run()
        # with chunks_now already advanced by the partial write. Re-snapshotting
        # would push the watermark PAST those partial chunks, and embed would
        # then skip them via --min-chunk-id (silent data loss). The first snapshot
        # is the authoritative one.
        before = c.state_get("chunk_id_before")
        if before is None:
            before = _max_chunk_id(session_factory)
            c.state_set(chunk_id_before=before)
        else:
            sink(f"resume: reusing chunk_id watermark = {before}")
        sink(f"preindex {c.venue} {c.year}: chunk_id watermark = {before}")
        run_local(build_preindex_cmd(c), c.project_root, sink)

    return StageSpec(
        key="preindex", title=f"Preindex {ctx.venue} {ctx.year} → SQL",
        run=run,
        probe=lambda c: preindex_probe(c, session_factory=session_factory),
        gate=lambda c: preindex_gate(c, session_factory=session_factory),
        # Global single-slot: SQLite WAL handles concurrent readers but a single
        # writer at a time keeps commit batches predictable + avoids surprise
        # contention with /api routes that also write.
        resource_key="sql",
    )


# ===================================================================== STAGE 4
def build_embed_dump_cmd(ctx: StageContext, *, chunk_id_before: int, start_shard: int) -> list[str]:
    return [
        sys.executable, "scripts/remote_embed/dump_chunks.py",
        "--min-chunk-id", str(chunk_id_before + 1),
        "--start-shard", str(start_shard),
    ]


def build_embed_remote_cmd(ctx: StageContext) -> str:
    return f"bash {_embed_sh(ctx)}"


def embed_env_prefix(ctx: StageContext) -> str:
    gpu_ids = ctx.state_get("gpu_ids", "")
    gpus = " ".join(gpu_ids.split(",")) if gpu_ids else ""
    return f'GPUS="{gpus}" '


def embed_preflight(ctx: StageContext) -> GateResult:
    # BGE-M3 (bf16) + reranker + working batch fits comfortably in ~8GB; we
    # demand 12GB free to leave headroom for activations and any same-rank
    # spillover. Same protective principle as extract_preflight.
    gpu = gpu_health(ctx.ssh_target, min_free_mb=12288, min_free_ratio=0.85)
    if not gpu:
        logger.error(
            f"stage.embed.preflight.no_gpu target={ctx.ssh_target} "
            f"(no GPU ≥85% idle with ≥12GB free)"
        )
        return GateResult(
            False,
            f"no GPU on {ctx.ssh_target} is ≥85% idle with ≥12GB free; "
            f"retry once a card is idle or pick another ssh_target",
        )
    ctx.state_set(gpu_ids=",".join(map(str, gpu)))
    logger.info(f"stage.embed.preflight.ok gpus={gpu}")
    return GateResult(True)


def embed_smb_probe(ctx: StageContext) -> Progress:
    # W82: poll the locally-mounted SMB share (where KSS writes), NOT the C-drive
    # copy. The old probe counted data/remote_embed/vectors/, which stays empty
    # until ship-down (step 5) — so step 4's poll saw done=0 for all 40 stall
    # iterations and dead-waited the full ~20 min before giving up. The SMB share
    # is the same NAS the worker writes to, so Arc sees shards the instant they
    # land and the poll completes on real embed progress. NUMBER-based
    # [shard_start, shard_end) (W81): the shared dir holds every venue's parquet,
    # so count only OUR shard span; before shard_end is known (mid-run) fall back
    # to #chunk shards dumped at-or-after the anchor.
    shard_start = int(ctx.state_get("shard_start", 0) or 0)
    shard_end_raw = ctx.state_get("shard_end")
    shard_end = int(shard_end_raw) if shard_end_raw is not None else None
    done = _count_shards_in_range(_embed_smb_vectors_dir(ctx), _VECTORS_GLOB, _SHARD_RE, shard_start, shard_end)
    if shard_end is not None:
        total = max(shard_end - shard_start, 0)
    else:
        total = _count_shards_in_range(_embed_chunks_dir(ctx), _CHUNKS_GLOB, _CHUNK_SHARD_RE, shard_start, None)
    return Progress(done=done, total=max(total, done))


def embed_gate(ctx: StageContext) -> GateResult:
    shard_start = int(ctx.state_get("shard_start", 0) or 0)
    shard_end = int(ctx.state_get("shard_end", shard_start) or shard_start)
    if shard_end > shard_start:
        return GateResult(True)
    return GateResult(False, f"no new parquet produced (shard_start={shard_start}, shard_end={shard_end})")


def make_embed_stage(
    ctx: StageContext,
    *,
    probe_fn: Optional[PollFn] = None,
    sleep_fn: SleepFn = time.sleep,
    poll_s: float = 30.0,
) -> StageSpec:
    probe = probe_fn or embed_smb_probe  # W82: mounted SMB share, not empty C-drive copy

    def run(c: StageContext, sink: EventSink) -> None:
        chunk_id_before = int(c.state_get("chunk_id_before", 0) or 0)
        # SINGLE shard-NUMBER anchor (first-write-wins for resume). The anchor is
        # the next free chunks shard index; vectors are named identically (1:1),
        # so the SAME number bounds chunks AND vectors for THIS task. dump's
        # --start-shard and ingest's [shard_start, shard_end) are this one number.
        #
        # W81: the old code derived shard_start from a COUNT of the shared
        # vectors/ dir and dump_start_shard from a COUNT of the shared chunks/
        # dir — two different numbers, both polluted by other concurrent venues'
        # files. ingest then sliced by the wrong bounds and grabbed/skipped the
        # wrong shards. Unify to the chunks anchor (contiguous, appended-only,
        # serialized by the per-box GPU lock so no peer dumps concurrently).
        prev_anchor = c.state_get("dump_start_shard")
        if prev_anchor is None:
            start_shard = _count_files(_embed_chunks_dir(c), _CHUNKS_GLOB)
            c.state_set(dump_start_shard=start_shard, shard_start=start_shard)
        else:
            start_shard = int(prev_anchor)
            sink(f"resume: reusing shard anchor = {start_shard}")

        sink(f"embed step 1/5: dump chunks (min_chunk_id={chunk_id_before + 1}, start_shard={start_shard})")
        run_local(build_embed_dump_cmd(c, chunk_id_before=chunk_id_before, start_shard=start_shard), c.project_root, sink)

        # Authoritative per-task upper bound: anchor + #chunk shards THIS task
        # just dumped (NUMBER >= anchor, measured now while the per-box GPU lock
        # is held so no peer is dumping). NEVER recount vectors/ after ship-down:
        # ship.sh down rsyncs EVERY venue's parquet from the shared SMB dir, so a
        # count there inflates the range and ingest spills onto other venues.
        n_dumped = _count_shards_in_range(_embed_chunks_dir(c), _CHUNKS_GLOB, _CHUNK_SHARD_RE, start_shard)
        shard_end = start_shard + n_dumped
        c.state_set(shard_end=shard_end)

        sink("embed step 2/5: ship chunks up to KSS")
        run_local(["bash", "scripts/remote_embed/ship.sh", "up"], c.project_root, sink, env={"MIN_SHARD": str(start_shard)})

        sink("embed step 3/5: run BGE-M3 embed on KSS")
        run_remote(c.ssh_target, build_embed_remote_cmd(c), sink, env_prefix=embed_env_prefix(c))
        sink("embed step 4/5: poll parquet output (SMB share)")
        _poll_until_complete(c, sink, probe, sleep_fn=sleep_fn, poll_s=poll_s)

        sink("embed step 5/5: ship vectors back down")
        run_local(["bash", "scripts/remote_embed/ship.sh", "down"], c.project_root, sink)

        sink(f"embed done: shards [{start_shard}, {shard_end}) = {n_dumped} produced")

    return StageSpec(
        key="embed", title=f"Embed {ctx.venue} {ctx.year} (KSS BGE-M3)",
        run=run, probe=probe, gate=embed_gate, preflight=embed_preflight,
        poll_interval_s=poll_s,
        # vectors/ is shared across venues — a count-based probe sees other
        # venues' shards as "done" and would skip the real embed (W72).
        skip_when_satisfied=False,
        # Shares the GPU-box mutex with extract: two pipelines on the same
        # ssh_target can't simultaneously hit GPUs (marker 9G × 6 + embed 8G+
        # would OOM a 24G card).
        resource_key=f"gpu-{ctx.ssh_target}",
    )


# ================================================= STAGE 4 (LOCAL / self-hosted)
def build_embed_worker_local_cmd(ctx: StageContext, *, model_dir: str) -> list[str]:
    """Single-GPU local embed pass (world_size=1, rank 0). Reads the shared
    chunks dir, writes the shared vectors dir. No ship.sh up/down — both dirs
    are already local on the self-hosted box (W106). Mirrors embed_worker.py's
    exact arg names (--input/--output/--model-dir/--rank/--world-size)."""
    return [
        sys.executable, "scripts/remote_embed/embed_worker.py",
        "--input", "data/remote_embed/chunks",
        "--output", "data/remote_embed/vectors",
        "--model-dir", model_dir,
        "--rank", "0",
        "--world-size", "1",
    ]


def _bge_m3_model_dir() -> str:
    import os
    return os.environ.get("BGE_M3_PATH") or _BGE_M3_FALLBACK


def embed_local_preflight(ctx: StageContext) -> GateResult:
    """LOCAL gpu_health (no ssh) ∩ PIPELINE_LOCAL_GPU_IDS allowlist: same
    thresholds as remote embed (≥12GB free, ≥85% idle). run() pins the embed
    subprocess to gpu_ids[0] (W106/W70)."""
    gpu = _local_idle_gpus(min_free_mb=12288, min_free_ratio=0.85)
    if not gpu:
        logger.error(
            "stage.embed.local.preflight.no_gpu "
            "(no allowed local GPU passed the configured free-memory constraints)"
        )
        return GateResult(
            False,
            "no allowed local GPU passed PIPELINE_LOCAL_MIN_FREE_MB / "
            "PIPELINE_LOCAL_MIN_FREE_RATIO; inspect runner.gpu_health logs, wait "
            "for memory to become free, or correct PIPELINE_LOCAL_GPU_IDS",
        )
    ctx.state_set(gpu_ids=",".join(map(str, gpu)))
    logger.info(f"stage.embed.local.preflight.ok gpus={gpu}")
    return GateResult(True)


def embed_local_probe(ctx: StageContext) -> Progress:
    """NUMBER-based [shard_start, shard_end) over the LOCAL vectors dir (where
    the local embed worker writes directly — no ship-down). Same accounting as
    embed_smb_probe but pointed at the C-drive copy, which IS the live output in
    self-hosted mode (W106 / W81)."""
    shard_start = int(ctx.state_get("shard_start", 0) or 0)
    shard_end_raw = ctx.state_get("shard_end")
    shard_end = int(shard_end_raw) if shard_end_raw is not None else None
    done = _count_shards_in_range(_embed_vectors_dir(ctx), _VECTORS_GLOB, _SHARD_RE, shard_start, shard_end)
    if shard_end is not None:
        total = max(shard_end - shard_start, 0)
    else:
        total = _count_shards_in_range(_embed_chunks_dir(ctx), _CHUNKS_GLOB, _CHUNK_SHARD_RE, shard_start, None)
    return Progress(done=done, total=max(total, done))


def make_embed_stage_local(
    ctx: StageContext,
    *,
    probe_fn: Optional[PollFn] = None,
    sleep_fn: SleepFn = time.sleep,
    poll_s: float = 30.0,
) -> StageSpec:
    """LOCAL embed: dump chunks → run BGE-M3 embed_worker via run_local. No
    ship.sh up/down (chunks + vectors are local). SAME shard-anchor / chunk_id
    state logic as the remote embed (W106)."""
    probe = probe_fn or embed_local_probe

    def run(c: StageContext, sink: EventSink) -> None:
        chunk_id_before = int(c.state_get("chunk_id_before", 0) or 0)
        # SAME single shard-NUMBER anchor (first-write-wins for resume) as the
        # remote embed: next free chunks shard index, unifying shard_start.
        prev_anchor = c.state_get("dump_start_shard")
        if prev_anchor is None:
            start_shard = _count_files(_embed_chunks_dir(c), _CHUNKS_GLOB)
            c.state_set(dump_start_shard=start_shard, shard_start=start_shard)
        else:
            start_shard = int(prev_anchor)
            sink(f"resume: reusing shard anchor = {start_shard}")

        sink(f"embed.local step 1/2: dump chunks (min_chunk_id={chunk_id_before + 1}, start_shard={start_shard})")
        run_local(build_embed_dump_cmd(c, chunk_id_before=chunk_id_before, start_shard=start_shard), c.project_root, sink)

        # Authoritative per-task upper bound: anchor + #chunk shards THIS task
        # just dumped (NUMBER >= anchor). Same as remote embed.
        n_dumped = _count_shards_in_range(_embed_chunks_dir(c), _CHUNKS_GLOB, _CHUNK_SHARD_RE, start_shard)
        shard_end = start_shard + n_dumped
        c.state_set(shard_end=shard_end)

        model_dir = _bge_m3_model_dir()
        # Pin embed_worker (hardcoded cuda:0) to the allowed idle GPU; without this
        # it inherits the backend's CUDA_VISIBLE_DEVICES and lands on the live-
        # search card (W106/W70).
        gpu_env = _local_gpu_env(c)
        pinned = gpu_env.get("CUDA_VISIBLE_DEVICES") if gpu_env else "inherit"
        sink(f"embed.local step 2/2: run BGE-M3 embed locally (model_dir={model_dir}, gpu={pinned})")
        run_local(
            build_embed_worker_local_cmd(c, model_dir=model_dir),
            c.project_root,
            sink,
            env=gpu_env,
        )

        sink(f"embed.local done: shards [{start_shard}, {shard_end}) = {n_dumped} produced")

    return StageSpec(
        key="embed", title=f"Embed {ctx.venue} {ctx.year} (local BGE-M3)",
        run=run, probe=probe, gate=embed_gate, preflight=embed_local_preflight,
        poll_interval_s=poll_s,
        # vectors/ is shared across venues — never satisfied-skip (W72). Same as
        # remote embed.
        skip_when_satisfied=False,
        resource_key="gpu-local",
    )


# ===================================================================== STAGE 5
def _qdrant_reachable() -> bool:
    """Probe the configured papers collection endpoint. True for HTTP 2xx."""
    import os
    import urllib.request

    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
    collection_url = f"{qdrant_url}/collections/papers"
    try:
        req = urllib.request.Request(collection_url)
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 (localhost only)
            return 200 <= resp.status < 300
    except Exception as exc:
        logger.warning(f"stage.ingest.qdrant.unreachable url={collection_url}: {exc}")
        return False


def build_ingest_cmd(ctx: StageContext) -> list[str]:
    cmd = [
        sys.executable, "scripts/remote_embed/local_ingest.py",
        "--shard-start", str(int(ctx.state_get("shard_start", 0) or 0)),
    ]
    shard_end = ctx.state_get("shard_end")
    if shard_end is not None:
        cmd += ["--shard-end", str(int(shard_end))]
    return cmd


def ingest_preflight(ctx: StageContext) -> GateResult:
    if not _qdrant_reachable():
        import os

        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
        return GateResult(False, f"Qdrant unreachable at {qdrant_url}")
    return GateResult(True)


def _ingest_done_shards(ctx: StageContext, shard_start: int, shard_end: Optional[int] = None) -> int:
    """Count distinct shard NUMBERS in [shard_start, shard_end) recorded in
    ingest_done.txt. `shard_end=None` → no upper bound. NUMBER-based so other
    venues' shards (>= shard_end) sharing the done-file are excluded (W81)."""
    done_file = _embed_vectors_dir(ctx) / "ingest_done.txt"
    if not done_file.exists():
        return 0
    seen: set[int] = set()
    try:
        for line in done_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            match = _SHARD_RE.search(line)
            idx = int(match.group(1)) if match else (int(line) if line.isdigit() else None)
            if idx is not None and idx >= shard_start and (shard_end is None or idx < shard_end):
                seen.add(idx)
    except OSError as exc:
        logger.warning(f"stage.ingest.done_file.error: {exc}")
        return 0
    return len(seen)


def ingest_probe(ctx: StageContext) -> Progress:
    # NUMBER-based (W81): total is OUR shard span [shard_start, shard_end), not a
    # raw count of the shared vectors/ dir (which holds every venue's parquet).
    shard_start = int(ctx.state_get("shard_start", 0) or 0)
    shard_end_raw = ctx.state_get("shard_end")
    shard_end = int(shard_end_raw) if shard_end_raw is not None else None
    done = _ingest_done_shards(ctx, shard_start, shard_end)
    if shard_end is not None:
        total = max(shard_end - shard_start, 0)
    else:
        # legacy / no upper bound: count OUR shards present (NUMBER >= start).
        total = _count_shards_in_range(_embed_vectors_dir(ctx), _VECTORS_GLOB, _SHARD_RE, shard_start, None)
    return Progress(done=done, total=total)


def ingest_gate(ctx: StageContext) -> GateResult:
    pr = ingest_probe(ctx)
    if pr.total > 0 and pr.done >= pr.total:
        return GateResult(True)
    return GateResult(False, f"only {pr.done}/{pr.total} new shards ingested")


def make_ingest_stage(ctx: StageContext) -> StageSpec:
    def run(c: StageContext, sink: EventSink) -> None:
        cmd = build_ingest_cmd(c)
        sink(f"ingest shards from {c.state_get('shard_start', 0)} → Qdrant")
        run_local(cmd, c.project_root, sink)

    return StageSpec(
        key="ingest", title=f"Ingest {ctx.venue} {ctx.year} → Qdrant",
        run=run, probe=ingest_probe, gate=ingest_gate, preflight=ingest_preflight,
        # ingest_done.txt / vectors/ are shared across venues — never satisfied-
        # skip; build_ingest_cmd already scopes to [shard_start, shard_end). W72.
        skip_when_satisfied=False,
        # Single global slot: Qdrant commit charge headroom is tight (W58/59/60).
        # Two concurrent ingests would race for mmap commits → OOM.
        resource_key="qdrant",
    )
