"""The 6 ingest stages, each expressed as pure builders + a thin run().

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
  6 cleanup   LOCAL   delete the vectors + SQL document a re-indexed paper replaced,
                      sync withdrawn flags to Qdrant (reads the preindex report), and
                      remove the task's own dump/vector files (data/remote_embed/tasks/<id>)

Everything is generic for ANY (venue, year): no conference is special-cased and
GPU ids are always discovered, never hardcoded.
"""
from __future__ import annotations

import functools
import json
import re
import shlex
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

from loguru import logger
from sqlalchemy import func

from core.db import get_db_session
from core.db.models import Paper as DBPaper
from core.indexing.embedding_profile import get_profile
from app.services.pipeline.context import StageContext
from app.services.pipeline.spec import GateResult, Progress, StageSpec
from app.services.pipeline.runners import (
    EventSink,
    gpu_health,
    run_local,
    run_remote,
)
from pipeline.openlist_reader import VolumeNotFound, VolumeReader
from pipeline.paper_crawler.slice_bundle import (
    _SAFE_NAME,
    SIDECAR_SUFFIX,
    bundle_file_name,
    inbox_dir,
    receive_slice_bundle,
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

RECEIVE_POLL_SECONDS = 15.0
# Stage option -> the kind of bundle that stage receives when it is "push".
_RECEIVE_OPTIONS = {"download_from": "pdf", "extract_from": "md"}
# What each option may name besides "local": "push" waits for a bundle, "volume" copies the
# files out of a 115 volume (the downloading host wrote them there; download only).
_RECEIVE_SOURCES = {"download_from": ("push", "volume"), "extract_from": ("push",)}
VOLUME_PROGRESS_EVERY = 50
# Files taken from a conference slice's volume directory: the PDF and the record beside it.
VOLUME_SUFFIXES = (".pdf", SIDECAR_SUFFIX)
# The volume a conference slice is received from when options.volume is absent: the 115 volume the
# downloading host (cfscan) fills. Mirrors arxiv_stages.DEFAULT_VOLUME.
DEFAULT_CONFERENCE_VOLUME = "/cfs-tier1"
# This host (.88) never fetches from the internet: cfscan downloads into the 115 volume and the host
# only receives. Every refusal of a local download starts with this text.
HOST_NEVER_DOWNLOADS = "downloads run on cfscan into the 115 volume; this host only receives"
# What the downloading host leaves in the volume besides the PDFs, per slice, and where each lands in the
# corpus tree: a record per paper it tried and could not get, and the downloader's identity rejections.
ATTEMPTS_VOLUME_DIR = "attempts"
ATTEMPTS_CORPUS_DIR = "download_attempts"


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


def _next_free_shard(*dirs: Path) -> int:
    """One past the highest shard number in dirs (0 when empty). Unlike a file
    count, this never lands on a number a partial or earlier dump already wrote."""
    top = -1
    for directory in dirs:
        if not directory.exists():
            continue
        for p in directory.iterdir():
            m = _CHUNK_SHARD_RE.search(p.name) or _SHARD_RE.search(p.name)
            if m:
                top = max(top, int(m.group(1)))
    return top + 1


_SQL_IN_BATCH = 900  # stay under SQLite's historical 999-variable limit per IN list


def _embed_root(ctx: StageContext) -> Path:
    return ctx.project_root / "data" / "remote_embed"


def _embed_task_dir(ctx: StageContext) -> Path:
    """Where one task's dump, embed and ingest files live: <remote_embed>/tasks/<task_id>/
    {chunks,vectors}. Task-scoped so a run only ever reads what it wrote itself: the
    embed worker embeds every chunks shard without a parquet and the ingest loads every
    parquet not in its done list, so a leftover (truncated, stale, another venue's) in a
    shared directory would break or pollute every later run."""
    return _embed_root(ctx) / "tasks" / str(ctx.task_id)


def _embed_chunks_dir(ctx: StageContext) -> Path:
    # The retired "remote" topology moves shards with scripts/remote_embed/ship.sh, which
    # hard-codes the shared directories on both hosts, so it keeps them (not scopable
    # without reviving it).
    base = _embed_root(ctx) if ctx.execution_mode == "remote" else _embed_task_dir(ctx)
    return base / "chunks"


def _embed_vectors_dir(ctx: StageContext) -> Path:
    base = _embed_root(ctx) if ctx.execution_mode == "remote" else _embed_task_dir(ctx)
    return base / "vectors"


def _embed_smb_vectors_dir(ctx: StageContext) -> Path:
    """The locally-mounted SMB share where the KSS embed worker writes parquets.
    Same NAS the worker writes to (zero-copy, cross-host same-source), readable
    from Arc with a plain glob (no ssh) — the SAME mechanism extract uses via
    smb_md_dir. Polled by embed_smb_probe so step 4 tracks live progress instead
    of the C-drive copy, which stays empty until ship-down (W82)."""
    return Path(ctx.smb_vectors_dir)


def _count_papers(ctx: StageContext, session_factory=get_db_session) -> int:
    # A context whose slice is not (venue, year) — arXiv month slices — says
    # which rows are its own.
    clause_fn = getattr(ctx, "paper_count_clause", None)
    clause = (
        clause_fn(DBPaper) if clause_fn is not None
        else (DBPaper.venue == ctx.venue) & (DBPaper.year == ctx.year)
    )
    with session_factory() as db:
        return int(db.query(func.count(DBPaper.paper_id)).filter(clause).scalar() or 0)


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


def _record_dir(ctx: StageContext, kind: str) -> Path:
    return ctx.corpus_root / kind / ctx.conf_lower / str(ctx.year)


def _read_record(path: Path) -> dict:
    """A downloading-host record as a dict; one that is missing or does not parse proves nothing."""
    from pipeline.paper_crawler.runtime_files import load_json_with_recovery

    try:
        record = load_json_with_recovery(path, None)
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _tried_papers(ctx: StageContext, paper_ids: dict[str, str]) -> set[str]:
    """The file stems of ``paper_ids`` (stem -> paper_id) the downloading host tried: an attempt record
    (download_attempts/) or an identity-rejection record (rejected_identity/) that names the paper and
    holds at least one attempt or rejection."""
    from pipeline.paper_crawler.ccf_a_main_corpus import REJECTED_IDENTITY_DIR

    tried: set[str] = set()
    for stem, paper_id in paper_ids.items():
        attempts = _read_record(_record_dir(ctx, ATTEMPTS_CORPUS_DIR) / f"{stem}.json")
        rejected = _read_record(_record_dir(ctx, REJECTED_IDENTITY_DIR) / f"{stem}.json")
        if (attempts.get("paper_id") == paper_id and isinstance(attempts.get("attempts"), list)
                and attempts["attempts"]) or (
                rejected.get("paper_id") == paper_id and isinstance(rejected.get("rejections"), list)
                and rejected["rejections"]):
            tried.add(stem)
    return tried


def download_gate(ctx: StageContext) -> GateResult:
    """The slice's PDFs are in: at least 90% of the manifest slice, or every paper still missing
    was tried on the downloading host (and stays on the refill list). Only the slice's own PDFs
    count, not whatever else shares the directory."""
    from pipeline.paper_crawler.ccf_a_main_corpus import sanitize_file_component

    total = ctx.manifest_slice_total()
    if total == 0:
        return GateResult(True)
    wanted = {sanitize_file_component(r["paper_id"]): r["paper_id"] for r in ctx.manifest_slice() if r.get("paper_id")}
    directory = ctx.pdf_dir
    held = {p.stem for p in directory.glob("*.pdf")} if directory.is_dir() else set()
    present = wanted.keys() & held
    if len(present) / total >= 0.9:
        return GateResult(True)
    missing = {stem: wanted[stem] for stem in wanted.keys() - present}
    tried = _tried_papers(ctx, missing)
    if missing and len(tried) == len(missing):
        return GateResult(True, f"{len(present)}/{total} PDFs; {len(missing)} still missing after download "
                                "attempts (kept on the refill list)")
    return GateResult(False, f"only {len(present)}/{total} PDFs downloaded "
                             f"({len(tried)} tried, {len(missing) - len(tried)} never tried)")


def receive_mode(options: Optional[dict], key: str) -> Optional[str]:
    """``"push"`` when options[key] says the stage's output is pushed here by the host that
    produced it, ``"volume"`` when the stage copies it out of a 115 volume; None when the
    stage produces it itself (the default)."""
    value = (options or {}).get(key)
    if value in (None, "", "local"):
        return None
    allowed = _RECEIVE_SOURCES.get(key, ("push",))
    if value not in allowed:
        raise ValueError(
            f"options.{key} must be 'local' or {' or '.join(repr(v) for v in allowed)}, got {value!r}")
    return value


def _download_from_volume(options: Optional[dict]) -> bool:
    """True when the download stage copies out of the volume: asked for, or no download_from at
    all (a conference slice's default; 'local' selects the internet download)."""
    requested = (options or {}).get("download_from")
    return requested in (None, "") or receive_mode(options, "download_from") == "volume"


def validate_receive_options(options: Optional[dict], execution_mode: Optional[str], *, conference: bool) -> None:
    """Refuse receive options that this run cannot honour, before a task exists."""
    asked = [key for key in _RECEIVE_OPTIONS if receive_mode(options, key)]
    if asked and not conference:
        raise ValueError(f"options {asked} apply to conference slices only")
    if asked and execution_mode == "remote":
        raise ValueError(f"options {asked} are not available in the retired remote execution mode")
    if conference and (options or {}).get("download_from") == "local":
        raise ValueError(f"{HOST_NEVER_DOWNLOADS} (download_from='volume' or 'push')")
    if conference and _download_from_volume(options):
        # options.volume is optional (DEFAULT_CONFERENCE_VOLUME) but must be well-formed when given.
        volume = (options or {}).get("volume")
        if volume is not None and (
                not isinstance(volume, str) or not volume.startswith("/") or ".." in volume.split("/")):
            raise ValueError(
                "options.volume must be the absolute path of the volume on this host's OpenList node "
                f"(e.g. {DEFAULT_CONFERENCE_VOLUME}), got {volume!r}")
        prefix = (options or {}).get("volume_prefix")
        if prefix is not None and (not isinstance(prefix, str) or ".." in prefix.split("/")):
            raise ValueError(f"options.volume_prefix must be a path inside the volume without '..', got {prefix!r}")


def receive_bundle_path(ctx: StageContext, kind: str) -> Path:
    """Where the producing host pushes this slice's bundle of ``kind`` ("pdf" | "md")."""
    return inbox_dir(ctx.corpus_root) / bundle_file_name(ctx.conf_lower, ctx.year, kind)


def _receive_run(kind: str, dest: Callable[[StageContext], Path], poll_s: Optional[float], sleep_fn: SleepFn):
    def run(c: StageContext, sink: EventSink) -> None:
        # options.receive_poll_seconds tunes how often the inbox is looked at.
        interval = poll_s if poll_s is not None else float(
            (c.options or {}).get("receive_poll_seconds", RECEIVE_POLL_SECONDS))
        receive_slice_bundle(
            receive_bundle_path(c, kind), dest(c), venue=c.venue, year=c.year, kind=kind,
            abort=c.abort, log=sink, poll_s=interval, sleep_fn=sleep_fn)
    return run


def make_download_stage_received(ctx: StageContext, *, poll_s: Optional[float] = None,
                                 sleep_fn: SleepFn = time.sleep) -> StageSpec:
    """The download stage when another host downloads: wait for its PDF bundle, verify
    it and land the PDFs in pdf_dir. Same probe and gate as the local download."""
    return StageSpec(
        key="download", title=f"Receive {ctx.venue} {ctx.year} PDFs (pushed)",
        run=_receive_run("pdf", lambda c: c.pdf_dir, poll_s, sleep_fn),
        probe=download_probe, gate=download_gate,
        # Waiting holds nothing, so no slot is taken.
        resource_key="",
    )


@dataclass
class VolumeReceipt:
    """What one receive_volume_files call did."""

    fetched: int = 0
    skipped: int = 0
    bytes: int = 0
    pdfs: list[str] = field(default_factory=list)  # every PDF the volume holds there, relative to the directory


def volume_root(ctx: StageContext) -> str:
    return str((ctx.options or {}).get("volume") or DEFAULT_CONFERENCE_VOLUME).rstrip("/")


def volume_source_dir(ctx: StageContext) -> str:
    """The volume directory of this slice: options.volume (default DEFAULT_CONFERENCE_VOLUME) +
    options.volume_prefix (default ``<conf lower>/<year>``, the layout the writer uses)."""
    prefix = str((ctx.options or {}).get("volume_prefix") or f"{ctx.conf_lower}/{ctx.year}").strip("/")
    return f"{volume_root(ctx)}/{prefix}"


def _volume_order(rel: str, suffixes: Sequence[str]) -> tuple[str, int]:
    """Sort key putting a paper's companion files before its PDF, the writer's own order,
    so a PDF on disk always has its record."""
    for suffix in suffixes:
        if rel.endswith(suffix):
            return rel[: -len(suffix)], int(suffix == ".pdf")
    return rel, 0


def receive_volume_files(
    reader: VolumeReader, src_dir: str, dest_dir: Path, *, suffixes: Sequence[str] = VOLUME_SUFFIXES,
    abort: threading.Event, log: Callable[[str], None],
) -> VolumeReceipt:
    """Copy the files below ``src_dir`` on the volume that end in one of ``suffixes`` into
    ``dest_dir``, keeping their relative paths.

    A file that is already here with the volume's size is left alone, so a rerun moves
    only what is missing; every other one lands atomically (VolumeReader.fetch).

    A directory the volume does not have holds no files: nothing was uploaded there yet, and the
    stage's gate decides on what is already on disk."""
    try:
        found = reader.list_files(src_dir)
    except VolumeNotFound:
        log(f"volume {src_dir.rstrip('/')}: no such directory, 0 files")
        return VolumeReceipt()
    listing = {rel: size for rel, size in found.items() if rel.endswith(tuple(suffixes))}
    for rel in listing:
        if not all(_SAFE_NAME.match(part) for part in rel.split("/")):
            raise ValueError(f"unsafe file name on the volume: {rel!r}")
    receipt = VolumeReceipt(pdfs=sorted(rel for rel in listing if rel.endswith(".pdf")))
    source = src_dir.rstrip("/")
    for done, rel in enumerate(sorted(listing, key=lambda r: _volume_order(r, suffixes)), 1):
        if abort.is_set():
            raise RuntimeError("cancelled while receiving from the volume")
        dest = Path(dest_dir) / rel
        if dest.is_file() and dest.stat().st_size == listing[rel]:
            receipt.skipped += 1
        else:
            receipt.bytes += reader.fetch(f"{source}/{rel}", dest, size=listing[rel])
            receipt.fetched += 1
        if done % VOLUME_PROGRESS_EVERY == 0:
            log(f"volume {source}: received {done:,} of {len(listing):,} files "
                f"({receipt.bytes / 1e6:.1f} MB fetched)")
    log(f"volume {source}: {receipt.fetched:,} fetched ({receipt.bytes / 1e6:.1f} MB), "
        f"{receipt.skipped:,} already here")
    return receipt


def make_download_stage_volume(ctx: StageContext, *, reader: Optional[VolumeReader] = None) -> StageSpec:
    """The download stage when the files are in a 115 volume: copy them into pdf_dir through
    the read-only OpenList user. Same probe and gate as the local download; a gate failure also
    says where the rest has to come from, because this host does not download."""

    def gate(c: StageContext) -> GateResult:
        verdict = download_gate(c)
        if verdict.ok:
            return verdict
        return GateResult(False, f"{verdict.reason}; the missing PDFs must be downloaded into the volume (cfscan) first")

    def run(c: StageContext, sink: EventSink) -> None:
        from pipeline.paper_crawler.ccf_a_main_corpus import REJECTED_IDENTITY_DIR, sanitize_file_component

        volume = reader or VolumeReader.from_env()
        source = volume_source_dir(c)
        sink(f"receive {c.venue} {c.year} PDFs from the volume {source}")
        receipt = receive_volume_files(volume, source, c.pdf_dir, abort=c.abort, log=sink)
        wanted = {f"{sanitize_file_component(r['paper_id'])}.pdf" for r in c.manifest_slice()}
        held = {Path(rel).name for rel in receipt.pdfs}
        # Files the volume holds beyond the slice land too; adding them to the manifest is
        # the collect step's job, not the download's.
        sink(f"{len(receipt.pdfs):,} PDFs on the volume; {len(held - wanted):,} not in the manifest slice "
             f"(kept), {len(wanted - held):,} of the slice not on the volume")
        # What the downloading host tried and could not get comes with the PDFs: the gate lets a slice
        # go on once every paper it still lacks was tried there.
        for volume_dir, corpus_dir, suffixes, label in (
                (ATTEMPTS_VOLUME_DIR, ATTEMPTS_CORPUS_DIR, (".json",), "download attempt"),
                (REJECTED_IDENTITY_DIR, REJECTED_IDENTITY_DIR, (".json", ".pdf"), "identity rejection")):
            records = receive_volume_files(
                volume, f"{volume_root(c)}/{volume_dir}/{c.conf_lower}/{c.year}", _record_dir(c, corpus_dir),
                suffixes=suffixes, abort=c.abort, log=sink)
            sink(f"{records.fetched + records.skipped - len(records.pdfs):,} {label} records for {c.venue} {c.year}")

    return StageSpec(
        key="download", title=f"Receive {ctx.venue} {ctx.year} PDFs (volume)",
        run=run, probe=download_probe, gate=gate,
        # The copy is network traffic like a download: the same single slot.
        resource_key="net",
    )


def make_download_stage(ctx: StageContext) -> StageSpec:
    """Download is received from the volume unless options.download_from says 'push'. This host
    never downloads: 'local' is refused here too, for a task stored before validate_receive_options did."""
    if receive_mode(ctx.options, "download_from") == "push":
        return make_download_stage_received(ctx)
    if _download_from_volume(ctx.options):
        return make_download_stage_volume(ctx)
    raise ValueError(f"{HOST_NEVER_DOWNLOADS} (download_from='volume' or 'push')")


def make_download_stage_local(ctx: StageContext) -> StageSpec:
    """The internet download on the host that runs it. No dispatcher reaches it on this host
    (make_download_stage refuses 'local'); it stays for a host that downloads."""

    def run(c: StageContext, sink: EventSink) -> None:
        cmd = build_download_cmd(c)
        sink(
            f"download {c.venue} {c.year} "
            f"(concurrency={download_effective_concurrency(c)}, "
            f"min_interval={download_min_interval_seconds(c):g}s)"
        )
        run_local(cmd, c.project_root, sink, abort=c.abort)

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
        run_remote(c.ssh_target, cmd, sink, env_prefix=prefix, abort=c.abort)
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
        run_remote(ctx.ssh_target, cmd, sink, abort=ctx.abort)
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
    run_local(cmd, ctx.project_root, sink, abort=ctx.abort)


# ================================================= STAGE 2 (LOCAL / self-hosted)
def _local_gpu_allowlist() -> Optional[set[int]]:
    """Optional physical-GPU allowlist for self-hosted local stages, from env
    PIPELINE_LOCAL_GPU_IDS (e.g. "2,3" or "3"). Restricts the extract to these
    cards so a SHARED box never lands it on a reserved GPU — e.g. GPU0
    is a co-tenant's card on the 4090 host (W70 "don't touch"). Unset → no
    restriction (dedicated box). Set it in the backend launch env."""
    import os
    raw = os.environ.get("PIPELINE_LOCAL_GPU_IDS", "").strip()
    if not raw:
        return None
    ids = {int(x) for x in raw.replace(" ", ",").split(",") if x.strip().isdigit()}
    return ids or None


def _local_idle_gpus(*, min_free_mb: int) -> list[int]:
    """Local GPUs (no ssh) with at least ``min_free_mb`` MiB free ∩ the PIPELINE_LOCAL_GPU_IDS
    allowlist (if set). The intersection is the W106 shared-box safety gate: a card must have
    room AND be permitted before the extract may use it.

    The gate is absolute free memory only. The card always carries the resident vLLM services,
    so it is never "85% idle" and an idle ratio would refuse every run; the floor is the
    registered extractor's own figure (`extract_local_min_free_mb`).
    """
    gpus = gpu_health(None, min_free_mb=min_free_mb)
    allow = _local_gpu_allowlist()
    if allow is not None:
        kept = [g for g in gpus if g in allow]
        logger.info(f"stage.local.gpu_allowlist allow={sorted(allow)} idle={gpus} kept={kept}")
        return kept
    return gpus


def build_extract_local_cmd(ctx: StageContext, *, gpu: int, rank: int, world_size: int) -> list[str]:
    """ONE extract rank, bound to a physical GPU, writing to the LOCAL md dir.

    No run_marker_kss.sh / tmux / ssh — a plain local subprocess per rank.
    Mirrors marker_worker.py's exact arg names (--input/--output/--gpu/--rank/
    --world-size/--recursive). The worker is the program PIPELINE_LOCAL_EXTRACTOR registers
    (`require_local_extractor`); Marker is retired, so there is no built-in default."""
    marker = require_local_extractor()
    return [
        marker.python, marker.worker,
        "--input", str(ctx.pdf_dir),
        "--output", str(ctx.local_md_dir),
        "--gpu", str(gpu),
        "--rank", str(rank),
        "--world-size", str(world_size),
        "--recursive",
        *marker.args,
    ]


def local_extractor_env(marker: Optional["MarkerHost"]) -> Optional[dict]:
    """The registered worker's env laid over the backend's own; None (inherit) when it
    declares none."""
    import os
    if marker is None or not marker.env:
        return None
    return {**os.environ, **{key: str(value) for key, value in marker.env.items()}}


def _local_worker_label(marker: Optional["MarkerHost"]) -> str:
    return "no extractor registered" if marker is None else Path(marker.worker).stem


def extract_local_min_free_mb() -> int:
    """The free-memory floor (MiB) the local extract needs: the registered extractor's own
    figure, the only place it is set (deploy/compute88/local_extractor.json). Raises
    ``RetiredInferencePath`` when no extractor is registered (Marker was the default)."""
    return require_local_extractor().extract_min_free_mb


def extract_local_preflight(ctx: StageContext) -> GateResult:
    """LOCAL gpu_health (no ssh): a card with at least the registered extractor's floor free
    (`extract_local_min_free_mb`, no idle ratio), intersected with the PIPELINE_LOCAL_GPU_IDS
    allowlist (W106/W70). Records the GPU ids in state for run() to distribute ranks."""
    try:
        floor = extract_local_min_free_mb()
    except ValueError as exc:
        return GateResult(False, str(exc))
    gpu = _local_idle_gpus(min_free_mb=floor)
    if not gpu:
        logger.error(
            "stage.extract.local.preflight.no_gpu "
            f"(no allowed local GPU with ≥{floor}MB free)"
        )
        # Busy now, free later: the orchestrator waits and asks again (it fails the stage with this
        # same message only after its deadline), so one held card cannot fail every queued task.
        return GateResult(
            False,
            f"no allowed local GPU has ≥{floor}MB free; retry once memory is free "
            "or widen PIPELINE_LOCAL_GPU_IDS",
            transient=True,
        )
    ctx.state_set(gpu_ids=",".join(map(str, gpu)))
    logger.info(f"stage.extract.local.preflight.ok gpus={gpu}")
    return GateResult(True)


def _extract_local_total(ctx: StageContext) -> int:
    """What the Markdown count is measured against: the PDFs on disk, or - when
    none are here because the Markdown was produced elsewhere and shipped alone -
    the manifest slice. A zero total reads as "unknown" and never completes."""
    return ctx.pdf_count() or ctx.manifest_slice_total()


# A verifying extractor (pvl_worker) refuses a paper it cannot vouch for, so a refusal is a decision and not a
# missing output. More than this share of a slice refused means the converter or the checker is broken, not the papers.
MAX_REFUSED_SHARE = 0.25


def extract_local_probe(ctx: StageContext) -> Progress:
    refused = len(_reports_without_markdown(ctx))
    return Progress(done=ctx.md_count_local(), total=_extract_local_total(ctx),
                    message=f"{refused} refused" if refused else "")


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
    total = _extract_local_total(ctx)
    if not _has_quality_reports(ctx):
        # Marker writes whatever it can: a gap is a miss.
        if total > 0 and done / total < 0.95:
            return GateResult(False, f"only {done}/{total} MD extracted{_quality_failure_summary(ctx)}")
        return GateResult(True)

    # A verifying extractor accounts for every paper: Markdown, or a failed report saying why not.
    refused = _refused_papers(ctx)
    summary = _failure_summary(refused)
    pdfs = {p.stem for p in ctx.pdf_dir.glob("*.pdf")} if ctx.pdf_dir.is_dir() else set()
    if pdfs:
        written = {f.stem for f in md_dir.glob("*.md")}
        unaccounted = sorted(pdfs - written - refused.keys())
        if unaccounted:
            return GateResult(False, f"{len(unaccounted)} PDF(s) have neither Markdown nor a failed report "
                                     f"(crashed or never reached): {', '.join(unaccounted[:5])}{summary}")
        total, refused_n = len(pdfs), len(pdfs & refused.keys())
    else:  # Markdown shipped without its PDFs: no names to match, only the counts
        refused_n = len(refused)
        if total > 0 and (total - done - refused_n) > 0.05 * total:
            return GateResult(False, f"only {done}/{total} MD extracted, {refused_n} refused{summary}")
    if refused_n > MAX_REFUSED_SHARE * total:
        return GateResult(False, f"only {done}/{total} MD extracted, {refused_n} refused is over the "
                                 f"{MAX_REFUSED_SHARE:.0%} cap{summary}")
    if not refused_n:
        return GateResult(True)
    return GateResult(True, f"{refused_n}/{total} refused verification and stay retryable; "
                            f"{_refusal_list(refused)}")


def _has_quality_reports(ctx: StageContext) -> bool:
    """Whether the extractor left `_quality/*.json`: only a verifying one does."""
    quality = ctx.local_md_dir / "_quality"
    return quality.is_dir() and any(quality.glob("*.json"))


def _reports_without_markdown(ctx: StageContext) -> list[Path]:
    """Per-paper reports whose Markdown is absent (the `_run`/`_server` files are not papers)."""
    quality = ctx.local_md_dir / "_quality"
    if not quality.is_dir():
        return []
    return [r for r in sorted(quality.glob("*.json"))
            if not r.stem.startswith("_") and not (ctx.local_md_dir / f"{r.stem}.md").is_file()]


def _refused_papers(ctx: StageContext) -> dict[str, tuple[str, str]]:
    """stem -> (paper, first reason) for the papers a verifying worker refused, from its
    `_quality/<paper>.json` reports.

    A report only counts while its Markdown is absent: a later run that converted the
    paper rewrites the report and writes the Markdown."""
    refused: dict[str, tuple[str, str]] = {}
    for report in _reports_without_markdown(ctx):
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("status") != "failed":
            continue
        reasons = [str(r) for r in data.get("failure_reasons") or ["no reason recorded"]]
        # A pipeline that died under the paper is the reason to name, ahead of any page it had judged by then.
        crashed = [r for r in reasons if r.startswith(_PIPELINE_FAILURES)]
        refused[report.stem] = (str(data.get("paper") or report.stem), (crashed or reasons)[0])
    return refused


# The errors paddleocr_vl.pipeline_failure writes: the extraction pipeline (or its server) died under the paper.
# Such a paper is retryable like a refused one, but no check judged it, so the gate names them apart.
_PIPELINE_FAILURES = ("pipeline stream failed", "page retry failed", "fallback model failed")


def _refusal_list(refused: dict[str, tuple[str, str]], *, listed: int = 5) -> str:
    return "; ".join(f"{paper}: {reason}" for paper, reason in list(refused.values())[:listed])


def _failure_summary(refused: dict[str, tuple[str, str]], *, listed: int = 5) -> str:
    """"; N paper(s) failed verification: paper: first reason; ...; M paper(s) were not converted, the extraction
    pipeline failed: paper: reason; ..." (either part only when it has papers), or "" when none was refused.
    TPAMI 2022 (2026-11-21) read "59 paper(s) failed verification" when 45 of them were GPU out-of-memory crashes."""
    crashed = {k: v for k, v in refused.items() if v[1].startswith(_PIPELINE_FAILURES)}
    judged = {k: v for k, v in refused.items() if k not in crashed}
    summary = f"; {len(judged)} paper(s) failed verification: {_refusal_list(judged, listed=listed)}" if judged else ""
    if crashed:
        summary += (f"; {len(crashed)} paper(s) were not converted, the extraction pipeline failed: "
                    f"{_refusal_list(crashed, listed=min(listed, 2))}")
    return summary


def _quality_failure_summary(ctx: StageContext, *, listed: int = 5) -> str:
    return _failure_summary(_refused_papers(ctx), listed=listed)


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
    # A registered worker that owns its card (one inference server per GPU) fixes the
    # stacking itself; otherwise the context's figure applies.
    marker = load_local_extractor()
    if marker is not None and marker.ranks_per_gpu:
        per = marker.ranks_per_gpu
    else:
        per = ctx.ranks_per_gpu if ctx.ranks_per_gpu > 0 else 1
    world_size = len(gpus) * per
    return [(r, gpus[r % len(gpus)], world_size) for r in range(world_size)]


def _sync_manifest_md_paths_local(ctx: StageContext, sink: EventSink) -> None:
    """Scoped manifest backfill for THIS (conf, year) from the LOCAL filesystem.
    NOT sync_new_conf_md_smb.py (SMB). Writes only the slice (W106)."""
    cmd = [
        sys.executable, "scripts/sync/sync_local_conf_md.py",
        "--conf", ctx.conf_lower, "--year", str(ctx.year),
        # The paths the stages themselves read, not the script's own defaults: a
        # context with another corpus root must never fall back to the canonical one.
        "--manifest-path", str(ctx.manifest_path),
        "--corpus-root", str(ctx.corpus_root),
    ]
    run_local(cmd, ctx.project_root, sink, abort=ctx.abort)


def make_extract_stage_received(ctx: StageContext, *, poll_s: Optional[float] = None,
                                sleep_fn: SleepFn = time.sleep) -> StageSpec:
    """The extract stage when another host ran the marker: wait for its Markdown
    bundle, verify it and land the files in local_md_dir. Same probe and gate as
    the local extract; no GPU is needed, so no preflight and no GPU lock."""
    return StageSpec(
        key="extract", title=f"Receive {ctx.venue} {ctx.year} Markdown (pushed)",
        run=_receive_run("md", lambda c: c.local_md_dir, poll_s, sleep_fn),
        probe=extract_local_probe, gate=extract_local_gate,
        resource_key="",
    )


def make_extract_stage_local(
    ctx: StageContext,
    *,
    probe_fn: Optional[PollFn] = None,
    sleep_fn: SleepFn = time.sleep,
    poll_s: float = 30.0,
) -> StageSpec:
    """LOCAL extract: launch N worker ranks via run_local (one process per rank,
    distributed across idle GPUs), then backfill manifest md paths from local FS.
    The worker is the one PIPELINE_LOCAL_EXTRACTOR registers; with none registered the stage
    raises ``RetiredInferencePath`` (Marker is retired) before it launches anything.
    No run_marker_kss.sh / ssh / SMB (W106). With options.extract_from == "push"
    the Markdown is received instead (make_extract_stage_received).

    A worker that exits non-zero because some PDFs failed does not discard the rest:
    the Markdown it wrote is synced and the gate judges the total, as in the ssh-host
    extract. Only a cancel aborts the stage."""
    if receive_mode(ctx.options, "extract_from"):
        return make_extract_stage_received(ctx)
    probe = probe_fn or extract_local_probe
    label = _local_worker_label(load_local_extractor())

    def run(c: StageContext, sink: EventSink) -> None:
        marker = require_local_extractor()
        plan = _plan_local_ranks(c)
        sink(f"extract.local: launch {len(plan)} {_local_worker_label(marker)} rank(s) on local GPUs")
        for rank, gpu, world_size in plan:
            cmd = build_extract_local_cmd(c, gpu=gpu, rank=rank, world_size=world_size)
            sink(f"extract.local rank {rank}/{world_size} gpu={gpu}: {' '.join(map(str, cmd))}")
            try:
                run_local(cmd, c.project_root, sink, env=local_extractor_env(marker), abort=c.abort)
            except RuntimeError as exc:
                if c.abort.is_set():
                    raise
                sink(f"extract.local rank {rank} did not finish cleanly ({str(exc)[:200]}); "
                     "keeping the Markdown it wrote, the gate judges the total", level="warning")
        sink("extract.local: sync MD paths into manifest (local slice)")
        _sync_manifest_md_paths_local(c, sink)

    return StageSpec(
        key="extract", title=f"Extract {ctx.venue} {ctx.year} (local {label})",
        run=run, probe=probe, gate=extract_local_gate, preflight=extract_local_preflight,
        poll_interval_s=poll_s,
        # Local GPU box mutex — distinct from any ssh target's "gpu-{target}".
        resource_key="gpu-local",
    )


# ===================================================================== STAGE 3
def preindex_report_path(ctx: StageContext) -> Path:
    """The task's own preindex report (what cleanup reads). Per task, because the
    script's default path is shared by every run and concurrent tasks would
    overwrite each other's lists."""
    return ctx.corpus_root / "reports" / f"preindex_{ctx.task_id}.json"


def build_preindex_cmd(ctx: StageContext) -> list[str]:
    selection_fn = getattr(ctx, "preindex_selection", None)
    selection = (
        selection_fn() if selection_fn is not None
        else ["--conf", ctx.venue, "--year", str(ctx.year)]
    )
    return [
        sys.executable, "scripts/preindex_md_to_sql_only.py",
        *selection,
        "--report-path", str(preindex_report_path(ctx)),
        "--chunker-type", "section_l1",
    ]


def preindex_probe(ctx: StageContext, *, session_factory=get_db_session) -> Progress:
    done = _count_papers(ctx, session_factory)
    # The papers the manifest records Markdown for: a refused paper has none and is not owed to SQL. Before any
    # Markdown is recorded the PDFs are the measure, so an empty result still reads as a failure.
    total = ctx.manifest_slice_extracted() or ctx.pdf_count()
    return Progress(done=done, total=total)


def preindex_gate(ctx: StageContext, *, session_factory=get_db_session) -> GateResult:
    pr = preindex_probe(ctx, session_factory=session_factory)
    if pr.total > 0 and pr.done < 0.95 * pr.total:
        return GateResult(False, f"only {pr.done}/{pr.total} papers in SQL")
    return GateResult(True)


def preindex_syncs_manifest_paths(ctx: StageContext) -> bool:
    """Whether preindex must first record where this slice's PDFs/Markdown are.

    preindex selects papers by the manifest's ``markdown_path``, so Markdown that
    was produced elsewhere and shipped here (extract then skips as satisfied, or
    was marked supplied) is invisible to it until the manifest says where it is.
    The sync is scoped to the slice and additive. A remote-mode run keeps its
    Markdown on the SMB share, and a context that selects its own manifest (arXiv
    batches) writes that manifest itself."""
    return getattr(ctx, "preindex_selection", None) is None and ctx.execution_mode != "remote"


def make_preindex_stage(ctx: StageContext, *, session_factory=get_db_session) -> StageSpec:
    def run(c: StageContext, sink: EventSink) -> None:
        # A preindex run can change this task's SQL text (a new or replaced paper), so
        # any dump an earlier run of the task made is stale: forget its shard range and
        # remove its files. The next embed dumps the task's documents again from the
        # preindex report (task_doc_ids). No chunk-id watermark is kept: the chunks table
        # is shared, and a (before, after] range written days ago and re-closed by a retry
        # covers every other writer's chunks in between. A remote-mode run keeps its
        # shards in the shared directories ship.sh hard-codes: never deleted here.
        c.state_set(chunk_id_before=None, chunk_id_after=None, dump_start_shard=None,
                    shard_start=None, shard_end=None)
        if c.execution_mode != "remote":
            task_dir = _embed_task_dir(c)
            if task_dir.exists():
                n_shards = _count_files(task_dir / "chunks", _CHUNKS_GLOB)
                shutil.rmtree(task_dir)
                sink(f"preindex {c.venue} {c.year}: removed the earlier dump {task_dir} "
                     f"({n_shards} chunk shards); embed dumps again from the report")
        if preindex_syncs_manifest_paths(c):
            sink(f"preindex {c.venue} {c.year}: record Markdown/PDF paths in the manifest slice")
            _sync_manifest_md_paths_local(c, sink)
        run_local(build_preindex_cmd(c), c.project_root, sink, abort=c.abort)

    return StageSpec(
        key="preindex", title=f"Preindex {ctx.venue} {ctx.year} → SQL",
        run=run,
        probe=lambda c: preindex_probe(c, session_factory=session_factory),
        gate=lambda c: preindex_gate(c, session_factory=session_factory),
        # The script is an idempotent upsert (new / replaced / unchanged), so it
        # always runs: "papers already in SQL" no longer means "nothing to do" —
        # a new version of one of them must replace its text.
        skip_when_satisfied=False,
        # Global single-slot: SQLite WAL handles concurrent readers but a single
        # writer at a time keeps commit batches predictable + avoids surprise
        # contention with /api routes that also write.
        resource_key="sql",
    )


# ===================================================================== STAGE 4
def _preindex_report(ctx: StageContext) -> Optional[dict]:
    """The task's preindex report, or None when preindex never ran for it (an external
    start at embed/ingest, or a pipeline with no preindex)."""
    path = preindex_report_path(ctx)
    if not path.exists():
        return None
    from pipeline.paper_crawler.runtime_files import load_json_with_recovery

    return load_json_with_recovery(path, {})


def task_doc_ids(ctx: StageContext, *, session_factory=get_db_session) -> Optional[list[int]]:
    """The documents this task's preindex wrote, sorted: the current ``papers.doc_id`` of every
    paper in the report's ``new`` + ``replaced``. None when the task has no preindex report.

    The report accumulates new/replaced across re-runs at the same path, so a retried
    task still covers what its earlier runs wrote."""
    report = _preindex_report(ctx)
    if report is None:
        return None
    paper_ids = sorted(set(report.get("new", [])) | set(report.get("replaced", [])))
    doc_ids: set[int] = set()
    with session_factory() as db:
        for start in range(0, len(paper_ids), _SQL_IN_BATCH):
            rows = (db.query(DBPaper.doc_id)
                    .filter(DBPaper.paper_id.in_(paper_ids[start:start + _SQL_IN_BATCH])).all())
            doc_ids.update(int(row[0]) for row in rows if row[0] is not None)
    return sorted(doc_ids)


def nothing_to_embed(ctx: StageContext) -> bool:
    """True once this task's preindex has run and wrote nothing to embed. A daily update
    with no new or replaced paper is such a run; embed and ingest have nothing to do and
    must neither need a GPU nor fail for lack of a new shard.

    With a preindex report: nothing iff its ``new`` + ``replaced`` are empty. Without one
    (an external start at embed/ingest) the chunk-id range from the task state decides:
    nothing iff chunk_id_after did not move past chunk_id_before; a task with no
    chunk_id_after is never 'nothing'."""
    report = _preindex_report(ctx)
    if report is not None:
        return not (report.get("new") or report.get("replaced"))
    before, after = ctx.state_get("chunk_id_before"), ctx.state_get("chunk_id_after")
    return before is not None and after is not None and int(after) <= int(before)


def _unless_nothing_to_embed(spec: StageSpec) -> StageSpec:
    """`spec` that does nothing, and passes its checks, when nothing_to_embed(ctx)."""
    def run(ctx, sink):
        if nothing_to_embed(ctx):
            sink(f"{spec.key}: preindex wrote no chunks, nothing to do")
            return
        spec.run(ctx, sink)

    def check(fn):
        if fn is None:
            return None
        return lambda ctx: GateResult(True) if nothing_to_embed(ctx) else fn(ctx)

    return replace(spec, run=run, preflight=check(spec.preflight), gate=check(spec.gate))


def embeds_only_new_chunks(make):
    """Maker decorator: the stage it builds skips itself when nothing_to_embed."""
    @functools.wraps(make)
    def wrapper(*args, **kwargs):
        return _unless_nothing_to_embed(make(*args, **kwargs))
    return wrapper


def build_embed_dump_cmd(ctx: StageContext, *, start_shard: int, shard_size: Optional[int] = None,
                         session_factory=get_db_session) -> list[str]:
    """The dump of the chunks this task embeds.

    A task that ran preindex dumps its own documents (``--doc-ids-file``, written to
    <task dir>/doc_ids.txt from the report): the chunks table is shared, so a chunk-id
    range also holds whatever any other writer added in between. Only a task with no
    preindex report (an external start at embed with an explicit ``chunk_id_before``)
    dumps the chunk-id range of its state."""
    cmd = [
        sys.executable, "scripts/remote_embed/dump_chunks.py",
        "--out", str(_embed_chunks_dir(ctx)),
    ]
    doc_ids = task_doc_ids(ctx, session_factory=session_factory)
    if doc_ids is not None:
        ids_file = _embed_task_dir(ctx) / "doc_ids.txt"
        ids_file.parent.mkdir(parents=True, exist_ok=True)
        ids_file.write_text("".join(f"{doc_id}\n" for doc_id in doc_ids), encoding="utf-8")
        cmd += ["--doc-ids-file", str(ids_file)]
    else:
        chunk_id_before = int(ctx.state_get("chunk_id_before", 0) or 0)
        cmd += ["--min-chunk-id", str(chunk_id_before + 1)]
        after = ctx.state_get("chunk_id_after")
        if after is not None:
            cmd += ["--max-chunk-id", str(int(after))]
    cmd += ["--start-shard", str(start_shard)]
    if shard_size is not None:
        cmd += ["--shard-size", str(shard_size)]
    return cmd


def dump_selection_label(cmd: list[str]) -> str:
    """What a ``build_embed_dump_cmd`` command dumps, for the stage's event line."""
    if "--doc-ids-file" in cmd:
        ids_file = Path(cmd[cmd.index("--doc-ids-file") + 1])
        n_docs = sum(1 for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip())
        return f"documents of this task: {n_docs}"
    return f"chunk ids > {int(cmd[cmd.index('--min-chunk-id') + 1]) - 1}"


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


@embeds_only_new_chunks
def make_embed_stage(
    ctx: StageContext,
    *,
    probe_fn: Optional[PollFn] = None,
    sleep_fn: SleepFn = time.sleep,
    poll_s: float = 30.0,
) -> StageSpec:
    probe = probe_fn or embed_smb_probe  # W82: mounted SMB share, not empty C-drive copy

    def run(c: StageContext, sink: EventSink) -> None:
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

        dump_cmd = build_embed_dump_cmd(c, start_shard=start_shard)
        sink(f"embed step 1/5: dump chunks ({dump_selection_label(dump_cmd)}, start_shard={start_shard})")
        run_local(dump_cmd, c.project_root, sink, abort=c.abort)

        # Authoritative per-task upper bound: anchor + #chunk shards THIS task
        # just dumped (NUMBER >= anchor, measured now while the per-box GPU lock
        # is held so no peer is dumping). NEVER recount vectors/ after ship-down:
        # ship.sh down rsyncs EVERY venue's parquet from the shared SMB dir, so a
        # count there inflates the range and ingest spills onto other venues.
        n_dumped = _count_shards_in_range(_embed_chunks_dir(c), _CHUNKS_GLOB, _CHUNK_SHARD_RE, start_shard)
        shard_end = start_shard + n_dumped
        c.state_set(shard_end=shard_end)

        sink("embed step 2/5: ship chunks up to KSS")
        run_local(["bash", "scripts/remote_embed/ship.sh", "up"], c.project_root, sink, env={"MIN_SHARD": str(start_shard)}, abort=c.abort)

        sink("embed step 3/5: run BGE-M3 embed on KSS")
        run_remote(c.ssh_target, build_embed_remote_cmd(c), sink, env_prefix=embed_env_prefix(c), abort=c.abort)
        sink("embed step 4/5: poll parquet output (SMB share)")
        _poll_until_complete(c, sink, probe, sleep_fn=sleep_fn, poll_s=poll_s)

        sink("embed step 5/5: ship vectors back down")
        run_local(["bash", "scripts/remote_embed/ship.sh", "down"], c.project_root, sink, abort=c.abort)

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
@functools.lru_cache(maxsize=1)
def _embed_worker():
    """scripts/remote_embed/embed_worker.py, loaded by path: its ``embed_shards`` is the one
    shard loop the worker CLI (remote and host stages) and the local stage both run. Loading
    it imports neither FlagEmbedding nor torch and leaves the environment alone."""
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "scripts" / "remote_embed" / "embed_worker.py"
    spec = importlib.util.spec_from_file_location("embed_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shared_encode(texts):
    """The backend's shared embedder (the vLLM embedding service) as the shard loop's ``encode``:
    background priority, so it yields to business requests and waits for the foreground to idle."""
    from core.indexing.embedders import get_embedder

    result = get_embedder().embed_texts(list(texts), priority="background")
    return result.dense, result.sparse


def embed_local_preflight(ctx: StageContext) -> GateResult:
    """The local embed stage is a client of the vLLM embedding service and puts nothing on a
    GPU, so the only precondition is that the service answers and serves this profile
    (`check_service`, no wait). No GPU query, no free-memory floor."""
    from config import settings
    from core.indexing.embedders import ServiceError, get_embedder

    url = settings.embedding.service_url.rstrip("/")
    try:
        up = get_embedder().check_service(wait_s=0)
    except ServiceError as exc:
        logger.error(f"stage.embed.local.preflight.wrong_service {exc}")
        return GateResult(False, str(exc))
    if not up:
        logger.error(f"stage.embed.local.preflight.service_down url={url}")
        return GateResult(False, f"embedding service not reachable at {url}; start arc-vllm-embed.service")
    logger.info(f"stage.embed.local.preflight.ok url={url}")
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


@embeds_only_new_chunks
def make_embed_stage_local(
    ctx: StageContext,
    *,
    probe_fn: Optional[PollFn] = None,
    sleep_fn: SleepFn = time.sleep,
    poll_s: float = 30.0,
    encode: Optional[Callable] = None,
    dump: Optional[Callable] = None,
    title: Optional[str] = None,
) -> StageSpec:
    """LOCAL embed: dump chunks → embed them in this process through the backend's shared
    embedder, a client of the vLLM embedding service (no model here, no worker subprocess;
    the stage holds no GPU). No ship.sh up/down (chunks + vectors
    are local). SAME shard-anchor logic and dump selection (the task's preindexed documents,
    else its chunk-id range) as the remote embed (W106).

    ``encode`` replaces the shared embedder (``encode(texts) -> (dense, sparse)``); the
    tests pass a recording function, production leaves it None.

    ``dump(ctx, sink, None)`` replaces the SQL chunk dump the way it does in
    ``make_embed_stage_host`` (same contract: it leaves ``dump_start_shard`` and ``shard_end``
    in the task state), so a pipeline that owns its shards embeds them here or on a host with
    one dump function."""
    probe = probe_fn or embed_local_probe

    def run(c: StageContext, sink: EventSink) -> None:
        if dump is not None:
            if c.state_get("dump_start_shard") is None or c.state_get("shard_end") is None:
                dump(c, sink, None)
            else:
                sink(f"resume: reusing shard anchor = {int(c.state_get('dump_start_shard'))}")
            start_shard = int(c.state_get("dump_start_shard"))
            shard_end = int(c.state_get("shard_end", start_shard) or start_shard)
            n_dumped = shard_end - start_shard
        else:
            # SAME single shard-NUMBER anchor (first-write-wins for resume) as the
            # remote embed: next free chunks shard index, unifying shard_start.
            prev_anchor = c.state_get("dump_start_shard")
            previous_end = c.state_get("shard_end")
            if prev_anchor is not None and previous_end is not None:
                # The dump is already durable. Re-running it after a backend restart
                # wastes minutes and can overwrite a task's own shard set while the
                # embed stage is only resuming unfinished parquet files.
                start_shard = int(prev_anchor)
                shard_end = int(previous_end)
                n_dumped = shard_end - start_shard
                sink(f"resume: reusing dumped shards [{start_shard}, {shard_end})")
            elif prev_anchor is None:
                start_shard = _count_files(_embed_chunks_dir(c), _CHUNKS_GLOB)
                c.state_set(dump_start_shard=start_shard, shard_start=start_shard)
                dump_cmd = build_embed_dump_cmd(c, start_shard=start_shard)
                sink(f"embed.local step 1/2: dump chunks ({dump_selection_label(dump_cmd)}, start_shard={start_shard})")
                run_local(dump_cmd, c.project_root, sink, abort=c.abort)
                n_dumped = _count_shards_in_range(_embed_chunks_dir(c), _CHUNKS_GLOB, _CHUNK_SHARD_RE, start_shard)
                shard_end = start_shard + n_dumped
                c.state_set(shard_end=shard_end)
            else:
                start_shard = int(prev_anchor)
                sink(f"resume: reusing shard anchor = {start_shard}")
                dump_cmd = build_embed_dump_cmd(c, start_shard=start_shard)
                sink(f"embed.local step 1/2: dump chunks ({dump_selection_label(dump_cmd)}, start_shard={start_shard})")
                run_local(dump_cmd, c.project_root, sink, abort=c.abort)
                n_dumped = _count_shards_in_range(_embed_chunks_dir(c), _CHUNKS_GLOB, _CHUNK_SHARD_RE, start_shard)
                shard_end = start_shard + n_dumped
                c.state_set(shard_end=shard_end)

        sink(f"embed.local step 2/2: embed shards [{start_shard}, {shard_end}) "
             "via the vLLM embedding service")
        finished = _embed_worker().embed_shards(
            _embed_chunks_dir(c), _embed_vectors_dir(c), encode or _shared_encode,
            rank=0, world_size=1, encode_batch_size=get_profile().encode_batch_size,
            should_stop=c.abort.is_set, log=sink)
        if not finished:
            # What run_local raises when the abort fires under its subprocess: the orchestrator
            # checks its own Event first and records a cancellation, not a failure.
            raise RuntimeError("cancelled")

        sink(f"embed.local done: shards [{start_shard}, {shard_end}) = {n_dumped} produced")

    return StageSpec(
        key="embed", title=title or f"Embed {ctx.venue} {ctx.year} (vLLM embedding service)",
        run=run, probe=probe, gate=embed_gate, preflight=embed_local_preflight,
        poll_interval_s=poll_s,
        # vectors/ is shared across venues — never satisfied-skip (W72). Same as
        # remote embed.
        skip_when_satisfied=False,
        # One embed at a time (the dump and tokenizing are CPU/RAM work on a 4-core box), but it takes
        # no card: the extract's ``gpu-local`` is not held against it.
        resource_key="embed-service",
    )


# ============================================ STAGE 4 (SSH EMBED HOST)
# A GPU box reachable over ssh that shares no filesystem with the orchestrator
# (unlike kss, whose output the orchestrator reads over SMB). Each chunk shard
# makes one round trip: scp up, embed_worker over ssh, scp the parquet back
# into the local vectors dir. The probe counts pulled parquets, so the Tasks
# panel advances shard by shard. Hosts are deployment facts, not code: one
# JSON registry (no credentials; ssh keys live in the orchestrator's
# ~/.ssh/config under the host's ssh_target alias).
_EMBED_HOSTS_FILE_DEFAULT = Path(__file__).resolve().parents[3] / "deploy" / "compute88" / "embed_hosts.json"


# Registry strings reach the host inside a double-quoted command, so none of the
# characters that any of the host shells (sh, cmd, PowerShell) or Python's repr
# would treat specially may appear in them; paths use forward slashes.
_HOST_FORBIDDEN = set('"\'\\$`\n\r')


def _checked_host_value(value, what: str) -> str:
    text = str(value)
    bad = sorted(_HOST_FORBIDDEN & set(text))
    if bad:
        raise ValueError(f"embed host registry: {what}={text!r} contains {bad}; "
                         "use forward slashes and no quotes, $ or backticks")
    return text


@dataclass(frozen=True)
class MarkerHost:
    """How a registered host turns PDFs into Markdown (the extract stage in host mode).

    ``worker`` is any program honouring marker_worker.py's contract: ``--input DIR
    --output DIR --gpu N`` writing ``<pdf stem>.md`` into the output directory for
    each PDF; it may exit non-zero when some PDFs failed. ``args`` and ``env`` are
    handed to it verbatim, so another PDF-to-Markdown tool is a registry change."""

    python: str
    worker: str
    env: dict = field(default_factory=dict)
    args: tuple = ()
    # Marker needs ~8 GB at the working point; the local extract floor (16 GB) is
    # the conservative default for a card shared with desktop apps.
    extract_min_free_mb: int = 16384
    batch_size: int = 40  # PDFs per scp-up / worker / scp-down round trip
    # Worker processes per GPU in the local extract; None = the task's own setting.
    # A worker that runs an inference server of its own needs the whole card: 1.
    ranks_per_gpu: Optional[int] = None

    def __post_init__(self) -> None:
        _checked_host_value(self.python, "marker.python")
        _checked_host_value(self.worker, "marker.worker")
        for key, value in self.env.items():
            _checked_host_value(key, "marker.env key")
            _checked_host_value(value, f"marker.env[{key}]")
        for arg in self.args:
            _checked_host_value(arg, "marker.args")
        if self.batch_size < 1:
            raise ValueError("embed host registry: marker.batch_size must be >= 1")
        if self.ranks_per_gpu is not None and self.ranks_per_gpu < 1:
            raise ValueError("embed host registry: marker.ranks_per_gpu must be >= 1")


@dataclass(frozen=True)
class EmbedHost:
    name: str
    ssh_target: str
    python: str
    worker: str
    model_dir: str
    workdir: str
    batch_size: int = 64
    shard_size: int = 25_000
    # An embed host can be a desktop whose display apps hold a shifting slice of
    # the card, so the gate is absolute free memory (BGE-M3 fp16 at batch 64
    # peaks near 6 GB), not a share of the card, and a busy moment is waited
    # out rather than failing the batch.
    min_free_mb: int = 12288
    min_free_ratio: float = 0.0
    preflight_wait_s: float = 300.0
    shard_attempts: int = 4
    # Shard round trips in flight at once. With 2, one shard's upload, worker start and parquet
    # download overlap the other's encoding on the host's one embedding service, which otherwise
    # idles ~42 s of every ~180 s shard (arc99, 2026-11-14).
    concurrent_shards: int = 1
    marker: Optional[MarkerHost] = None
    # Where an ingest task in host mode embeds: "host" = on this host (the embed stage that ships shards
    # over ssh); "local" = through this backend's vLLM embedding service, so the host only converts.
    # Re-embed always embeds on the host it names.
    ingest_embed: str = "host"

    def __post_init__(self) -> None:
        if self.ingest_embed not in ("host", "local"):
            raise ValueError(f"embed host registry: {self.name}: ingest_embed must be 'host' or 'local', "
                             f"not {self.ingest_embed!r}")


def load_embed_host(name: str) -> EmbedHost:
    import json
    import os

    path = Path(os.environ.get("PIPELINE_EMBED_HOSTS") or _EMBED_HOSTS_FILE_DEFAULT)
    try:
        hosts = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"embed host registry unreadable: {path}: {exc}") from exc
    if name not in hosts:
        raise ValueError(f"unknown embed host {name!r}; known: {sorted(hosts)} ({path})")
    entry = dict(hosts[name])
    marker = entry.pop("marker", None)
    return EmbedHost(name=name, marker=None if marker is None else _marker_host(marker), **entry)


def _marker_host(block: dict) -> MarkerHost:
    block = dict(block)
    block["args"] = tuple(block.get("args", ()))
    return MarkerHost(**block)


def load_local_extractor() -> Optional[MarkerHost]:
    """The worker the LOCAL extract stage runs, from the JSON file that
    PIPELINE_LOCAL_EXTRACTOR names (a registry ``marker`` block, validated by the same
    `MarkerHost`); None when unset (`require_local_extractor` turns that into an error).

    Read on every call, so the next stage run sees an edited file. A configured but
    unreadable or invalid file raises instead of falling back to Marker: a typo must
    not silently change which converter produces the corpus."""
    import json
    import os

    path = os.environ.get("PIPELINE_LOCAL_EXTRACTOR", "").strip()
    if not path:
        return None
    try:
        return _marker_host(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"local extractor registry unreadable or invalid: {path}: {exc}") from exc


def require_local_extractor() -> MarkerHost:
    """The registered local extractor; ``RetiredInferencePath`` when none is.

    Unset used to mean the built-in Marker worker. Marker is retired (user, 2026-11-14: "Marker 报错，
    Paddle 保留"), so an unset PIPELINE_LOCAL_EXTRACTOR is an error that names the variable, never a
    quiet switch to another converter."""
    from core.inference_retired import EXTRACTOR_INSTEAD, RetiredInferencePath

    marker = load_local_extractor()
    if marker is None:
        raise RetiredInferencePath("the built-in Marker extractor of the local extract stage "
                                   "(PIPELINE_LOCAL_EXTRACTOR is not set)", EXTRACTOR_INSTEAD)
    return marker


def _host_py(host: EmbedHost, code: str) -> str:
    """One Python statement on the host: its own interpreter does filesystem
    work, so the command is the same whatever the host's shell is."""
    return f'{host.python} -c "{code}"'


def embed_host_preflight(ctx: StageContext, *, sleep_fn: SleepFn = time.sleep,
                         stage: str = "embed") -> GateResult:
    """A GPU on the host with enough free memory for ``stage`` ("embed" or "extract").

    The floor is the host's ``min_free_mb`` for embed and its marker block's
    ``extract_min_free_mb`` for extract. A busy card is waited out (preflight_wait_s)."""
    try:
        host = load_embed_host(ctx.ssh_target)
    except ValueError as exc:
        return GateResult(False, str(exc))
    floor = host.min_free_mb
    if stage == "extract":
        if host.marker is None:
            return GateResult(False, f"host {host.name} has no 'marker' block in the registry, "
                                     "so it cannot run extract")
        floor = host.marker.extract_min_free_mb
    deadline = time.monotonic() + host.preflight_wait_s
    while True:
        gpu = gpu_health(host.ssh_target, min_free_mb=floor, min_free_ratio=host.min_free_ratio)
        if gpu:
            ctx.state_set(gpu_ids=",".join(map(str, gpu)))
            return GateResult(True)
        if time.monotonic() >= deadline:
            return GateResult(False, f"no GPU on {stage} host {host.name} had >={floor} MB free "
                                     f"for {host.preflight_wait_s:.0f}s (or ssh {host.ssh_target} failed)")
        sleep_fn(15.0)


def _embed_host_shard(c: StageContext, host: EmbedHost, sink: EventSink, name: str,
                      target: Path, work: str) -> None:
    """One shard's round trip: upload, embed on the host GPU, pull back, clean."""
    rin, rout = f"{work}/in", f"{work}/out"
    run_remote(host.ssh_target, _host_py(
        host, f"import os; [os.makedirs(d, exist_ok=True) for d in ('{rin}', '{rout}')]"), sink,
        abort=c.abort)
    run_local(["scp", "-q", str(_embed_chunks_dir(c) / f"chunks_{name}.jsonl.gz"),
               f"{host.ssh_target}:{rin}/"], c.project_root, sink, abort=c.abort)
    run_remote(host.ssh_target,
               f"{host.python} {host.worker} --input {rin} --output {rout} "
               f"--model-dir {host.model_dir} --batch-size {host.batch_size}", sink, abort=c.abort)
    part = target.with_name(target.name + ".part")
    run_local(["scp", "-q", f"{host.ssh_target}:{rout}/vectors_{name}.parquet", str(part)],
              c.project_root, sink, abort=c.abort)
    part.replace(target)
    run_remote(host.ssh_target, _host_py(
        host, f"import shutil; shutil.rmtree('{work}', ignore_errors=True)"), sink, abort=c.abort)


EmbedDump = Callable[[StageContext, EventSink, EmbedHost], None]


def _dump_sql_chunks(c: StageContext, sink: EventSink, host: EmbedHost) -> None:
    """The ingest pipelines' dump: this task's SQL chunks (its preindexed documents, else its
    chunk-id range) into chunk shards.

    A dump leaves ``dump_start_shard``, ``shard_start`` and ``shard_end`` in the task state, the
    shard range the host stage embeds."""
    # No finished dump (first run, or one killed midway, e.g. by a restart): dump
    # again past every existing shard, so the killed dump's partial files are
    # neither reused nor overwritten, nor a later task's shards clobbered.
    start_shard = _next_free_shard(_embed_chunks_dir(c), _embed_vectors_dir(c))
    c.state_set(dump_start_shard=start_shard, shard_start=start_shard)
    dump_cmd = build_embed_dump_cmd(c, start_shard=start_shard, shard_size=host.shard_size)
    sink(f"embed.host step 1/2: dump chunks ({dump_selection_label(dump_cmd)}, "
         f"start_shard={start_shard}, shard_size={host.shard_size})")
    run_local(dump_cmd, c.project_root, sink, abort=c.abort)
    n_dumped = _count_shards_in_range(_embed_chunks_dir(c), _CHUNKS_GLOB, _CHUNK_SHARD_RE, start_shard)
    c.state_set(shard_end=start_shard + n_dumped)


@embeds_only_new_chunks
def make_embed_stage_host(ctx: StageContext, *, poll_s: float = 15.0,
                          sleep_fn: SleepFn = time.sleep, dump: EmbedDump = _dump_sql_chunks,
                          title: Optional[str] = None) -> StageSpec:
    """The embed stage on an ssh embed host.

    ``dump`` produces the chunk shards to embed (default: the SQL chunk dump of an ingest
    pipeline); a pipeline that already has its own shards, like the re-embed pipeline's select
    stage, passes a ``dump`` that only records their range.

    The registry's ``concurrent_shards`` round trips run at once (default 1, one after another)."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    host = load_embed_host(ctx.ssh_target)
    if host.concurrent_shards < 1:
        raise ValueError(f"embed host {host.name!r}: concurrent_shards must be at least 1, not {host.concurrent_shards}")

    def run(c: StageContext, sink: EventSink) -> None:
        prev_anchor = c.state_get("dump_start_shard")
        if prev_anchor is None or c.state_get("shard_end") is None:
            dump(c, sink, host)
            start_shard = int(c.state_get("dump_start_shard"))
        else:
            start_shard = int(prev_anchor)
            sink(f"resume: reusing shard anchor = {start_shard}")
        shard_end = int(c.state_get("shard_end", start_shard) or start_shard)

        vectors = _embed_vectors_dir(c)
        vectors.mkdir(parents=True, exist_ok=True)
        base = f"{host.workdir}/task_{c.task_id}"
        sink(f"embed.host step 2/2: shards [{start_shard}, {shard_end}) on {host.name} (gpu {c.state_get('gpu_ids')}, "
             f"{host.concurrent_shards} at a time)")

        def embed_shard(name: str, target: Path) -> None:
            for attempt in range(1, host.shard_attempts + 1):
                task = c.store.get_task(c.task_id) if c.store is not None else None
                if task and task.get("status") == "cancelled":
                    raise RuntimeError(f"cancelled before shard {name}")
                try:
                    _embed_host_shard(c, host, sink, name, target, f"{base}/a{attempt}_{name}")
                    return
                except RuntimeError as exc:
                    # The host link is an overlay tunnel whose sessions can stall
                    # (sshd: "Read error ... Connection timed out") while ping
                    # stays clean. Each attempt gets fresh host dirs, so it never
                    # reads a dead attempt's half-written output.
                    if attempt == host.shard_attempts:
                        raise
                    delay = 30 * attempt
                    sink(f"embed.host shard {name} attempt {attempt}/{host.shard_attempts} failed: "
                         f"{str(exc)[:200]}; retrying in {delay}s", level="warning")
                    sleep_fn(delay)

        # Shards pulled by an earlier attempt are skipped. The rest run ``concurrent_shards`` at a time; a
        # failure submits nothing more, lets the shards already running finish (their parquet is kept) and
        # is raised after them, so a failed stage leaves no round trip running behind it.
        todo = [idx for idx in range(start_shard, shard_end) if not (vectors / f"vectors_{idx:04d}.parquet").exists()]
        finished = shard_end - start_shard - len(todo)
        failure: Optional[BaseException] = None
        with ThreadPoolExecutor(max_workers=host.concurrent_shards, thread_name_prefix="embed-host") as pool:
            queue = iter(todo)
            running: dict = {}

            def submit_next() -> None:
                idx = next(queue, None)
                if idx is not None:
                    running[pool.submit(embed_shard, f"{idx:04d}", vectors / f"vectors_{idx:04d}.parquet")] = idx

            for _ in range(host.concurrent_shards):
                submit_next()
            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    idx = running.pop(future)
                    try:
                        future.result()
                    except BaseException as exc:
                        failure = failure or exc
                        continue
                    finished += 1
                    sink(f"embed.host shard {idx:04d} done ({finished}/{shard_end - start_shard})")
                    if failure is None:
                        submit_next()
        if failure is not None:
            raise failure
        run_remote(host.ssh_target, _host_py(
            host, f"import shutil; shutil.rmtree('{base}', ignore_errors=True)"), sink, abort=c.abort)

    return StageSpec(
        key="embed", title=title or f"Embed {ctx.venue} {ctx.year} (BGE-M3 on {host.name})",
        run=run, probe=embed_local_probe, gate=embed_gate, preflight=embed_host_preflight,
        poll_interval_s=poll_s,
        skip_when_satisfied=False,  # shared vectors/ dir, same as the other embed stages (W72)
        resource_key=f"gpu-{host.name}",
    )


def make_embed_stage_for(ctx: StageContext) -> StageSpec:
    """The embed stage for ctx.execution_mode: local GPU, an ssh embed host (or this backend's
    embedding service when the host's registry entry says ``ingest_embed: local``), or kss."""
    if ctx.execution_mode == "local":
        return make_embed_stage_local(ctx)
    if ctx.execution_mode == "host":
        if load_embed_host(ctx.ssh_target).ingest_embed == "local":
            return make_embed_stage_local(ctx)
        return make_embed_stage_host(ctx)
    return make_embed_stage(ctx)


# ============================================ STAGE 2 (SSH HOST: extract)
def build_marker_host_cmd(marker: MarkerHost, *, rin: str, rout: str, gpu: int) -> str:
    """The command that converts the PDFs in ``rin`` into Markdown in ``rout`` on the host.

    Without env it is the plain argument list. With env, the host's own Python sets
    the variables and runs the worker, because no environment syntax is common to
    sh, cmd and PowerShell (the registry values are validated for exactly this)."""
    argv = [marker.python, marker.worker, "--input", rin, "--output", rout, "--gpu", str(gpu), *marker.args]
    if not marker.env:
        return " ".join(argv)
    code = (f"import os,subprocess,sys; os.environ.update({ {k: str(v) for k, v in marker.env.items()}!r}); "
            f"sys.exit(subprocess.call({argv!r}))")
    return f'{marker.python} -c "{code}"'


def _py_cmd(python: str, code: str) -> str:
    """One Python statement run by the host's own interpreter: the same command
    whatever the host's shell is."""
    return f'{python} -c "{code}"'


def _extract_host_batch(c: StageContext, host: EmbedHost, batch: list[Path], sink: EventSink, work: str,
                        gpu: int, ssh: Sequence[str],
                        scp: Sequence[str]) -> tuple[int, Optional[RuntimeError], set[str]]:
    """One round trip for ``batch``: upload, convert on the host GPU, pull back every
    Markdown that exists (also after a worker that exited non-zero), clean the host dir.

    The worker's sidecars travel with it, because the host dir is deleted: the kept pages of a
    refused paper (``_pages/<stem>.json.gz``) go up before the worker so it judges them again
    instead of converting the paper again, and the batch's ``_quality/<stem>.json`` reports and
    kept pages come back with the Markdown (the gate reads the reports; without them every
    refusal would be a miss). A paper whose Markdown landed drops its local kept pages, as the
    worker does on its side.

    Returns (Markdown files landed, the worker's failure if it exited non-zero, the stems whose
    "failed" report this round trip brought back: the worker judged those papers and refused them)."""
    marker = host.marker
    rin, rout = f"{work}/in", f"{work}/out"
    kept_dir = c.local_md_dir / "_pages"
    kept = [kept_dir / f"{p.stem}.json.gz" for p in batch if (kept_dir / f"{p.stem}.json.gz").is_file()]

    def remote(command: str, out_sink: EventSink = sink, *, hold_stdin: bool = False) -> None:
        run_local([*ssh, host.ssh_target, command], c.project_root, out_sink, abort=c.abort, hold_stdin=hold_stdin)

    def host_py(code: str) -> str:
        return _py_cmd(marker.python, code)

    dirs = ", ".join(f"'{d}'" for d in (rin, rout, *([f"{rout}/_pages"] if kept else [])))
    remote(host_py(f"import os; [os.makedirs(d, exist_ok=True) for d in ({dirs})]"))
    run_local([*scp, "-q", *map(str, batch), f"{host.ssh_target}:{rin}/"], c.project_root, sink, abort=c.abort)
    if kept:
        run_local([*scp, "-q", *map(str, kept), f"{host.ssh_target}:{rout}/_pages/"], c.project_root, sink,
                  abort=c.abort)
    worker_error: Optional[RuntimeError] = None
    try:
        # The worker's stdin stays open for as long as this ssh client lives: a cancelled stage, a restarted
        # backend or a dead client close it, and that EOF is how the far end learns the session is over (the
        # remote sshd does not signal what the session started).
        remote(build_marker_host_cmd(marker, rin=rin, rout=rout, gpu=gpu), hold_stdin=True)
    except RuntimeError as exc:
        if c.abort.is_set():
            raise
        worker_error = exc  # keep what it did convert; the caller decides about the rest
    listing: list[str] = []
    remote(host_py(f"import os; r='{rout}'; print(' '.join(sorted((s+'/'+f if s else f) "
                   "for s in ('', '_quality', '_pages') if os.path.isdir(r+'/'+s) for f in os.listdir(r+'/'+s))))"),
           lambda message, *args, **kw: listing.append(message))
    wanted = {name for p in batch for name in (f"{p.stem}.md", f"_quality/{p.stem}.json", f"_pages/{p.stem}.json.gz")}
    produced = [n for n in (listing[-1].split() if listing else []) if n in wanted]
    landed = 0
    landed_stems: list[str] = []
    refused: set[str] = set()
    if produced:
        stage_dir = c.local_md_dir.parent / f".incoming-{c.local_md_dir.name}-{uuid.uuid4().hex[:8]}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_local([*scp, "-q", *[f"{host.ssh_target}:{rout}/{n}" for n in produced], f"{stage_dir}/"],
                      c.project_root, sink, abort=c.abort)
            c.local_md_dir.mkdir(parents=True, exist_ok=True)
            for n in produced:
                sub, _, name = n.rpartition("/")
                if sub:  # a sidecar: this run's report / kept pages replace the earlier ones
                    (c.local_md_dir / sub).mkdir(exist_ok=True)
                    (stage_dir / name).replace(c.local_md_dir / sub / name)
                    if sub == "_quality" and _report_status(c.local_md_dir / sub / name) == "failed":
                        refused.add(name[:-len(".json")])
                elif (stage_dir / name).stat().st_size > 0:
                    (stage_dir / name).replace(c.local_md_dir / name)
                    landed += 1
                    landed_stems.append(name[:-len(".md")])
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
    for stem in landed_stems:
        (kept_dir / f"{stem}.json.gz").unlink(missing_ok=True)
    remote(host_py(f"import shutil; shutil.rmtree('{work}', ignore_errors=True)"))
    return landed, worker_error, refused


def _report_status(report: Path) -> Optional[str]:
    try:
        return json.loads(report.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError, AttributeError):
        return None


def make_extract_stage_host(ctx: StageContext, *, poll_s: float = 30.0, sleep_fn: SleepFn = time.sleep,
                            ssh: Sequence[str] = ("ssh",), scp: Sequence[str] = ("scp",)) -> StageSpec:
    """Extract on an ssh host that shares no filesystem with the orchestrator.

    The PDFs without Markdown go in batches (``marker.batch_size``): scp up, the
    registered worker on the host GPU, scp the Markdown back, clean the host
    directory. A batch is retried (fresh host dirs) up to ``shard_attempts``
    times for what it did not convert; the task's cancellation is checked between
    batches. A paper the worker refused (its fresh ``failed`` report came back) is accounted for and
    not sent again: the gate judges refusals, and a resend would redo the model's work. Same probe and
    gate as the local extract. With
    ``options.extract_from == "push"`` the Markdown is received instead.
    ``ssh`` / ``scp`` are the executables (or command prefixes) to run."""
    if receive_mode(ctx.options, "extract_from"):
        return make_extract_stage_received(ctx)
    host = load_embed_host(ctx.ssh_target)
    if host.marker is None:
        raise ValueError(f"host {host.name!r} has no 'marker' block in the registry, so it cannot run extract")

    def run(c: StageContext, sink: EventSink) -> None:
        todo = [p for p in sorted(c.pdf_dir.glob("*.pdf")) if not (c.local_md_dir / f"{p.stem}.md").is_file()]
        gpu_ids = [g for g in str(c.state_get("gpu_ids", "")).split(",") if g.strip()]
        if not gpu_ids:
            # Never guess a card on a shared host (W70): the preflight chooses it.
            raise RuntimeError("extract.host: no gpu_ids in state; the preflight must select a GPU before run()")
        gpu = int(gpu_ids[0])
        size = host.marker.batch_size
        batches = [todo[i:i + size] for i in range(0, len(todo), size)]
        sink(f"extract.host: {len(todo)} PDFs to convert in {len(batches)} batch(es) on {host.name} "
             f"(gpu {gpu}, worker {host.marker.worker})")
        base = f"{host.workdir}/marker/task_{c.task_id}"
        for number, batch in enumerate(batches, start=1):
            remaining = batch
            produced_here = 0
            refused_here: set[str] = set()
            last_error: Optional[RuntimeError] = None
            for attempt in range(1, host.shard_attempts + 1):
                task = c.store.get_task(c.task_id) if c.store is not None else None
                if c.abort.is_set() or (task and task.get("status") == "cancelled"):
                    raise RuntimeError(f"cancelled before batch {number}")
                try:
                    landed, last_error, refused = _extract_host_batch(
                        c, host, remaining, sink, f"{base}/b{number:04d}_a{attempt}", gpu, ssh, scp)
                    refused_here |= refused
                except RuntimeError as exc:
                    if c.abort.is_set():
                        raise
                    landed, last_error = 0, exc
                    sink(f"extract.host batch {number} attempt {attempt}/{host.shard_attempts} failed: "
                         f"{str(exc)[:200]}", level="warning")
                produced_here += landed
                remaining = [p for p in remaining
                             if not (c.local_md_dir / f"{p.stem}.md").is_file() and p.stem not in refused_here]
                if not remaining:
                    break
                if attempt < host.shard_attempts:
                    delay = 30 * attempt
                    sink(f"extract.host batch {number}: {len(remaining)} PDF(s) without Markdown "
                         f"(attempt {attempt}/{host.shard_attempts}); retrying in {delay}s", level="warning")
                    sleep_fn(delay)
            if remaining and produced_here == 0 and not refused_here:
                raise RuntimeError(f"extract.host batch {number}: no Markdown after {host.shard_attempts} "
                                   f"attempts: {last_error}")
            if refused_here:
                sink(f"extract.host batch {number}: {len(refused_here)} paper(s) refused verification; their "
                     "reports and kept pages are here, the gate judges them", level="warning")
            if remaining:
                sink(f"extract.host batch {number}: {len(remaining)} PDF(s) produced no Markdown "
                     f"(e.g. {remaining[0].name}); the gate judges the total", level="warning")
            sink(f"extract.host batch {number}/{len(batches)} done ({c.md_count_local()} Markdown files)")
        run_local([*ssh, host.ssh_target,
                   _py_cmd(host.marker.python, f"import shutil; shutil.rmtree('{base}', ignore_errors=True)")],
                  c.project_root, sink, abort=c.abort)

    return StageSpec(
        key="extract", title=f"Extract {ctx.venue} {ctx.year} (marker on {host.name})",
        run=run, probe=extract_local_probe, gate=extract_local_gate,
        preflight=lambda c: embed_host_preflight(c, stage="extract"),
        poll_interval_s=poll_s,
        # The host's GPU is shared with its embed stage: one lock per host.
        resource_key=f"gpu-{host.name}",
    )


def make_extract_stage_for(ctx: StageContext) -> StageSpec:
    """The extract stage for ctx.execution_mode: local GPU, an ssh host, or kss."""
    if ctx.execution_mode == "local":
        return make_extract_stage_local(ctx)
    if ctx.execution_mode == "host":
        return make_extract_stage_host(ctx)
    return make_extract_stage(ctx)


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
        "--vectors", str(_embed_vectors_dir(ctx)),
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


@embeds_only_new_chunks
def make_ingest_stage(ctx: StageContext) -> StageSpec:
    def run(c: StageContext, sink: EventSink) -> None:
        cmd = build_ingest_cmd(c)
        sink(f"ingest shards from {c.state_get('shard_start', 0)} → Qdrant")
        run_local(cmd, c.project_root, sink, abort=c.abort)

    return StageSpec(
        key="ingest", title=f"Ingest {ctx.venue} {ctx.year} → Qdrant",
        run=run, probe=ingest_probe, gate=ingest_gate, preflight=ingest_preflight,
        # ingest_done.txt lives in the task's vectors dir; never satisfied-skip, the
        # command is scoped to [shard_start, shard_end). W72.
        skip_when_satisfied=False,
        # Single global slot: Qdrant commit charge headroom is tight (W58/59/60).
        # Two concurrent ingests would race for mmap commits → OOM.
        resource_key="qdrant",
    )


# ===================================================================== STAGE 6
# What a re-index leaves behind, and the one flag that follows a paper's state:
#   superseded   the preindex report names the Documents that replaced papers
#                (replaced) stopped referencing. Their vectors are deleted from
#                every paper collection, then the SQL document (chunks cascade).
#   withdrawn    papers whose withdrawn flag changed get the optional `withdrawn`
#                payload key set (True) or removed (False) on their points.
# Everything is derived from durable files (the preindex report, the context's
# frozen delta), so a crashed stage re-runs from the top and finishes the job.
@dataclass
class CleanupPlan:
    superseded: dict[int, str]      # old doc_id -> paper_id that moved off it
    withdrawn: dict[str, bool]      # paper_id -> flag it must end with
    report: dict                    # the preindex report, for the result counts


def cleanup_state_path(ctx: StageContext) -> Path:
    return ctx.corpus_root / "reports" / f"cleanup_{ctx.task_id}.json"


def load_cleanup_plan(ctx: StageContext) -> CleanupPlan:
    from pipeline.paper_crawler.runtime_files import load_json_with_recovery

    report = load_json_with_recovery(preindex_report_path(ctx), {})
    withdrawn = {e["paper_id"]: bool(e["withdrawn"]) for e in report.get("withdrawn_changed", [])}
    only = getattr(ctx, "withdrawn_only", None)
    withdrawn.update(only() if only is not None else {})
    return CleanupPlan(
        superseded={int(e["doc_id"]): e["paper_id"] for e in report.get("superseded", [])},
        withdrawn=withdrawn, report=report,
    )


def _read_cleanup_state(path: Path) -> dict:
    from pipeline.paper_crawler.runtime_files import load_json_with_recovery

    state = load_json_with_recovery(path, {})
    return {"superseded_done": sorted(set(state.get("superseded_done", []))),
            "superseded_kept": sorted(set(state.get("superseded_kept", []))),
            "withdrawn_done": bool(state.get("withdrawn_done", False))}


def _cleanup_units(plan: CleanupPlan) -> int:
    return len(plan.superseded) + (1 if plan.withdrawn else 0)


def cleanup_probe(ctx: StageContext) -> Progress:
    plan = load_cleanup_plan(ctx)
    state = _read_cleanup_state(cleanup_state_path(ctx))
    settled = set(state["superseded_done"]) | set(state["superseded_kept"])
    done = len(settled & set(plan.superseded)) + (
        1 if plan.withdrawn and state["withdrawn_done"] else 0)
    return Progress(done=done, total=_cleanup_units(plan))


def cleanup_gate(ctx: StageContext) -> GateResult:
    pr = cleanup_probe(ctx)
    if pr.done < pr.total:
        return GateResult(False, f"cleanup finished {pr.done}/{pr.total} units")
    return GateResult(True)


def _count_doc_points(client, collections: list[str], doc_id: int) -> int:
    from qdrant_client import models

    selector = models.Filter(must=[models.FieldCondition(
        key="doc_id", match=models.MatchValue(value=str(doc_id)))])
    return sum(client.count(collection_name=name, count_filter=selector, exact=True).count
               for name in collections)


def apply_cleanup(
    plan: CleanupPlan,
    *,
    session_factory,
    client,
    collections: list[str],
    state_path: Path,
    sink: EventSink = lambda *a, **k: None,
) -> dict[str, int]:
    """Carry out `plan` against SQL and Qdrant; safe to repeat after any crash.

    A superseded document is deleted only when (1) no `papers` row references it
    any more, so a report entry whose preindex commit never happened (or was
    rolled back) cannot delete a live document, and (2) the paper's current
    document has points in Qdrant, so the old vectors are never the last copy.
    Qdrant first, SQL second: the reverse order could orphan vectors forever.
    """
    from datetime import datetime, timezone

    from core.db.models import Document as DBDocument, SourceType
    from core.storage.qdrant_operations import delete_points_by_doc_id, set_paper_withdrawn
    from pipeline.paper_crawler.runtime_files import write_json_atomic

    state = _read_cleanup_state(state_path)
    deleted = kept = 0

    def save() -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(state_path, state)

    if plan.withdrawn and not state["withdrawn_done"]:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        reported = {e["paper_id"] for e in plan.report.get("withdrawn_changed", [])}
        effective = {}
        with session_factory() as db:
            for pid, flag in plan.withdrawn.items():
                row = db.get(DBPaper, pid)
                # Preindex already committed this flag. An old task must sync
                # today's canonical value, rather than undo a subsequent revision.
                if row is not None and pid in reported:
                    flag = bool(row.withdrawn)
                if row is not None and bool(row.withdrawn) != flag:
                    row.withdrawn = flag
                    row.withdrawn_at = stamp if flag else None
                if row is not None:
                    effective[pid] = flag
        for name in collections:
            set_paper_withdrawn(client, name, [p for p, f in effective.items() if f], True)
            set_paper_withdrawn(client, name, [p for p, f in effective.items() if not f], False)
        state["withdrawn_done"] = True
        save()
        sink(f"cleanup: withdrawn flag synced for {len(plan.withdrawn)} paper(s)")

    for doc_id, paper_id in plan.superseded.items():
        if doc_id in state["superseded_done"]:
            continue
        with session_factory() as db:
            current = db.query(DBPaper.doc_id).filter(DBPaper.paper_id == paper_id).scalar()
            referenced = db.query(DBPaper.paper_id).filter(DBPaper.doc_id == doc_id).first()
        if referenced is not None:
            kept += 1
            state["superseded_kept"] = sorted(set(state["superseded_kept"]) | {doc_id})
            save()
            sink(f"cleanup: document {doc_id} still belongs to {referenced[0]}; kept", level="warning")
            continue
        actual = _count_doc_points(client, collections, current) if current is not None else 0
        if actual == 0:
            raise RuntimeError(
                f"{paper_id}: its current document {current} has no points in {collections}; "
                f"keeping the old document {doc_id} (embed/ingest did not finish)")
        from core.db.models import Chunk as DBChunk
        with session_factory() as db:
            expected = db.query(DBChunk.id).filter(DBChunk.doc_id == current).count()
        if actual != expected:
            raise RuntimeError(
                f"{paper_id}: current document {current} has {actual}/{expected} points; "
                f"keeping old document {doc_id} (ingest is incomplete)")
        for name in collections:
            delete_points_by_doc_id(client, name, str(doc_id))
        with session_factory() as db:
            db.query(DBDocument).filter(
                DBDocument.id == doc_id, DBDocument.source_type == SourceType.paper).delete()
        state["superseded_done"] = sorted(set(state["superseded_done"]) | {doc_id})
        save()
        deleted += 1
    if plan.superseded:
        sink(f"cleanup: {deleted} replaced document(s) deleted, {kept} still referenced")
    return {"superseded_deleted": deleted, "superseded_kept": kept}


def cleanup_result_counts(plan: CleanupPlan, outcome: dict, ctx: StageContext) -> dict:
    report = plan.report
    extra = getattr(ctx, "result_extra", None)
    return {
        "new": len(report.get("new", [])),
        "replaced": len(report.get("replaced", [])),
        "unchanged": len(report.get("unchanged", [])),
        "withdrawn": sum(1 for flag in plan.withdrawn.values() if flag),
        "revived": sum(1 for flag in plan.withdrawn.values() if not flag),
        "preindex_failed": int(report.get("failed", 0)),
        **outcome,
        **(extra() if extra is not None else {}),
    }


def remove_embed_task_dir(ctx: StageContext, sink: EventSink) -> bool:
    """Delete the task's dump/vector files once its ingest finished: the vectors are in
    Qdrant and the dump is reproducible from SQL. A task whose ingest has not completed
    keeps them, so a re-drive resumes from its own shards."""
    task_dir = _embed_task_dir(ctx)
    if not task_dir.exists():
        return False
    if not nothing_to_embed(ctx) and not ingest_gate(ctx).ok:
        sink(f"cleanup: ingest not finished, keeping {task_dir}", level="warning")
        return False
    shutil.rmtree(task_dir)
    sink(f"cleanup: removed {task_dir}")
    return True


def _production_qdrant():
    """(client, existing paper collection names) of the running deployment."""
    from core.storage.corpus_layout import paper_collections
    from core.storage.qdrant_store import QdrantStore

    names = paper_collections()
    client = QdrantStore(collection_name=names["papers"], create_if_missing=False).client
    return client, [name for name in names.values() if client.collection_exists(name)]


def make_cleanup_stage(ctx: StageContext, *, session_factory=get_db_session,
                       qdrant=_production_qdrant) -> StageSpec:
    def run(c: StageContext, sink: EventSink) -> None:
        plan = load_cleanup_plan(c)
        sink(f"cleanup: {len(plan.superseded)} replaced document(s), "
             f"{len(plan.withdrawn)} withdrawn flag(s) to apply")
        client, collections = qdrant()
        outcome = apply_cleanup(plan, session_factory=session_factory, client=client,
                                collections=collections, state_path=cleanup_state_path(c), sink=sink)
        remove_embed_task_dir(c, sink)
        if c.store is not None:
            c.store.update_result(c.task_id, cleanup_result_counts(plan, outcome, c))

    return StageSpec(
        key="cleanup", title=f"Clean up {ctx.venue} {ctx.year} replaced documents",
        run=run, probe=cleanup_probe, gate=cleanup_gate,
        # Deletes vectors and writes flags in Qdrant, so it queues with ingest on the
        # same key. The few SQL rows it touches rely on SQLite's own write lock (busy
        # timeout 30 s): StageSpec carries one resource key, and holding "sql" for the
        # stage would not make these short writes any safer than the lock already does.
        resource_key="qdrant",
    )
