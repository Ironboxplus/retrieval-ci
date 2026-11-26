"""arXiv CS ingest pipeline: the same five stage keys, one task per batch.

``venue="arXiv"`` with ``year=<key>`` selects this pipeline. The key is a
batch ``YYYYMMPP`` (a fixed id range of one month, ~2,000 papers; see
arxiv_corpus) or a whole month ``YYYYMM`` / the legacy slice 202311.
Streaming comes from the task system, not from this module: one task per
batch, all started together, and ``StageLockSet`` lets batch N embed while
batch N+1 converts and batch N+2 is still downloading.

  download  receive: the downloading host (which holds the proxy pool) pushes
            arxiv_<key>.tar into <harvest root>/inbox once the batch is fully
            resolved, and a progress file every round in between
            (arxiv_batches.BatchPusher). This stage shows that progress, waits
            for the bundle and imports rows + HTML. No resource lock while
            waiting.
  extract   scripts/collect/arxiv_cs_corpus.py convert --slice <key> (HTML ->
            JSON + Markdown on CPU, then the key's manifest)
  preindex  shared stage, pointed at the key's manifest
  embed     shared stage
  ingest    shared stage

Every probe counts real artifacts while its stage runs, so the Tasks panel
shows live progress per batch and stage.

The daily update is the same pipeline with ``year = DELTA_KEY`` (0): one task
per run, no month or batch, and it downloads for itself instead of receiving:

  download  enumerate (OAI-PMH from the stored watermark), download every 2026+
            paper that needs it through the first hop alone, then FREEZE the run:
            the papers whose HTML is new against SQL (``ids``), the withdrawn
            flags that differ without a new text (``withdrawn_only``), the papers
            arXiv has no HTML for yet (``html_missing``) -> corpus/deltas/<task>.json
  extract   convert exactly the frozen ids, manifest per run
  ...       preindex / embed / ingest as for a batch
  cleanup   shared stage: drop what replaced versions left behind, sync flags

Everything downstream reads the frozen file, never the live harvest database,
which keeps changing while the run proceeds.

This host never downloads: the downloading host (cfscan) does (``options.source`` is "volume", the
default; "local" is refused), and leaves its bundles (``arxiv_delta_<UTC>.tar``, the rows it
changed plus their HTML, each with a .sha256) and a heartbeat in the 115 volume
``options.volume`` (default /cfs-arxiv). The download stage waits for a run it has not consumed
(its heartbeat), imports every bundle it has not taken (``receive_volume_bundles``), freezes
the run exactly as above, and then records the heartbeat as consumed. A run that changed
nothing is an empty delta that completes. ``make_local_arxiv_delta_download_stage`` keeps the
enumerate-and-download stage described above for a host that downloads; no dispatcher reaches it.

Papers land with venue "arXiv", venue_tier/track_type "preprint".
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tarfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from loguru import logger
from sqlalchemy import func, select

from app.services.pipeline.context import StageContext
from app.services.pipeline.store import TaskStore
from app.services.pipeline.runners import EventSink, run_local
from app.services.pipeline.spec import GateResult, Progress, StageSpec
from app.services.pipeline.stages import HOST_NEVER_DOWNLOADS
from pipeline.openlist_reader import VolumeNotFound, VolumeReader
from pipeline.paper_crawler.arxiv_batches import (
    DELTA_BUNDLE_RE,
    HEARTBEAT_FILE,
    VOLUME_SCHEMA,
    bundle_name,
    bundle_rows,
    checksum_file_name,
    file_sha256,
    heartbeat_is_new,
    import_bundle,
    inbox_dir,
    parse_checksum,
    read_progress,
    record_heartbeat_consumed,
    record_volume_import,
    record_volume_rejection,
    volume_imports,
    volume_rejections,
)
from pipeline.paper_crawler.arxiv_corpus import (
    ARXIV_VENUE,
    DELTA_KEY,
    DELTA_MIN_ID_PREFIX,
    LEGACY_SLICE,
    MAX_BATCH_PART,
    SlicePaths,
    batch_range,
    corpus_root as default_corpus_root,
    harvest_root as default_harvest_root,
    is_batch_key,
    month_of,
    paper_id_for,
    rejected_ids,
    slice_label,
    slice_sql,
    _needs_convert,
)
from pipeline.paper_crawler.arxiv_html import (
    CONTENT_VERSION_SQL,
    CONVERT_REJECTED_SQL,
    HTML_CUTOFF,
    NO_HTML_RETRY_DAYS,
    HarvestStore,
    html_path,
)
from pipeline.paper_crawler.runtime_files import load_json_with_recovery, write_json_atomic

DOWNLOAD_MAX_ATTEMPTS = 3
RECEIVE_POLL_SECONDS = 15.0
DELTA_POLL_SECONDS = 15.0
# Tries per request in a delta download. Without a proxy pool every try reuses the
# single first-hop route (each at most the HTML timeout), so the CLI default of 12
# would let one dead page hold a worker for minutes (slice A measurement, plan §5 C).
DELTA_MAX_EXITS = 3
# Where the delta's HTML comes from: the downloading host's packages in a 115 volume. "local" (this host
# downloads) is refused by validate_source_option.
SOURCES = ("volume",)
DEFAULT_VOLUME = "/cfs-arxiv"
VOLUME_DEADLINE_HOURS = 6.0
VOLUME_POLL_SECONDS = 60.0
# `papers.paper_id` range of the delta scope in SQL: new-style ids from DELTA_MIN_ID_PREFIX
# on. ":" sorts right after "9", so the upper bound spans every "arxiv-<digits>..." id
# and excludes old-style "arxiv-cs-0101001" ids, the same set as slice_sql(DELTA_KEY).
_DELTA_SQL_LOW = paper_id_for(DELTA_MIN_ID_PREFIX)
_DELTA_SQL_HIGH = "arxiv-:"


def is_arxiv_venue(venue: str) -> bool:
    return venue.lower() == ARXIV_VENUE.lower()


def validate_source_option(options: Optional[dict]) -> str:
    """options.source of an arXiv run: absent means the volume; this host never downloads, so 'local' is
    refused, and so is anything else. Raises ValueError, before a task exists when called by the runner."""
    value = (options or {}).get("source") or "volume"
    if value == "local":
        raise ValueError(f"{HOST_NEVER_DOWNLOADS} (source='volume')")
    if value not in SOURCES:
        raise ValueError(f"options.source must be 'volume', got {value!r}")
    return value


def validate_slice_key(key: int) -> None:
    """Format check for an arXiv batch/slice key; raises ValueError."""
    if key in (LEGACY_SLICE, DELTA_KEY):
        return
    month = month_of(key)
    part_ok = not is_batch_key(key) or key % 100 <= MAX_BATCH_PART
    if not (202312 <= month <= 209912 and 1 <= month % 100 <= 12 and part_ok):
        raise ValueError(
            f"arXiv pipelines take a batch YYYYMMPP (PP <= {MAX_BATCH_PART}), a month slice "
            f"YYYYMM >= 202312, {LEGACY_SLICE} for pre-2023-12 ids as year, or {DELTA_KEY} "
            f"for the daily update, got {key}"
        )


@dataclass
class ArxivStageContext(StageContext):
    """StageContext for one arXiv month slice, batch or delta run; ``year`` carries the key."""

    corpus_root: Path = field(default_factory=default_corpus_root)
    harvest_root: Path = field(default_factory=default_harvest_root)
    convert_workers: int = max(1, min(8, (os.cpu_count() or 2) - 1))

    def __post_init__(self) -> None:
        validate_slice_key(self.year)
        self.source  # an unknown source is refused when the context is built, not mid-run

    # ------------------------------------------------------------- delta source
    @property
    def source(self) -> str:
        return validate_source_option(self.options)

    @property
    def volume(self) -> str:
        value = str((self.options or {}).get("volume") or DEFAULT_VOLUME).rstrip("/")
        if not value.startswith("/"):
            raise ValueError(f"options.volume must be the absolute path of the volume on the OpenList node, got {value!r}")
        return value

    @property
    def receive_deadline_hours(self) -> float:
        return float((self.options or {}).get("receive_deadline_hours", VOLUME_DEADLINE_HOURS))

    # ------------------------------------------------------------ slice paths
    @property
    def slice_key(self) -> int:
        return self.year

    @property
    def is_delta(self) -> bool:
        return self.slice_key == DELTA_KEY

    @property
    def paths(self) -> SlicePaths:
        # A delta run's manifest and report are named by the task, so runs never overwrite each other.
        return SlicePaths(self.corpus_root, self.slice_key, self.task_id if self.is_delta else None)

    @property
    def delta_file(self) -> Path:
        return self.corpus_root / "deltas" / f"{self.task_id}.json"

    def frozen(self) -> dict:
        """The run's frozen selection (see freeze_delta); empty before the download stage wrote it."""
        return load_json_with_recovery(self.delta_file, {}) if self.is_delta else {}

    def withdrawn_only(self) -> dict[str, bool]:
        """paper_id -> flag for papers whose withdrawn flag changed without a new text.
        They are not converted or re-indexed; cleanup writes the flag (SQL and Qdrant)."""
        return {pid: bool(flag) for pid, flag in self.frozen().get("withdrawn_only", {}).items()}

    def convert_report(self) -> dict:
        """What this run's convert wrote (arxiv_corpus.convert_rows); empty before it ran."""
        return load_json_with_recovery(self.paths.report, {})

    def rejected_in_run(self) -> list[str]:
        """Papers of this run whose HTML has no usable article: the ones the freeze left out
        because an earlier run found that, and the ones this run's convert just found."""
        found = list(self.frozen().get("convert_rejected", [])) + list(self.convert_report().get("rejected", []))
        return list(dict.fromkeys(found))

    def result_extra(self) -> dict[str, int]:
        """Counts only the delta download and convert know, merged into the task result by cleanup."""
        counts = self.frozen().get("counts", {})
        extra = {key: int(counts[key]) for key in
                 ("selected", "withdrawn_only", "html_missing", "download_errors") if key in counts}
        if self.is_delta and counts:
            extra["convert_rejected"] = len(self.rejected_in_run())
            extra["convert_errors"] = len(self.convert_report().get("errors", []))
        return extra

    @property
    def harvest_db(self) -> Path:
        return self.harvest_root / "arxiv_cs.sqlite3"

    @property
    def bundle(self) -> Path:
        return inbox_dir(self.harvest_root) / bundle_name(self.slice_key)

    def ok_ids(self) -> list[str]:
        """Received papers of this key that have HTML (a delta run: the frozen ids)."""
        if self.is_delta:
            return list(self.frozen().get("ids", []))
        if not self.harvest_db.exists():
            return []
        where, params = slice_sql(self.slice_key)
        db = sqlite3.connect(f"file:{self.harvest_db}?mode=ro", uri=True)
        try:
            return [r[0] for r in db.execute(
                f"SELECT arxiv_id FROM papers WHERE latest_version_date >= ? AND {where} AND status='ok'",
                [HTML_CUTOFF.isoformat(), *params])]
        finally:
            db.close()

    @property
    def manifest_path(self) -> Path:
        if self.manifest_path_override is not None:
            return self.manifest_path_override
        return self.paths.manifest

    @property
    def local_md_dir(self) -> Path:
        # A delta run spans months: the corpus-level root, one paper at a time via paths.markdown(id).
        return self.corpus_root / "markdown" / "arxiv" if self.is_delta else self.paths.markdown_dir

    def md_count_local(self) -> int:
        if self.is_delta:
            return sum(1 for a in self.ok_ids() if self.paths.markdown(a).exists())
        return super().md_count_local()

    # --------------------------------------------------------------- manifest
    def manifest_slice(self) -> list[dict]:
        records = load_json_with_recovery(self.manifest_path, [])
        return records if isinstance(records, list) else []

    def pdf_count(self) -> int:
        return 0  # HTML source; no PDFs

    # ------------------------------------------------- shared-stage hooks
    def preindex_selection(self) -> list[str]:
        """Replaces preindex's --conf/--year: the slice manifest is the slice."""
        return ["--manifest-path", str(self.manifest_path), "--conf", ARXIV_VENUE]

    def paper_count_clause(self, paper_model):
        """SQL clause selecting this slice's rows in ``papers`` (paper_id is
        ``arxiv-<id>``; the slice is the id's month; a delta run: its frozen ids)."""
        from sqlalchemy import Integer, and_, cast, or_

        pid = paper_model.paper_id
        if self.is_delta:
            # One bound JSON array instead of one variable per id: a backlog run holds
            # tens of thousands of ids, above SQLite's variable limit (32766).
            ids = func.json_each(json.dumps([paper_id_for(a) for a in self.ok_ids()])).table_valued("value")
            return pid.in_(select(ids.c.value))
        if self.slice_key == LEGACY_SLICE:
            month = func.substr(pid, 7, 4)
            return and_(pid.like("arxiv-%"), or_(month < "2312", pid.like("arxiv-%-%")))
        in_month = pid.like(f"arxiv-{month_of(self.slice_key) % 10000:04d}.%")
        if not is_batch_key(self.slice_key):
            return in_month
        lo, hi = batch_range(self.slice_key)
        # arxiv-YYMM.NNNNN: the number starts at character 12
        return and_(in_month, cast(func.substr(pid, 12), Integer).between(lo, hi))

    # -------------------------------------------------------- harvest counts
    def harvest_counts(self) -> dict[str, int]:
        """Download state of the slice: {status: n} plus retryable errors. A delta
        run reports what it froze."""
        if self.is_delta:
            counts = self.frozen().get("counts", {})
            return {status: counts[key] for status, key in
                    (("ok", "selected"), ("no_html", "html_missing"), ("error", "download_errors"))
                    if counts.get(key)}
        if not self.harvest_db.exists():
            return {}
        where, params = slice_sql(self.slice_key)
        db = sqlite3.connect(f"file:{self.harvest_db}?mode=ro", uri=True)
        try:
            rows = db.execute(
                f"SELECT status, attempts < ? AS retryable, COUNT(*) FROM papers "
                f"WHERE latest_version_date >= ? AND {where} GROUP BY status, retryable",
                [DOWNLOAD_MAX_ATTEMPTS, HTML_CUTOFF.isoformat(), *params],
            ).fetchall()
        finally:
            db.close()
        counts: dict[str, int] = {}
        for status, retryable, n in rows:
            counts[status] = counts.get(status, 0) + n
            if status == "error" and retryable:
                counts["error_retryable"] = counts.get("error_retryable", 0) + n
        return counts


# ===================================================================== download
def arxiv_download_probe(ctx: ArxivStageContext) -> Progress:
    """Before the bundle arrives: the downloading host's progress file for this
    key (live download progress). After: the rows received into the local
    harvest DB."""
    c = ctx.harvest_counts()
    source = "received"
    if not c:
        c = read_progress(inbox_dir(ctx.harvest_root)).get(str(ctx.slice_key), {})
        source = "downloading"
    total = sum(v for k, v in c.items() if k != "error_retryable")
    unresolved = c.get("pending", 0) + c.get("error_retryable", 0)
    done = total - unresolved
    if source == "downloading":
        # The downloading host's word is progress, never completion: the stage
        # is done only once the bundle is imported here. (The pusher writes
        # progress.json before the round's bundles, so the last round's batches
        # stay "fully resolved" in it forever; reading that as complete skipped
        # the import and the convert then found no HTML.)
        done = min(done, max(total - 1, 0))
    return Progress(
        done=done, total=total,
        message=(f"{source}: ok {c.get('ok', 0):,} · no_html {c.get('no_html', 0):,} · "
                 f"unresolved {unresolved:,}"),
    )


def _cancelled(ctx: ArxivStageContext) -> bool:
    task = ctx.store.get_task(ctx.task_id) if ctx.store is not None else None
    return bool(task and task.get("status") == "cancelled")


def receive_bundle(ctx: ArxivStageContext, sink: EventSink, *, poll_s: float = RECEIVE_POLL_SECONDS,
                   sleep_fn=time.sleep) -> dict[str, int]:
    """Wait for the downloading host to push this key's bundle, then import it."""
    waited = 0.0
    while not ctx.bundle.is_file():
        if _cancelled(ctx):
            raise RuntimeError("cancelled while waiting for the bundle")
        if waited % 600 < poll_s:
            sink(f"waiting for {ctx.bundle.name} ({arxiv_download_probe(ctx).message})")
        sleep_fn(poll_s)
        waited += poll_s
    store = HarvestStore(ctx.harvest_db)
    try:
        counts = import_bundle(store, ctx.harvest_root, ctx.bundle)
    finally:
        store.close()
    sink(f"received {ctx.bundle.name}: {counts['rows']:,} rows, {counts['html']:,} html")
    return counts


def make_arxiv_download_stage(ctx: ArxivStageContext) -> StageSpec:
    def run(c: ArxivStageContext, sink: EventSink) -> None:
        receive_bundle(c, sink)

    def gate(c: ArxivStageContext) -> GateResult:
        pr = arxiv_download_probe(c)
        if pr.total == 0:
            return GateResult(False, f"{c.bundle.name} held no eligible papers")
        if pr.done < pr.total:
            return GateResult(False, f"{pr.total - pr.done:,} papers still unresolved ({pr.message})")
        return GateResult(True)

    return StageSpec(
        key="download", title=f"Download arXiv {slice_label(ctx.slice_key)} HTML",
        run=run, probe=arxiv_download_probe, gate=gate,
        poll_interval_s=RECEIVE_POLL_SECONDS,
        # Waiting holds nothing: other batches keep flowing through every stage.
        resource_key="",
    )


# ============================================================== delta download
def _delta_options(ctx: ArxivStageContext) -> tuple[int, int, int]:
    options = ctx.options or {}
    return (int(options.get("html_concurrency", 3)),
            int(options.get("no_html_retry_days", NO_HTML_RETRY_DAYS)),
            int(options.get("limit", 0)))


def _first_hop_proxy() -> str:
    """The configured proxy's host:port as the SOCKS5 first hop the harvest CLI chains through.

    settings.proxy is an http:// URL, but ChainedProxyPool takes only socks5/socks5h
    (pipeline/download_control._SUPPORTED_CHAIN_EXIT_SCHEMES); the listener on that
    port speaks both, so only the scheme changes."""
    from urllib.parse import urlsplit

    from config import settings

    endpoint = urlsplit(settings.proxy.https_proxy)
    return f"socks5://{endpoint.hostname}:{endpoint.port}"


def build_delta_enumerate_cmd(ctx: ArxivStageContext) -> list[str]:
    return [sys.executable, "scripts/collect/arxiv_cs_html.py", "enumerate",
            "--root", str(ctx.harvest_root), "--first-hop-proxy", _first_hop_proxy()]


def build_delta_download_cmd(ctx: ArxivStageContext) -> list[str]:
    """Delta scope, no proxy pool file (the first hop alone), a short retry budget."""
    concurrency, retry_days, limit = _delta_options(ctx)
    cmd = [sys.executable, "scripts/collect/arxiv_cs_html.py", "download",
           "--root", str(ctx.harvest_root), "--first-hop-proxy", _first_hop_proxy(),
           "--scope", "delta", "--concurrency", str(concurrency),
           "--no-html-retry-days", str(retry_days), "--max-exits", str(DELTA_MAX_EXITS)]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def freeze_delta(harvest_db: Path, session_factory, *, started: str) -> dict:
    """Decide what this run indexes and write nothing: returns the selection.

    Against the SQL papers (the state the index is in) the harvest rows of the delta
    scope split into:
      ids             status ok (HTML of the content version on disk) and SQL has no such
                      paper or holds another source_version -> convert, index, replace
      convert_rejected  would be in ids, but its HTML of this version was already converted
                      and found unusable (arxiv_html.CONVERT_REJECTED_SQL): selecting it
                      again would fail every run's markdown gate. A new version clears it.
      withdrawn_only  SQL has the paper and its withdrawn flag differs from the harvest,
                      whatever the HTML status, and the paper is not in ``ids`` (those carry
                      the flag through their manifest). A newer version whose HTML is not
                      there yet lands here too: its old text stays, only the flag moves.
      html_missing    judged no_html by this run (fetched since ``started``)
    A SQL paper without source_version means the backfill never ran; selecting on it
    would re-index the whole corpus, so that is refused instead.
    """
    from core.db.models import Paper as DBPaper

    with session_factory() as db:
        unbackfilled = db.query(func.count(DBPaper.paper_id)).filter(
            DBPaper.paper_id >= _DELTA_SQL_LOW, DBPaper.paper_id < _DELTA_SQL_HIGH,
            DBPaper.source_version.is_(None)).scalar()
        if unbackfilled:
            raise RuntimeError(
                f"{unbackfilled:,} arXiv papers in SQL have no source_version; run "
                f"scripts/maintenance/backfill_arxiv_source_version.py before the first delta run")
        have = {pid: (version, bool(withdrawn)) for pid, version, withdrawn in db.query(
            DBPaper.paper_id, DBPaper.source_version, DBPaper.withdrawn).filter(
            DBPaper.paper_id >= _DELTA_SQL_LOW, DBPaper.paper_id < _DELTA_SQL_HIGH)}

    where, params = slice_sql(DELTA_KEY)
    harvest = sqlite3.connect(f"file:{harvest_db}?mode=ro", uri=True)
    try:
        rows = harvest.execute(
            f"SELECT arxiv_id, status, {CONTENT_VERSION_SQL}, withdrawn, html_version, fetched_at, "
            f"{CONVERT_REJECTED_SQL} FROM papers WHERE {where} ORDER BY arxiv_id", params).fetchall()
    finally:
        harvest.close()

    ids: list[str] = []
    withdrawn_only: dict[str, bool] = {}
    html_missing: list[str] = []
    convert_rejected: list[str] = []
    errors = 0
    for arxiv_id, status, version, withdrawn, html_version, fetched_at, rejected in rows:
        pid = paper_id_for(arxiv_id)
        known = have.get(pid)
        if status == "ok" and html_version in (None, version) and (known is None or known[0] != version):
            if rejected:
                convert_rejected.append(arxiv_id)
            else:
                ids.append(arxiv_id)
            continue
        if known is not None and known[1] != bool(withdrawn):
            withdrawn_only[pid] = bool(withdrawn)
        this_run = bool(fetched_at) and fetched_at >= started
        if status == "no_html" and this_run:
            html_missing.append(arxiv_id)
        elif status == "error" and this_run:
            errors += 1
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ids": ids, "withdrawn_only": withdrawn_only, "html_missing": html_missing,
        "convert_rejected": convert_rejected,
        "counts": {"selected": len(ids), "withdrawn_only": len(withdrawn_only),
                   "html_missing": len(html_missing), "convert_rejected": len(convert_rejected),
                   "download_errors": errors},
    }


def arxiv_delta_download_probe(ctx: ArxivStageContext) -> Progress:
    """Papers the download attempted since the stage started, of those it queued.
    Capped one short of done until the run is frozen: the frozen file, not the
    downloads, is what later stages read."""
    queued = ctx.state_get("delta_queued")
    started = ctx.state_get("delta_started")
    if not queued or not started or not ctx.harvest_db.exists():
        return Progress(done=0, total=0)
    where, params = slice_sql(DELTA_KEY)
    db = sqlite3.connect(f"file:{ctx.harvest_db}?mode=ro", uri=True)
    try:
        attempted = db.execute(
            f"SELECT COUNT(*) FROM papers WHERE {where} AND fetched_at >= ?", [*params, started]).fetchone()[0]
    finally:
        db.close()
    done = min(attempted, queued)
    if not ctx.delta_file.exists():
        done = min(done, queued - 1)
    return Progress(done=done, total=queued, message=f"attempted {attempted:,} of {queued:,} queued")


def _freeze_delta_run(c: ArxivStageContext, sink: EventSink, session_factory, started: str) -> None:
    frozen = freeze_delta(c.harvest_db, session_factory, started=started)
    frozen["task_id"] = c.task_id
    write_json_atomic(c.delta_file, frozen)
    sink("frozen: " + ", ".join(f"{k}={v:,}" for k, v in frozen["counts"].items()))


def _delta_gate(c: ArxivStageContext) -> GateResult:
    if not c.delta_file.exists():
        return GateResult(False, f"{c.delta_file.name} was not written")
    return GateResult(True)


# ----------------------------------------------------- delta download from the volume
@dataclass
class VolumeDelivery:
    """What one run of the downloading host left, as the stage took it in."""

    imported: list[str]  # bundle names imported by this call, oldest first
    heartbeat: dict  # the run's heartbeat; the stage records it as consumed after the freeze


class _BundleDamaged(Exception):
    pass


def _volume_inbox(ctx: ArxivStageContext) -> str:
    return f"{ctx.volume}/inbox"


def _read_heartbeat(reader: VolumeReader, inbox: str) -> Optional[dict]:
    """The job's heartbeat, or None when there is none (yet) or it is not one."""
    try:
        beat = json.loads(reader.read_bytes(f"{inbox}/{HEARTBEAT_FILE}", limit=64 * 1024))
    except (VolumeNotFound, ValueError):
        return None
    ok = isinstance(beat, dict) and beat.get("run_id") and beat.get("finished_at")
    return beat if ok and isinstance(beat.get("bundles", []), list) else None


def _volume_bundles(reader: VolumeReader, inbox: str) -> dict[str, int]:
    """{name: size} of the volume's delta bundles, oldest first. An inbox the job has not
    created yet is an empty one; the .sha256 files and unfinished copies are not bundles."""
    try:
        listing = reader.list_files(inbox)
    except VolumeNotFound:
        return {}
    return {name: size for name, size in sorted(listing.items()) if DELTA_BUNDLE_RE.match(name)}


def _no_run_message(db, inbox: str, beat: Optional[dict], missing: list[str], hours: float) -> str:
    """Why waiting ended with nothing: where, what the job last said, what was last imported."""
    taken, rejected = volume_imports(db), volume_rejections(db)
    parts = [f"no new run of the downloading host finished in {inbox} after {hours:g} h"]
    if beat is None:
        parts.append("no heartbeat on the volume")
    else:
        parts.append(f"last heartbeat on the volume: {json.dumps(beat, ensure_ascii=False)[:300]}"
                     + ("" if heartbeat_is_new(db, beat) else " (already consumed)"))
    if missing:
        parts.append(f"it names bundle(s) not on the volume: {', '.join(missing)}")
    last = max(taken.items(), key=lambda item: item[1], default=None)
    parts.append(f"last bundle imported: {last[0]} at {last[1]}" if last else "no bundle has been imported yet")
    if rejected:
        parts.append("rejected bundles: " + "; ".join(f"{n} ({r})" for n, r in rejected.items()))
    parts.append("look at the GitHub run of workflow paper-download (job arxiv-to-115)")
    return "; ".join(parts)


def _take_bundle(ctx: ArxivStageContext, store: HarvestStore, reader: VolumeReader, inbox: str,
                 name: str, size: int, sink: EventSink) -> tuple[list[dict], dict[str, int]]:
    """Fetch ``name``, check it against its .sha256 and import it: (its rows, import counts).

    A bad copy is fetched once more; still bad, _BundleDamaged says why. A failed fetch is
    not the bundle's fault and propagates (nothing is recorded)."""
    local = inbox_dir(ctx.harvest_root) / name
    checksum = checksum_file_name(name)
    reason = ""
    try:
        for attempt in (1, 2):
            sink(f"fetch {name} ({size / 1e6:.1f} MB)" + (" again" if attempt == 2 else ""))
            reader.fetch(f"{inbox}/{name}", local, size=size)
            try:
                expected = parse_checksum(reader.read_bytes(f"{inbox}/{checksum}", limit=4096))
            except VolumeNotFound:
                reason = f"no {checksum} beside the bundle"
                continue
            except ValueError as exc:
                reason = f"{checksum} holds no checksum ({exc})"
                continue
            actual = file_sha256(local)
            if actual != expected:
                reason = f"sha256 {actual[:12]} differs from the {expected[:12]} in {checksum}"
                continue
            try:
                return bundle_rows(local), import_bundle(store, ctx.harvest_root, local)
            except (tarfile.TarError, KeyError, ValueError) as exc:
                reason = f"not a readable bundle ({type(exc).__name__}: {str(exc)[:200]})"
        raise _BundleDamaged(reason)
    finally:
        local.unlink(missing_ok=True)


def receive_volume_bundles(ctx: ArxivStageContext, sink: EventSink, reader: VolumeReader, *,
                           sleep_fn=time.sleep, clock=time.monotonic) -> VolumeDelivery:
    """Wait for a run of the downloading host this host has not consumed, then import its bundles.

    "A run" is its heartbeat (written last, on every run, bundles or none): one that is not
    recorded as consumed and finished after the last consumed one. A run that wrote no bundle
    is a normal, empty delta. The bundles taken are every one not yet imported or rejected,
    oldest first, so those of an earlier task whose freeze failed come along; the heartbeat
    names the run's own, and the call keeps waiting until all of them are listed.

    Fails at ``options.receive_deadline_hours`` saying where it looked, what the job last wrote
    and what was last imported. A bundle that stays damaged after one refetch is recorded as
    rejected and fails the stage; later calls skip it and say so."""
    inbox = _volume_inbox(ctx)
    store = HarvestStore(ctx.harvest_db)
    try:
        store.db.executescript(VOLUME_SCHEMA)
        poll_s = float((ctx.options or {}).get("receive_poll_seconds", VOLUME_POLL_SECONDS))
        deadline = clock() + ctx.receive_deadline_hours * 3600
        last_logged = None
        while True:
            beat = _read_heartbeat(reader, inbox)
            listing = _volume_bundles(reader, inbox)
            taken, rejected = volume_imports(store.db), volume_rejections(store.db)
            missing: list[str] = []
            if beat is not None and heartbeat_is_new(store.db, beat):
                missing = [b for b in beat.get("bundles", []) if b not in listing and b not in taken
                           and b not in rejected]
                if not missing:
                    break
            if _cancelled(ctx):
                raise RuntimeError("cancelled while waiting for the downloading host")
            if clock() >= deadline:
                raise RuntimeError(_no_run_message(store.db, inbox, beat, missing, ctx.receive_deadline_hours))
            if last_logged is None or clock() - last_logged >= 600:
                last_logged = clock()
                seen = f"{beat['run_id']} at {beat['finished_at']}" if beat else "none"
                sink(f"waiting for a run of the downloading host to finish in {inbox} (last heartbeat: {seen})")
            sleep_fn(poll_s)

        for name, reason in rejected.items():
            sink(f"rejected bundle {name} stays out ({reason})", level="warning")
        new = {name: size for name, size in listing.items() if name not in taken and name not in rejected}
        base = int(ctx.state_get("volume_imported", 0) or 0)
        ctx.state_set(volume_run=beat["run_id"], volume_packages=base + len(new))
        known = ctx.state_get("delta_started")
        if beat.get("started_at") and (known is None or beat["started_at"] < known):
            ctx.state_set(delta_started=beat["started_at"])
        imported: list[str] = []
        for i, (name, size) in enumerate(new.items(), 1):
            if _cancelled(ctx):
                raise RuntimeError("cancelled while importing bundles")
            try:
                rows, counts = _take_bundle(ctx, store, reader, inbox, name, size, sink)
            except _BundleDamaged as exc:
                record_volume_rejection(store.db, name, size, str(exc))
                raise RuntimeError(
                    f"{name} rejected ({exc}); recorded in volume_imports as rejected, later runs skip it") from None
            record_volume_import(store.db, name, size)
            # The rows were fetched by the job before this stage began; they are still this
            # run's, which freeze_delta tells by "fetched since started".
            fetched = [r["fetched_at"] for r in rows if r.get("fetched_at")]
            known = ctx.state_get("delta_started")
            if fetched and (known is None or min(fetched) < known):
                ctx.state_set(delta_started=min(fetched))
            ctx.state_set(volume_imported=base + i)
            imported.append(name)
            sink(f"imported {name}: {counts['rows']:,} rows, {counts['html']:,} html")
        return VolumeDelivery(imported, beat)
    finally:
        store.close()


def arxiv_delta_volume_probe(ctx: ArxivStageContext) -> Progress:
    """Bundles imported, then the freeze, of those the run left; capped one short until the run
    is frozen (the frozen file, not the import, is what later stages read)."""
    inbox = _volume_inbox(ctx)
    if not ctx.state_get("volume_run"):
        return Progress(done=0, total=0, message=f"waiting for a run of the downloading host in {inbox}")
    bundles = int(ctx.state_get("volume_packages", 0) or 0)
    done = int(ctx.state_get("volume_imported", 0) or 0)
    total = bundles + 1
    return Progress(done=done + int(ctx.delta_file.exists()), total=total,
                    message=f"imported {done} of {bundles} bundle(s) from {inbox}")


def _make_volume_delta_download_stage(ctx: ArxivStageContext, session_factory, reader: Optional[VolumeReader]) -> StageSpec:
    def run(c: ArxivStageContext, sink: EventSink) -> None:
        delivery = receive_volume_bundles(c, sink, reader or VolumeReader.from_env())
        _freeze_delta_run(c, sink, session_factory, c.state_get("delta_started") or datetime.now(timezone.utc).isoformat())
        # Consumed only now: a task that fails before this point leaves the run for the next one.
        store = HarvestStore(c.harvest_db)
        try:
            record_heartbeat_consumed(store.db, delivery.heartbeat["run_id"], delivery.heartbeat["finished_at"])
        finally:
            store.close()

    return StageSpec(
        key="download", title="Receive arXiv delta HTML (volume)",
        run=run, probe=arxiv_delta_volume_probe, gate=_delta_gate,
        poll_interval_s=DELTA_POLL_SECONDS,
        # Waiting holds nothing: the bundles come from the local node, not from arXiv.
        resource_key="",
    )


def make_arxiv_delta_download_stage(ctx: ArxivStageContext, *, session_factory=None,
                                    reader: Optional[VolumeReader] = None) -> StageSpec:
    from core.db import get_db_session

    session_factory = session_factory or get_db_session
    # ctx.source is "volume": a context with any other source is refused when it is built.
    return _make_volume_delta_download_stage(ctx, session_factory, reader)


def make_local_arxiv_delta_download_stage(ctx: ArxivStageContext, *, session_factory=None) -> StageSpec:
    """Enumerate and download the delta on the host that runs it. No dispatcher reaches it on this
    host (make_arxiv_delta_download_stage always receives from the volume); it stays for a host that downloads."""
    from core.db import get_db_session

    session_factory = session_factory or get_db_session

    def run(c: ArxivStageContext, sink: EventSink) -> None:
        # First write wins: a resumed run keeps the original start, so progress and the
        # "judged no_html by this run" set do not reset.
        started = c.state_get("delta_started")
        if started is None:
            started = datetime.now(timezone.utc).isoformat()
            c.state_set(delta_started=started)
        sink("enumerate: OAI-PMH from the stored watermark")
        run_local(build_delta_enumerate_cmd(c), c.project_root, sink, abort=c.abort)
        _, retry_days, limit = _delta_options(c)
        store = HarvestStore(c.harvest_db)
        try:
            queued = len(store.pending(max_attempts=DOWNLOAD_MAX_ATTEMPTS, limit=limit or None,
                                       scope=slice_sql(DELTA_KEY), no_html_retry_days=retry_days))
        finally:
            store.close()
        c.state_set(delta_queued=queued)
        sink(f"download: {queued:,} papers queued")
        run_local(build_delta_download_cmd(c), c.project_root, sink, abort=c.abort)
        _freeze_delta_run(c, sink, session_factory, started)

    return StageSpec(
        key="download", title="Download arXiv delta HTML",
        run=run, probe=arxiv_delta_download_probe, gate=_delta_gate,
        poll_interval_s=DELTA_POLL_SECONDS,
        # HTTP to arXiv through the first hop: the same single slot as a conference download.
        resource_key="net",
    )


# ====================================================================== extract
def arxiv_extract_probe(ctx: ArxivStageContext) -> Progress:
    """Markdown files written for this key's papers (live while converting),
    capped one short of done until the manifest exists — preindex reads the
    manifest, so converted files without it are not a finished stage.

    Papers whose HTML was found to have no usable article (the persisted convert
    outcome of their current version) can never get Markdown: they are neither
    done nor expected, so a run holding them still completes."""
    ids = ctx.ok_ids()
    rejected = rejected_ids(ctx.harvest_db, ids) if ids else set()
    ids = [a for a in ids if a not in rejected]
    if ctx.is_delta:
        written = sum(1 for a in ids if _converted_from_current_html(ctx, a))
    else:
        written = sum(1 for a in ids if ctx.paths.markdown(a).exists())
    if not ctx.manifest_path.exists():
        written = min(written, max(len(ids) - 1, 0))
    return Progress(done=written, total=len(ids),
                    message=f"{len(rejected):,} rejected (no usable article), not expected" if rejected else "")


def _converted_from_current_html(ctx: ArxivStageContext, arxiv_id: str) -> bool:
    """A delta paper is converted when its outputs are not older than its HTML.

    "The Markdown exists" would be true for every paper that got a new version
    (the old version's file is still there) and skip the conversion that matters.
    Same test the converter itself applies (arxiv_corpus._needs_convert)."""
    html = html_path(ctx.harvest_root, arxiv_id)
    md, doc = ctx.paths.markdown(arxiv_id), ctx.paths.doc(arxiv_id)
    return html.exists() and md.exists() and doc.exists() and not _needs_convert(html, md, doc)


def build_arxiv_extract_cmd(ctx: ArxivStageContext) -> list[str]:
    # `convert` writes the manifest after converting, for a slice and for a delta run alike.
    target = (["--ids-file", str(ctx.delta_file), "--run-id", str(ctx.task_id)] if ctx.is_delta
              else ["--slice", str(ctx.slice_key)])
    return [
        sys.executable, "scripts/collect/arxiv_cs_corpus.py", "convert",
        *target, "--workers", str(ctx.convert_workers),
    ]


def arxiv_extract_env(ctx: ArxivStageContext) -> dict:
    env = dict(os.environ)
    env["ARXIV_HARVEST_ROOT"] = str(ctx.harvest_root)
    env["ARXIV_CORPUS_ROOT"] = str(ctx.corpus_root)
    return env


def make_arxiv_extract_stage(ctx: ArxivStageContext) -> StageSpec:
    def run(c: ArxivStageContext, sink: EventSink) -> None:
        if c.is_delta and not c.ok_ids():
            # The converter exits non-zero for an empty selection (0 of 0 is below its
            # markdown ratio), and there is nothing to preindex: no manifest, no work.
            sink("no new HTML in this run: nothing to convert")
            return
        sink(f"convert arXiv {slice_label(c.slice_key)} HTML -> JSON + Markdown (workers={c.convert_workers})")
        run_local(build_arxiv_extract_cmd(c), c.project_root, sink, env=arxiv_extract_env(c),
                  abort=c.abort)
        report = load_json_with_recovery(c.paths.report, {})
        sink(f"convert: {report.get('markdown', 0):,}/{report.get('convertible', 0):,} convertible have Markdown, "
             f"{len(report.get('rejected', [])):,} rejected, {len(report.get('errors', [])):,} errors")

    def gate(c: ArxivStageContext) -> GateResult:
        pr = arxiv_extract_probe(c)
        if pr.total and pr.done < 0.95 * pr.total:
            return GateResult(False, f"only {pr.done:,}/{pr.total:,} convertible papers converted ({pr.message})")
        return GateResult(True)

    return StageSpec(
        key="extract", title=f"Convert arXiv {slice_label(ctx.slice_key)} HTML → Markdown",
        run=run, probe=arxiv_extract_probe, gate=gate,
        resource_key="cpu-convert",
    )


def build_arxiv_pipeline(ctx: ArxivStageContext) -> list[StageSpec]:
    from app.services.pipeline.stages import (
        make_cleanup_stage,
        make_embed_stage_for,
        make_ingest_stage,
        make_preindex_stage,
    )

    specs = [
        make_arxiv_delta_download_stage(ctx) if ctx.is_delta else make_arxiv_download_stage(ctx),
        make_arxiv_extract_stage(ctx),
        make_preindex_stage(ctx),
        make_embed_stage_for(ctx),
        make_ingest_stage(ctx),
        make_cleanup_stage(ctx),
    ]
    logger.info(
        f"registry.build_arxiv_pipeline slice={ctx.slice_key} execution_mode={ctx.execution_mode} "
        f"stages={[s.key for s in specs]}"
    )
    return specs


# ================================================================= run listing
ARXIV_RUN_KINDS = ("new", "replaced", "withdrawn", "revived", "html_missing", "convert_rejected")


@dataclass(frozen=True)
class RunSources:
    """Where a delta run's records live: its task row, SQL papers, corpus and harvest roots."""

    store: "TaskStore"
    session_factory: Callable
    corpus_root: Path
    harvest_root: Path


def default_run_sources() -> RunSources:
    from core.db import get_db_session

    return RunSources(TaskStore(), get_db_session, default_corpus_root(), default_harvest_root())


class NotADeltaRun(LookupError):
    """The task does not exist or is not an arXiv delta run."""


def delta_run_lists(sources: RunSources, task_id: int) -> dict[str, list[str]]:
    """paper_id lists of one delta run, per ARXIV_RUN_KINDS, from the files the run wrote:
    the preindex report (new / replaced / the flags it flipped) and the frozen selection
    (flags that moved without a new text, papers arXiv has no HTML for yet) and the convert
    report (papers whose HTML has no usable article)."""
    from app.services.pipeline.stages import preindex_report_path

    task = sources.store.get_task(task_id)
    params = (task or {}).get("params") or {}
    if task is None or params.get("venue") != ARXIV_VENUE or params.get("year") != DELTA_KEY:
        raise NotADeltaRun(f"task {task_id} is not an arXiv delta run")
    ctx = ArxivStageContext(venue=ARXIV_VENUE, year=DELTA_KEY, store=sources.store, task_id=task_id,
                            corpus_root=sources.corpus_root, harvest_root=sources.harvest_root)
    report = load_json_with_recovery(preindex_report_path(ctx), {})
    flags = {e["paper_id"]: bool(e["withdrawn"]) for e in report.get("withdrawn_changed", [])}
    flags.update(ctx.withdrawn_only())
    return {
        "new": list(report.get("new", [])),
        "replaced": list(report.get("replaced", [])),
        "withdrawn": [pid for pid, flag in flags.items() if flag],
        "revived": [pid for pid, flag in flags.items() if not flag],
        "html_missing": [paper_id_for(a) for a in ctx.frozen().get("html_missing", [])],
        "convert_rejected": [paper_id_for(a) for a in ctx.rejected_in_run()],
    }


def delta_run_page(sources: RunSources, task_id: int, kind: str, offset: int, limit: int) -> dict:
    """One page of ``kind`` papers of a delta run, titled from SQL (the harvest database
    for papers that are not indexed yet), plus every kind's count for the tab labels."""
    from core.db.models import Paper as DBPaper

    lists = delta_run_lists(sources, task_id)
    page = lists[kind][offset:offset + limit]
    with sources.session_factory() as db:
        rows = {r.paper_id: r for r in db.query(
            DBPaper.paper_id, DBPaper.arxiv_id, DBPaper.title, DBPaper.source_version,
            DBPaper.withdrawn).filter(DBPaper.paper_id.in_(page))}
    unindexed = [pid for pid in page if pid not in rows]
    titles: dict[str, str] = {}
    db_file = sources.harvest_root / "arxiv_cs.sqlite3"
    if unindexed and db_file.exists():
        harvest = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        try:
            ids = [pid[len("arxiv-"):] for pid in unindexed]
            titles = {paper_id_for(a): t for a, t in harvest.execute(
                f"SELECT arxiv_id, title FROM papers WHERE arxiv_id IN ({','.join('?' * len(ids))})", ids)}
        finally:
            harvest.close()
    papers = []
    for pid in page:
        row = rows.get(pid)
        papers.append({
            "paper_id": pid,
            "arxiv_id": row.arxiv_id if row else pid[len("arxiv-"):],
            "title": row.title if row else titles.get(pid),
            "source_version": row.source_version if row else None,
            "withdrawn": bool(row.withdrawn) if row else None,
        })
    return {"task_id": task_id, "kind": kind, "offset": offset, "limit": limit,
            "total": len(lists[kind]), "counts": {k: len(v) for k, v in lists.items()},
            "papers": papers}


def discover_batch_keys(harvest_root: Path, months: list[int]) -> list[int]:
    """Batch keys of the given months known on this host, newest first: from
    the downloading host's progress file (still downloading), the inbox
    (pushed) and the local harvest DB (already received)."""
    for month in months:
        validate_slice_key(month)
        if is_batch_key(month) or month == LEGACY_SLICE:
            raise ValueError(f"months must be YYYYMM month slices, got {month}")
    wanted = set(months)
    keys = {int(k) for k in read_progress(inbox_dir(harvest_root))}
    inbox = inbox_dir(harvest_root)
    if inbox.is_dir():
        for bundle in inbox.glob("arxiv_*.tar"):
            stem = bundle.name[len("arxiv_"):-len(".tar")]
            if stem.isdigit():
                keys.add(int(stem))
    db_path = harvest_root / "arxiv_cs.sqlite3"
    if db_path.exists():
        from pipeline.paper_crawler.arxiv_corpus import batch_of

        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            for (arxiv_id,) in db.execute("SELECT arxiv_id FROM papers WHERE arxiv_id NOT LIKE '%/%'"):
                key = batch_of(arxiv_id)
                if key is not None and month_of(key) in wanted:
                    keys.add(key)
        finally:
            db.close()
    return sorted((k for k in keys if is_batch_key(k) and month_of(k) in wanted), reverse=True)


def arxiv_context(**kw) -> ArxivStageContext:
    ctx = ArxivStageContext(**kw)
    ctx.venue = ARXIV_VENUE
    return ctx


def maybe_arxiv_context(venue: str, **kw) -> Optional[ArxivStageContext]:
    return arxiv_context(venue=venue, **kw) if is_arxiv_venue(venue) else None
