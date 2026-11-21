"""Orchestrate medical corpus preparation (catalog + plan; optional execute).

Default collect keeps **top medical journals only** (see
:mod:`pipeline.medical_corpus.top_journals`). Non-flagship OA dumps are dropped.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from filelock import FileLock, Timeout

from pipeline.medical_corpus.europe_pmc import DEFAULT_RECENT_QUERY, EuropePMCClient
from pipeline.medical_corpus.models import MedicalPaperRecord
from pipeline.medical_corpus.pdf_fetch import (
    DEFAULT_MAX_BYTES_PER_SECOND,
    DownloadDiskGuardStopped,
    DownloadProgressCallback,
    DownloadRateLimiter,
    download_pdf,
    plan_pdf_fetch,
    planned_pdf_path,
)
from pipeline.medical_corpus.proxy import MedicalProxyConfig, load_medical_proxy_from_settings
from pipeline.medical_corpus.top_journals import (
    default_top_journal_query,
    filter_top_journal_records,
)
from pipeline.paper_crawler.runtime_files import write_json_atomic
from pipeline.paper_crawler.runtime_files import build_write_in_progress_path, is_valid_pdf_artifact

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path(os.environ.get("MEDICAL_DATA_ROOT", "data/medical"))
DEFAULT_MIN_FREE_BYTES = int(os.environ.get("MEDICAL_MIN_FREE_BYTES", str(20 * 1024**3)))


def require_absolute_medical_data_root(data_root: Path) -> Path:
    """Return the durable artifact root or fail before a download can write anywhere.

    A relative root silently depends on the service/CLI working directory and can
    put a multi-terabyte corpus on the wrong disk.  The production unit supplies
    ``/mnt/lsb/retrival/medical`` via MEDICAL_DATA_ROOT; tests pass a temporary
    absolute root explicitly.
    """
    root = Path(data_root).expanduser()
    if not root.is_absolute():
        raise ValueError("MEDICAL_DATA_ROOT must be an absolute path")
    return root.resolve()


@dataclass(slots=True)
class PrepareConfig:
    data_root: Path = field(default_factory=lambda: DEFAULT_DATA_ROOT)
    # Prefer top-journal prefiltered query; fall back to recent OA if overridden.
    query: str = field(default_factory=default_top_journal_query)
    max_records: int = 1000
    page_size: int = 100
    execute_download: bool = False
    allow_cf_bypass: bool = True
    download_limit: int = 0  # 0 = all planned when execute
    min_interval_s: float = 0.35
    # Hard gate: drop anything not on the top-journal whitelist (default ON).
    top_journals_only: bool = True
    proxy: Optional[MedicalProxyConfig] = None
    max_bytes_per_second: int = DEFAULT_MAX_BYTES_PER_SECOND
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES

    @property
    def manifest_path(self) -> Path:
        return self.data_root / "manifest" / "medical_manifest.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.data_root / "manifest" / "medical_prepare_summary.json"

    @property
    def pdf_root(self) -> Path:
        return self.data_root / "pdf"


_REPLACE_ATTEMPTS = 12
_REPLACE_BACKOFF_S = 0.05


def _atomic_replace(tmp: Path, path: Path) -> None:
    """``os.replace`` with a bounded retry for a concurrently-open destination.

    POSIX replaces a file that readers hold open; Windows refuses with
    PermissionError [WinError 5]. That is not hypothetical here: the pipeline's
    poller thread reads this exact manifest through ``_manifest_progress`` ->
    ``_load_jsonl`` -> ``Path.read_text()`` every ``poll_interval_s``, and its
    first probe fires as the stage starts, so reader and writer overlap by
    construction. When the collision landed on the disk-guard write, the
    PermissionError escaped ``run_download_pass`` before the summary was stored,
    so the stage failed for the wrong reason and callers reading
    ``state["summary"]`` got a KeyError — observed once in four full-suite runs
    on 2026-09-02, never in isolation, because load widens the window.

    Retrying keeps the guarantee rather than weakening it: the destination is
    only ever swapped whole, and a collision that outlasts the budget still
    raises instead of silently leaving the checkpoint unwritten.
    """
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Only Windows produces this for an open destination, and only
            # transiently — the reader closes microseconds later.
            if attempt + 1 == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))


def _write_jsonl(path: Path, records: list[MedicalPaperRecord]) -> None:
    """Atomically replace the whole manifest after every durable checkpoint.

    A process kill can leave an old complete manifest, never a half-written JSONL
    file that would make restart recovery treat good PDFs as unknown records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in records]
    tmp = path.with_name(f".{path.name}.write_in_progress")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))
        handle.flush()
        os.fsync(handle.fileno())
    _atomic_replace(tmp, path)


def _load_jsonl(path: Path) -> list[MedicalPaperRecord]:
    if not path.exists():
        return []
    out: list[MedicalPaperRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}: line {line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"{path}: line {line_number}: medical manifest record must be a JSON object, "
                f"got {type(payload).__name__}"
            )
        try:
            out.append(MedicalPaperRecord.from_dict(payload))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: line {line_number}: invalid medical manifest record: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return out


def _downloaded_artifact_error(record: MedicalPaperRecord) -> str | None:
    """Validate a manifest's claimed completed PDF before resume skips it."""
    if not record.pdf_path:
        return "manifest has no pdf_path"
    path = Path(record.pdf_path)
    try:
        size = path.stat().st_size
    except OSError:
        return "PDF path is missing or unreadable"
    if record.pdf_bytes is None or size != int(record.pdf_bytes):
        return f"size mismatch manifest={record.pdf_bytes!r} disk={size}"
    if not record.pdf_sha256:
        return "manifest has no sha256"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record.pdf_sha256:
        return "sha256 mismatch"
    if not is_valid_pdf_artifact(path):
        return "PDF parser validation failed"
    return None


def _cleanup_stale_write_in_progress(cfg: PrepareConfig, record: MedicalPaperRecord) -> None:
    """Remove only this record's deterministic temp path before safe resume."""
    dest = planned_pdf_path(cfg.pdf_root, record)
    build_write_in_progress_path(dest).unlink(missing_ok=True)


def collect_manifest_v2(cfg: PrepareConfig) -> dict[str, Any]:
    """Fetch EuropePMC catalog and write JSONL manifest (network via proxy).

    When ``top_journals_only`` is True (default), non-flagship journals are
    dropped client-side even if the EuropePMC query was broader.
    """
    proxy = cfg.proxy or load_medical_proxy_from_settings()
    t0 = time.perf_counter()
    raw_records: list[MedicalPaperRecord] = []
    hit_count = 0
    with EuropePMCClient(proxy=proxy, min_interval_s=cfg.min_interval_s) as client:
        page0 = client.search_page(cfg.query, page_size=1)
        hit_count = page0.hit_count
        for rec in client.iter_records(
            cfg.query, page_size=cfg.page_size, max_records=cfg.max_records,
        ):
            raw_records.append(rec)

    journal_stats: dict[str, Any] = {
        "top_journals_only": bool(cfg.top_journals_only),
        "raw_fetched": len(raw_records),
    }
    if cfg.top_journals_only:
        records, filter_stats = filter_top_journal_records(raw_records)
        journal_stats.update(filter_stats)
    else:
        records = raw_records
        journal_stats["kept"] = len(records)
        journal_stats["dropped_not_top"] = 0
        journal_stats["dropped_no_journal"] = 0

    for rec in records:
        plan = plan_pdf_fetch(rec, pdf_root=cfg.pdf_root)
        if plan.action == "would_download":
            rec.download_status = "planned"
        elif plan.action == "skip_exists":
            rec.download_status = "skipped"
            rec.pdf_path = str(plan.dest_path)
        elif plan.action == "no_url":
            rec.download_status = "failed"
            rec.download_error = plan.note

    _write_jsonl(cfg.manifest_path, records)
    summary = {
        "stage": "collect",
        "query": cfg.query,
        "hit_count_reported": hit_count,
        "records_written": len(records),
        "journal_filter": journal_stats,
        "manifest_path": str(cfg.manifest_path),
        "pdf_root": str(cfg.pdf_root),
        "proxy": proxy.http_proxy,
        "execute_download": bool(cfg.execute_download),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "planned_pdf": sum(1 for r in records if r.download_status == "planned"),
        "skipped_existing": sum(1 for r in records if r.download_status == "skipped"),
        "no_pdf_url": sum(1 for r in records if not r.pdf_url),
    }
    write_json_atomic(cfg.summary_path, summary)
    logger.info(
        "medical collect done records=%s (raw=%s top_only=%s) path=%s",
        len(records), len(raw_records), cfg.top_journals_only, cfg.manifest_path,
    )
    return summary


# public alias
collect_manifest = collect_manifest_v2


DownloadRecordCheckpoint = Callable[[MedicalPaperRecord, int, int], None]
DownloadCancellationProbe = Callable[[], bool]


def _run_download_pass_locked(
    cfg: PrepareConfig,
    *,
    progress_callback: DownloadProgressCallback | None = None,
    checkpoint_callback: DownloadRecordCheckpoint | None = None,
    rate_limiter: DownloadRateLimiter | None = None,
    should_cancel: DownloadCancellationProbe | None = None,
    abort_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Download planned rows with a checkpoint after every completed record.

    The caller can safely restart after any checkpoint: downloaded rows retain
    their verified path/hash/size and are skipped rather than fetched again.
    """
    cfg.data_root = require_absolute_medical_data_root(cfg.data_root)
    proxy = cfg.proxy or load_medical_proxy_from_settings()
    records = _load_jsonl(cfg.manifest_path)
    if not records:
        raise FileNotFoundError(f"no manifest at {cfg.manifest_path}; run collect first")
    t0 = time.perf_counter()
    updated: list[MedicalPaperRecord] = []
    downloaded = planned = skipped = failed = blocked = 0
    stopped_low_disk = False
    disk_free_bytes: int | None = None
    disk_guard_reason: str | None = None
    budget = cfg.download_limit if cfg.download_limit > 0 else len(records)
    used = 0
    attempted = 0
    limiter = rate_limiter or DownloadRateLimiter(cfg.max_bytes_per_second)
    for record_index, rec in enumerate(records):
        if should_cancel and should_cancel():
            raise RuntimeError("medical download cancelled")
        # This is intentionally before the verified-artifact fast path: a
        # crash can leave a sibling temp even when the final PDF was renamed.
        _cleanup_stale_write_in_progress(cfg, rec)
        if rec.download_status in {"downloaded", "skipped"} and rec.pdf_path:
            artifact_error = _downloaded_artifact_error(rec)
            if artifact_error is None:
                updated.append(rec)
                skipped += 1
                continue
            # Do not silently retain a broken completed status.  The regular
            # pending/error flow below either re-downloads it or checkpoints a
            # durable explanatory failure when no source URL remains.
            rec.download_status = "pending"
            rec.download_error = f"resume revalidation failed: {artifact_error}"
        if used >= budget:
            updated.append(rec)
            continue
        # Check the target mount before opening each new network transfer.  A
        # threshold breach is a controlled stop: the last JSONL checkpoint is
        # already durable and a future run can resume remaining rows.
        disk_free_bytes = shutil.disk_usage(cfg.data_root).free
        if disk_free_bytes < cfg.min_free_bytes:
            stopped_low_disk = True
            disk_guard_reason = (
                f"free_bytes={disk_free_bytes} below reserved_bytes={cfg.min_free_bytes}"
            )
            updated.append(rec)
            updated.extend(records[record_index + 1:])
            _write_jsonl(cfg.manifest_path, updated)
            logger.warning(
                "medical download stopped before %s: free=%s below floor=%s",
                rec.paper_id, disk_free_bytes, cfg.min_free_bytes,
            )
            break
        if not rec.pdf_url:
            rec.download_status = "failed"
            rec.download_error = rec.download_error or "no pdf_url"
            failed += 1
            updated.append(rec)
            _write_jsonl(cfg.manifest_path, updated + records[record_index + 1:])
            if checkpoint_callback:
                checkpoint_callback(rec, attempted, budget)
            continue
        used += 1
        attempted += 1
        def reserve_artifact_space(remaining_bytes: int | None) -> str | None:
            """Protect the configured floor both before and during a stream."""
            nonlocal disk_free_bytes
            disk_free_bytes = shutil.disk_usage(cfg.data_root).free
            if disk_free_bytes < cfg.min_free_bytes:
                return (
                    f"free_bytes={disk_free_bytes} below reserved_bytes={cfg.min_free_bytes}"
                )
            if remaining_bytes is not None and disk_free_bytes - remaining_bytes < cfg.min_free_bytes:
                return (
                    f"free_bytes={disk_free_bytes} minus remaining_bytes={remaining_bytes} "
                    f"would cross reserved_bytes={cfg.min_free_bytes}"
                )
            return None

        try:
            out = download_pdf(
                rec,
                pdf_root=cfg.pdf_root,
                proxy=proxy,
                execute=cfg.execute_download,
                allow_cf_bypass=cfg.allow_cf_bypass,
                max_bytes_per_second=cfg.max_bytes_per_second,
                rate_limiter=limiter,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
                abort_event=abort_event,
                disk_guard=reserve_artifact_space,
            )
        except DownloadDiskGuardStopped as exc:
            # The fetcher already removed its .write_in_progress file.  Keep
            # this paper pending and atomically checkpoint the old facts so a
            # later resume can retry without pretending it completed.
            stopped_low_disk = True
            disk_guard_reason = str(exc)
            rec.download_status = "pending"
            rec.download_error = f"disk guard stopped stream: {exc}"
            updated.append(rec)
            updated.extend(records[record_index + 1:])
            _write_jsonl(cfg.manifest_path, updated)
            logger.warning("medical download stopped during %s: %s", rec.paper_id, exc)
            break
        if out.download_status == "downloaded":
            downloaded += 1
        elif out.download_status == "planned":
            planned += 1
        elif out.download_status == "skipped":
            skipped += 1
        elif out.download_status == "blocked":
            blocked += 1
        else:
            failed += 1
        updated.append(out)
        # Checkpoint each completed source, not every 50.  This is intentionally
        # more I/O than the crawler because PDFs are expensive and a resume must
        # never silently lose the verified success immediately before a restart.
        _write_jsonl(cfg.manifest_path, updated + records[record_index + 1:])
        if checkpoint_callback:
            checkpoint_callback(out, attempted, budget)

    _write_jsonl(cfg.manifest_path, updated)
    # Summary is a manifest state snapshot for the UI, not merely this worker's
    # delta.  This makes a restarted job show previous verified downloads too.
    status_counts = {
        status: sum(1 for row in updated if row.download_status == status)
        for status in ("downloaded", "planned", "skipped", "failed", "blocked")
    }
    unknown_sizes = [row for row in updated if row.download_status not in {"downloaded", "skipped"} and not row.pdf_bytes]
    estimated_remaining_bytes = (
        None if unknown_sizes else sum(int(row.pdf_bytes or 0) for row in updated if row.download_status not in {"downloaded", "skipped"})
    )
    summary = {
        "stage": "download",
        "execute_download": cfg.execute_download,
        "records": len(updated),
        "attempted": attempted,
        "budget": budget,
        "downloaded": status_counts["downloaded"],
        "planned": status_counts["planned"],
        "skipped": status_counts["skipped"],
        "failed": status_counts["failed"],
        "blocked": status_counts["blocked"],
        "cf_bypass_enabled": cfg.allow_cf_bypass,
        "proxy": proxy.http_proxy,
        "max_bytes_per_second": cfg.max_bytes_per_second,
        "min_free_bytes": cfg.min_free_bytes,
        "reserved_bytes": cfg.min_free_bytes,
        "disk_free_bytes": disk_free_bytes if disk_free_bytes is not None else shutil.disk_usage(cfg.data_root).free,
        "stopped_low_disk": stopped_low_disk,
        "disk_guard_reason": disk_guard_reason,
        "estimated_remaining_bytes": estimated_remaining_bytes,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "manifest_path": str(cfg.manifest_path),
    }
    write_json_atomic(cfg.summary_path, summary)
    return summary


def run_download_pass(
    cfg: PrepareConfig,
    *,
    progress_callback: DownloadProgressCallback | None = None,
    checkpoint_callback: DownloadRecordCheckpoint | None = None,
    rate_limiter: DownloadRateLimiter | None = None,
    should_cancel: DownloadCancellationProbe | None = None,
    abort_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run one pass under the cross-process corpus-root transfer lock.

    This is deliberately here rather than only in the FastAPI service so the
    CLI cannot accidentally overlap a UI-triggered pipeline and double the
    aggregate proxy bandwidth.  The lock file is on ``MEDICAL_DATA_ROOT``'s
    artifact mount, therefore distinct .88 service processes also contend.
    """
    lock = FileLock(str(cfg.manifest_path.parent / "medical-download.lock"))
    try:
        with lock.acquire(timeout=0):
            return _run_download_pass_locked(
                cfg,
                progress_callback=progress_callback,
                checkpoint_callback=checkpoint_callback,
                rate_limiter=rate_limiter,
                should_cancel=should_cancel,
                abort_event=abort_event,
            )
    except Timeout as exc:
        raise RuntimeError("another process owns the medical download lock") from exc


def prepare(cfg: PrepareConfig) -> dict[str, Any]:
    """Collect catalog then download pass (still dry-run unless execute_download)."""
    collect_summary = collect_manifest_v2(cfg)
    download_summary = run_download_pass(cfg)
    return {"collect": collect_summary, "download": download_summary}
