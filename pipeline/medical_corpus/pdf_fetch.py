"""PDF fetch planner + optional real download (proxy-mandatory, CF-aware)."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import httpx

from pipeline.download_control import (
    AggregateDownloadRateLimiter as DownloadRateLimiter,
    DEFAULT_DOWNLOAD_BYTES_PER_SECOND,
    DEFAULT_STREAM_CHUNK_BYTES,
)
from pipeline.medical_corpus.cf_bypass import CfBypassResult, solve_cloudflare_for_url_sync
from pipeline.medical_corpus.models import MedicalPaperRecord, sanitize_id_component
from pipeline.medical_corpus.proxy import MedicalProxyConfig, load_medical_proxy_from_settings
from pipeline.medical_corpus.shield import classify_http_shield, retry_delay_seconds
from pipeline.paper_crawler.runtime_files import build_write_in_progress_path, is_valid_pdf_artifact

logger = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
# User-facing contract: decimal network MB/s, not binary MiB/s.
DEFAULT_MAX_BYTES_PER_SECOND = DEFAULT_DOWNLOAD_BYTES_PER_SECOND
_STREAM_CHUNK_BYTES = DEFAULT_STREAM_CHUNK_BYTES


_GLOBAL_DOWNLOAD_LIMITER = DownloadRateLimiter()


@dataclass(slots=True)
class PdfFetchPlan:
    paper_id: str
    pdf_url: str
    dest_path: Path
    action: str  # skip_exists | would_download | download | blocked | no_url
    note: str = ""


def planned_pdf_path(pdf_root: Path, record: MedicalPaperRecord) -> Path:
    year = str(record.year or "unknown")
    journal = sanitize_id_component((record.journal or "unknown")[:40]) or "unknown"
    stem = sanitize_id_component(record.paper_id)
    return pdf_root / year / journal / f"{stem}.pdf"


def plan_pdf_fetch(
    record: MedicalPaperRecord,
    *,
    pdf_root: Path,
    skip_existing: bool = True,
) -> PdfFetchPlan:
    dest = planned_pdf_path(pdf_root, record)
    if not record.pdf_url:
        return PdfFetchPlan(record.paper_id, "", dest, "no_url", "record has no pdf_url")
    if skip_existing and is_valid_pdf_artifact(dest):
        return PdfFetchPlan(
            record.paper_id, record.pdf_url, dest, "skip_exists", "valid PDF already on disk",
        )
    return PdfFetchPlan(
        record.paper_id, record.pdf_url, dest, "would_download", "dry-run plan only",
    )


DownloadProgressCallback = Callable[[MedicalPaperRecord, int, int | None], None]
DownloadDiskReservationProbe = Callable[[int | None], str | None]


class DownloadCancelled(RuntimeError):
    """Raised by the owning durable Task when a user cancels its transfer."""


class DownloadDiskGuardStopped(RuntimeError):
    """Raised before a partial PDF can consume the artifact mount reserve."""


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel and should_cancel():
        raise DownloadCancelled("medical download cancelled")


def _wait_with_abort(
    timeout_s: float,
    *,
    abort_event: threading.Event | None,
    should_cancel: Callable[[], bool] | None,
) -> None:
    """Back off without making a cancellation wait for the retry deadline."""
    _raise_if_cancelled(should_cancel)
    if timeout_s <= 0:
        return
    if abort_event is not None:
        if abort_event.wait(timeout_s):
            raise DownloadCancelled("medical download cancelled during retry backoff")
        _raise_if_cancelled(should_cancel)
        return
    # The standalone library/CLI path may have only a probe.  Poll in short
    # waits so it stays cancellation-responsive without a busy loop.
    until = time.monotonic() + timeout_s
    while True:
        _raise_if_cancelled(should_cancel)
        remaining = until - time.monotonic()
        if remaining <= 0:
            return
        threading.Event().wait(min(0.1, remaining))


def _check_disk_reservation(
    disk_guard: DownloadDiskReservationProbe | None,
    remaining_bytes: int | None,
) -> None:
    if disk_guard is None:
        return
    reason = disk_guard(remaining_bytes)
    if reason:
        raise DownloadDiskGuardStopped(reason)


def _read_prefix_and_remainder(
    response: httpx.Response,
    *,
    limiter: DownloadRateLimiter,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[bytes, object]:
    """Read enough bytes to classify a response, returning its live iterator too."""
    iterator = response.iter_bytes(chunk_size=_STREAM_CHUNK_BYTES)
    prefix_parts: list[bytes] = []
    prefix_size = 0
    while prefix_size < 12_000:
        if should_cancel and should_cancel():
            raise DownloadCancelled("medical download cancelled")
        limiter.wait_for_next_read()
        try:
            chunk = next(iterator)
        except StopIteration:
            break
        if chunk:
            prefix_parts.append(chunk)
            prefix_size += len(chunk)
        if prefix_size >= 5 and b"".join(prefix_parts)[:5] == b"%PDF-":
            # A valid PDF header is sufficient to safely start streaming; only
            # error/challenge responses need the longer diagnostic prefix.
            break
    return b"".join(prefix_parts), iterator


def download_pdf(
    record: MedicalPaperRecord,
    *,
    pdf_root: Path,
    proxy: MedicalProxyConfig | None = None,
    execute: bool = False,
    allow_cf_bypass: bool = True,
    max_retries: int = 3,
    timeout_s: float = 120.0,
    min_bytes: int = 1024,
    max_bytes_per_second: int = DEFAULT_MAX_BYTES_PER_SECOND,
    rate_limiter: DownloadRateLimiter | None = None,
    progress_callback: DownloadProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    abort_event: threading.Event | None = None,
    disk_guard: DownloadDiskReservationProbe | None = None,
) -> MedicalPaperRecord:
    """Download one PDF when ``execute=True``; otherwise only plan + annotate.

    Never silently bypasses the proxy. On CF/403/HTML-challenge, optionally
    invokes TurnstileBypass (minimal headless) once, then retries with cookies.
    """
    proxy = proxy or load_medical_proxy_from_settings()
    limiter = rate_limiter
    if limiter is None:
        limiter = (_GLOBAL_DOWNLOAD_LIMITER if max_bytes_per_second == DEFAULT_MAX_BYTES_PER_SECOND
                   else DownloadRateLimiter(max_bytes_per_second))
    plan = plan_pdf_fetch(record, pdf_root=pdf_root)
    record = MedicalPaperRecord.from_dict(record.to_dict())

    if plan.action == "no_url":
        record.download_status = "failed"
        record.download_error = plan.note
        return record
    if plan.action == "skip_exists":
        record.download_status = "skipped"
        record.pdf_path = str(plan.dest_path)
        record.notes = list(record.notes) + [plan.note]
        return record
    if not execute:
        record.download_status = "planned"
        record.notes = list(record.notes) + [f"dry-run: would GET {plan.pdf_url} -> {plan.dest_path}"]
        return record

    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "application/pdf,*/*;q=0.8",
    }
    session_cookies: dict[str, str] = {}
    last_error: Optional[str] = None
    cf_tried = False

    for attempt in range(1, max_retries + 1):
        try:
            _raise_if_cancelled(should_cancel)
            with httpx.Client(
                proxy=proxy.as_httpx_proxy(),
                timeout=timeout_s,
                follow_redirects=True,
                headers=headers,
                cookies=session_cookies or None,
            ) as client, client.stream("GET", plan.pdf_url) as resp:
                prefix, iterator = _read_prefix_and_remainder(
                    resp, limiter=limiter, should_cancel=should_cancel,
                )
                decision = classify_http_shield(
                    status_code=resp.status_code,
                    content_type=resp.headers.get("content-type"),
                    body=prefix[:12000],
                    expect_pdf=True,
                    headers=resp.headers,
                )
                if decision.kind.value == "none" and prefix.startswith(b"%PDF"):
                    dest = plan.dest_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = build_write_in_progress_path(dest)
                    total = int(resp.headers.get("content-length") or 0) or None
                    written = 0
                    digest = hashlib.sha256()
                    try:
                        # Content-Length is checked before opening the temp
                        # file; every write below rechecks the remaining bytes.
                        _check_disk_reservation(disk_guard, total)
                        with tmp.open("wb") as output:
                            if prefix:
                                _check_disk_reservation(
                                    disk_guard,
                                    None if total is None else max(0, total - len(prefix)),
                                )
                                output.write(prefix)
                                digest.update(prefix)
                                written += len(prefix)
                                if progress_callback:
                                    progress_callback(record, written, total)
                            while True:
                                _raise_if_cancelled(should_cancel)
                                # Reserve BEFORE reading the next socket chunk.  A
                                # post-read sleep only paces disk writes while the
                                # proxy has already delivered an unbounded burst.
                                limiter.wait_for_next_read()
                                try:
                                    chunk = next(iterator)
                                except StopIteration:
                                    break
                                if not chunk:
                                    continue
                                _check_disk_reservation(
                                    disk_guard,
                                    None if total is None else max(0, total - written - len(chunk)),
                                )
                                output.write(chunk)
                                digest.update(chunk)
                                written += len(chunk)
                                if progress_callback:
                                    progress_callback(record, written, total)
                    except (DownloadCancelled, DownloadDiskGuardStopped):
                        tmp.unlink(missing_ok=True)
                        raise
                    except Exception:
                        # Filesystem write/flush/progress failures must never
                        # leave a resumable-looking partial artifact behind.
                        tmp.unlink(missing_ok=True)
                        raise
                    if written < min_bytes:
                        tmp.unlink(missing_ok=True)
                        last_error = f"PDF too small ({written} bytes)"
                        continue
                    # A valid header and byte count are insufficient: durable
                    # manifest success requires the same real PdfReader check
                    # that a later resume will use before skipping this file.
                    if not is_valid_pdf_artifact(tmp):
                        tmp.unlink(missing_ok=True)
                        last_error = "PDF parser validation failed before rename"
                        continue
                    try:
                        tmp.replace(dest)
                    except OSError as exc:
                        tmp.unlink(missing_ok=True)
                        last_error = f"PDF rename failed: {type(exc).__name__}: {exc}"
                        continue
                    record.pdf_path = str(dest)
                    record.pdf_bytes = written
                    record.pdf_sha256 = digest.hexdigest()
                    record.download_status = "downloaded"
                    record.download_error = None
                    record.shield_status = "ok"
                    return record

                record.shield_status = decision.kind.value
                last_error = decision.message

            if decision.retryable:
                _wait_with_abort(
                    retry_delay_seconds(decision, attempt),
                    abort_event=abort_event,
                    should_cancel=should_cancel,
                )
                continue

            if allow_cf_bypass and decision.needs_cf_bypass and not cf_tried:
                cf_tried = True
                logger.info("cf_bypass attempt for %s shield=%s", record.paper_id, decision.kind.value)
                # Prefer page URL (html) for dual-path classification; fall back to pdf URL.
                target = record.html_url or plan.pdf_url
                bypass: CfBypassResult = solve_cloudflare_for_url_sync(
                    target,
                    proxy=proxy,
                    path="auto",
                    require_clearance=True,
                    headless=True,
                    verify=True,
                )
                record.notes = list(record.notes) + [
                    f"cf_bypass path={bypass.path} ok={bypass.ok} backend={bypass.backend} ms={bypass.ms:.0f}"
                ]
                if not bypass.ok:
                    record.download_status = "blocked"
                    record.download_error = bypass.error or decision.message
                    record.shield_status = decision.kind.value
                    return record
                if bypass.user_agent:
                    headers["User-Agent"] = bypass.user_agent
                if bypass.cookie_header:
                    # httpx cookies from header string
                    for part in bypass.cookie_header.split(";"):
                        part = part.strip()
                        if "=" in part:
                            k, v = part.split("=", 1)
                            session_cookies[k.strip()] = v.strip()
                # retry download with cookies
                continue

            record.download_status = "blocked" if decision.blocked else "failed"
            record.download_error = last_error
            return record
        except (DownloadCancelled, DownloadDiskGuardStopped):
            raise
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("pdf download attempt=%s paper=%s err=%s", attempt, record.paper_id, exc)
            _wait_with_abort(
                min(30.0, 1.5 * (2 ** (attempt - 1))),
                abort_event=abort_event,
                should_cancel=should_cancel,
            )

    record.download_status = "failed"
    record.download_error = last_error or "unknown download failure"
    return record
