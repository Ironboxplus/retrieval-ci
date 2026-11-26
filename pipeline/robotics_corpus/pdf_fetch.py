"""PDF planner + CF-aware downloader for robotics proceedings."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from pipeline.http_retry import exponential_backoff_seconds
from pipeline.medical_corpus.cf_bypass import CfBypassResult, solve_cloudflare_for_url_sync
from pipeline.medical_corpus.shield import classify_http_shield, retry_delay_seconds
from pipeline.paper_crawler.runtime_files import build_write_in_progress_path, is_valid_pdf_artifact
from pipeline.robotics_corpus.models import RoboticsPaperRecord, sanitize_id_component
from pipeline.robotics_corpus.proxy import RoboticsProxyConfig, load_robotics_proxy_from_settings

logger = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass(slots=True)
class PdfFetchPlan:
    paper_id: str
    pdf_url: str
    dest_path: Path
    action: str  # skip_exists | would_download | no_url
    note: str = ""
    candidate_urls: list[str] | None = None


def planned_pdf_path(pdf_root: Path, record: RoboticsPaperRecord) -> Path:
    venue = sanitize_id_component(record.venue.value.lower())
    year = str(record.year)
    stem = sanitize_id_component(record.paper_id)
    return pdf_root / venue / year / f"{stem}.pdf"


def candidate_pdf_urls(record: RoboticsPaperRecord) -> list[str]:
    urls: list[str] = []
    for u in [record.pdf_url, *list(record.alternate_pdf_urls or [])]:
        if not u:
            continue
        u = str(u).strip()
        if u and u not in urls:
            urls.append(u)
    return urls


def plan_pdf_fetch(
    record: RoboticsPaperRecord,
    *,
    pdf_root: Path,
    skip_existing: bool = True,
) -> PdfFetchPlan:
    dest = planned_pdf_path(pdf_root, record)
    urls = candidate_pdf_urls(record)
    if not urls:
        return PdfFetchPlan(record.paper_id, "", dest, "no_url", "record has no pdf candidate", [])
    if skip_existing and is_valid_pdf_artifact(dest):
        return PdfFetchPlan(
            record.paper_id, urls[0], dest, "skip_exists", "valid PDF already on disk", urls,
        )
    return PdfFetchPlan(
        record.paper_id, urls[0], dest, "would_download", "dry-run plan only", urls,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_pdf(
    record: RoboticsPaperRecord,
    *,
    pdf_root: Path,
    proxy: RoboticsProxyConfig | None = None,
    execute: bool = False,
    allow_cf_bypass: bool = True,
    max_retries: int = 3,
    timeout_s: float = 120.0,
    min_bytes: int = 1024,
) -> RoboticsPaperRecord:
    """Download one PDF when ``execute=True``; otherwise only plan + annotate.

    Tries primary + alternate URLs. On CF/403/418/HTML interstitial, optionally
    invokes TurnstileBypass (minimal headless) once per paper, then retries.
    """
    proxy = proxy or load_robotics_proxy_from_settings()
    plan = plan_pdf_fetch(record, pdf_root=pdf_root)
    record = RoboticsPaperRecord.from_dict(record.to_dict())

    if plan.action == "no_url":
        record.download_status = "no_pdf"
        record.download_error = plan.note
        return record
    if plan.action == "skip_exists":
        record.download_status = "skipped"
        record.pdf_path = str(plan.dest_path)
        record.notes = list(record.notes) + [plan.note]
        return record
    if not execute:
        record.download_status = "planned"
        record.notes = list(record.notes) + [
            f"dry-run: would GET {len(plan.candidate_urls or [])} candidate(s) -> {plan.dest_path}"
        ]
        return record

    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "application/pdf,*/*;q=0.8",
    }
    session_cookies: dict[str, str] = {}
    last_error: Optional[str] = None
    cf_tried = False
    urls = plan.candidate_urls or [plan.pdf_url]

    for candidate_url in urls:
        for attempt in range(1, max_retries + 1):
            try:
                with httpx.Client(
                    proxy=proxy.as_httpx_proxy(),
                    timeout=timeout_s,
                    follow_redirects=True,
                    headers=headers,
                    cookies=session_cookies or None,
                ) as client:
                    resp = client.get(candidate_url)

                decision = classify_http_shield(
                    status_code=resp.status_code,
                    content_type=resp.headers.get("content-type"),
                    body=resp.content[:12000],
                    expect_pdf=True,
                    headers=resp.headers,
                )
                # IEEE often returns 418 for bot walls
                if resp.status_code == 418:
                    decision = classify_http_shield(
                        status_code=403,
                        content_type=resp.headers.get("content-type"),
                        body=resp.content[:12000],
                        expect_pdf=True,
                        headers=resp.headers,
                    )

                if decision.kind.value == "none" and resp.content.startswith(b"%PDF"):
                    if len(resp.content) < min_bytes:
                        last_error = f"PDF too small ({len(resp.content)} bytes)"
                        continue
                    dest = plan.dest_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = build_write_in_progress_path(dest)
                    tmp.write_bytes(resp.content)
                    tmp.replace(dest)
                    record.pdf_path = str(dest)
                    record.pdf_bytes = len(resp.content)
                    record.pdf_sha256 = _sha256(resp.content)
                    record.pdf_url = candidate_url
                    record.download_status = "downloaded"
                    record.download_error = None
                    record.shield_status = "ok"
                    return record

                record.shield_status = decision.kind.value
                last_error = f"{candidate_url}: {decision.message}"

                if decision.retryable:
                    time.sleep(retry_delay_seconds(decision, attempt))
                    continue

                if allow_cf_bypass and decision.needs_cf_bypass and not cf_tried:
                    cf_tried = True
                    logger.info(
                        "cf_bypass attempt for %s url=%s shield=%s",
                        record.paper_id, candidate_url, decision.kind.value,
                    )
                    target = record.paper_page_url or candidate_url
                    bypass: CfBypassResult = solve_cloudflare_for_url_sync(
                        target,
                        proxy=proxy.as_httpx_proxy(),
                        path="auto",
                        require_clearance=True,
                        headless=True,
                        verify=True,
                    )
                    record.notes = list(record.notes) + [
                        f"cf_bypass path={bypass.path} ok={bypass.ok} "
                        f"backend={bypass.backend} ms={bypass.ms:.0f}"
                    ]
                    if bypass.ok:
                        if bypass.user_agent:
                            headers["User-Agent"] = bypass.user_agent
                        if bypass.cookie_header:
                            for part in bypass.cookie_header.split(";"):
                                part = part.strip()
                                if "=" in part:
                                    k, v = part.split("=", 1)
                                    session_cookies[k.strip()] = v.strip()
                        continue
                    # CF failed — try next candidate URL
                    break

                # non-retryable without CF success — try next candidate
                break
            except Exception as exc:
                last_error = f"{candidate_url}: {type(exc).__name__}: {exc}"
                logger.warning(
                    "pdf download attempt=%s paper=%s err=%s",
                    attempt, record.paper_id, exc,
                )
                # Only the wait curve is shared: the loop rotates candidate URLs and
                # swaps headers and cookies after a Cloudflare solve.
                time.sleep(exponential_backoff_seconds(attempt))

    record.download_status = "blocked" if cf_tried else "failed"
    record.download_error = last_error or "unknown download failure"
    return record
