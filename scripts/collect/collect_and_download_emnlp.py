"""Collect EMNLP papers and merge into existing manifest, then download PDFs.

Usage:
    python scripts/collect_and_download_emnlp.py --stage collect   # collect only
    python scripts/collect_and_download_emnlp.py --stage download  # download only (requires prior collect)
    python scripts/collect_and_download_emnlp.py --stage all       # collect + download
    python scripts/collect_and_download_emnlp.py --stage all --limit 5  # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from urllib.parse import quote, urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx

from pipeline.download_control import ChainedProxyPool
from pipeline.medical_corpus.cf_bypass import solve_cloudflare_for_url
from pipeline.paper_crawler.ccf_a_main_corpus import (
    EmnlpMainConferenceSource,
    OfficialProceedingsClient,
    CorpusPaper,
    build_default_corpus_root,
    configure_logging,
    sanitize_file_component,
    is_valid_pdf_artifact,
    build_write_in_progress_path,
)
from pipeline.paper_crawler.runtime_files import write_json_atomic

LOGGER = logging.getLogger(__name__)

DEFAULT_PROXY = "http://127.0.0.1:10808"
YEARS = (2022, 2023, 2024, 2025)
EMNLP_2026_OPENREVIEW_INVITATION = "EMNLP/2026/Conference/-/Submission"
EMNLP_2026_OPENREVIEW_VENUE = "EMNLP 2026 MainConference"
EMNLP_2026_OPENREVIEW_API = "https://api2.openreview.net/notes"
TWO_HOP_PDF_PROBE = "https://aclanthology.org/2025.emnlp-main.1.pdf"


async def collect_emnlp_2026_openreview_notes(
    *,
    chained_proxy_pool: ChainedProxyPool,
    xray_executable: Path,
    challenge_retries: int = 12,
) -> list[dict]:
    """Read public EMNLP accepted-main notes in the solved browser session.

    OpenReview protects the notes API with a browser challenge and binds the
    clearance to the egress identity.  The API fetch therefore stays inside the
    Camoufox session reached through the same first-hop -> landing-exit chain.
    """
    required_healthy = min(
        chained_proxy_pool.configured_count,
        max(1, min(6, int(challenge_retries))),
    )
    probe = await chained_proxy_pool.probe_pdf(
        TWO_HOP_PDF_PROBE,
        required_healthy=required_healthy,
        max_candidates=min(96, chained_proxy_pool.configured_count),
        concurrency=min(6, chained_proxy_pool.configured_count),
        timeout_seconds=8.0,
    )
    if probe.healthy_count < required_healthy:
        raise RuntimeError(
            f"only {probe.healthy_count}/{required_healthy} landing exits passed "
            "the real two-hop PDF probe; refusing direct or first-hop-only access"
        )

    base_query = {
        "invitation": EMNLP_2026_OPENREVIEW_INVITATION,
        "content.venue": EMNLP_2026_OPENREVIEW_VENUE,
        "limit": "1000",
    }
    excluded_ordinals: set[int] = set()

    async def _read_page(offset: int) -> dict:
        query = {**base_query, "offset": str(offset)}
        api_url = f"{EMNLP_2026_OPENREVIEW_API}?{urlencode(query)}"
        challenge_url = (
            "https://openreview.net/challenge?redirect="
            + quote(api_url, safe="")
        )

        for attempt in range(1, max(1, int(challenge_retries)) + 1):
            payload_holder: dict[str, dict] = {}

            async def _read_redirected_response(page, _initial_download=None):
                # OpenReview clearance is request-scoped.  The redirect target
                # already contains the authorised JSON; issuing another fetch
                # from the solved page immediately receives a fresh 403.
                raw = await asyncio.wait_for(
                    page.evaluate(
                        "document.documentElement.innerText || "
                        "document.body?.textContent || ''"
                    ),
                    timeout=30,
                )
                if not str(raw or "").strip():
                    page_url = str(getattr(page, "url", "") or "")
                    html = await page.content()
                    raise RuntimeError(
                        f"OpenReview redirected response was empty at {page_url}; "
                        f"html={html[:240]!r}"
                    )
                try:
                    payload = json.loads(raw or "{}")
                except json.JSONDecodeError as exc:
                    page_url = str(getattr(page, "url", "") or "")
                    raise RuntimeError(
                        f"OpenReview redirected response was not JSON at {page_url}; "
                        f"text={str(raw)[:240]!r}"
                    ) from exc
                if not isinstance(payload.get("notes"), list):
                    raise RuntimeError("OpenReview redirected response has no notes list")
                payload_holder["value"] = payload
                return {
                    "notes": len(payload["notes"]),
                    "expected": int(payload.get("count", len(payload["notes"]))),
                }

            lease = None
            try:
                async with chained_proxy_pool.open_browser_proxy(
                    xray_executable=xray_executable,
                    excluded_ordinals=excluded_ordinals,
                ) as active_lease:
                    lease = active_lease
                    result = await solve_cloudflare_for_url(
                        challenge_url,
                        proxy=active_lease.proxy_url,
                        path="managed",
                        require_clearance=False,
                        headless=False,
                        verify=True,
                        timeout_s=60,
                        overall_timeout_s=240,
                        backend="camoufox",
                        verified_session_callback=_read_redirected_response,
                    )
                    payload = payload_holder.get("value")
                    if not result.ok or not result.verified or payload is None:
                        raise RuntimeError(
                            result.error or "OpenReview solved page returned no accepted notes"
                        )
                    return payload
            except Exception as exc:
                if lease is not None:
                    excluded_ordinals.add(lease.exit_ordinal)
                LOGGER.warning(
                    "EMNLP OpenReview offset %d attempt %d failed on landing "
                    "exit #%s: %s: %s",
                    offset,
                    attempt,
                    lease.exit_ordinal if lease is not None else "unavailable",
                    type(exc).__name__,
                    exc,
                )

        raise RuntimeError(
            f"EMNLP OpenReview offset {offset} produced no usable page after "
            f"{max(1, int(challenge_retries))} attempt(s)"
        )

    notes: list[dict] = []
    expected_count: int | None = None
    while expected_count is None or len(notes) < expected_count:
        payload = await _read_page(len(notes))
        batch = payload["notes"]
        expected_count = int(payload.get("count", len(batch)))
        if not batch:
            break
        notes.extend(batch)
    if expected_count is None or len(notes) != expected_count:
        raise RuntimeError(
            f"OpenReview catalog incomplete: received {len(notes)} of {expected_count} notes"
        )
    return notes


def load_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def backup_manifest(manifest_path: Path) -> Path:
    import time
    bak = manifest_path.with_suffix(f".json.bak.emnlp-{int(time.time())}")
    import shutil
    shutil.copy2(manifest_path, bak)
    LOGGER.info("Backed up manifest to %s (%d bytes)", bak.name, bak.stat().st_size)
    return bak


def collect_emnlp(proxy_url: str, limit: int | None = None) -> list[CorpusPaper]:
    source = EmnlpMainConferenceSource()
    client = OfficialProceedingsClient(proxy_url=proxy_url)
    all_papers: list[CorpusPaper] = []
    try:
        for year in YEARS:
            LOGGER.info("Collecting EMNLP %d...", year)
            papers = source.collect_papers(year, client)
            papers.sort(key=lambda p: p.paper_id)
            if limit:
                papers = papers[:limit]
            all_papers.extend(papers)
            LOGGER.info("  %d papers", len(papers))
    finally:
        client.close()
    LOGGER.info("Total EMNLP collected: %d", len(all_papers))
    return all_papers


def merge_into_manifest(manifest_path: Path, new_papers: list[CorpusPaper]) -> int:
    existing = load_manifest(manifest_path)
    existing_ids = {r["paper_id"] for r in existing}
    added = 0
    for paper in new_papers:
        if paper.paper_id not in existing_ids:
            existing.append(paper.to_manifest_record())
            existing_ids.add(paper.paper_id)
            added += 1
    write_json_atomic(manifest_path, existing)
    LOGGER.info("Merged %d new EMNLP papers (total manifest: %d)", added, len(existing))
    return added


async def _download_emnlp_pdfs(
    items: list[tuple[CorpusPaper, Path]],
    proxy_url: str,
    concurrency: int,
) -> list[tuple[CorpusPaper, Path]]:
    """Download PDFs from ACL Anthology via httpx (no manifest checkpointing)."""
    semaphore = asyncio.Semaphore(concurrency)
    client_opts = {
        "proxy": proxy_url,
        "timeout": 120,
        "follow_redirects": True,
        "headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    }

    async def download_one(paper: CorpusPaper, pdf_path: Path):
        async with semaphore:
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = build_write_in_progress_path(pdf_path)
            for attempt in range(1, 4):
                try:
                    async with httpx.AsyncClient(**client_opts) as client:
                        async with client.stream("GET", paper.pdf_url) as resp:
                            resp.raise_for_status()
                            with tmp.open("wb") as f:
                                async for chunk in resp.aiter_bytes():
                                    f.write(chunk)
                    if is_valid_pdf_artifact(tmp):
                        tmp.replace(pdf_path)
                        paper.pdf_path = str(pdf_path)
                        paper.downloaded_from_url = paper.pdf_url
                        paper.pdf_resolution_status = "downloaded"
                        paper.download_error = None
                        return
                    else:
                        raise ValueError("PDF validation failed")
                except Exception as e:
                    if tmp.exists():
                        tmp.unlink()
                    if attempt == 3:
                        paper.download_error = str(e)
                        LOGGER.warning("Failed %s: %s", paper.paper_id, e)
                    else:
                        await asyncio.sleep(2 * attempt)

    tasks = [download_one(paper, pdf_path) for paper, pdf_path in items]
    await asyncio.gather(*tasks)
    return items


def main():
    parser = argparse.ArgumentParser(description="Collect + download EMNLP papers (merge-safe)")
    parser.add_argument("--stage", choices=("collect", "download", "all"), default="all")
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    parser.add_argument("--limit", type=int, default=None, help="Limit per year (smoke test)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    corpus_root = build_default_corpus_root()
    manifest_path = corpus_root / "manifests" / "paper_manifest.json"

    if args.stage in ("collect", "all"):
        backup_manifest(manifest_path)
        emnlp_papers = collect_emnlp(args.proxy, limit=args.limit)
        added = merge_into_manifest(manifest_path, emnlp_papers)
        if added == 0:
            LOGGER.info("No new papers to add")

    if args.stage in ("download", "all"):
        all_records = load_manifest(manifest_path)
        emnlp_records = [r for r in all_records if r.get("conference") == "EMNLP"]
        emnlp_papers = [CorpusPaper.from_manifest_record(r) for r in emnlp_records]

        pdf_root = corpus_root / "pdf"
        needs_download = []
        for p in emnlp_papers:
            pdf_path = pdf_root / "emnlp" / str(p.year) / f"{sanitize_file_component(p.paper_id)}.pdf"
            if is_valid_pdf_artifact(pdf_path):
                p.pdf_path = str(pdf_path)
                p.pdf_resolution_status = p.pdf_resolution_status or "downloaded"
            else:
                needs_download.append((p, pdf_path))

        LOGGER.info("EMNLP: %d need download, %d already on disk",
                     len(needs_download), len(emnlp_papers) - len(needs_download))

        if needs_download:
            downloaded = asyncio.run(_download_emnlp_pdfs(
                needs_download, args.proxy, args.concurrency,
            ))
            ok = sum(1 for p, _ in downloaded if p.pdf_path)
            fail = sum(1 for p, _ in downloaded if p.download_error)
            LOGGER.info("Download complete: %d OK, %d failed", ok, fail)

        record_map = {r["paper_id"]: r for r in all_records}
        for p in emnlp_papers:
            rec = record_map.get(p.paper_id)
            if rec:
                rec["pdf_path"] = p.pdf_path
                rec["downloaded_from_url"] = p.downloaded_from_url
                rec["pdf_resolution_status"] = p.pdf_resolution_status
                rec["download_error"] = p.download_error
        write_json_atomic(manifest_path, list(record_map.values()))
        LOGGER.info("Manifest updated (%d total records)", len(record_map))


if __name__ == "__main__":
    main()
