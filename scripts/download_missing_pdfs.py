"""Fast missing-PDF downloader with global rate limiting.

Loads full manifest, finds missing PDFs, downloads only those through
proxy 127.0.0.1:10808 with a global token bucket (2 req/s) to avoid
OpenReview 429 rate-limiting. Concurrency-friendly: workers overlap I/O
but request rate is hard-capped.

Usage:
    python scripts/download_missing_pdfs.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.http_retry import retry_after_from
from pipeline.paper_crawler.ccf_a_main_corpus import (
    MainConferenceCorpusBuilder,
    CorpusPaper,
    sanitize_file_component,
)
from pipeline.paper_crawler.runtime_files import (
    build_write_in_progress_path,
    is_valid_pdf_artifact,
)


PROGRESS_INTERVAL = 25
# OpenReview rate-limits to 5 req / 60s shared across openreview.net + api.openreview.net.
# Sustained rate = 5/60 = 0.083 req/s. Use 0.07 to stay under the limit.
RATE_PER_SEC = 0.07
BURST_CAPACITY = 3
CONCURRENCY = 2
CHECKPOINT_INTERVAL_SEC = 180


def rewrite_openreview_pdf_url(url: str, year: int) -> list[str]:
    """Rewrite openreview.net/pdf URL to api.openreview.net/attachment.

    The /pdf endpoint is rate-limited (5 req/min anon), but the API
    /attachment endpoint allows much higher throughput. Returns list of
    candidate URLs to try in order.
    """
    if "openreview.net/pdf" not in url:
        return [url]
    import re
    m = re.search(r"[?&]id=([^&]+)", url)
    if not m:
        return [url]
    forum_id = m.group(1)
    # ICLR 2022/2023 -> v1; 2024/2025 -> v2
    if year <= 2023:
        primary = f"https://api.openreview.net/attachment?id={forum_id}&name=pdf"
        fallback = f"https://api2.openreview.net/attachment?id={forum_id}&name=pdf"
    else:
        primary = f"https://api2.openreview.net/attachment?id={forum_id}&name=pdf"
        fallback = f"https://api.openreview.net/attachment?id={forum_id}&name=pdf"
    return [primary, fallback, url]


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate = rate_per_sec
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait)
                self.tokens = 0.0
                self.last_refill = time.monotonic()
            else:
                self.tokens -= 1.0


def resolve_pdf_path(builder: MainConferenceCorpusBuilder, paper: CorpusPaper) -> Path:
    return (
        builder.pdf_root
        / paper.conference.lower()
        / str(paper.year)
        / f"{sanitize_file_component(paper.paper_id)}.pdf"
    )


def compute_missing(builder: MainConferenceCorpusBuilder, papers: list[CorpusPaper]) -> list[CorpusPaper]:
    missing: list[CorpusPaper] = []
    for paper in papers:
        pdf_path = resolve_pdf_path(builder, paper)
        if pdf_path.exists() and pdf_path.stat().st_size > 1024:
            continue
        missing.append(paper)
    return missing


async def download_one(
    client: httpx.AsyncClient,
    bucket: TokenBucket,
    builder: MainConferenceCorpusBuilder,
    paper: CorpusPaper,
    cooldown_until: list[float],
    retries: int = 3,
) -> tuple[bool, Optional[str]]:
    pdf_path = resolve_pdf_path(builder, paper)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    # Disk-skip: another script (e.g. arxiv pass) may have downloaded this
    if pdf_path.exists() and pdf_path.stat().st_size > 1024:
        paper.pdf_path = str(pdf_path)
        if paper.pdf_resolution_status in (None, "download_failed"):
            paper.pdf_resolution_status = "downloaded"
            paper.pdf_resolution_note = "Already present on disk (parallel script)"
        paper.download_error = None
        return True, None
    tmp_path = build_write_in_progress_path(pdf_path)

    raw_urls = [paper.pdf_url, *paper.alternate_pdf_urls]
    raw_urls = [u.strip() for u in raw_urls if u and u.strip()]
    # Expand each url through rewrite (OpenReview /pdf -> api /attachment)
    candidate_urls: list[str] = []
    for u in raw_urls:
        for rewritten in rewrite_openreview_pdf_url(u, paper.year):
            if rewritten not in candidate_urls:
                candidate_urls.append(rewritten)

    last_error: Optional[str] = None
    for candidate_url in candidate_urls:
        for attempt in range(1, retries + 1):
            # Honor global cooldown set by recent 429 events
            now = time.monotonic()
            if cooldown_until[0] > now:
                await asyncio.sleep(cooldown_until[0] - now)
            # Acquire 1 token from the global rate limiter
            await bucket.acquire()
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
                async with client.stream("GET", candidate_url) as response:
                    response.raise_for_status()
                    with tmp_path.open("wb") as fp:
                        async for chunk in response.aiter_bytes():
                            fp.write(chunk)
                if not is_valid_pdf_artifact(tmp_path):
                    raise ValueError("invalid PDF artifact")
                tmp_path.replace(pdf_path)
                paper.pdf_path = str(pdf_path)
                paper.downloaded_from_url = candidate_url
                paper.pdf_resolution_status = "downloaded"
                paper.pdf_resolution_note = f"Resolved PDF from {candidate_url}"
                paper.download_error = None
                return True, None
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                last_error = f"HTTPStatusError {code}"
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                if code == 429:
                    wait_s = max(retry_after_from(exc.response, default=20.0), 15.0)
                    # Set GLOBAL cooldown so all workers pause
                    cooldown_until[0] = max(cooldown_until[0], time.monotonic() + wait_s)
                    if attempt < retries:
                        continue  # global cooldown will be respected at next iter top
                elif code == 404:
                    break
                else:
                    if attempt < retries:
                        await asyncio.sleep(min(2 ** attempt, 10))
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                if attempt < retries:
                    await asyncio.sleep(min(2 ** attempt, 10))
    paper.download_error = last_error or "no candidate URLs"
    paper.pdf_resolution_status = "download_failed"
    paper.pdf_resolution_note = last_error
    return False, last_error


async def main() -> None:
    builder = MainConferenceCorpusBuilder()
    print(f"manifest: {builder.manifest_path}")
    print(f"proxy: {builder.proxy_url}")
    print(f"rate-limit: {RATE_PER_SEC} req/s, burst={BURST_CAPACITY}, concurrency={CONCURRENCY}")

    t0 = time.perf_counter()
    raw_records = json.loads(builder.manifest_path.read_text(encoding="utf-8"))
    all_papers = [CorpusPaper.from_manifest_record(r) for r in raw_records]
    print(f"manifest entries: {len(all_papers)}")

    missing = compute_missing(builder, all_papers)
    print(f"missing PDFs: {len(missing)}")
    by_cy = Counter((p.conference, p.year) for p in missing)
    for k, v in sorted(by_cy.items()):
        print(f"  {k}: {v}")

    if not missing:
        print("Nothing to download.")
        return

    paper_by_id = {p.paper_id: p for p in all_papers}

    semaphore = asyncio.Semaphore(CONCURRENCY)
    bucket = TokenBucket(RATE_PER_SEC, BURST_CAPACITY)
    cooldown_until = [0.0]  # mutable holder for global cooldown timestamp

    success_count = 0
    fail_count = 0
    last_checkpoint = time.perf_counter()
    completed = 0
    log_lock = asyncio.Lock()

    client_options = dict(
        proxy=builder.proxy_url,
        timeout=httpx.Timeout(120.0, connect=30.0),
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )

    async with httpx.AsyncClient(**client_options) as client:

        async def worker(paper: CorpusPaper) -> None:
            nonlocal success_count, fail_count, completed, last_checkpoint
            async with semaphore:
                ok, _err = await download_one(client, bucket, builder, paper, cooldown_until)
            async with log_lock:
                completed += 1
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                if completed % PROGRESS_INTERVAL == 0 or completed == len(missing):
                    elapsed = time.perf_counter() - t0
                    rate = completed / max(elapsed, 1e-9)
                    eta = (len(missing) - completed) / max(rate, 1e-9)
                    cd_remain = max(0, cooldown_until[0] - time.monotonic())
                    print(
                        f"  [{completed}/{len(missing)}] "
                        f"ok={success_count} fail={fail_count} "
                        f"rate={rate*60:.1f}/min eta={eta/60:.1f}m cd={cd_remain:.0f}s",
                        flush=True,
                    )
                if time.perf_counter() - last_checkpoint > CHECKPOINT_INTERVAL_SEC:
                    last_checkpoint = time.perf_counter()
                    builder.save_manifest(list(paper_by_id.values()))
                    print(f"  [checkpoint] manifest saved", flush=True)

        await asyncio.gather(*(worker(p) for p in missing))

    builder.save_manifest(list(paper_by_id.values()))
    elapsed = time.perf_counter() - t0
    print(f"\nDONE: {success_count} ok / {fail_count} fail / {elapsed:.0f}s "
          f"({elapsed/60:.1f}m)")

    still_missing = compute_missing(builder, list(paper_by_id.values()))
    print(f"still-missing on disk: {len(still_missing)}")
    if still_missing and len(still_missing) <= 50:
        for p in still_missing[:50]:
            print(f"  - {p.paper_id}: {p.download_error}")


if __name__ == "__main__":
    asyncio.run(main())
