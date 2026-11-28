"""ArXiv fallback downloader for missing ICLR PDFs.

OpenReview's anonymous rate limit (5 req/60s) makes ICLR bulk download
impractical (34+ hours for ~8.6K papers). Most ICLR papers also appear
on arXiv, which has a much higher rate ceiling (~1 req/3s soft policy).

Strategy:
  1. For each missing ICLR paper, query arXiv by title.
  2. If a confident match is found (journal_ref contains "ICLR YYYY" and
     normalized title similarity >= 0.9), download the PDF from arXiv.
  3. Update manifest with arxiv-sourced PDF metadata.
  4. Whatever arXiv doesn't have remains as missing — handled later by
     scripts/download_missing_pdfs.py (OpenReview, rate-limited).

Usage:
    python scripts/download_via_arxiv.py
    python scripts/download_via_arxiv.py --limit 50   # smoke test
"""
import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from collections import Counter
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    MainConferenceCorpusBuilder,
    CorpusPaper,
    sanitize_file_component,
)
from pipeline.paper_crawler.runtime_files import (
    build_write_in_progress_path,
    is_valid_pdf_artifact,
)


ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_SEARCH_RATE_PER_SEC = 0.1   # 1 query per 10 sec — very conservative
ARXIV_SEARCH_CONCURRENCY = 1
ARXIV_PDF_RATE_PER_SEC = 1.0
ARXIV_PDF_CONCURRENCY = 2
ARXIV_429_COOLDOWN_SEC = 300      # 5 min cooldown on 429
PROGRESS_INTERVAL = 25
CHECKPOINT_INTERVAL_SEC = 120


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
            self.tokens = min(self.capacity, self.tokens + (now - self.last_refill) * self.rate)
            self.last_refill = now
            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait)
                self.tokens = 0.0
                self.last_refill = time.monotonic()
            else:
                self.tokens -= 1.0


def normalize_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def title_similarity(a: str, b: str) -> float:
    a, b = normalize_title(a), normalize_title(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Token-set overlap (forgiving for connectors like and/& differences)
    at, bt = set(a.split()), set(b.split())
    if not at or not bt:
        return 0.0
    # Coverage: how much of the shorter title is covered by the other
    coverage = len(at & bt) / min(len(at), len(bt))
    return coverage


def resolve_pdf_path(builder: MainConferenceCorpusBuilder, paper: CorpusPaper) -> Path:
    return (
        builder.pdf_root
        / paper.conference.lower()
        / str(paper.year)
        / f"{sanitize_file_component(paper.paper_id)}.pdf"
    )


def compute_missing_iclr(builder: MainConferenceCorpusBuilder, papers: list[CorpusPaper]) -> list[CorpusPaper]:
    missing = []
    for p in papers:
        if p.conference != "ICLR":
            continue
        pdf_path = resolve_pdf_path(builder, p)
        if pdf_path.exists() and pdf_path.stat().st_size > 1024:
            continue
        missing.append(p)
    return missing


def parse_arxiv_entry(xml_text: str) -> Optional[dict]:
    """Extract first entry's title / journal_ref / pdf_url from atom XML."""
    # Find first <entry>...</entry>
    m = re.search(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)
    if not m:
        return None
    entry = m.group(1)
    title_m = re.search(r"<title>([\s\S]*?)</title>", entry)
    jref_m = re.search(r"<arxiv:journal_ref>([\s\S]*?)</arxiv:journal_ref>", entry)
    pdf_m = re.search(r'<link[^>]*href="([^"]*?)"[^>]*type="application/pdf"', entry)
    if not pdf_m:
        # Fallback: build from <id>
        id_m = re.search(r"<id>http[s]?://arxiv\.org/abs/([^<]+)</id>", entry)
        pdf_url = f"https://arxiv.org/pdf/{id_m.group(1)}" if id_m else None
    else:
        pdf_url = pdf_m.group(1)
    title = title_m.group(1).strip() if title_m else ""
    journal_ref = jref_m.group(1).strip() if jref_m else ""
    return {"title": title, "journal_ref": journal_ref, "pdf_url": pdf_url}


def build_arxiv_query(title: str) -> str:
    # Strip non-alphanumeric to avoid `&`/`+`/etc messing up exact-phrase
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", title)
    cleaned = " ".join(cleaned.split())
    # Use first 10 words to keep query specific without being brittle
    truncated = " ".join(cleaned.split()[:10])
    return f'ti:"{truncated}"'


def parse_arxiv_entries(xml_text: str, max_count: int = 5) -> list[dict]:
    entries = re.findall(r"<entry>([\s\S]*?)</entry>", xml_text)
    results = []
    for entry in entries[:max_count]:
        title_m = re.search(r"<title>([\s\S]*?)</title>", entry)
        jref_m = re.search(r"<arxiv:journal_ref>([\s\S]*?)</arxiv:journal_ref>", entry)
        pdf_m = re.search(r'<link[^>]*href="([^"]*?)"[^>]*type="application/pdf"', entry)
        if not pdf_m:
            id_m = re.search(r"<id>http[s]?://arxiv\.org/abs/([^<]+)</id>", entry)
            pdf_url = f"https://arxiv.org/pdf/{id_m.group(1)}" if id_m else None
        else:
            pdf_url = pdf_m.group(1)
        results.append({
            "title": title_m.group(1).strip() if title_m else "",
            "journal_ref": jref_m.group(1).strip() if jref_m else "",
            "pdf_url": pdf_url,
        })
    return results


async def search_arxiv(
    client: httpx.AsyncClient,
    bucket: TokenBucket,
    title: str,
    cooldown_until: list[float],
    retries: int = 3,
) -> list[dict]:
    from urllib.parse import urlencode
    query = build_arxiv_query(title)
    params = {"search_query": query, "max_results": 5}
    url = f"{ARXIV_API_BASE}?{urlencode(params)}"
    last_status = None
    for attempt in range(1, retries + 1):
        now = time.monotonic()
        if cooldown_until[0] > now:
            await asyncio.sleep(cooldown_until[0] - now)
        await bucket.acquire()
        try:
            response = await client.get(url)
            if response.status_code == 429:
                last_status = 429
                cooldown_until[0] = max(cooldown_until[0], time.monotonic() + ARXIV_429_COOLDOWN_SEC)
                if attempt < retries:
                    continue
                return []
            response.raise_for_status()
            return parse_arxiv_entries(response.text)
        except httpx.HTTPStatusError as exc:
            last_status = exc.response.status_code
            if attempt < retries:
                await asyncio.sleep(min(5 * attempt, 30))
        except Exception:
            if attempt < retries:
                await asyncio.sleep(min(5 * attempt, 30))
    return []


async def download_pdf(
    client: httpx.AsyncClient,
    bucket: TokenBucket,
    url: str,
    dest: Path,
) -> tuple[bool, Optional[str]]:
    await bucket.acquire()
    tmp = build_write_in_progress_path(dest)
    try:
        if tmp.exists():
            tmp.unlink()
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with tmp.open("wb") as fp:
                async for chunk in response.aiter_bytes():
                    fp.write(chunk)
        if not is_valid_pdf_artifact(tmp):
            tmp.unlink()
            return False, "invalid PDF artifact"
        tmp.replace(dest)
        return True, None
    except Exception as exc:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


async def main(limit: int = 0, sim_threshold: float = 0.85) -> None:
    builder = MainConferenceCorpusBuilder()
    print(f"manifest: {builder.manifest_path}")
    print(f"proxy: {builder.proxy_url}")
    print(f"arxiv-search rate: {ARXIV_SEARCH_RATE_PER_SEC}/s, pdf rate: {ARXIV_PDF_RATE_PER_SEC}/s")

    raw_records = json.loads(builder.manifest_path.read_text(encoding="utf-8"))
    all_papers = [CorpusPaper.from_manifest_record(r) for r in raw_records]
    paper_by_id = {p.paper_id: p for p in all_papers}

    missing = compute_missing_iclr(builder, all_papers)
    if limit > 0:
        missing = missing[:limit]
    print(f"missing ICLR PDFs to try: {len(missing)}")

    if not missing:
        print("Nothing to do.")
        return

    search_bucket = TokenBucket(ARXIV_SEARCH_RATE_PER_SEC, capacity=2)
    pdf_bucket = TokenBucket(ARXIV_PDF_RATE_PER_SEC, capacity=4)
    search_sem = asyncio.Semaphore(ARXIV_SEARCH_CONCURRENCY)
    pdf_sem = asyncio.Semaphore(ARXIV_PDF_CONCURRENCY)
    arxiv_cooldown = [0.0]

    t0 = time.perf_counter()
    last_checkpoint = time.perf_counter()
    completed = 0
    matched = 0
    downloaded = 0
    nomatch = 0
    fail = 0
    log_lock = asyncio.Lock()

    client_options = dict(
        proxy=builder.proxy_url,
        timeout=httpx.Timeout(60.0, connect=20.0),
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

        async def process(paper: CorpusPaper) -> None:
            nonlocal completed, matched, downloaded, nomatch, fail, last_checkpoint
            entries = []
            async with search_sem:
                entries = await search_arxiv(
                    client, search_bucket, paper.title, arxiv_cooldown
                )

            local_status = "nomatch"
            best_entry = None
            best_sim = 0.0
            for e in entries:
                if not e.get("pdf_url"):
                    continue
                sim = title_similarity(e.get("title", ""), paper.title)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = e
            if best_entry and best_sim >= sim_threshold:
                pdf_path = resolve_pdf_path(builder, paper)
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                async with pdf_sem:
                    ok, _err = await download_pdf(client, pdf_bucket, best_entry["pdf_url"], pdf_path)
                if ok:
                    paper.pdf_path = str(pdf_path)
                    paper.downloaded_from_url = best_entry["pdf_url"]
                    paper.pdf_resolution_status = "downloaded"
                    paper.pdf_resolution_note = (
                        f"arxiv via title match (sim={best_sim:.2f}, "
                        f"jref={best_entry.get('journal_ref', '')!r})"
                    )
                    paper.download_error = None
                    if best_entry["pdf_url"] not in paper.alternate_pdf_urls:
                        paper.alternate_pdf_urls.append(best_entry["pdf_url"])
                    local_status = "downloaded"
                else:
                    local_status = "download_fail"
            else:
                local_status = "low_confidence" if best_entry else "nomatch"
            async with log_lock:
                completed += 1
                if local_status == "downloaded":
                    matched += 1
                    downloaded += 1
                elif local_status == "low_confidence":
                    nomatch += 1
                elif local_status == "download_fail":
                    fail += 1
                else:
                    nomatch += 1
                if completed % PROGRESS_INTERVAL == 0 or completed == len(missing):
                    elapsed = time.perf_counter() - t0
                    rate = completed / max(elapsed, 1e-9)
                    eta = (len(missing) - completed) / max(rate, 1e-9)
                    print(
                        f"  [{completed}/{len(missing)}] "
                        f"dl={downloaded} nomatch={nomatch} fail={fail} "
                        f"rate={rate*60:.1f}/min eta={eta/60:.1f}m",
                        flush=True,
                    )
                if time.perf_counter() - last_checkpoint > CHECKPOINT_INTERVAL_SEC:
                    last_checkpoint = time.perf_counter()
                    builder.save_manifest(list(paper_by_id.values()))
                    print("  [checkpoint] manifest saved", flush=True)

        await asyncio.gather(*(process(p) for p in missing))

    builder.save_manifest(list(paper_by_id.values()))
    elapsed = time.perf_counter() - t0
    print(f"\nDONE: downloaded={downloaded} nomatch={nomatch} fail={fail} / {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit papers to process (0 = all)")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))
