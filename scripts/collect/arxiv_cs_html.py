"""Harvest arXiv's native HTML for every CS paper that has one.

    python scripts/collect/arxiv_cs_html.py enumerate           # OAI-PMH cs set -> SQLite (resumable, incremental)
    python scripts/collect/arxiv_cs_html.py download --concurrency 6 --proxy-pool-file POOL
    python scripts/collect/arxiv_cs_html.py download --scope delta --concurrency 3   # 2026+ ids, no pool
    python scripts/collect/arxiv_cs_html.py status

Metadata goes through the single local first hop (sanctioned OAI-PMH channel);
``enumerate`` continues from the stored watermark (--from-date is only the start
of a first walk). HTML goes through the credentialed two-hop pool, the pool file
read at runtime only so no credential reaches logs or the database; without a
pool file it goes through the first hop alone. Output root defaults to
data/arxiv/cs_html (gitignored): arxiv_cs.sqlite3 + html/<yymm>/<id>.html.gz.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.download_control import ChainedProxyPool, require_explicit_proxy, safe_proxy_label  # noqa: E402
from pipeline.paper_crawler.arxiv_batches import BatchPusher  # noqa: E402
from pipeline.paper_crawler.arxiv_corpus import DELTA_KEY, slice_sql  # noqa: E402
from pipeline.paper_crawler.arxiv_html import (  # noqa: E402
    HTML_TIMEOUT_S,
    NO_HTML_RETRY_DAYS,
    HarvestStore,
    download_pending,
    harvest_oai,
    iter_status_lines,
)

PROBE_PDF = "https://export.arxiv.org/pdf/2609.08019"


def _log_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return log


def _scope(slices: list[int]) -> tuple[str, list[object]] | None:
    """OR of the given month slices' WHERE fragments; None = every slice."""
    if not slices:
        return None
    parts = [slice_sql(key) for key in slices]
    return "(" + " OR ".join(sql for sql, _ in parts) + ")", [p for _, params in parts for p in params]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["enumerate", "download", "status"])
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "arxiv" / "cs_html")
    parser.add_argument("--first-hop-proxy", default="socks5://127.0.0.1:10808")
    parser.add_argument("--proxy-pool-file", type=Path, default=None,
                        help="download: credentialed exit list; empty or missing = first hop only")
    parser.add_argument("--from-date", default="2023-12-01",
                        help="enumerate: where a first walk starts; later walks continue from the watermark")
    parser.add_argument("--max-pages", type=int, default=0, help="enumerate: stop after N pages (0 = all)")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="download: at most N papers (0 = all pending)")
    parser.add_argument("--max-attempts", type=int, default=3, help="download: retries per paper across runs")
    parser.add_argument("--push-to", default="",
                        help="download: hand each finished batch to the ingest host, "
                             "<ssh target>:<its harvest root>/inbox (e.g. 4090-88:/srv/.../inbox)")
    parser.add_argument("--pdf-fallback", action="store_true",
                        help="download: fetch the PDF of papers without HTML (pdf/<yymm>/<id>.pdf)")
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument("--slice", type=int, nargs="*", default=[],
                             help="download: only these month slices (YYYYMM, 202311 = legacy ids); none = all")
    scope_group.add_argument("--scope", choices=["delta"],
                             help="download: delta = every 2026+ paper (the daily update)")
    parser.add_argument("--html-timeout", type=float, default=HTML_TIMEOUT_S,
                        help="download: total seconds for one HTML request before it is retried")
    parser.add_argument("--max-exits", type=int, default=12,
                        help="download: tries per request; without a proxy pool every try reuses the "
                             "one route (each at most --html-timeout plus a cooldown)")
    parser.add_argument("--no-html-retry-days", type=int, default=NO_HTML_RETRY_DAYS,
                        help="download: retry no_html papers whose content version is this recent")
    parser.add_argument("--follow", action="store_true",
                        help="download: keep picking up rows while enumerate is still running")
    parser.add_argument("--probe-healthy", type=int, default=12)
    parser.add_argument("--probe-candidates", type=int, default=96)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 6:
        parser.error("--concurrency must be between 1 and 6")
    if args.max_exits < 1:
        parser.error("--max-exits must be at least 1")

    store = HarvestStore(args.root / "arxiv_cs.sqlite3")
    log = _log_to(args.root / "logs" / f"{args.command}.log")
    first_hop = require_explicit_proxy(args.first_hop_proxy)
    try:
        if args.command == "status":
            for line in iter_status_lines(store.summary(no_html_retry_days=args.no_html_retry_days)):
                print(line)
            return 0
        if args.command == "enumerate":
            log(f"enumerate start: set=cs via {safe_proxy_label(first_hop)}")
            walk = asyncio.run(harvest_oai(
                store, first_hop_proxy=first_hop, from_date=args.from_date, log=log,
                max_pages=args.max_pages or None,
            ))
            log(f"enumerate stop after {walk['pages']} page(s), {walk['records']:,} record(s)")
        else:
            use_exits = bool(args.proxy_pool_file) and args.proxy_pool_file.is_file()
            if use_exits:
                pool = ChainedProxyPool.from_file(args.proxy_pool_file, first_hop_proxy=first_hop)
            else:
                log(f"no proxy pool file ({args.proxy_pool_file}): downloading through "
                    f"{safe_proxy_label(first_hop)} alone, pool probe skipped")
                pool = ChainedProxyPool(first_hop, [])
            pusher = None
            if args.push_to:
                target, _, inbox = args.push_to.partition(":")
                pusher = BatchPusher(args.root / "arxiv_cs.sqlite3", args.root, ssh_target=target,
                                     remote_inbox=inbox, scope=_scope(args.slice),
                                     max_attempts=args.max_attempts, log=log)

            scope = slice_sql(DELTA_KEY) if args.scope == "delta" else _scope(args.slice)

            async def run() -> dict[str, int]:
                if use_exits:
                    report = await pool.probe_pdf(
                        PROBE_PDF, required_healthy=args.probe_healthy,
                        max_candidates=args.probe_candidates, concurrency=12,
                    )
                    log(f"pool probe: configured={report.configured_count} attempted={report.attempted_count} "
                        f"healthy={report.healthy_count} in {report.elapsed_seconds:.1f}s")
                    if report.healthy_count == 0:
                        raise RuntimeError("no healthy two-hop exit reached arXiv")
                return await download_pending(
                    store, pool, args.root, concurrency=args.concurrency,
                    max_attempts=args.max_attempts, limit=args.limit or None, log=log,
                    follow=args.follow,
                    max_exits=args.max_exits,
                    scope=scope, probe_url=PROBE_PDF if use_exits else None,
                    pdf_fallback=args.pdf_fallback, pusher=pusher,
                    no_html_retry_days=args.no_html_retry_days, html_timeout_s=args.html_timeout,
                )
            counts = asyncio.run(run())
            log(f"download stop: {counts}")
            if args.slice:
                # Exit non-zero while the slice still has retryable rows, so a
                # pipeline stage running this reports the gap instead of passing.
                left = store.pending(max_attempts=args.max_attempts, scope=_scope(args.slice),
                                     no_html_retry_days=args.no_html_retry_days)
                if args.pdf_fallback:
                    left += store.pdf_pending(max_attempts=args.max_attempts, scope=_scope(args.slice))
                if left:
                    log(f"slices {args.slice}: {len(left):,} papers still pending or retryable")
                    return 2
        for line in iter_status_lines(store.summary(no_html_retry_days=args.no_html_retry_days)):
            log(line)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
