"""Download a slice of manifest papers (subset by conference / year).

Wraps MainConferenceCorpusBuilder.download_papers but only feeds it the
filtered subset, so we can run targeted backfills like ICLR 2025 without
re-scanning all 64K manifest entries on every retry.

Usage:
    python scripts/download_subset.py --conf ICLR --year 2025 --concurrency 8
    python scripts/download_subset.py --conf ICLR --year 2024 --concurrency 8
    python scripts/download_subset.py --conf CVPR --year 2022 --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    MainConferenceCorpusBuilder,
    CorpusPaper,
    load_json_with_recovery,
    sanitize_file_component,
)
from pipeline.download_control import (
    ChainedProxyPool,
    DEFAULT_DOWNLOAD_BYTES_PER_SECOND,
    require_explicit_proxy,
    safe_proxy_label,
)
from config import settings


def _alive(p: str | None) -> bool:
    return bool(p) and Path(p).exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="Conference name (e.g. ICLR, CVPR)")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent downloads (default: 8). OpenReview CDN tolerates 8-12 well.")
    parser.add_argument("--min-interval-seconds", type=float, default=0.0,
                        help="Minimum seconds between starting PDF GET requests. Use 10 for rate-sensitive sources.")
    parser.add_argument(
        "--max-bytes-per-second",
        type=int,
        default=DEFAULT_DOWNLOAD_BYTES_PER_SECOND,
        help="Aggregate byte-rate cap shared by every worker; 0 disables it (default: 2000000).",
    )
    parser.add_argument(
        "--proxy-pool-file",
        type=Path,
        default=None,
        help="Credentialed SOCKS exit pool. Each exit is reached through --first-hop-proxy.",
    )
    parser.add_argument(
        "--proxy-allowlist-file",
        type=Path,
        default=None,
        help="Optional host:port allowlist selecting permitted exits from the pool file.",
    )
    parser.add_argument(
        "--first-hop-proxy",
        default="socks5://127.0.0.1:10808",
        help="Mandatory first hop for chained pool mode (default: socks5://127.0.0.1:10808).",
    )
    parser.add_argument(
        "--proxy-attempts-per-url",
        type=int,
        default=24,
        help="Maximum distinct exits tried for one PDF URL (default: 24).",
    )
    parser.add_argument(
        "--proxy-probe-candidates",
        type=int,
        default=64,
        help="Maximum exits tested against a real PDF before downloading (default: 64).",
    )
    parser.add_argument(
        "--proxy-probe-timeout-seconds",
        type=float,
        default=8.0,
        help="Per-exit two-hop health-probe timeout (default: 8).",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Corpus root override. Primarily used by isolated runtime smoke tests.",
    )
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=None,
        help="PDF artifact root override; the manifest remains under --corpus-root.",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Max papers to enqueue (0 = all missing). Useful for smoke test.")
    args = parser.parse_args()
    if args.conf.upper() == "ICML" and args.year == 2026:
        args.concurrency = 1
        args.min_interval_seconds = max(args.min_interval_seconds, 10.0)

    # 强制走部署配置中的代理（无直连回退）。代理不可达时显式失败，
    # 不静默改直连（见 config.ProxySettings）。
    proxy_url = require_explicit_proxy(
        args.first_hop_proxy if args.proxy_pool_file else settings.proxy.http_proxy
    )
    if args.max_bytes_per_second < 0:
        parser.error("--max-bytes-per-second must be non-negative")
    if args.proxy_attempts_per_url <= 0:
        parser.error("--proxy-attempts-per-url must be positive")
    if args.proxy_allowlist_file and not args.proxy_pool_file:
        parser.error("--proxy-allowlist-file requires --proxy-pool-file")
    chained_pool = (
        ChainedProxyPool.from_file(
            args.proxy_pool_file,
            first_hop_proxy=proxy_url,
            allowlist_path=args.proxy_allowlist_file,
        )
        if args.proxy_pool_file
        else None
    )
    builder = MainConferenceCorpusBuilder(
        corpus_root=args.corpus_root,
        pdf_root=args.pdf_root,
        proxy_url=proxy_url,
    )
    print(f"[{time.strftime('%H:%M:%S')}] corpus_root={builder.corpus_root} proxy={safe_proxy_label(proxy_url)}")
    print(f"           pdf_root={builder.pdf_root}")
    if chained_pool is not None:
        print(f"           strict two-hop exits={chained_pool.configured_count:,}")
    print(f"           manifest={builder.manifest_path}")

    records = load_json_with_recovery(builder.manifest_path, default_value=[])
    if not records:
        print("[!] manifest empty / unreadable", file=sys.stderr)
        return 2
    print(f"           manifest entries = {len(records):,}")

    # Filter to target slice + missing PDFs
    subset_records = [
        r for r in records
        if r.get("conference") == args.conf and r.get("year") == args.year
        and not _alive(r.get("pdf_path"))
        and r.get("pdf_url")
    ]
    total_in_slice = sum(
        1 for r in records
        if r.get("conference") == args.conf and r.get("year") == args.year
    )
    print(f"           slice {args.conf} {args.year}: {total_in_slice:,} total, "
          f"{len(subset_records):,} need download")

    if args.limit:
        subset_records = subset_records[: args.limit]
        print(f"           limit applied: {len(subset_records):,}")

    if not subset_records:
        print("[done] nothing to download")
        return 0

    papers = [CorpusPaper.from_manifest_record(r) for r in subset_records]

    # CRITICAL: builder.save_manifest_checkpoint / save_manifest will
    # overwrite the WHOLE manifest with only the subset we feed in
    # (verified 2026-05-17 00:00: smoke 5-paper run reduced 64,421-entry
    # manifest to 5 entries). Suppress them — manifest sync is now handled
    # by scripts/sync_manifest_pdf_md_paths.py against the filesystem after
    # downloads finish, so downloader's job is just "drop PDFs on disk".
    builder.save_manifest_checkpoint = lambda *a, **k: None  # type: ignore
    builder.save_manifest = lambda *a, **k: None  # type: ignore

    print(f"[{time.strftime('%H:%M:%S')}] starting download_papers "
          f"max_concurrent={args.concurrency} min_interval={args.min_interval_seconds:g}s "
          f"max_bytes_per_second={args.max_bytes_per_second} "
          f"subset={len(papers):,} "
          f"(manifest-write suppressed; sync after via sync_manifest_pdf_md_paths.py)")
    t0 = time.perf_counter()
    async def run_download() -> None:
        if chained_pool is not None:
            required_healthy = min(
                chained_pool.configured_count,
                max(args.concurrency, args.concurrency * 2),
            )
            probe = await chained_pool.probe_pdf(
                papers[0].pdf_url,
                required_healthy=required_healthy,
                max_candidates=args.proxy_probe_candidates,
                concurrency=args.concurrency,
                timeout_seconds=args.proxy_probe_timeout_seconds,
            )
            print(
                f"[{time.strftime('%H:%M:%S')}] two-hop probe "
                f"healthy={probe.healthy_count}/{probe.attempted_count} "
                f"configured={probe.configured_count} elapsed={probe.elapsed_seconds:.1f}s"
            )
            if probe.healthy_count < args.concurrency:
                raise RuntimeError(
                    f"only {probe.healthy_count} two-hop exits passed the real-PDF probe; "
                    f"need at least {args.concurrency}"
                )
        await builder.download_papers(
            papers,
            max_concurrent=args.concurrency,
            min_request_interval_seconds=args.min_interval_seconds,
            max_bytes_per_second=args.max_bytes_per_second,
            chained_proxy_pool=chained_pool,
            proxy_attempts_per_url=args.proxy_attempts_per_url,
        )

    asyncio.run(run_download())
    elapsed = time.perf_counter() - t0
    print(f"[{time.strftime('%H:%M:%S')}] done in {elapsed:.0f}s "
          f"({len(papers) / max(elapsed, 0.001):.2f} papers/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
