"""Download a slice of manifest papers (subset by conference / year).

Wraps MainConferenceCorpusBuilder.download_papers but only feeds it the
filtered subset, so we can run targeted backfills like ICLR 2025 without
re-scanning all 64K manifest entries on every retry.

Usage:
    python scripts/download_subset.py --conf ICLR --year 2025 --concurrency 8
    python scripts/download_subset.py --conf ICLR --year 2024 --concurrency 8
    python scripts/download_subset.py --conf CVPR --year 2022 --concurrency 4
    python scripts/download_subset.py --target JMLR:2024 --target VLDB:2025 --host-turns \
        --min-interval-seconds 5

Several --target slices download as one serial queue with --host-turns: two
consecutive requests never hit the same host while another host has work,
and a host that refuses is cooled down instead of retried.
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
    verify_full_text_identity,
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


def verify_identity(builder: MainConferenceCorpusBuilder, records: list[dict], targets, dry_run: bool = False) -> dict:
    """Page-1 title / page-count check of every target PDF on disk, before the
    extract stage sees them. Rejected non-publisher copies, and any copy whose URL names
    another paper's DOI, leave the PDF tree; ``dry_run`` only lists them."""
    wanted = set(targets)
    papers = [
        CorpusPaper.from_manifest_record(r) for r in records
        if (r.get("conference"), r.get("year")) in wanted and r.get("paper_id")
    ]
    report = verify_full_text_identity(
        papers, corpus_root=builder.corpus_root, pdf_root=builder.pdf_root, dry_run=dry_run,
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] identity check: ok={report['ok']} "
        f"kept_mismatch={report['kept_mismatch']} rejected={report['rejected']} "
        f"already_checked={report['already_checked']}"
    )
    for item in report["kept_mismatches"]:
        print(f"           kept ({item['full_text_source'] or 'origin unknown'}): "
              f"{item['paper_id']}: {item['problem']}")
    for item in report["rejections"]:
        print(f"           {'would reject' if dry_run else 'rejected'}: {item['paper_id']} "
              f"({item['full_text_source'] or 'origin unknown'} {item['url']}): {item['problem']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", help="Conference name (e.g. ICLR, CVPR)")
    parser.add_argument("--year", type=int)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="CONF:YEAR",
        help="Slice to download; repeatable. Combined with --conf/--year when both are given.",
    )
    parser.add_argument(
        "--host-turns",
        action="store_true",
        help="Serial download that alternates hosts and cools down refusing hosts.",
    )
    parser.add_argument(
        "--host-cooldown-seconds",
        type=float,
        default=300.0,
        help="First cooldown after a host refuses; doubles per consecutive refusal (default: 300).",
    )
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
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="No download: only the full-text identity check of the target PDFs already on disk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --verify-only: list the copies that would be rejected; move and write nothing.",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Max papers to enqueue (0 = all missing). Useful for smoke test.")
    args = parser.parse_args()
    if args.dry_run and not args.verify_only:
        parser.error("--dry-run only applies to --verify-only")
    targets = parse_targets(args.target, args.conf, args.year)
    if not targets:
        parser.error("give --conf and --year, or at least one --target CONF:YEAR")
    if args.host_turns:
        args.concurrency = 1
    if any(conf.upper() == "ICML" and year == 2026 for conf, year in targets):
        args.concurrency = 1
        args.min_interval_seconds = max(args.min_interval_seconds, 10.0)

    # 强制走部署配置中的代理（无直连回退）。代理不可达时显式失败，
    # 不静默改直连（见 config.ProxySettings）。--verify-only 不发任何请求。
    proxy_url = "" if args.verify_only else require_explicit_proxy(
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
    print(f"[{time.strftime('%H:%M:%S')}] corpus_root={builder.corpus_root} "
          f"proxy={safe_proxy_label(proxy_url) if proxy_url else 'none (verify only)'}")
    print(f"           pdf_root={builder.pdf_root}")
    if chained_pool is not None:
        print(f"           strict two-hop exits={chained_pool.configured_count:,}")
    print(f"           manifest={builder.manifest_path}")

    records = load_json_with_recovery(builder.manifest_path, default_value=[])
    if not records:
        print("[!] manifest empty / unreadable", file=sys.stderr)
        return 2
    print(f"           manifest entries = {len(records):,}")
    if args.verify_only:
        verify_identity(builder, records, targets, dry_run=args.dry_run)
        return 0

    # Filter to the target slices + missing PDFs
    wanted = set(targets)
    subset_records = [
        r for r in records
        if (r.get("conference"), r.get("year")) in wanted
        and not _alive(r.get("pdf_path"))
        and r.get("pdf_url")
    ]
    for conf, year in targets:
        total_in_slice = sum(
            1 for r in records if r.get("conference") == conf and r.get("year") == year
        )
        need = sum(1 for r in subset_records if r.get("conference") == conf and r.get("year") == year)
        print(f"           slice {conf} {year}: {total_in_slice:,} total, {need:,} need download")

    if args.limit:  # per slice, so a multi-target smoke covers every slice
        taken: dict[tuple, int] = {}
        limited = []
        for r in subset_records:
            key = (r.get("conference"), r.get("year"))
            if taken.get(key, 0) < args.limit:
                taken[key] = taken.get(key, 0) + 1
                limited.append(r)
        subset_records = limited
        print(f"           limit applied: {len(subset_records):,}")

    if not subset_records:
        print("[done] nothing to download")
        verify_identity(builder, records, targets)
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
            host_turns=args.host_turns,
            host_cooldown_seconds=args.host_cooldown_seconds,
        )

    asyncio.run(run_download())
    elapsed = time.perf_counter() - t0
    print(f"[{time.strftime('%H:%M:%S')}] done in {elapsed:.0f}s "
          f"({len(papers) / max(elapsed, 0.001):.2f} papers/s)")
    verify_identity(builder, records, targets)
    return 0



def parse_targets(raw_targets: list[str], conf: str | None, year: int | None) -> list[tuple[str, int]]:
    targets: list[tuple[str, int]] = []
    if conf and year:
        targets.append((conf, int(year)))
    for raw in raw_targets:
        name, sep, raw_year = raw.rpartition(":")
        if not sep or not name or not raw_year.isdigit():
            raise SystemExit(f"--target must look like CONF:YEAR, got {raw!r}")
        if (name, int(raw_year)) not in targets:
            targets.append((name, int(raw_year)))
    return targets


if __name__ == "__main__":
    sys.exit(main())
