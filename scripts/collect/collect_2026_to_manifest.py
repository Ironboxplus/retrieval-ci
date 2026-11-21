"""Collect a single (conference, year) slice and merge it into the existing manifest.

`MainConferenceCorpusBuilder.collect_manifest` writes the manifest from a fresh
empty dict, which would destroy unrelated entries (e.g. all 2022-2025 papers)
when run with a narrow conferences/years scope. This script loads the existing
manifest, calls one source's collect_papers, and merges by paper_id, then
writes atomically with an automatic backup.

Usage:
    python scripts/collect_2026_to_manifest.py --conf CVPR --year 2026
    python scripts/collect_2026_to_manifest.py --conf ICLR --year 2026
    python scripts/collect_2026_to_manifest.py --conf AAAI --year 2026
    python scripts/collect_2026_to_manifest.py --conf EMNLP --year 2026 --announced-snapshot
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    MainConferenceCorpusBuilder,
    OfficialProceedingsClient,
    normalize_title_key,
)
from pipeline.paper_crawler.runtime_files import (
    load_json_with_recovery,
    write_json_atomic,
)
from config import settings


def maybe_resolve_open_access(source, papers, client, *, enabled: bool):
    """Opt-in publisher replacement; ACM DL remains canonical by default."""
    if not enabled:
        return None
    resolve_oa = getattr(source, "resolve_open_access_pdf_urls", None)
    if resolve_oa is None:
        return None
    return resolve_oa(papers, client)


def announced_prior_year_replay(
    candidates,
    manifest_records,
    *,
    conference: str,
    year: int,
) -> dict[str, float | int]:
    """Measure whether an announced page is actually last year's paper list.

    ACL Anthology HTML can inject spaces inside styled title fragments, so the
    comparison removes all normalized whitespace after punctuation folding.
    This is intentionally only a replay detector; it does not promote records.
    """
    target_conference = conference.upper()
    prior_year_titles = {
        normalize_title_key(str(record.get("title") or "")).replace(" ", "")
        for record in manifest_records
        if str(record.get("conference") or record.get("venue") or "").upper()
        == target_conference
        and record.get("year") == year - 1
        and record.get("title")
    }
    candidate_titles = {
        normalize_title_key(str(candidate.title)).replace(" ", "")
        for candidate in candidates
        if getattr(candidate, "title", "")
    }
    overlap_count = len(candidate_titles & prior_year_titles)
    denominator = max(1, len(candidate_titles))
    return {
        "candidate_count": len(candidate_titles),
        "prior_year_count": len(prior_year_titles),
        "overlap_count": overlap_count,
        "overlap_ratio": overlap_count / denominator,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="Conference name (e.g. CVPR, ICLR, AAAI)")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap collected papers (0 = no cap). Useful for smoke.")
    parser.add_argument(
        "--announced-snapshot",
        action="store_true",
        help=(
            "Write a provisional accepted-paper snapshot under reports/ without "
            "touching the canonical manifest. The source must explicitly support it."
        ),
    )
    parser.add_argument(
        "--resolve-open-access",
        action="store_true",
        help="Explicitly replace publisher PDF URLs with resolved OA copies.",
    )
    args = parser.parse_args()

    # 强制走代理（无直连回退）：外网采集必须经 10808。代理只在 Arc 上 → 采集跑在 Arc。
    # 代理不可达时显式失败，不静默改直连（见 config.ProxySettings）。
    proxy_url = settings.proxy.http_proxy
    builder = MainConferenceCorpusBuilder(proxy_url=proxy_url)
    manifest_path = builder.manifest_path
    print(f"[{time.strftime('%H:%M:%S')}] manifest_path={manifest_path} proxy={proxy_url or 'direct'}")

    sources = builder.sources
    if args.conf not in sources:
        print(f"[!] unknown conference {args.conf!r}; known: {sorted(sources)}", file=sys.stderr)
        return 2
    source = sources[args.conf]
    if not source.supports_year(args.year):
        print(f"[!] {args.conf} source does not support year {args.year}", file=sys.stderr)
        return 2

    if args.announced_snapshot:
        collect_announced = getattr(source, "collect_announced_papers", None)
        if collect_announced is None:
            print(
                f"[!] {args.conf} has no announced-paper snapshot source",
                file=sys.stderr,
            )
            return 2
        print(f"[{time.strftime('%H:%M:%S')}] collecting announced {args.conf} {args.year} ...")
        t0 = time.perf_counter()
        client = OfficialProceedingsClient(proxy_url=proxy_url)
        try:
            candidates = collect_announced(args.year, client)
        finally:
            client.close()
        manifest_records = load_json_with_recovery(manifest_path, default_value=[])
        replay = announced_prior_year_replay(
            candidates,
            manifest_records,
            conference=args.conf,
            year=args.year,
        )
        if replay["candidate_count"] and replay["overlap_ratio"] >= 0.95:
            print(
                f"[!] rejected stale {args.conf} {args.year} announcement: "
                f"{replay['overlap_count']:,}/{replay['candidate_count']:,} titles "
                f"replay the canonical {args.year - 1} corpus; no snapshot written",
                file=sys.stderr,
            )
            return 3
        if args.limit:
            candidates = candidates[: args.limit]
        snapshot_path = builder.report_dir / f"{args.conf.lower()}-{args.year}-announced.json"
        write_json_atomic(snapshot_path, [row.to_snapshot_record() for row in candidates])
        print(
            f"           wrote {len(candidates):,} candidates to {snapshot_path} "
            f"in {time.perf_counter() - t0:.1f}s; canonical manifest unchanged"
        )
        return 0

    existing = load_json_with_recovery(manifest_path, default_value=[])
    print(f"           manifest has {len(existing):,} existing entries")

    backup_path = manifest_path.with_suffix(f".bak.{args.conf.lower()}-{args.year}-{int(time.time())}")
    shutil.copy2(manifest_path, backup_path)
    print(f"           backup -> {backup_path.name}")

    print(f"[{time.strftime('%H:%M:%S')}] collecting {args.conf} {args.year} ...")
    t0 = time.perf_counter()
    client = OfficialProceedingsClient(proxy_url=proxy_url)
    try:
        new_papers = source.collect_papers(args.year, client)
        resolution = maybe_resolve_open_access(
            source,
            new_papers,
            client,
            enabled=args.resolve_open_access,
        )
        if resolution is not None:
            print(f"           open-access resolution: {resolution}")
    finally:
        client.close()
    elapsed = time.perf_counter() - t0
    print(f"           collected {len(new_papers):,} papers in {elapsed:.1f}s")
    if args.limit:
        new_papers = new_papers[: args.limit]
        print(f"           limit applied: {len(new_papers):,}")

    # Merge by paper_id. A full slice collection replaces stale target-slice
    # entries (for example, ICML 2026 all-events schedule rows that are no
    # longer in the main-conference papers page). Limited smoke runs must not
    # delete existing rows.
    target_conf = args.conf.upper()
    removed_stale = 0
    if args.limit:
        kept_existing = existing
    else:
        kept_existing = []
        for record in existing:
            record_conf = (record.get("conference") or record.get("venue") or "").upper()
            if record_conf == target_conf and record.get("year") == args.year:
                removed_stale += 1
                continue
            kept_existing.append(record)

    by_id = {p["paper_id"]: p for p in kept_existing}
    added = 0
    updated = 0
    for paper in new_papers:
        d = asdict(paper)
        # Match the field convention used by other manifest entries.
        d.setdefault("conference", paper.conference)
        d.setdefault("year", paper.year)
        if paper.paper_id in by_id:
            updated += 1
        else:
            added += 1
        by_id[paper.paper_id] = d

    merged = list(by_id.values())
    print(
        f"           merged: removed {removed_stale} stale target-slice entries, "
        f"+{added} new, {updated} updated, total {len(merged):,}"
    )

    write_json_atomic(manifest_path, merged)
    print(f"[{time.strftime('%H:%M:%S')}] wrote {manifest_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
