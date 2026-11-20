"""
Verify conference paper sources — open & extensible.

Usage:
    python scripts/verify_2025_sources.py                    # verify all conferences, all target years
    python scripts/verify_2025_sources.py --year 2025        # verify specific year
    python scripts/verify_2025_sources.py --conf AAAI CVPR   # verify specific conferences
    python scripts/verify_2025_sources.py --year 2025 --conf NeurIPS --dry-run  # just check availability
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    DEFAULT_PROXY_URL,
    DEFAULT_TARGET_YEARS,
    OfficialProceedingsClient,
    build_sources,
)


@dataclass
class SourceVerification:
    conference: str
    year: int
    paper_count: int
    elapsed_sec: float
    sample_title: str
    error: Optional[str] = None
    track_type: str = "main_conference"


def verify_source(
    name: str,
    source,
    year: int,
    client: OfficialProceedingsClient,
) -> SourceVerification:
    t0 = time.perf_counter()
    try:
        if not source.supports_year(year):
            return SourceVerification(
                conference=name, year=year, paper_count=0,
                elapsed_sec=0, sample_title="",
                error=f"Year {year} not supported (e.g. ICCV is biennial)",
            )
        papers = source.collect_papers(year, client)
        elapsed = time.perf_counter() - t0
        return SourceVerification(
            conference=name, year=year, paper_count=len(papers),
            elapsed_sec=round(elapsed, 1),
            sample_title=papers[0].title[:80] if papers else "",
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return SourceVerification(
            conference=name, year=year, paper_count=0,
            elapsed_sec=round(elapsed, 1), sample_title="",
            error=str(e)[:120],
        )


def main():
    parser = argparse.ArgumentParser(description="Verify conference paper sources")
    parser.add_argument("--year", type=int, nargs="+", default=None,
                        help="Years to verify (default: all target years)")
    parser.add_argument("--conf", type=str, nargs="+", default=None,
                        help="Conferences to verify (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only check supports_year(), skip actual collection")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON report to file")
    args = parser.parse_args()

    years = args.year or list(DEFAULT_TARGET_YEARS)
    all_sources = build_sources()
    conf_names = args.conf or list(all_sources.keys())

    client = OfficialProceedingsClient(proxy_url=DEFAULT_PROXY_URL)
    results = []

    for name in conf_names:
        source = all_sources.get(name)
        if source is None:
            print(f"WARNING: Unknown conference '{name}', skipping")
            continue
        for year in years:
            if args.dry_run:
                supported = source.supports_year(year)
                print(f"{name:10s} {year}: {'supported' if supported else 'NOT supported'}")
                continue

            result = verify_source(name, source, year, client)
            results.append(result)
            status = f"{result.paper_count:>5d} papers" if not result.error else f"ERROR: {result.error}"
            print(f"{name:10s} {year}: {status}  ({result.elapsed_sec}s)")
            if result.sample_title:
                print(f"{'':10s}       sample: {result.sample_title}")

    client.close()

    if not args.dry_run:
        total = sum(r.paper_count for r in results)
        errors = sum(1 for r in results if r.error)
        print(f"\nTotal: {total} papers across {len(results)} source-year combos, {errors} errors")

    if args.output and results:
        report = [asdict(r) for r in results]
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
