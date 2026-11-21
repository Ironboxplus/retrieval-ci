#!/usr/bin/env python3
"""Prepare robotics conference PDF corpus: ICRA + CoRL + RSS, 2022–2026 full.

Default is **dry-run collect of the entire bulk window** (all three venues ×
every year 2022..2026). Real PDF bytes require ``--execute-download`` (still
via 127.0.0.1:10808). Cloudflare / IEEE 418 edges use TurnstileBypass minimal
headless when CF bypass is enabled (default).

Examples:
  # Full catalog for all venues and years (no PDF bytes)
  python scripts/robotics/prepare_robotics_corpus.py collect

  # Smoke: 5 papers per venue-year only
  python scripts/robotics/prepare_robotics_corpus.py collect --sample-per-cell 5

  # Plan downloads for entire manifest
  python scripts/robotics/prepare_robotics_corpus.py download

  # ONE-SHOT full download of everything planned (after catalog is green)
  python scripts/robotics/prepare_robotics_corpus.py download --execute-download

  # Subset: only CoRL+RSS 2023-2024
  python scripts/robotics/prepare_robotics_corpus.py collect --venues CoRL RSS --years 2023 2024
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.robotics_corpus.models import RoboticsVenue  # noqa: E402
from pipeline.robotics_corpus.prepare import (  # noqa: E402
    PrepareConfig,
    collect_manifest,
    prepare,
    run_download_pass,
)
from pipeline.robotics_corpus.proxy import load_robotics_proxy_from_settings  # noqa: E402
from pipeline.robotics_corpus.sources import DEFAULT_TARGET_YEARS, DEFAULT_VENUES  # noqa: E402
from pipeline.medical_corpus.cf_bypass import (  # noqa: E402
    minimal_backend_available,
    turnstile_available,
)


def _parse_venues(raw: list[str] | None) -> list[RoboticsVenue]:
    if not raw:
        return list(DEFAULT_VENUES)
    out: list[RoboticsVenue] = []
    for token in raw:
        key = token.strip()
        # accept icra / ICRA / CoRL / corl / RSS
        mapping = {
            "icra": RoboticsVenue.ICRA,
            "corl": RoboticsVenue.CORL,
            "rss": RoboticsVenue.RSS,
        }
        venue = mapping.get(key.lower())
        if venue is None:
            venue = RoboticsVenue(key)  # may raise
        out.append(venue)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-v", "--verbose", action="store_true")
        sp.add_argument("--data-root", type=Path, default=Path("data/robotics"))
        sp.add_argument(
            "--years",
            type=int,
            nargs="+",
            default=list(DEFAULT_TARGET_YEARS),
            help=f"Years to include (default: full bulk {list(DEFAULT_TARGET_YEARS)})",
        )
        sp.add_argument(
            "--venues",
            nargs="+",
            default=None,
            help="Venues: ICRA CoRL RSS (default: all three)",
        )
        sp.add_argument(
            "--sample-per-cell",
            type=int,
            default=0,
            help="Smoke only: cap papers per venue-year (0 = FULL year, default)",
        )
        sp.add_argument(
            "--min-interval",
            type=float,
            default=0.25,
            help="Seconds between catalog HTTP requests (default 0.25; raises on 429)",
        )
        sp.add_argument(
            "--download-concurrency",
            type=int,
            default=8,
            help="Parallel PDF workers for execute-download (default 8; OA-friendly)",
        )
        sp.add_argument(
            "--execute-download",
            action="store_true",
            help="Actually download PDF bytes (default: plan only)",
        )
        sp.add_argument(
            "--download-limit",
            type=int,
            default=0,
            help="Cap downloads this pass (0 = ENTIRE manifest — full bulk)",
        )
        sp.add_argument(
            "--no-cf-bypass",
            action="store_true",
            help="Disable TurnstileBypass on CF/403/418 edges",
        )

    sp = sub.add_parser("collect", help="Build full-year robotics JSONL manifest")
    add_common(sp)
    sp = sub.add_parser("download", help="Plan or execute PDF downloads from manifest")
    add_common(sp)
    sp = sub.add_parser("all", help="collect then download pass")
    add_common(sp)

    sp = sub.add_parser("status", help="Show proxy/CF readiness and default scope")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.add_argument("--data-root", type=Path, default=Path("data/robotics"))

    return p


def _cfg_from_args(args: argparse.Namespace) -> PrepareConfig:
    return PrepareConfig(
        data_root=args.data_root,
        years=tuple(args.years),
        venues=_parse_venues(getattr(args, "venues", None)),
        execute_download=bool(getattr(args, "execute_download", False)),
        allow_cf_bypass=not bool(getattr(args, "no_cf_bypass", False)),
        download_limit=int(getattr(args, "download_limit", 0) or 0),
        min_interval_s=float(getattr(args, "min_interval", 0.25)),
        download_concurrency=int(getattr(args, "download_concurrency", 8) or 8),
        sample_limit_per_venue_year=int(getattr(args, "sample_per_cell", 0) or 0),
        proxy=load_robotics_proxy_from_settings(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "status":
        proxy = load_robotics_proxy_from_settings()
        payload = {
            "proxy": proxy.http_proxy,
            "default_years": list(DEFAULT_TARGET_YEARS),
            "default_venues": [v.value for v in DEFAULT_VENUES],
            "scope": "FULL bulk 2022-2026 × ICRA+CoRL+RSS (not a smoke subset)",
            "turnstile_importable": turnstile_available(),
            "minimal_backend_available": minimal_backend_available(),
            "data_root": str(args.data_root),
            "manifest": str(args.data_root / "manifest" / "robotics_manifest.jsonl"),
            "pdf_root": str(args.data_root / "pdf"),
            "note": (
                "collect without --sample-per-cell builds the full multi-year catalog; "
                "download --execute-download without --download-limit pulls every planned PDF."
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    cfg = _cfg_from_args(args)
    print(
        f"proxy={cfg.proxy.http_proxy if cfg.proxy else load_robotics_proxy_from_settings().http_proxy}"
    )
    print(f"years={list(cfg.years)} venues={[v.value for v in cfg.resolved_venues()]}")
    print(
        f"sample_per_cell={cfg.sample_limit_per_venue_year} "
        f"execute_download={cfg.execute_download} "
        f"download_limit={cfg.download_limit} cf_bypass={cfg.allow_cf_bypass}"
    )
    if cfg.sample_limit_per_venue_year == 0 and args.cmd in {"collect", "all"}:
        print("NOTE: full bulk catalog (all papers per venue-year), not a sample.")
    if cfg.execute_download and cfg.download_limit == 0:
        print("NOTE: full bulk download of entire manifest (download_limit=0).")

    if args.cmd == "collect":
        summary = collect_manifest(cfg)
    elif args.cmd == "download":
        summary = run_download_pass(cfg)
    elif args.cmd == "all":
        summary = prepare(cfg)
    else:
        parser.error(f"unknown cmd {args.cmd}")
        return 2

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
