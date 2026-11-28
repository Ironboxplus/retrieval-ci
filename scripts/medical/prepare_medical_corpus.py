#!/usr/bin/env python3
"""Prepare medical OA PDF corpus (EuropePMC-first, proxy-mandatory).

Default is **dry-run**: build a manifest of OA medical papers with PDF URLs and
plan download paths. Real PDF bytes require ``--execute-download`` (still via
127.0.0.1:10808). Cloudflare edges use TurnstileBypass minimal headless when
``--allow-cf-bypass`` is on (default).

Examples:
  # Catalog smoke (network, no PDF bytes)
  python scripts/medical/prepare_medical_corpus.py collect --max-records 20

  # Plan downloads from existing manifest
  python scripts/medical/prepare_medical_corpus.py download

  # Real download of at most 3 PDFs (only after smoke)
  python scripts/medical/prepare_medical_corpus.py download --execute-download --download-limit 3

  # Probe CF dual-path on a blocked URL (Chrome + selenium-driverless)
  python scripts/medical/prepare_medical_corpus.py cf-probe --url https://example.com/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.medical_corpus.cf_bypass import (  # noqa: E402
    minimal_backend_available,
    solve_cloudflare_for_url_sync,
    turnstile_available,
)
from pipeline.medical_corpus.europe_pmc import (  # noqa: E402
    DEFAULT_BULK_QUERY,
    DEFAULT_RECENT_QUERY,
)
from pipeline.medical_corpus.prepare import DEFAULT_DATA_ROOT, PrepareConfig, collect_manifest_v2, prepare, run_download_pass
from pipeline.medical_corpus.pdf_fetch import DEFAULT_MAX_BYTES_PER_SECOND
from pipeline.medical_corpus.proxy import load_medical_proxy_from_settings
from pipeline.medical_corpus.top_journals import (  # noqa: E402
    TOP_MEDICAL_JOURNALS,
    default_top_journal_query,
)
from pipeline.medical_corpus.executor import require_designated_medical_cli_execution  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-v", "--verbose", action="store_true")
        sp.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
        sp.add_argument(
            "--query",
            default=None,
            help="EuropePMC query override (default: OA+PDF MED top-journal clause 2020-2026)",
        )
        sp.add_argument(
            "--bulk-query",
            action="store_true",
            help=f"Use full OA+PDF MED query without journal clause: {DEFAULT_BULK_QUERY}",
        )
        sp.add_argument(
            "--no-top-journals",
            action="store_true",
            help="Disable top-journal whitelist (NOT recommended; keeps mid-tier OA noise)",
        )
        sp.add_argument("--max-records", type=int, default=100)
        sp.add_argument("--page-size", type=int, default=50)
        sp.add_argument("--min-interval", type=float, default=0.35,
                        help="Seconds between EuropePMC requests")
        sp.add_argument("--execute-download", action="store_true",
                        help="Actually download PDF bytes (default: plan only)")
        sp.add_argument("--download-limit", type=int, default=0,
                        help="Cap real/planned downloads this pass (0=all)")
        sp.add_argument("--no-cf-bypass", action="store_true",
                        help="Disable TurnstileBypass on CF/403 edges")

    sp = sub.add_parser("collect", help="Build/refresh EuropePMC manifest JSONL")
    add_common(sp)
    sp = sub.add_parser("download", help="Plan or execute PDF downloads from manifest")
    add_common(sp)
    sp = sub.add_parser("all", help="collect then download pass")
    add_common(sp)

    sp = sub.add_parser("status", help="Show proxy/CF SDK readiness and paths")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)

    sp = sub.add_parser("cf-probe", help="Run TurnstileBypass dual-path on one URL")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.add_argument("--url", required=True)
    sp.add_argument("--no-verify", action="store_true")
    sp.add_argument("--path", default="auto", choices=["auto", "soft", "widget", "managed"])

    return p


def _cfg_from_args(args: argparse.Namespace) -> PrepareConfig:
    top_only = not bool(getattr(args, "no_top_journals", False))
    if getattr(args, "bulk_query", False):
        query = DEFAULT_BULK_QUERY
    elif getattr(args, "query", None):
        query = args.query
    elif top_only:
        query = default_top_journal_query()
    else:
        query = DEFAULT_RECENT_QUERY
    return PrepareConfig(
        data_root=args.data_root,
        query=query,
        max_records=args.max_records,
        page_size=args.page_size,
        execute_download=bool(getattr(args, "execute_download", False)),
        allow_cf_bypass=not bool(getattr(args, "no_cf_bypass", False)),
        download_limit=int(getattr(args, "download_limit", 0) or 0),
        min_interval_s=float(getattr(args, "min_interval", 0.35)),
        top_journals_only=top_only,
        proxy=load_medical_proxy_from_settings(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "status":
        proxy = load_medical_proxy_from_settings()
        payload = {
            "proxy": proxy.http_proxy,
            "turnstile_importable": turnstile_available(),
            "minimal_backend_available": minimal_backend_available(),
            "top_journals_only_default": True,
            "top_journal_count": len(TOP_MEDICAL_JOURNALS),
            "top_journals_sample": list(TOP_MEDICAL_JOURNALS[:12]),
            "default_query": default_top_journal_query(),
            "data_root": str(args.data_root),
            "manifest": str(args.data_root / "manifest" / "medical_manifest.jsonl"),
            "pdf_root": str(args.data_root / "pdf"),
            "max_bytes_per_second": DEFAULT_MAX_BYTES_PER_SECOND,
            "max_megabytes_per_second": DEFAULT_MAX_BYTES_PER_SECOND / 1_000_000,
            "vendor": str(ROOT / "vendor" / "TurnstileBypass"),
            "note": (
                "Default collect keeps top journals only. "
                "Install selenium-driverless for minimal FO; "
                "clone TurnstileBypass into vendor/ if missing."
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "cf-probe":
        proxy = load_medical_proxy_from_settings()
        print(f"proxy={proxy.http_proxy} turnstile={turnstile_available()} "
              f"minimal={minimal_backend_available()}")
        res = solve_cloudflare_for_url_sync(
            args.url,
            proxy=proxy,
            path=args.path,
            require_clearance=True,
            headless=True,
            verify=not args.no_verify,
        )
        print(json.dumps({
            "ok": res.ok,
            "path": res.path,
            "page_url": res.page_url,
            "final_url": res.final_url,
            "has_clearance": bool(res.cf_clearance),
            "has_cookie_header": bool(res.cookie_header),
            "has_token": bool(res.token),
            "backend": res.backend,
            "verified": res.verified,
            "ms": res.ms,
            "error": res.error,
            "user_agent": res.user_agent[:80] if res.user_agent else "",
        }, ensure_ascii=False, indent=2))
        return 0 if res.ok else 2

    cfg = _cfg_from_args(args)
    if cfg.execute_download:
        # Dry-run remains usable from an operator workstation.  Real PDF bytes
        # are deliberately restricted to the deployment-owned .88 mount/proxy.
        cfg.data_root = require_designated_medical_cli_execution(
            cfg.data_root,
            cfg.proxy or load_medical_proxy_from_settings(),
        )
    print(f"proxy={cfg.proxy.http_proxy if cfg.proxy else load_medical_proxy_from_settings().http_proxy}")
    print(f"query={cfg.query!r}")
    print(
        f"max_records={cfg.max_records} top_journals_only={cfg.top_journals_only} "
        f"execute_download={cfg.execute_download} cf_bypass={cfg.allow_cf_bypass}"
    )

    if args.cmd == "collect":
        summary = collect_manifest_v2(cfg)
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
