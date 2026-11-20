"""Push one conference slice's PDFs or Markdown to the ingest host as a verified tar.

Run on the machine that produced the files (a downloading host, a GPU box that
ran the marker). The bundle lands in the ingest host's inbox, where an ingest
pipeline started with ``options={"download_from": "push"}`` (PDFs) or
``{"extract_from": "push"}`` (Markdown) is waiting for it; see
pipeline/paper_crawler/slice_bundle.py for the format and the checks.

Usage:
    python scripts/sync/push_slice_bundle.py --conf CVPR --year 2026 --kind pdf \
        --ssh-target 4090-88 --remote-inbox /data/papers/ccf_a_main_conference_corpus/inbox
    python scripts/sync/push_slice_bundle.py --conf CVPR --year 2026 --kind md \
        --corpus-root data/papers --ssh-target 4090-88 --remote-inbox /path/to/inbox
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.paper_crawler.inbox_transfer import InboxTransfer  # noqa: E402
from pipeline.paper_crawler.slice_bundle import push_slice_files  # noqa: E402

CORPUS_ROOT = ROOT / "data" / "papers" / "ccf_a_main_conference_corpus"
KIND_DIR = {"pdf": "pdf", "md": "markdown"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", required=True, help="venue as in the manifest, e.g. CVPR")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--kind", choices=sorted(KIND_DIR), required=True)
    ap.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    ap.add_argument("--ssh-target", required=True, help="ssh alias of the ingest host")
    ap.add_argument("--remote-inbox", required=True, help="<ingest corpus root>/inbox on the ingest host")
    ap.add_argument("--ssh-command", nargs="+", help="ssh executable, or interpreter + script (default: the OpenSSH client)")
    ap.add_argument("--scp-command", nargs="+", help="scp executable, or interpreter + script (default: the OpenSSH client)")
    args = ap.parse_args(argv)

    conf_lower = args.conf.lower()
    source_dir = args.corpus_root / KIND_DIR[args.kind] / conf_lower / str(args.year)
    if not source_dir.is_dir():
        print(f"no {args.kind} directory for {args.conf} {args.year}: {source_dir}", file=sys.stderr)
        return 2
    transfer = InboxTransfer(
        ssh_target=args.ssh_target, remote_inbox=args.remote_inbox,
        ssh=args.ssh_command, scp=args.scp_command)
    try:
        info = push_slice_files(
            source_dir, args.corpus_root / "outbox", transfer,
            venue=args.conf, year=args.year, kind=args.kind, conf_lower=conf_lower)
    except (ValueError, RuntimeError) as exc:
        print(f"push failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"venue": args.conf, "year": args.year, "kind": args.kind, **info}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
