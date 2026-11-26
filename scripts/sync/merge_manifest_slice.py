"""Merge a payload of manifest records into a host's WORKING manifest, by paper_id.

Why this exists: the manifest is not one global file. `.88` keeps a small working
manifest for the slice currently being ingested (5,194 records on 2026-09-16 --
ACL 2026 plus a SIGIR canary); the 136 MB full manifest was archived beside it on
2026-08-04. Shipping the local 106,032-record manifest over would destroy that
working state, and every path in it points at Windows locations this host cannot
read.

So: export the slice on the authoring host, rewrite its paths for the target, and
merge here. A record whose paper_id already exists is REPLACED (the payload is
newer by construction); anything else is appended. Records outside the payload are
never read or written, so other venues keep whatever paths they have.

Usage:
    python scripts/sync/merge_manifest_slice.py --payload /tmp/slice.json
    python scripts/sync/merge_manifest_slice.py --payload /tmp/slice.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT / "data" / "papers" / "ccf_a_main_conference_corpus" / "manifests" / "paper_manifest.json"
)


def merge_slice(base: list[dict], payload: list[dict]) -> dict:
    """Mutate `base` in place: replace-by-paper_id, else append. Returns counts."""
    index = {r.get("paper_id"): i for i, r in enumerate(base) if r.get("paper_id")}
    replaced = appended = skipped = 0

    for rec in payload:
        pid = rec.get("paper_id")
        if not pid:
            skipped += 1  # a record with no key cannot be merged idempotently
            continue
        hit = index.get(pid)
        if hit is None:
            index[pid] = len(base)
            base.append(rec)
            appended += 1
        else:
            base[hit] = rec
            replaced += 1

    return {
        "payload_records": len(payload),
        "replaced": replaced,
        "appended": appended,
        "skipped_no_paper_id": skipped,
        "total_after": len(base),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, help="JSON list of manifest records")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = parser.parse_args()

    mp = Path(args.manifest_path)
    base = json.loads(mp.read_text(encoding="utf-8"))
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    print(f"[{time.strftime('%H:%M:%S')}] {mp}: {len(base):,} records; payload {len(payload):,}")

    counts = merge_slice(base, payload)
    print(
        f"  replaced={counts['replaced']} appended={counts['appended']} "
        f"skipped_no_paper_id={counts['skipped_no_paper_id']} total_after={counts['total_after']:,}"
    )

    if args.dry_run:
        print("[dry-run] manifest not written")
        return 0

    bak = mp.with_suffix(f".json.bak.merge.{int(time.time())}")
    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
    mp.rename(bak)
    tmp.rename(mp)
    print(f"[execute] wrote {mp}  (backup: {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
