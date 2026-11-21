"""Migrate the `papers` Qdrant collection payload to the D2/D3 contract.

D2 (docs/papers_payload_contract_plan.md): normalize `doc_id` to `str`.
Qdrant's KEYWORD index only indexes string values, so the ~6.9M production
paper points that store `doc_id` as `int` are invisible to any doc_id
keyword filter — silent 0-match, not an error.

D3: remove `data_source` from `paper`/`review` points. It is not part of
that payload profile (core.storage.papers_payload.PROFILE_BY_SOURCE_TYPE);
it stays on `upload`/`webpage`, where it is a real business field.

Nothing is deleted or re-inserted: the only writes are set_payload (which
merges one key, leaving the rest of the payload alone) and delete_payload on
`data_source`. Vectors are never read (`with_vectors=False`) and never
written. Measured against a real Qdrant in
tests/test_migrate_papers_payload_real_qdrant.py, which asserts both that
vectors are byte-identical across the migration and that the rewritten points
actually enter the KEYWORD index.

SAFETY:
  - Default is DRY-RUN (scan + report only). --execute is required to write.
  - A doc_id that is neither int nor str aborts the run. str() would produce a
    plausible but wrong key ("2.0", "True") that the read-back would then
    happily confirm, so this is checked before the write, not after.
  - --collection has NO default and rejects "papers" outright. The
    production migration is a separate, operator-approved step (plan.md D2);
    this script is verified against an isolated collection first.
  - Every write is set_payload/delete_payload with wait=True and bounded
    exponential backoff (AGENTS.md 2.1 iron law), then re-read back and
    verified — a write call that raises nothing but silently no-ops (the W4
    failure mode wait=True exists to prevent) is treated as a failure, not
    trusted.
  - A checkpoint file records the last fully-scanned scroll offset so a
    crashed run can resume without rescanning from the start. Both writes
    this script issues are idempotent (setting doc_id to the same str twice,
    or deleting an already-absent key, are no-ops), so a resumed run
    reprocessing a page it already touched is safe — the checkpoint only
    saves time, it is never required for correctness.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.storage.papers_payload import PROFILE_BY_SOURCE_TYPE  # noqa: E402
from core.storage.qdrant_operations import retry_qdrant_operation  # noqa: E402

OPERATIONS = ("doc_id", "data_source")

# `papers` is the 13.5M-point production collection. Writing to it needs this
# exact token, so neither a slipped default nor a copy-pasted invocation can
# reach it; a read-only dry run needs nothing.
PRODUCTION_COLLECTION = "papers"
PRODUCTION_ACK = "yes-migrate-production-papers"


class MigrationReport:
    """Accumulates scan facts. Plain data, no I/O — easy to unit test."""

    def __init__(self) -> None:
        self.total_points = 0
        self.by_source_type: Counter = Counter()
        self.keys_by_source_type: Dict[Any, set] = {}
        self.doc_id_type_by_source_type: Counter = Counter()
        self.doc_id_missing = 0
        self.doc_id_needs_str = 0
        self.data_source_by_source_type: Counter = Counter()
        self.data_source_to_remove = 0

    def observe(self, payload: Dict[str, Any]) -> None:
        self.total_points += 1
        source_type = payload.get("source_type")
        self.by_source_type[source_type] += 1
        self.keys_by_source_type.setdefault(source_type, set()).update(payload.keys())

        doc_id = payload.get("doc_id")
        if doc_id is None:
            self.doc_id_missing += 1
        else:
            self.doc_id_type_by_source_type[(source_type, type(doc_id).__name__)] += 1
            if not isinstance(doc_id, str):
                self.doc_id_needs_str += 1

        if "data_source" in payload:
            self.data_source_by_source_type[(source_type, payload["data_source"])] += 1
            profile = PROFILE_BY_SOURCE_TYPE.get(source_type)
            # Only paper/review carrying data_source is contamination (D3).
            # An unrecognized source_type isn't this script's problem to fix.
            if profile is not None and "data_source" not in profile:
                self.data_source_to_remove += 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_points": self.total_points,
            "by_source_type": dict(self.by_source_type),
            "keys_by_source_type": {k: sorted(v) for k, v in self.keys_by_source_type.items()},
            "doc_id_type_by_source_type": {
                f"{st}:{tn}": n for (st, tn), n in self.doc_id_type_by_source_type.items()
            },
            "doc_id_missing": self.doc_id_missing,
            "doc_id_needs_str": self.doc_id_needs_str,
            "data_source_by_source_type": {
                f"{st}:{v}": n for (st, v), n in self.data_source_by_source_type.items()
            },
            "data_source_to_remove": self.data_source_to_remove,
        }


def build_plan(report: MigrationReport) -> Dict[str, int]:
    """Migration plan derived purely from the scan report (plan.md 3.3.2)."""
    return {
        "doc_id_str_conversions": report.doc_id_needs_str,
        "data_source_removals": report.data_source_to_remove,
    }


def normalized_doc_id(value: Any, point_id: Any) -> str:
    """Return the string form of a stored doc_id, or refuse to guess.

    Only `int` converts safely. `str()` on any other type yields a
    plausible-looking WRONG key — a float 2.0 becomes "2.0", a bool True
    becomes "True" — and the post-write read-back compares the stored value
    against that same wrong key, so it would pass. Verification structurally
    cannot catch this class of error; the guard has to run before the write.

    It raises rather than skipping the point: production doc_ids come from a
    SQL integer primary key (index_service.py writes str(row.doc_id)), so a
    doc_id that is neither int nor str means the write contract was broken in
    some way this script has not accounted for. That is something to look at,
    not something to step over quietly on 6.9M points.
    """
    if isinstance(value, str):
        return value
    # bool is a subclass of int, and str(True) == "True" — check it first.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"point {point_id}: doc_id={value!r} is {type(value).__name__}, not int "
            f"or str — refusing to guess its string form. Investigate the write "
            f"path that produced it before migrating."
        )
    return str(value)


class Checkpoint:
    """Resume support: persists the last fully-processed scroll offset."""

    def __init__(self, path: Optional[Path]):
        self.path = path

    def load(self) -> Optional[Any]:
        if self.path is None or not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8")).get("offset")

    def save(self, offset: Any) -> None:
        if self.path is None:
            return
        self.path.write_text(json.dumps({"offset": offset}), encoding="utf-8")

    def clear(self) -> None:
        if self.path is not None and self.path.exists():
            self.path.unlink()


def migrate(
    client,
    collection: str,
    *,
    execute: bool = False,
    scan_batch: int = 2000,
    apply_batch: int = 500,
    limit: int = 0,
    checkpoint: Optional[Checkpoint] = None,
    max_retries: int = 5,
    operations: tuple = OPERATIONS,
    progress_every: int = 250_000,
    log=None,
    sleep=time.sleep,
) -> Dict[str, Any]:
    """Single-pass scan (+ optional apply) over `collection`.

    Returns {"report": ..., "plan": ..., "applied": {...}}. `applied` counts
    what DID change; `plan` counts what the scan found, for every operation,
    whether or not it was selected — so one dry run sizes both.

    `operations` selects which of D2 (doc_id) / D3 (data_source) to apply. They
    are separable on purpose: D2 only rewrites a key's type, while D3 removes a
    field outright, and an operator repairing doc_id has not thereby asked for
    an irreversible deletion.
    """
    unknown = [op for op in operations if op not in OPERATIONS]
    if unknown:
        raise ValueError(f"unknown operation(s) {unknown}; expected any of {list(OPERATIONS)}")
    do_doc_id = "doc_id" in operations
    do_data_source = "data_source" in operations

    report = MigrationReport()
    pending_doc_id: Dict[str, List[Any]] = defaultdict(list)
    pending_doc_id_points = 0  # counted in points, not in distinct doc_id values
    pending_data_source_removal: List[Any] = []
    applied = {
        "doc_id_points_updated": 0,
        "data_source_points_updated": 0,
        "batches_applied": 0,
        "batches_failed": 0,
    }

    def flush_doc_id(pending: Dict[str, List[Any]]) -> None:
        """Write every pending doc_id value in ONE request, then verify once.

        A batch has to be measured in POINTS. Counting it per distinct doc_id
        value — flush when THIS value has accumulated apply_batch points —
        bounds nothing on production data: a single 120,000-point scroll window
        of `papers` holds 34,571 distinct int doc_ids (measured 2026-09-02), so
        no one value reaches the threshold and the whole 6.87M-id backlog stays
        resident until the final flush, which then issues one set_payload plus
        one read-back per distinct value.

        batch_update_points carries one SetPayloadOperation per value in a
        single call, so a flush costs one update plus one verification read no
        matter how many values it spans, and memory stays bounded by
        apply_batch.
        """
        if not pending:
            return
        expected = {pid: value for value, ids in pending.items() for pid in ids}
        if execute:
            from qdrant_client.http import models as qmodels

            ops = [
                qmodels.SetPayloadOperation(set_payload=qmodels.SetPayload(
                    payload={"doc_id": value}, points=list(ids)))
                for value, ids in pending.items()
            ]
            # AGENTS.md 2.1: every write is wait=True + bounded retry (W4/W14).
            retry_qdrant_operation(
                f"batch_update_points(collection={collection} operations={len(ops)})",
                lambda: client.batch_update_points(
                    collection, update_operations=ops, wait=True,
                ),
                max_attempts=max_retries, sleep=sleep,
            )
            # Only the key under test comes back: a `papers` point carries 25
            # payload keys including `content` and `abstract`, and pulling all
            # of them would move tens of GB across a 6.9M-point run to compare
            # one field.
            verify = client.retrieve(collection, ids=list(expected), with_payload=["doc_id"])
            # Each id is checked against ITS OWN expected value: one batch now
            # spans many values, so "some doc_id is present" proves nothing.
            # `missing` also catches a point that disappeared entirely.
            bad = [v.id for v in verify if (v.payload or {}).get("doc_id") != expected[v.id]]
            missing = sorted(set(expected) - {v.id for v in verify})
            if bad or missing:
                applied["batches_failed"] += 1
                raise RuntimeError(
                    f"post-batch_update validation failed over {len(expected)} points: "
                    f"doc_id did not stick for {bad}; points absent on read-back: {missing}"
                )
            applied["batches_applied"] += 1
        applied["doc_id_points_updated"] += len(expected)
        pending.clear()

    def flush_data_source(ids: List[Any]) -> None:
        if not ids:
            return
        if execute:
            retry_qdrant_operation(
                f"delete_payload(collection={collection} data_source points={len(ids)})",
                lambda: client.delete_payload(
                    collection, keys=["data_source"], points=list(ids), wait=True,
                ),
                max_attempts=max_retries, sleep=sleep,
            )
            verify = client.retrieve(collection, ids=list(ids), with_payload=["data_source"])
            bad = [v.id for v in verify if "data_source" in (v.payload or {})]
            if bad:
                applied["batches_failed"] += 1
                raise RuntimeError(
                    f"post-delete_payload validation failed: data_source "
                    f"still present on point ids {bad}"
                )
            applied["batches_applied"] += 1
        applied["data_source_points_updated"] += len(ids)

    offset = checkpoint.load() if checkpoint else None
    scanned = 0
    started = time.time()
    next_progress = progress_every
    while True:
        points, next_offset = client.scroll(
            collection, limit=scan_batch, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            report.observe(payload)
            scanned += 1

            doc_id = payload.get("doc_id")
            if doc_id is not None and not isinstance(doc_id, str):
                # The type guard runs even when D2 is not selected: a doc_id
                # this script cannot explain is worth stopping for either way.
                key = normalized_doc_id(doc_id, p.id)
                if do_doc_id:
                    pending_doc_id[key].append(p.id)
                    pending_doc_id_points += 1
                    if pending_doc_id_points >= apply_batch:
                        flush_doc_id(pending_doc_id)
                        pending_doc_id_points = 0

            if do_data_source and "data_source" in payload:
                profile = PROFILE_BY_SOURCE_TYPE.get(payload.get("source_type"))
                if profile is not None and "data_source" not in profile:
                    pending_data_source_removal.append(p.id)
                    if len(pending_data_source_removal) >= apply_batch:
                        flush_data_source(pending_data_source_removal[:])
                        pending_data_source_removal.clear()

        if log is not None and scanned >= next_progress:
            elapsed = max(time.time() - started, 1e-9)
            log(
                f"scanned={scanned} elapsed={elapsed:.0f}s rate={scanned / elapsed:.0f}/s "
                f"doc_id_converted={applied['doc_id_points_updated']} "
                f"data_source_removed={applied['data_source_points_updated']} "
                f"batches={applied['batches_applied']}"
            )
            next_progress = scanned + progress_every

        offset = next_offset
        if checkpoint:
            checkpoint.save(offset)
        if offset is None:
            break
        if limit and scanned >= limit:
            break

    flush_doc_id(pending_doc_id)
    pending_doc_id_points = 0
    flush_data_source(pending_data_source_removal)

    if checkpoint and offset is None:
        checkpoint.clear()  # full pass completed — nothing left to resume

    return {
        "report": report.as_dict(),
        "plan": build_plan(report),
        "applied": applied,
    }


def validate_args(args) -> None:
    """Gate writes to the production collection behind an explicit token.

    The gate is on --execute, not on the collection name: sizing production
    read-only is exactly what someone should do first, so a dry run against
    `papers` stays a one-liner. Only the ~6.9M-write run needs the token.
    """
    if args.collection == PRODUCTION_COLLECTION and args.execute:
        if args.confirm_production != PRODUCTION_ACK:
            raise SystemExit(
                f"refusing to --execute against '{PRODUCTION_COLLECTION}': that is the "
                f"production collection (13,556,378 points on 2026-09-02).\n"
                f"Verify against an isolated collection first, then re-run with "
                f"--confirm-production {PRODUCTION_ACK}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument(
        "--collection", required=True,
        help="Target collection. No default; writing to 'papers' additionally "
             "requires --confirm-production.",
    )
    parser.add_argument(
        "--confirm-production", default=None,
        help=f"Required to --execute against '{PRODUCTION_COLLECTION}'. "
             f"Must be exactly: {PRODUCTION_ACK}",
    )
    parser.add_argument(
        "--operations", nargs="+", choices=list(OPERATIONS), default=list(OPERATIONS),
        help="Which fixes to apply. D2 'doc_id' rewrites a type; D3 "
             "'data_source' removes a field. The scan reports both regardless.",
    )
    parser.add_argument("--execute", action="store_true",
                        help="Actually write. Default is dry-run (scan + report only).")
    parser.add_argument("--scan-batch", type=int, default=2000)
    parser.add_argument("--apply-batch", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop scanning after N points (0 = all)")
    parser.add_argument("--checkpoint-file", default=None,
                        help="Resume support: persists the last scanned offset")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=250_000,
                        help="Report scan progress every N points scanned")
    parser.add_argument("--report-path", default=None,
                        help="Also write the JSON result to this path")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)

    from qdrant_client import QdrantClient

    client = QdrantClient(url=args.url, timeout=180)
    mode = "EXECUTE (WRITE)" if args.execute else "DRY-RUN (read-only)"
    print(f"[{mode}] qdrant={args.url} collection={args.collection} "
          f"operations={','.join(args.operations)}", flush=True)

    checkpoint = Checkpoint(Path(args.checkpoint_file)) if args.checkpoint_file else None
    started = time.time()
    result = migrate(
        client, args.collection, execute=args.execute,
        scan_batch=args.scan_batch, apply_batch=args.apply_batch,
        limit=args.limit, checkpoint=checkpoint, max_retries=args.max_retries,
        operations=tuple(args.operations),
        progress_every=args.progress_every,
        log=lambda line: print(f"  .. {line}", flush=True),
    )
    result["elapsed_seconds"] = round(time.time() - started, 1)

    rendered = json.dumps(result, indent=2, default=str)
    print(rendered)
    if args.report_path:
        Path(args.report_path).write_text(rendered, encoding="utf-8")
    if not args.execute:
        print("\nDRY-RUN: nothing modified. Re-run with --execute to apply.")


if __name__ == "__main__":
    main()
