"""Drop the vectors and catalog rows of NON-canonical Windows scope spellings.

Why these rows exist
--------------------
Until 2026-09-01 (`4e94dff`) nothing case-folded an index scope on the write
path: `CodeIndexRequest.project` stored whatever the client typed, and
`normalize_scope` lower-cased only the drive letter, for ancestor comparison
only. So `C:\\Users\\Arc\\Prd\\retrival` and `C:\\Users\\Arc\\prd\\retrival`
became two independent scopes over one NTFS directory.

`4e94dff` added `canonicalize_scope_path` as a write-time validator, and it
reached production in the 2026-09-02 rollout. Writes now land on the fully
case-folded spelling — which for an existing repo means a *third* copy appears
the first time it is re-indexed, because canonicalising future writes does not
migrate past ones. That commit shipped
`scripts/merge_index_catalog_scope_variants.py` for exactly this, but it only
touches `index_catalog`, never Qdrant, and it skips any group whose variants
cover different rel_path sets — which is every group that kept being indexed.

Why deleting, not merging
-------------------------
The canonical spelling is re-indexed by the live client and is the most complete
copy; the stale spellings are frozen at whatever the last pre-canonicalisation
run left. Merging catalog rows from a frozen variant would advertise files whose
vectors are gone. Deleting the variant outright leaves exactly one truth.

Safety
------
* Dry-run by default; `--execute` additionally requires `--confirm-drop`.
* Refuses to touch a group unless the canonical spelling itself is present and
  non-empty. Several production groups have NO canonical member yet (every
  spelling is legacy), and there "drop the non-canonical ones" would mean "drop
  the whole repository".
* Names exact scope strings. `vector_store.delete_project` matches (owner,
  project) exactly, so alias expansion — which the DELETE /api/code route
  deliberately performs, and which would take the live copy with it — never runs.
* Counts before and after, and fails loudly if the canonical count moved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.code_kb.vector_store import CodeVectorStore  # noqa: E402
from app.services.index_catalog.service import get_index_catalog_service  # noqa: E402
from app.services.scope_paths import canonicalize_scope_path, is_windows_scope  # noqa: E402


def _count(store: CodeVectorStore, *, owner: str, project: str) -> int:
    qf = store.store._build_filter({"owner_user_id": owner, "project": project})
    return store.store.client.count(
        store.collection_name, count_filter=qf, exact=True).count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--canonical", required=True,
                    help="the spelling to KEEP; must equal canonicalize_scope_path(itself)")
    ap.add_argument("--drop", required=True, nargs="+",
                    help="exact stale spellings to remove")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--confirm-drop", action="store_true")
    args = ap.parse_args()

    canonical = args.canonical
    if not is_windows_scope(canonical):
        return _fail("only Windows scopes are case-variant; POSIX scopes are distinct paths")
    if canonicalize_scope_path(canonical) != canonical:
        return _fail(f"--canonical is not canonical; expected "
                     f"{canonicalize_scope_path(canonical)!r}")

    store = CodeVectorStore()
    keep_before = _count(store, owner=args.owner, project=canonical)
    print(f"KEEP    {canonical!r}: {keep_before} points")
    if keep_before == 0:
        return _fail("the canonical spelling holds no points; refusing to drop the "
                     "others, which would leave this scope unindexed")

    targets = []
    for scope in args.drop:
        if scope == canonical:
            return _fail(f"--drop lists the canonical spelling {scope!r}")
        if canonicalize_scope_path(scope) != canonical:
            return _fail(f"{scope!r} canonicalises to "
                         f"{canonicalize_scope_path(scope)!r}, not {canonical!r}; "
                         "it is a different directory, not a case variant")
        n = _count(store, owner=args.owner, project=scope)
        print(f"DROP    {scope!r}: {n} points")
        targets.append((scope, n))

    if not (args.execute and args.confirm_drop):
        print("\nDRY RUN — nothing written. Re-run with --execute --confirm-drop.")
        return 0

    catalog = get_index_catalog_service()
    removed_points = removed_rows = 0
    for scope, expected in targets:
        deleted = store.delete_project(owner_user_id=args.owner, project=scope)
        # kind="code" only: the same scope string can also carry journal document
        # rows, and this script is not authorised to touch those. host=None is
        # correct and deliberate — code vectors are host-agnostic (W148), so the
        # rows for one scope may sit under several hosts.
        rows = catalog.repository.delete_files(
            owner_user_id=args.owner, project=scope, host=None, kind="code",
            rel_paths=None)
        removed_points += deleted
        removed_rows += rows
        print(f"dropped {scope!r}: {deleted} points, {rows} catalog rows")
        left = _count(store, owner=args.owner, project=scope)
        if left:
            return _fail(f"{scope!r} still holds {left} points after delete")

    keep_after = _count(store, owner=args.owner, project=canonical)
    print(f"\nremoved {removed_points} points and {removed_rows} catalog rows")
    print(f"KEEP    {canonical!r}: {keep_before} -> {keep_after} points")
    if keep_after != keep_before:
        return _fail("the canonical scope's count changed; a delete escaped its filter")
    print("OK")
    return 0


def _fail(message: str) -> int:
    print(f"ABORT: {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
