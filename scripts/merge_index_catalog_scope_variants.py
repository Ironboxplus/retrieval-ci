"""Merge Windows-scope case-variant ``index_catalog`` rows into their canonical spelling.

W164/W2026-09-01: writers now canonicalise Windows scopes at write time
(``mcp_server/server.py::_canonical_windows_scope`` / ``canonicalize_scope_path``), but
older rows never got cleaned up. 236 projects, 25 duplicate-case groups measured
2026-09-01 (see docs/index_catalog_scope_and_inventory_plan.md B.1) — the same NTFS
directory shows up as several catalog "projects" (``C:/Users/Arc/Prd/retrival``,
``C:/Users/Arc/prd/retrival``, ``C:/users/arc/prd/retrival``, ...), each with its own
stale ``indexed_at``/file snapshot. Functionally harmless (search already unions legacy
Windows case spellings — ``app/services/code_kb/service.py::_project_filter``) but it
inflates the catalog and confuses the inventory display.

This script ONLY touches ``index_catalog`` rows (never Qdrant vectors, never
``index_scope_policy``/``code_scope_visibility`` rows — see plan.md D). It:

  * reuses ``app.services.scope_paths.canonicalize_scope_path`` for canonical spelling —
    no separate case-folding rule is invented here;
  * NEVER merges POSIX scopes (case-sensitive there — ``/repo/Foo`` != ``/repo/foo``);
  * groups candidate variants by (owner_user_id, host, kind, canonical spelling) — never
    across hosts, which are distinct physical machines;
  * within a group, compares each variant's case-folded rel_path SET; a mismatch means
    the variants do not actually cover the same files, so the whole group is reported
    and SKIPPED rather than silently dropping a file;
  * otherwise keeps the variant with the newest ``indexed_at`` (tie-broken by the
    lexicographically larger project spelling, for determinism), deletes every other
    variant's rows, then renames the survivor's rows to the canonical spelling. Deleting
    losers before renaming the survivor means the canonical spelling is never occupied by
    two variants at once, so this can never trip
    ``uq_index_catalog_file(owner_user_id, host, project, rel_path, kind)``.

Defaults to dry-run (report only). Writing requires BOTH ``--execute`` and
``--confirm-merge`` — a single flag is not enough to mutate the catalog. Not intended to
be run against production in this change; produced for local validation and human
sign-off (plan.md B.2 / D).

Usage::

    python scripts/merge_index_catalog_scope_variants.py                       # dry-run, all owners
    python scripts/merge_index_catalog_scope_variants.py --owner-user-id U     # scoped dry-run
    python scripts/merge_index_catalog_scope_variants.py --execute --confirm-merge
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.services.scope_paths import canonicalize_scope_path, is_windows_scope
from core.db.database import get_db_session
from core.db.models import IndexCatalog

SessionFactory = Callable[[], "Iterator[Session] | Any"]


def _fold_rel_path(rel_path: str) -> str:
    """Same identity fold ``report_files`` uses for Windows scopes (case-insensitive
    NTFS): compare rel_path sets by this key, never by literal string, so a variant
    that only differs by rel_path casing is correctly treated as covering the same file."""
    return rel_path.replace("\\", "/").casefold()


@dataclass
class VariantMember:
    project: str
    file_count: int
    last_indexed_at: datetime | None


@dataclass
class VariantGroup:
    owner_user_id: str
    host: str
    kind: str
    canonical: str
    members: list[VariantMember]
    status: str  # "merge" | "conflict"
    keep: str | None = None
    rows_deleted: int = 0
    conflict_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_user_id": self.owner_user_id,
            "host": self.host,
            "kind": self.kind,
            "canonical": self.canonical,
            "status": self.status,
            "keep": self.keep,
            "rows_deleted": self.rows_deleted,
            "conflict_detail": self.conflict_detail,
            "members": [
                {
                    "project": m.project,
                    "file_count": m.file_count,
                    "last_indexed_at": m.last_indexed_at.isoformat() if m.last_indexed_at else None,
                }
                for m in self.members
            ],
        }


@dataclass
class MergeReport:
    executed: bool
    groups: list[VariantGroup] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"executed": self.executed, "groups": [g.to_dict() for g in self.groups]}


def _distinct_project_scopes(
    session: Session, *, owner_user_id: str | None, host: str | None, kind: str | None,
) -> list[dict[str, Any]]:
    """(owner_user_id, host, kind, project) candidate scopes with file_count + freshness,
    for non-deleted rows only — the small candidate universe for variant grouping, not
    a per-file scan."""
    query = session.query(
        IndexCatalog.owner_user_id, IndexCatalog.host, IndexCatalog.kind, IndexCatalog.project,
        func.count(IndexCatalog.id).label("file_count"),
        func.max(IndexCatalog.indexed_at).label("last_indexed_at"),
    ).filter(IndexCatalog.deleted_at.is_(None))
    if owner_user_id is not None:
        query = query.filter(IndexCatalog.owner_user_id == owner_user_id)
    if host is not None:
        query = query.filter(IndexCatalog.host == host)
    if kind is not None:
        query = query.filter(IndexCatalog.kind == kind)
    rows = query.group_by(
        IndexCatalog.owner_user_id, IndexCatalog.host, IndexCatalog.kind, IndexCatalog.project,
    ).all()
    return [
        {"owner_user_id": r[0], "host": r[1], "kind": r[2], "project": r[3],
         "file_count": int(r[4] or 0), "last_indexed_at": r[5]}
        for r in rows
    ]


def _rel_path_identity_set(
    session: Session, *, owner_user_id: str, host: str, kind: str, project: str,
) -> set[str]:
    rows = session.query(IndexCatalog.rel_path).filter(
        IndexCatalog.owner_user_id == owner_user_id,
        IndexCatalog.host == host,
        IndexCatalog.kind == kind,
        IndexCatalog.project == project,
        IndexCatalog.deleted_at.is_(None),
    ).all()
    return {_fold_rel_path(r[0]) for r in rows}


def detect_variant_groups(
    session_factory: SessionFactory = get_db_session,
    *,
    owner_user_id: str | None = None,
    host: str | None = None,
    kind: str | None = None,
) -> list[VariantGroup]:
    """Find case-variant project groups. Read-only — never writes."""
    with session_factory() as session:
        scopes = _distinct_project_scopes(
            session, owner_user_id=owner_user_id, host=host, kind=kind)

        buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for row in scopes:
            project = row["project"] or ""
            if not project:
                continue
            canonical = canonicalize_scope_path(project)
            if not is_windows_scope(canonical):
                continue  # POSIX scopes are case-sensitive — never merged (plan.md B.2)
            key = (row["owner_user_id"], row["host"], row["kind"], canonical)
            buckets.setdefault(key, []).append(row)

        groups: list[VariantGroup] = []
        for (owner, host_name, kind_name, canonical), members in sorted(buckets.items()):
            if len(members) < 2:
                continue  # no case-variant duplicate — nothing to report
            member_infos = [
                VariantMember(project=m["project"], file_count=m["file_count"],
                              last_indexed_at=m["last_indexed_at"])
                for m in sorted(members, key=lambda m: m["project"])
            ]
            identity_sets = {
                m["project"]: _rel_path_identity_set(
                    session, owner_user_id=owner, host=host_name, kind=kind_name,
                    project=m["project"])
                for m in members
            }
            distinct_sets = {frozenset(s) for s in identity_sets.values()}
            if len(distinct_sets) > 1:
                mismatch = "; ".join(
                    f"{project!r} has {sorted(paths)}" for project, paths in identity_sets.items())
                groups.append(VariantGroup(
                    owner_user_id=owner, host=host_name, kind=kind_name, canonical=canonical,
                    members=member_infos, status="conflict",
                    conflict_detail=f"rel_path sets differ across variants: {mismatch}",
                ))
                continue
            winner = max(
                members, key=lambda m: (m["last_indexed_at"] or datetime.min, m["project"]))
            groups.append(VariantGroup(
                owner_user_id=owner, host=host_name, kind=kind_name, canonical=canonical,
                members=member_infos, status="merge", keep=winner["project"],
                rows_deleted=sum(
                    m["file_count"] for m in members if m["project"] != winner["project"]),
            ))
        return groups


def apply_merge(session_factory: SessionFactory, group: VariantGroup) -> int:
    """Delete every losing variant's rows, then rename the surviving variant's rows to
    the canonical spelling. Returns the number of rows deleted.

    Delete-before-rename is load-bearing: it guarantees the canonical spelling is never
    occupied by two variants' rows at the same time, so the rename can never trip
    uq_index_catalog_file(owner_user_id, host, project, rel_path, kind) — even when the
    canonical spelling already belongs to one of the LOSING (older) variants.
    """
    if group.status != "merge" or group.keep is None:
        raise ValueError(f"cannot apply a non-merge group (status={group.status!r})")
    deleted = 0
    with session_factory() as session:
        for member in group.members:
            if member.project == group.keep:
                continue
            deleted += session.query(IndexCatalog).filter(
                IndexCatalog.owner_user_id == group.owner_user_id,
                IndexCatalog.host == group.host,
                IndexCatalog.kind == group.kind,
                IndexCatalog.project == member.project,
            ).delete(synchronize_session=False)
        if group.keep != group.canonical:
            session.query(IndexCatalog).filter(
                IndexCatalog.owner_user_id == group.owner_user_id,
                IndexCatalog.host == group.host,
                IndexCatalog.kind == group.kind,
                IndexCatalog.project == group.keep,
            ).update({"project": group.canonical}, synchronize_session=False)
    return deleted


def run(
    session_factory: SessionFactory = get_db_session,
    *,
    owner_user_id: str | None = None,
    host: str | None = None,
    kind: str | None = None,
    execute: bool = False,
) -> MergeReport:
    """Detect, report, and (only if ``execute``) apply every mergeable group.

    Conflicting groups are always skipped, regardless of ``execute`` — they are never
    silently written to.
    """
    groups = detect_variant_groups(
        session_factory, owner_user_id=owner_user_id, host=host, kind=kind)
    if execute:
        for group in groups:
            if group.status == "merge":
                group.rows_deleted = apply_merge(session_factory, group)
    return MergeReport(executed=execute, groups=groups)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge Windows-scope case-variant index_catalog rows into their "
                    "canonical spelling. Dry-run by default; see plan.md B for the spec.")
    parser.add_argument("--owner-user-id", default=None, help="Restrict to one owner.")
    parser.add_argument("--host", default=None, help="Restrict to one host.")
    parser.add_argument("--kind", default=None, help="Restrict to one kind (code/doc).")
    parser.add_argument("--execute", action="store_true",
                        help="Write changes. Requires --confirm-merge too.")
    parser.add_argument("--confirm-merge", action="store_true",
                        help="Second, explicit confirmation required alongside --execute.")
    args = parser.parse_args(argv)

    if args.execute and not args.confirm_merge:
        parser.error("--execute also requires --confirm-merge (this deletes/renames "
                     "index_catalog rows) — refusing to write without both flags")

    report = run(
        owner_user_id=args.owner_user_id, host=args.host, kind=args.kind,
        execute=args.execute and args.confirm_merge,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    merge_count = sum(1 for g in report.groups if g.status == "merge")
    conflict_count = sum(1 for g in report.groups if g.status == "conflict")
    mode = "EXECUTED" if report.executed else "DRY-RUN"
    print(f"\n[{mode}] {len(report.groups)} variant group(s): "
         f"{merge_count} merge(able), {conflict_count} conflict(ing, skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
