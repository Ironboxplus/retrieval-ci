"""TDD for scripts/merge_index_catalog_scope_variants.py (W2026-09-01 B).

Real SQLite, real IndexCatalog rows — no mocks. See
docs/index_catalog_scope_and_inventory_plan.md section B for the spec: dry-run by
default, canonical spelling reused from app.services.scope_paths, POSIX scopes never
merged, conflicting rel_path sets reported and skipped, uq_index_catalog_file respected.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from core.db.models import IndexCatalog


def _session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        session = session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return factory


def _insert(session_factory, *, owner_user_id, host, project, kind, rel_path,
            indexed_at, content_hash=None, chunk_count=1):
    with session_factory() as session:
        session.add(IndexCatalog(
            owner_user_id=owner_user_id, host=host, project=project, kind=kind,
            root_dir=project, rel_path=rel_path, collection="code_chunks",
            chunk_count=chunk_count, content_hash=content_hash,
            indexed_at=indexed_at,
        ))


def _all_rows(session_factory):
    with session_factory() as session:
        rows = session.query(IndexCatalog).all()
        out = [
            {"owner_user_id": r.owner_user_id, "host": r.host, "project": r.project,
             "kind": r.kind, "rel_path": r.rel_path, "content_hash": r.content_hash}
            for r in rows
        ]
        for r in rows:
            session.expunge(r)
        return out


T0 = datetime(2026, 1, 1, 0, 0, 0)


def test_merges_windows_case_variants_keeping_newest_and_renaming_to_canonical():
    from scripts.merge_index_catalog_scope_variants import detect_variant_groups, apply_merge

    sf = _session_factory()
    _insert(sf, owner_user_id="u1", host="desktop", project="C:/Users/Arc/Prd/retrival",
            kind="code", rel_path="a.py", indexed_at=T0, content_hash="old-hash")
    _insert(sf, owner_user_id="u1", host="desktop", project="c:/users/arc/prd/retrival",
            kind="code", rel_path="a.py", indexed_at=T0 + timedelta(days=30),
            content_hash="new-hash")

    groups = detect_variant_groups(sf, owner_user_id="u1")
    assert len(groups) == 1
    group = groups[0]
    assert group.status == "merge"
    assert group.canonical == "C:/users/arc/prd/retrival"
    assert group.keep == "c:/users/arc/prd/retrival"  # newest indexed_at wins
    assert group.rows_deleted == 1
    assert {m.project for m in group.members} == {
        "C:/Users/Arc/Prd/retrival", "c:/users/arc/prd/retrival"}

    deleted = apply_merge(sf, group)
    assert deleted == 1

    rows = _all_rows(sf)
    assert len(rows) == 1
    assert rows[0]["project"] == "C:/users/arc/prd/retrival"
    assert rows[0]["content_hash"] == "new-hash"  # newest variant's data survived


def test_dry_run_detects_but_does_not_write():
    from scripts.merge_index_catalog_scope_variants import run

    sf = _session_factory()
    _insert(sf, owner_user_id="u1", host="desktop", project="C:/Repo",
            kind="code", rel_path="x.py", indexed_at=T0)
    _insert(sf, owner_user_id="u1", host="desktop", project="c:/repo",
            kind="code", rel_path="x.py", indexed_at=T0 + timedelta(days=1))

    report = run(sf, owner_user_id="u1", execute=False)
    assert report.executed is False
    assert len(report.groups) == 1
    assert report.groups[0].status == "merge"
    assert report.groups[0].rows_deleted == 1  # reported, but nothing written yet

    rows = _all_rows(sf)
    assert len(rows) == 2  # untouched
    assert {r["project"] for r in rows} == {"C:/Repo", "c:/repo"}


def test_execute_true_applies_all_mergeable_groups():
    from scripts.merge_index_catalog_scope_variants import run

    sf = _session_factory()
    _insert(sf, owner_user_id="u1", host="desktop", project="C:/Repo",
            kind="code", rel_path="x.py", indexed_at=T0)
    _insert(sf, owner_user_id="u1", host="desktop", project="c:/repo",
            kind="code", rel_path="x.py", indexed_at=T0 + timedelta(days=1))

    report = run(sf, owner_user_id="u1", execute=True)
    assert report.executed is True
    assert report.groups[0].status == "merge"
    assert report.groups[0].rows_deleted == 1

    rows = _all_rows(sf)
    assert len(rows) == 1
    assert rows[0]["project"] == "C:/repo"


def test_conflicting_rel_path_sets_are_reported_and_skipped():
    from scripts.merge_index_catalog_scope_variants import run

    sf = _session_factory()
    # Same canonical scope, but the two variants indexed DIFFERENT files — merging
    # would silently drop whichever variant's unique file is on the losing side.
    _insert(sf, owner_user_id="u1", host="desktop", project="C:/Repo",
            kind="code", rel_path="only_in_old.py", indexed_at=T0)
    _insert(sf, owner_user_id="u1", host="desktop", project="c:/repo",
            kind="code", rel_path="only_in_new.py", indexed_at=T0 + timedelta(days=1))

    report = run(sf, owner_user_id="u1", execute=True)  # even with execute=True
    assert len(report.groups) == 1
    group = report.groups[0]
    assert group.status == "conflict"
    assert group.conflict_detail  # non-empty explanation
    assert group.rows_deleted == 0

    rows = _all_rows(sf)
    assert len(rows) == 2  # both variants preserved — nothing silently dropped
    assert {r["project"] for r in rows} == {"C:/Repo", "c:/repo"}


def test_posix_scopes_are_never_merged_even_when_only_case_differs():
    from scripts.merge_index_catalog_scope_variants import detect_variant_groups

    sf = _session_factory()
    _insert(sf, owner_user_id="u1", host="linuxbox", project="/repo/Foo",
            kind="code", rel_path="a.py", indexed_at=T0)
    _insert(sf, owner_user_id="u1", host="linuxbox", project="/repo/foo",
            kind="code", rel_path="a.py", indexed_at=T0 + timedelta(days=1))

    groups = detect_variant_groups(sf, owner_user_id="u1")
    assert groups == []  # /repo/Foo and /repo/foo are DISTINCT POSIX directories


def test_merge_respects_unique_constraint_when_canonical_spelling_is_the_older_loser():
    """The exact canonical spelling already has rows (older, so it LOSES); a
    differently-cased newer variant WINS and must be renamed to that same canonical
    spelling. Renaming before deleting the loser would collide with
    uq_index_catalog_file; this proves delete-then-rename ordering avoids that."""
    from scripts.merge_index_catalog_scope_variants import detect_variant_groups, apply_merge

    sf = _session_factory()
    # This literal spelling IS already the canonical one.
    _insert(sf, owner_user_id="u1", host="desktop", project="C:/repo/x",
            kind="code", rel_path="a.py", indexed_at=T0, content_hash="canonical-old")
    # A differently-cased, newer variant with the SAME logical file.
    _insert(sf, owner_user_id="u1", host="desktop", project="C:/REPO/X",
            kind="code", rel_path="a.py", indexed_at=T0 + timedelta(days=1),
            content_hash="variant-new")

    groups = detect_variant_groups(sf, owner_user_id="u1")
    assert len(groups) == 1
    group = groups[0]
    assert group.canonical == "C:/repo/x"
    assert group.keep == "C:/REPO/X"  # the newer, non-canonically-spelled variant wins

    apply_merge(sf, group)  # must not raise IntegrityError

    rows = _all_rows(sf)
    assert len(rows) == 1
    assert rows[0]["project"] == "C:/repo/x"
    assert rows[0]["content_hash"] == "variant-new"


def test_report_includes_members_file_counts_and_last_indexed_at():
    from scripts.merge_index_catalog_scope_variants import run

    sf = _session_factory()
    _insert(sf, owner_user_id="u1", host="desktop", project="C:/Repo",
            kind="code", rel_path="a.py", indexed_at=T0)
    _insert(sf, owner_user_id="u1", host="desktop", project="C:/Repo",
            kind="code", rel_path="b.py", indexed_at=T0)
    _insert(sf, owner_user_id="u1", host="desktop", project="c:/repo",
            kind="code", rel_path="a.py", indexed_at=T0 + timedelta(days=2))
    _insert(sf, owner_user_id="u1", host="desktop", project="c:/repo",
            kind="code", rel_path="b.py", indexed_at=T0 + timedelta(days=2))

    report = run(sf, owner_user_id="u1", execute=False)
    as_dict = report.to_dict()
    assert len(as_dict["groups"]) == 1
    g = as_dict["groups"][0]
    members_by_project = {m["project"]: m for m in g["members"]}
    assert members_by_project["C:/Repo"]["file_count"] == 2
    assert members_by_project["c:/repo"]["file_count"] == 2
    assert members_by_project["C:/Repo"]["last_indexed_at"] is not None
    assert members_by_project["c:/repo"]["last_indexed_at"] is not None
    assert g["rows_deleted"] == 2
    assert g["keep"] == "c:/repo"


def test_scoping_by_owner_host_kind_isolates_groups():
    from scripts.merge_index_catalog_scope_variants import detect_variant_groups

    sf = _session_factory()
    _insert(sf, owner_user_id="u1", host="desktop", project="C:/Repo",
            kind="code", rel_path="a.py", indexed_at=T0)
    _insert(sf, owner_user_id="u1", host="desktop", project="c:/repo",
            kind="code", rel_path="a.py", indexed_at=T0 + timedelta(days=1))
    # Different owner: same canonical scope must NOT be merged into u1's group.
    _insert(sf, owner_user_id="u2", host="desktop", project="C:/Repo",
            kind="code", rel_path="a.py", indexed_at=T0)
    _insert(sf, owner_user_id="u2", host="desktop", project="c:/repo",
            kind="code", rel_path="a.py", indexed_at=T0 + timedelta(days=1))

    groups = detect_variant_groups(sf, owner_user_id="u1")
    assert len(groups) == 1
    assert groups[0].owner_user_id == "u1"

    all_groups = detect_variant_groups(sf)
    assert len(all_groups) == 2
    assert {g.owner_user_id for g in all_groups} == {"u1", "u2"}


def test_cli_requires_confirm_merge_alongside_execute(capsys):
    from scripts.merge_index_catalog_scope_variants import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--execute"])
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "--confirm-merge" in captured.err
