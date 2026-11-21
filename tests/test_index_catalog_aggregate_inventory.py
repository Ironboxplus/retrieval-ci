"""TDD for the SUMMARY (detail=False) inventory aggregation query (W2026-09-01 A).

Real SQLite, thousands of IndexCatalog rows across owners/hosts/projects/kinds — no
mocks. ``list_inventory()`` downloads every ORM entity (208,718 rows in production for
one 8s /api/index/inventory call); ``aggregate_inventory()`` must do the same summing in
SQL and return ~project-count rows instead. See docs/index_catalog_scope_and_inventory_plan.md.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from core.db.models import IndexCatalog
from app.services.index_catalog.repository import IndexCatalogRepository
from app.services.index_catalog.schemas import IndexFileRecord, IndexReportRequest
from app.services.index_catalog.service import IndexCatalogService


def _service():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return IndexCatalogService(repository=IndexCatalogRepository(session_factory=session_factory))


N_OWNERS = 5
N_HOSTS = 3
N_PROJECTS_PER_OWNER = 20
N_FILES_PER_PROJECT = 20  # 5 * 20 * 20 = 2000 code rows, plus doc rows below → thousands


def _seed_bulk_catalog(svc: IndexCatalogService) -> None:
    """Populate thousands of realistic rows spanning many owners/hosts/projects/kinds,
    with STRICTLY INCREASING indexed_at (via successive report() calls) and a distinct
    git_commit per project so freshness/last_commit equivalence is unambiguous (no ties)."""
    for oi in range(N_OWNERS):
        owner = f"user-{oi}"
        for hi in range(N_HOSTS):
            host = f"host-{hi}"
            for pi in range(N_PROJECTS_PER_OWNER):
                project = f"/repo/owner{oi}/host{hi}/proj{pi}"
                # Two reports per project: an older commit, then a newer one touching a
                # different file — mirrors test_inventory_last_commit_pairs_with_freshest_file.
                svc.report(IndexReportRequest(
                    host=host, project=project, root_dir=project, replace=False,
                    files=[IndexFileRecord(
                        rel_path=f"pkg/f{i}.py", kind="code", collection="code_chunks",
                        chunk_count=3, content_hash=f"hash{oi}{hi}{pi}{i}",
                        metadata={"git_commit": f"old{oi}{hi}{pi}"},
                    ) for i in range(N_FILES_PER_PROJECT - 1)],
                ), owner_user_id=owner)
                svc.report(IndexReportRequest(
                    host=host, project=project, root_dir=project, replace=False,
                    files=[IndexFileRecord(
                        rel_path=f"pkg/z_newest.py", kind="code", collection="code_chunks",
                        chunk_count=7, content_hash=f"hashnewest{oi}{hi}{pi}",
                        metadata={"git_commit": f"new{oi}{hi}{pi}"},
                    )],
                ), owner_user_id=owner)
                # One doc-kind row per project too, so aggregate grouping must span kinds.
                svc.report(IndexReportRequest(
                    host=host, project=project, root_dir=project, replace=False,
                    files=[IndexFileRecord(
                        rel_path="README.md", kind="doc", collection="journal_chunks",
                        chunk_count=1,
                    )],
                ), owner_user_id=owner)


def test_aggregate_inventory_returns_far_fewer_rows_than_full_scan():
    svc = _service()
    _seed_bulk_catalog(svc)

    full_rows = svc.repository.list_inventory(owner_user_id="user-0")
    agg_rows = svc.repository.aggregate_inventory(owner_user_id="user-0")

    total_file_rows = N_HOSTS * N_PROJECTS_PER_OWNER * (N_FILES_PER_PROJECT + 1)  # +doc row
    assert len(full_rows) == total_file_rows
    # One aggregate row per (host, project, kind): code + doc kinds → 2 per project.
    expected_agg_rows = N_HOSTS * N_PROJECTS_PER_OWNER * 2
    assert len(agg_rows) == expected_agg_rows
    assert len(agg_rows) < len(full_rows) / 5, (
        f"aggregate path fetched {len(agg_rows)} rows vs {len(full_rows)} full rows — "
        "not a meaningful reduction"
    )


def test_aggregate_inventory_rows_are_plain_dicts_not_orm_entities():
    svc = _service()
    _seed_bulk_catalog(svc)
    agg_rows = svc.repository.aggregate_inventory(owner_user_id="user-1")
    assert agg_rows, "expected at least one aggregate row"
    for row in agg_rows:
        assert isinstance(row, dict)
        assert not isinstance(row, IndexCatalog)


def test_inventory_summary_matches_detail_project_fields_exactly():
    """The whole point of A: SUMMARY output must be field-for-field identical to the
    DETAIL path at the project level (file_count/chunk_count/freshness/commit/code
    snapshot state) — only the (heavy) per-dir/file tree differs."""
    svc = _service()
    _seed_bulk_catalog(svc)

    summary = svc.inventory(owner_user_id="user-2", kind="code", detail=False)
    detail = svc.inventory(owner_user_id="user-2", kind="code", detail=True)

    def project_index(inv):
        return {
            (h.host, p.project): p
            for u in inv.users for h in u.hosts for p in h.projects
        }

    summary_projects = project_index(summary)
    detail_projects = project_index(detail)
    assert summary_projects.keys() == detail_projects.keys()
    assert len(summary_projects) == N_HOSTS * N_PROJECTS_PER_OWNER

    for key, sp in summary_projects.items():
        dp = detail_projects[key]
        assert sp.project == dp.project
        assert sp.host == dp.host
        assert sp.root_dir == dp.root_dir
        assert sp.file_count == dp.file_count, key
        assert sp.chunk_count == dp.chunk_count, key
        assert sp.last_indexed_at == dp.last_indexed_at, key
        assert sp.last_commit == dp.last_commit, key
        assert sp.code_snapshot_state == dp.code_snapshot_state, key
        assert sp.code_active_index_job_id == dp.code_active_index_job_id, key
        assert sp.code_previous_completed_snapshot_available == (
            dp.code_previous_completed_snapshot_available), key
        assert sp.code_recovery_required == dp.code_recovery_required, key
        # This is the one field that's SUPPOSED to differ.
        assert sp.dirs == []
        assert dp.dirs != []

    assert summary.totals == detail.totals


def test_inventory_summary_matches_detail_totals_across_full_multi_owner_dataset():
    svc = _service()
    _seed_bulk_catalog(svc)
    summary = svc.inventory(detail=False)
    detail = svc.inventory(detail=True)
    assert summary.totals == detail.totals
    assert summary.totals["files"] == N_OWNERS * N_HOSTS * N_PROJECTS_PER_OWNER * (
        N_FILES_PER_PROJECT + 1)


def test_summary_path_never_falls_back_to_full_per_file_scan():
    """Structural regression guard: detail=False must go through aggregate_inventory()
    only. If list_inventory() (the O(all files) path) is ever invoked for a summary
    call, that reintroduces the exact perf bug this change fixes."""
    svc = _service()
    _seed_bulk_catalog(svc)

    def _forbidden(*args, **kwargs):
        raise AssertionError("detail=False must not call the O(all files) list_inventory()")

    svc.repository.list_inventory = _forbidden
    # Must not raise.
    svc.inventory(owner_user_id="user-3", kind="code", detail=False)


def test_aggregate_inventory_root_dir_and_last_commit_use_freshest_row():
    svc = _service()
    svc.report(IndexReportRequest(
        host="h", project="/repo/p", root_dir="/repo/p", replace=False,
        files=[IndexFileRecord(rel_path="a.py", kind="code", collection="code_chunks",
                               chunk_count=1, metadata={"git_commit": "old111"})]),
        owner_user_id="u1")
    svc.report(IndexReportRequest(
        host="h", project="/repo/p", root_dir="/repo/p", replace=False,
        files=[IndexFileRecord(rel_path="z_newer.py", kind="code", collection="code_chunks",
                               chunk_count=1, metadata={"git_commit": "new999"})]),
        owner_user_id="u1")

    agg_rows = svc.repository.aggregate_inventory(owner_user_id="u1")
    assert len(agg_rows) == 1
    row = agg_rows[0]
    assert row["root_dir"] == "/repo/p"
    assert row["last_commit"] == "new999"
    assert row["file_count"] == 2
    assert row["chunk_count"] == 2


# --- indexed_at ties -------------------------------------------------------------
# One index run stamps a whole batch with a single timestamp, so ties on
# MAX(indexed_at) are the normal case, not an exotic one. The seeding above uses
# successive report() calls and so has STRICTLY INCREASING timestamps — it cannot
# catch a tie-break disagreement. These rows are inserted directly, with equal
# indexed_at and differing git_commit, to pin the arbitration rule on both paths.
#
# Regression: SUMMARY originally read root_dir/metadata_json as bare columns beside
# MAX(indexed_at). SQLite resolves those from "the" max row, but on a tie any tied
# row may win, so DETAIL and SUMMARY reported different last_commit for the same
# project. DETAIL's rule is "smallest rel_path among rows tied at the newest
# indexed_at"; aggregate_inventory() now reproduces it with ROW_NUMBER().

TIE_TS = datetime(2026, 9, 1, 12, 0, 0)


def _service_with_rows(rows):
    """Service over a real SQLite DB seeded with exact (kind, rel_path, commit) rows."""
    svc = _service()
    with svc.repository.session() as session:
        for project, kind, rel_path, commit in rows:
            session.add(IndexCatalog(
                owner_user_id="u1", host="h1", project=project, kind=kind,
                rel_path=rel_path, root_dir=f"/r/{project}", collection="code_chunks",
                chunk_count=1, indexed_at=TIE_TS, metadata_json={"git_commit": commit},
            ))
    return svc


def _commits(svc, detail):
    inv = svc.inventory(owner_user_id="u1", detail=detail)
    return {
        p.project: (p.file_count, p.chunk_count, p.last_indexed_at, p.last_commit)
        for u in inv.users for h in u.hosts for p in h.projects
    }


@pytest.mark.parametrize("label,rows,expected_commit", [
    # Insertion order reversed against rel_path order: SQLite's bare-column pick and
    # DETAIL's rel_path scan disagreed here before the fix.
    ("within-kind tie", [
        ("Q", "code", "z_dir/f.txt", "zzzcommit"),
        ("Q", "code", "a_dir/f.txt", "aaacommit"),
    ], "aaacommit"),
    # 'code' sorts before 'doc', but 'b.md' sorts before 'z.py' — merging the per-kind
    # aggregate rows in kind order would pick codeX while DETAIL picks docY.
    ("cross-kind tie", [
        ("R", "code", "z.py", "codeX"),
        ("R", "doc", "b.md", "docY"),
    ], "docY"),
    # Sorting by dir name is not sorting by rel_path: 'a-b/c.py' < 'a/z.py' by path
    # ('-' < '/'), but dir 'a' < dir 'a-b'. DETAIL now derives freshness from the
    # rel_path-ordered stream rather than the dir-grouped loop.
    ("dir-order vs path-order tie", [
        ("S", "code", "a/z.py", "slashcommit"),
        ("S", "code", "a-b/c.py", "dashcommit"),
    ], "dashcommit"),
])
def test_summary_and_detail_agree_on_last_commit_when_indexed_at_ties(
    label, rows, expected_commit,
):
    svc = _service_with_rows(rows)
    detail = _commits(svc, detail=True)
    summary = _commits(svc, detail=False)
    project = rows[0][0]
    assert detail == summary, f"{label}: SUMMARY diverged from DETAIL"
    assert detail[project][3] == expected_commit, label
