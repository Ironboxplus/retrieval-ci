"""`/api/index/inventory` at production size: ~583K live rows, ~341 scopes, 3 hosts, no
``sqlite_stat1`` (the .88 catalog; measured 2026-11-09: the compact summary took 5.8 s median
and 18 s worst, the MCP call shape was another full-owner scan).

Real SQLite file, the repository's own indexes and the service the route calls. Profile on the
fixture before the fix (this machine): summary 5.2 s, of which 4.5 s was the ROW_NUMBER winners
query sorting every row of the owner and 0.55 s the aggregate reading the table through a
non-covering index; the MCP shape read the owner's whole scope index for ORDER BY.
"""
from __future__ import annotations

import statistics
import time

import pytest

from tests.catalog_sqlite import CatalogDb, capture_catalog_plans, full_scans, unbounded_reads

OWNER = "internal-admin"
ROWS = 583_000
SCOPES = 341
PER_SCOPE = ROWS // SCOPES
SUMMARY_BUDGET_MS = 500


@pytest.fixture(scope="module")
def production_catalog(tmp_path_factory):
    catalog = CatalogDb(path=tmp_path_factory.mktemp("inventory") / "catalog.db")
    catalog.production_shaped(owner=OWNER, rows=ROWS, scopes=SCOPES, stamped=True)
    # Two more hosts, a doc kind in an already indexed scope, and tombstones.
    catalog.add_files("E:/work/extra-b", [f"src/b{i}.py" for i in range(300)],
                      owner=OWNER, host="host-b", chunk_count=2)
    catalog.add_files("E:/work/extra-c", [f"docs/c{i}.md" for i in range(300)],
                      owner=OWNER, host="host-c", kind="doc")
    catalog.add_files("E:/work/scope007", [f"notes/n{i}.md" for i in range(50)],
                      owner=OWNER, kind="doc")
    catalog.add_files("E:/work/scope007", ["gone/a.py", "gone/b.py"], owner=OWNER, deleted=True)
    return catalog


def _median_ms(call, runs: int = 5) -> float:
    call()  # warm the page cache and the statement cache
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def _summary(catalog):
    return catalog.service.inventory(owner_user_id=OWNER, detail=False)


def _mcp_shape(catalog, scope="E:/work/scope007"):
    return catalog.service.inventory(
        owner_user_id=OWNER, project=scope, kind="code", detail=True, lean=True)


def test_fixture_has_the_production_shape(production_catalog):
    with production_catalog.engine.connect() as conn:
        assert conn.exec_driver_sql(
            "SELECT count(*) FROM index_catalog WHERE deleted_at IS NULL").scalar() >= ROWS
        assert conn.exec_driver_sql(
            "SELECT count(*) FROM sqlite_master WHERE name = 'sqlite_stat1'").scalar() == 0
        assert conn.exec_driver_sql("SELECT count(DISTINCT host) FROM index_catalog").scalar() == 3


def test_compact_summary_is_correct(production_catalog):
    inventory = _summary(production_catalog)
    projects = {(h.host, p.project): p for u in inventory.users for h in u.hosts for p in h.projects}
    assert inventory.totals["hosts"] == 3
    assert inventory.totals["projects"] == SCOPES + 2
    assert inventory.totals["files"] == SCOPES * PER_SCOPE + 300 + 300 + 50  # tombstones excluded
    scope7 = projects[("host", "E:/work/scope007")]
    assert scope7.file_count == PER_SCOPE + 50
    assert scope7.chunk_count == PER_SCOPE + 50
    assert scope7.root_dir == "E:/work/scope007"
    scope12 = projects[("host", "E:/work/scope012")]
    assert scope12.last_commit == "c012" and scope12.root_dir == "E:/work/scope012"
    assert scope12.last_indexed_at.isoformat().startswith("2026-11-04T12:00:00")
    # The per-file tree of the same scope (freshness derived row by row, mixed kinds) agrees.
    detail = production_catalog.service.inventory(
        owner_user_id=OWNER, project="E:/work/scope007", detail=True)
    (detailed,) = [p for u in detail.users for h in u.hosts for p in h.projects]
    assert (detailed.root_dir, detailed.last_commit, detailed.last_indexed_at) == (
        scope7.root_dir, scope7.last_commit, scope7.last_indexed_at)
    assert (detailed.file_count, detailed.chunk_count) == (scope7.file_count, scope7.chunk_count)
    assert projects[("host-b", "E:/work/extra-b")].chunk_count == 600
    assert all(p.dirs == [] for p in projects.values())  # compact: no per-file nodes


def test_compact_summary_meets_the_latency_budget(production_catalog):
    median = _median_ms(lambda: _summary(production_catalog))
    assert median < SUMMARY_BUDGET_MS, f"compact summary took {median:.0f} ms (budget {SUMMARY_BUDGET_MS})"


def test_compact_summary_never_sorts_or_scans_the_files_of_the_owner(production_catalog):
    with capture_catalog_plans(production_catalog) as plans:
        _summary(production_catalog)
    lines = [line for _statement, details in plans for line in details]
    assert any("ix_index_catalog_live_summary" in line for line in lines), lines
    assert not [line for line in lines if "TEMP B-TREE FOR LAST" in line], lines


def test_mcp_call_shape_reads_one_scope_through_a_bounded_index(production_catalog):
    inventory = _mcp_shape(production_catalog)
    (project,) = [p for u in inventory.users for h in u.hosts for p in h.projects]
    assert project.project == "E:/work/scope007"
    assert sum(d.file_count for d in project.dirs) == PER_SCOPE  # code only, live only
    with capture_catalog_plans(production_catalog) as plans:
        _mcp_shape(production_catalog)
    assert full_scans(plans) == []
    assert unbounded_reads(plans) == []
    assert _median_ms(lambda: _mcp_shape(production_catalog)) < SUMMARY_BUDGET_MS / 5


def test_inventory_rows_stay_in_owner_host_project_path_order(tmp_path):
    catalog = CatalogDb()
    paths = ["b/z.py", "B/a.py", "a/m.py", "_x.py", "é/y.py", "a/b.py"]
    catalog.add_files("/repo", paths, owner=OWNER, host="h2")
    catalog.add_files("/repo", paths, owner=OWNER, host="h1")
    catalog.add_files("/other", ["z.py"], owner=OWNER, host="h1")

    expected = sorted(
        [(OWNER, host, "/repo", path) for host in ("h1", "h2") for path in paths])
    rows = catalog.repository.list_inventory(owner_user_id=OWNER, project="/repo")
    assert [(r.owner_user_id, r.host, r.project, r.rel_path) for r in rows] == expected
    baseline = catalog.repository.list_inventory_baseline(owner_user_id=OWNER, project="/repo")
    assert [(r.owner_user_id, r.host, r.project, r.rel_path) for r in baseline] == expected


def test_existing_database_gets_the_summary_index_from_the_idempotent_migrations(tmp_path, monkeypatch):
    from sqlalchemy import create_engine

    from core.db import database
    from core.db.database import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP INDEX ix_index_catalog_live_summary")
    monkeypatch.setattr(database, "engine", engine)
    database._apply_idempotent_migrations()
    database._apply_idempotent_migrations()  # re-runnable
    with engine.connect() as conn:
        sql = conn.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE name = 'ix_index_catalog_live_summary'").scalar()
    assert sql and "WHERE deleted_at IS NULL" in sql
    assert "indexed_at, rel_path, chunk_count" in sql
