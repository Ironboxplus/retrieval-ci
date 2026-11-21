"""TDD for the index-inventory catalog (user → host → project → dir → file)."""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from app.services.index_catalog.repository import IndexCatalogRepository
from app.services.index_catalog.schemas import (
    IndexFileRecord, IndexReportRequest, IndexScopePolicyUpsert,
)
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


def _req(host, project, paths, root_dir="/repo", replace=True, chunks=3):
    return IndexReportRequest(
        host=host, project=project, root_dir=root_dir, replace=replace,
        files=[IndexFileRecord(rel_path=p, kind="code", collection="code_chunks",
                               chunk_count=chunks) for p in paths],
    )


def test_report_roundtrips_content_hash_and_metadata():
    # W158: content_hash (incremental diff baseline) + git_commit metadata (freshness)
    # must survive report → inventory so the client can diff + display the indexed rev.
    svc = _service()
    req = IndexReportRequest(
        host="4090", project="retrival", root_dir="/repo", replace=True,
        files=[IndexFileRecord(rel_path="a.py", kind="code", collection="code_chunks",
                               chunk_count=4, content_hash="deadbeef",
                               metadata={"git_commit": "abc1234"})],
    )
    svc.report(req, owner_user_id="user-1")
    inv = svc.inventory(owner_user_id="user-1")
    node = inv.users[0].hosts[0].projects[0].dirs[0].files[0]
    assert node.content_hash == "deadbeef"
    assert node.metadata.get("git_commit") == "abc1234"
    assert node.indexed_at is not None


def test_windows_catalog_case_only_collision_is_rejected_and_legacy_row_is_reused():
    svc = _service()
    svc.report(_req("desktop", "C:/Repo", ["Docs/Readme.md"], root_dir="C:/Repo"),
               owner_user_id="user-1")
    # A later canonical spelling updates the same logical file, not a second row.
    svc.report(_req("desktop", "c:/repo", ["docs/readme.md"], root_dir="c:/repo"),
               owner_user_id="user-1")
    inv = svc.inventory(owner_user_id="user-1", detail=True)
    assert inv.totals["files"] == 1
    assert inv.users[0].hosts[0].projects[0].dirs[0].files[0].rel_path == "docs/readme.md"

    with pytest.raises(ValueError, match="case-insensitive catalog path collision"):
        svc.report(_req("desktop", "C:/Repo", ["Docs/A.md", "docs/a.md"], root_dir="C:/Repo"),
                   owner_user_id="user-1")


def test_code_scope_policy_roundtrips_auto_refresh_and_non_git_excludes():
    svc = _service()
    policy = svc.upsert_policy(IndexScopePolicyUpsert(
        host="desktop", project="C:/plain", root_dir="C:/plain", kind="code",
        auto_refresh=True, respect_gitignore=True,
        exclude_globs=["generated/**", "private/**"], min_check_interval_s=600,
    ), owner_user_id="user-1")
    assert policy.auto_refresh is True
    assert policy.exclude_globs == ["generated/**", "private/**"]

    loaded = svc.get_policy(
        owner_user_id="user-1", host="desktop", project="C:/plain", kind="code")
    assert loaded is not None
    assert loaded.root_dir == "C:/plain"
    assert loaded.respect_gitignore is True
    assert loaded.min_check_interval_s == 600
    assert svc.get_policy(
        owner_user_id="other", host="desktop", project="C:/plain", kind="code") is None


def test_policy_interval_defaults_to_five_minutes_and_rejects_anything_shorter():
    # 2026-11-15 user rule: automatic indexing runs at most once per 5 minutes per scope.
    from pydantic import ValidationError

    scope = dict(host="desktop", project="C:/plain", root_dir="C:/plain")
    assert IndexScopePolicyUpsert(**scope).min_check_interval_s == 300
    for too_short in (0, 10, 299):
        with pytest.raises(ValidationError):
            IndexScopePolicyUpsert(**scope, min_check_interval_s=too_short)
    for allowed in (300, 3600):
        assert IndexScopePolicyUpsert(
            **scope, min_check_interval_s=allowed).min_check_interval_s == allowed
    with pytest.raises(ValidationError):
        IndexScopePolicyUpsert(**scope, min_check_interval_s=3601)


def test_stored_policy_below_the_floor_is_reported_at_the_floor_never_fails():
    svc = _service()
    for project, stored in (("C:/legacy", 10), ("C:/custom", 600)):
        # the repository stores what it is given: this is a row written before the floor
        svc.repository.upsert_policy(
            owner_user_id="user-1", host="desktop", project=project, kind="code",
            root_dir=project, auto_refresh=True, respect_gitignore=True,
            exclude_globs=[], min_check_interval_s=stored)

    legacy = svc.get_policy(
        owner_user_id="user-1", host="desktop", project="C:/legacy", kind="code")
    custom = svc.get_policy(
        owner_user_id="user-1", host="desktop", project="C:/custom", kind="code")
    assert legacy is not None and legacy.min_check_interval_s == 300
    assert custom is not None and custom.min_check_interval_s == 600


def test_migration_raises_stored_policy_intervals_to_the_floor_and_is_rerunnable(
        tmp_path, monkeypatch):
    from core.db import database

    engine = create_engine(f"sqlite:///{(tmp_path / 'policies.db').as_posix()}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for project, stored in (("C:/ten", 10), ("C:/six-hundred", 600),
                                ("C:/just-below", 299), ("C:/floor", 300)):
            conn.execute(text(
                "INSERT INTO index_scope_policies (owner_user_id, host, project, kind, "
                "root_dir, auto_refresh, respect_gitignore, min_check_interval_s) "
                "VALUES ('user-1', 'desktop', :project, 'code', :project, 1, 1, :stored)"),
                {"project": project, "stored": stored})

    def stored_intervals() -> dict[str, int]:
        with engine.connect() as conn:
            return {row[0]: row[1] for row in conn.execute(text(
                "SELECT project, min_check_interval_s FROM index_scope_policies"))}

    monkeypatch.setattr(database, "engine", engine)
    database._apply_idempotent_migrations()
    expected = {"C:/ten": 300, "C:/six-hundred": 600, "C:/just-below": 300, "C:/floor": 300}
    assert stored_intervals() == expected
    database._apply_idempotent_migrations()  # a second start changes nothing
    assert stored_intervals() == expected


def test_inventory_shows_registered_code_scope_before_any_completed_snapshot():
    """A persisted auto-refresh policy is a registration, not evidence that vectors landed."""
    svc = _service()
    svc.upsert_policy(IndexScopePolicyUpsert(
        host="desktop", project="C:/plain", root_dir="C:/plain", kind="code",
    ), owner_user_id="user-1")

    inventory = svc.inventory(owner_user_id="user-1", kind="code", detail=False)

    project = inventory.users[0].hosts[0].projects[0]
    assert project.project == "C:/plain"
    assert project.file_count == 0 and project.chunk_count == 0
    assert project.code_snapshot_state == "registered"
    assert project.code_active_index_job_id is None
    assert project.code_previous_completed_snapshot_available is False


def test_inventory_overlays_active_code_job_as_preparing_not_completed():
    svc = _service()
    svc.report(_req("desktop", "/repo/sing-box", ["hysteria.go"], root_dir="/repo/sing-box"),
               owner_user_id="u1")

    inventory = svc.inventory(
        owner_user_id="u1",
        kind="code",
        detail=False,
        code_scope_states={
            ("u1", "desktop", "/repo/sing-box"): {
                "active_index_job_id": "indexing-sing-box",
                "previous_completed_snapshot_available": True,
            },
        },
    )

    project = inventory.users[0].hosts[0].projects[0]
    assert project.code_snapshot_state == "preparing"
    assert project.code_active_index_job_id == "indexing-sing-box"
    assert project.code_previous_completed_snapshot_available is True


def test_inventory_durable_recovery_wins_over_runtime_queued_overlay():
    """A fresh queued receipt cannot downgrade a durable interrupted-write boundary."""
    svc = _service()
    svc.report(_req("desktop", "/repo/sing-box", ["hysteria.go"], root_dir="/repo/sing-box"),
               owner_user_id="u1")
    svc.mark_code_scope_recovery_required(
        owner_user_id="u1", host="desktop", project="/repo/sing-box",
        job_id="failed-write-job",
    )

    inventory = svc.inventory(
        owner_user_id="u1",
        kind="code",
        detail=False,
        code_scope_states={
            ("u1", "desktop", "/repo/sing-box"): {
                "active_index_job_id": "newly-queued-recovery-job",
                "previous_completed_snapshot_available": True,
                "recovery_required": False,
            },
        },
    )

    project = inventory.users[0].hosts[0].projects[0]
    assert project.code_snapshot_state == "preparing"
    assert project.code_recovery_required is True
    assert project.code_active_index_job_id == "failed-write-job"


def test_inventory_overlays_fan_out_target_even_when_writer_host_differs():
    """Fan-out is scoped by owner/project; host is display metadata, not visibility.

    A parent writer can update a duplicate child scope whose inventory row carries a
    different host.  The child must still be reported as preparing instead of completed.
    """
    svc = _service()
    svc.report(_req("child-host", "/repo/child", ["a.py"], root_dir="/repo/child"),
               owner_user_id="u1")

    inventory = svc.inventory(
        owner_user_id="u1",
        kind="code",
        detail=False,
        code_scope_states={
            ("u1", "parent-writer-host", "/repo/child"): {
                "active_index_job_id": "parent-fan-out-job",
                "previous_completed_snapshot_available": True,
            },
        },
    )

    project = inventory.users[0].hosts[0].projects[0]
    assert project.host == "child-host"
    assert project.code_snapshot_state == "preparing"
    assert project.code_active_index_job_id == "parent-fan-out-job"


def test_host_filtered_inventory_overlays_durable_fan_out_target_marker():
    """A host filter selects display rows, not the durable marker namespace.

    The target's inventory row belongs to ``child-host`` while the parent job wrote
    its owner/project marker with ``parent-writer-host``.  Filtering marker reads by
    the writer host would incorrectly make this partial target look completed.
    """
    svc = _service()
    svc.report(_req("child-host", "/repo/child", ["a.py"], root_dir="/repo/child"),
               owner_user_id="u1")
    svc.mark_code_scope_preparing(
        owner_user_id="u1", host="parent-writer-host", project="/repo/child",
        job_id="parent-fan-out-job",
    )

    inventory = svc.inventory(
        owner_user_id="u1", host="child-host", kind="code", detail=False,
    )

    project = inventory.users[0].hosts[0].projects[0]
    assert project.host == "child-host"
    assert project.code_snapshot_state == "preparing"
    assert project.code_active_index_job_id == "parent-fan-out-job"


def test_durable_code_visibility_survives_catalog_reconstruction_and_is_owner_isolated():
    """A process-local queue loss must not erase an unsafe Qdrant write boundary."""
    first_process_catalog = _service()
    first_process_catalog.mark_code_scope_preparing(
        owner_user_id="owner-a", host="writer-host", project="/repo/crashed",
        job_id="job-partial-write",
    )

    # New service object, same SQLite-backed repository/session factory: this models the
    # backend restarting after Qdrant accepted some batches but before a completion marker.
    restarted_catalog = IndexCatalogService(repository=first_process_catalog.repository)
    marker = restarted_catalog.code_scope_visibility(
        owner_user_id="owner-a", projects=["/repo/crashed"])

    assert marker is not None
    assert marker["state"] == "preparing"
    assert marker["job_id"] == "job-partial-write"
    assert restarted_catalog.code_scope_visibility(
        owner_user_id="owner-b", projects=["/repo/crashed"]) is None

    restarted_catalog.mark_code_scope_preparing(
        owner_user_id="owner-a", host="writer-host", project="/repo/crashed",
        job_id="job-recovery",
    )
    restarted_catalog.mark_code_scope_completed(
        owner_user_id="owner-a", host="writer-host", project="/repo/crashed",
        job_id="job-recovery",
    )
    completed = restarted_catalog.code_scope_visibility(
        owner_user_id="owner-a", projects=["/repo/crashed"])
    assert completed is not None and completed["state"] == "completed"


def test_code_scope_policy_rejects_root_project_mismatch():
    svc = _service()
    with pytest.raises(ValueError, match="root_dir must match project"):
        svc.upsert_policy(IndexScopePolicyUpsert(
            host="desktop", project="C:/repo", root_dir="C:/other", kind="code",
        ), owner_user_id="user-1")


def test_delete_project_removes_code_refresh_policy():
    svc = _service()
    svc.upsert_policy(IndexScopePolicyUpsert(
        host="desktop", project="C:/plain", root_dir="C:/plain", kind="code",
        auto_refresh=True,
    ), owner_user_id="user-1")

    assert svc.delete_project(
        owner_user_id="user-1", project="C:/plain", kind="code") == 0
    assert svc.get_policy(
        owner_user_id="user-1", host="desktop", project="C:/plain", kind="code") is None


def test_inventory_summary_omits_files_but_keeps_counts_and_freshness():
    """SUMMARY (detail=False) is the tool's default view: per-project counts + server-side
    freshness, but NO per-file nodes. The full per-file content_hash tree ballooned to ~10MB
    and broke index_inventory over the wire (truncated download); the tool only ever renders
    project-level counts + 'last indexed', so summary ships just that (KB, not MB)."""
    svc = _service()
    req = IndexReportRequest(
        host="4090", project="retrival", root_dir="/repo", replace=True,
        files=[IndexFileRecord(rel_path="pkg/a.py", kind="code", collection="code_chunks",
                               chunk_count=4, content_hash="deadbeef",
                               metadata={"git_commit": "abc1234"})],
    )
    svc.report(req, owner_user_id="user-1")
    inv = svc.inventory(owner_user_id="user-1", detail=False)
    proj = inv.users[0].hosts[0].projects[0]
    assert proj.file_count == 1 and proj.chunk_count == 4          # counts preserved
    assert proj.dirs == []                                          # no per-file/-dir tree
    assert proj.last_commit == "abc1234"                            # freshness server-side
    assert proj.last_indexed_at is not None
    assert "deadbeef" not in inv.model_dump_json()                  # zero hashes on the wire


def test_inventory_detail_true_still_includes_files():
    svc = _service()
    svc.report(_req("4090", "retrival", ["pkg/a.py", "pkg/b.py"]), owner_user_id="user-1")
    inv = svc.inventory(owner_user_id="user-1", detail=True)
    proj = inv.users[0].hosts[0].projects[0]
    assert {f.rel_path for d in proj.dirs for f in d.files} == {"pkg/a.py", "pkg/b.py"}
    assert proj.file_count == 2


def test_inventory_last_commit_pairs_with_freshest_file():
    """last_commit is the commit of the NEWEST-indexed file, not the first-seen one — so the
    'last indexed <ts> @<commit>' line is internally consistent even across partial re-indexes."""
    svc = _service()
    svc.report(IndexReportRequest(
        host="h", project="p", root_dir="/repo", replace=False,
        files=[IndexFileRecord(rel_path="a.py", kind="code", collection="code_chunks",
                               chunk_count=1, metadata={"git_commit": "old111"})]),
        owner_user_id="u1")
    svc.report(IndexReportRequest(
        host="h", project="p", root_dir="/repo", replace=False,
        files=[IndexFileRecord(rel_path="z_newer.py", kind="code", collection="code_chunks",
                               chunk_count=1, metadata={"git_commit": "new999"})]),
        owner_user_id="u1")
    proj = svc.inventory(owner_user_id="u1", detail=False).users[0].hosts[0].projects[0]
    assert proj.last_commit == "new999"


def test_inventory_project_freshness_present_in_detail_mode_too():
    svc = _service()
    req = IndexReportRequest(
        host="4090", project="p", root_dir="/repo", replace=True,
        files=[IndexFileRecord(rel_path="a.py", kind="code", collection="code_chunks",
                               chunk_count=1, metadata={"git_commit": "c0ffee"})],
    )
    svc.report(req, owner_user_id="u1")
    proj = svc.inventory(owner_user_id="u1", detail=True).users[0].hosts[0].projects[0]
    assert proj.last_commit == "c0ffee" and proj.last_indexed_at is not None


def test_report_upserts_then_replace_soft_deletes_absent():
    svc = _service()
    r1 = svc.report(_req("4090", "retrival", ["a.py", "b.py"]), owner_user_id="user-1")
    assert (r1.upserted, r1.removed) == (2, 0)

    # Re-report without b.py + a new c.py → b.py removed, c.py added, a.py updated.
    r2 = svc.report(_req("4090", "retrival", ["a.py", "c.py"]), owner_user_id="user-1")
    assert r2.upserted == 2
    assert r2.removed == 1

    inv = svc.inventory(owner_user_id="user-1")
    paths = {
        f.rel_path
        for u in inv.users for h in u.hosts for p in h.projects for d in p.dirs for f in d.files
    }
    assert paths == {"a.py", "c.py"}


def test_incremental_report_does_not_delete():
    svc = _service()
    svc.report(_req("4090", "retrival", ["a.py"]), owner_user_id="user-1")
    r = svc.report(_req("4090", "retrival", ["b.py"], replace=False), owner_user_id="user-1")
    assert r.removed == 0
    inv = svc.inventory(owner_user_id="user-1")
    paths = {f.rel_path for u in inv.users for h in u.hosts for p in h.projects for d in p.dirs for f in d.files}
    assert paths == {"a.py", "b.py"}


def _projects(inv):
    return {p.project for u in inv.users for h in u.hosts for p in h.projects}


def test_delete_project_clears_all_files_across_hosts():
    # Code vectors are host-agnostic (point id carries no host, W148), so clearing a
    # project must sweep its catalog rows on EVERY host, and leave other projects intact.
    svc = _service()
    svc.report(_req("h1", "proj", ["a.py", "b.py"]), owner_user_id="u1")
    svc.report(_req("h2", "proj", ["c.py"]), owner_user_id="u1")
    svc.report(_req("h1", "other", ["z.py"]), owner_user_id="u1")

    removed = svc.delete_project(owner_user_id="u1", project="proj")
    assert removed == 3                                   # a.py+b.py (h1) + c.py (h2)
    assert _projects(svc.inventory(owner_user_id="u1")) == {"other"}


def test_delete_project_kind_filter_keeps_docs():
    # Deleting the CODE index of a directory must not wipe the journal docs indexed under
    # the same scope (kind=doc) — they're a different tool's inventory (W151).
    svc = _service()
    svc.report(_req("h1", "proj", ["a.py"]), owner_user_id="u1")           # kind=code
    doc = IndexReportRequest(host="h1", project="proj", root_dir="/repo", replace=False,
        files=[IndexFileRecord(rel_path="README.md", kind="doc",
                               collection="journal_chunks", chunk_count=1)])
    svc.report(doc, owner_user_id="u1")

    removed = svc.delete_project(owner_user_id="u1", project="proj", kind="code")
    assert removed == 1
    kinds = {f.kind for u in svc.inventory(owner_user_id="u1").users
             for h in u.hosts for p in h.projects for d in p.dirs for f in d.files}
    assert kinds == {"doc"}                               # code cleared, doc kept


def test_delete_project_scoped_to_owner():
    svc = _service()
    svc.report(_req("h1", "proj", ["a.py"]), owner_user_id="u1")
    svc.report(_req("h1", "proj", ["a.py"]), owner_user_id="u2")
    assert svc.delete_project(owner_user_id="u1", project="proj") == 1
    assert _projects(svc.inventory(owner_user_id="u2")) == {"proj"}  # u2 untouched


def test_empty_report_never_wipes_inventory():
    # An accidental index_code on a source-less dir (files=[]) must NOT soft-delete
    # everything, even with replace=True (the default).
    svc = _service()
    svc.report(_req("4090", "retrival", ["a.py", "b.py"]), owner_user_id="user-1")
    r = svc.report(_req("4090", "retrival", [], replace=True), owner_user_id="user-1")
    assert r.removed == 0
    inv = svc.inventory(owner_user_id="user-1")
    assert inv.totals["files"] == 2


def _req_kind(host, project, files, root_dir="/repo", replace=True):
    """files: list of (rel_path, kind)."""
    return IndexReportRequest(
        host=host, project=project, root_dir=root_dir, replace=replace,
        files=[IndexFileRecord(rel_path=p, kind=k,
                               collection=("code_chunks" if k == "code" else "journal_chunks"),
                               chunk_count=3) for p, k in files],
    )


def test_replace_sweep_is_kind_scoped():
    # W151: a code report (kind=code) full-sync must NOT soft-delete the project's
    # doc inventory (kind=doc) reported by a different tool under the same project.
    svc = _service()
    svc.report(_req_kind("4090", "retrival", [("notes.md", "doc")]), owner_user_id="user-1")
    svc.report(_req_kind("4090", "retrival", [("core/store.py", "code")]), owner_user_id="user-1")

    inv = svc.inventory(owner_user_id="user-1")
    got = {
        (f.rel_path, f.kind)
        for u in inv.users for h in u.hosts for p in h.projects for d in p.dirs for f in d.files
    }
    assert got == {("notes.md", "doc"), ("core/store.py", "code")}, "doc row wrongly swept by code report"


def test_replace_sweep_is_pattern_scoped():
    # W169b (catalog axis): a doc report for "*.txt" must NOT soft-delete the "*.md" doc
    # rows already indexed under the same scope — they're outside this run's pattern
    # envelope. Without this, the inventory drifts out of sync with the journal (whose
    # reconcile IS pattern-scoped), reviving the very stale-drift finding #3 guarded against.
    svc = _service()
    svc.report(_req_kind("4090", "retrival", [("notes.md", "doc"), ("guide.md", "doc")]),
               owner_user_id="user-1")

    txt = IndexReportRequest(
        host="4090", project="retrival", root_dir="/repo", replace=True, patterns=["*.txt"],
        files=[IndexFileRecord(rel_path="log.txt", kind="doc",
                               collection="journal_chunks", chunk_count=2)])
    r = svc.report(txt, owner_user_id="user-1")

    assert r.removed == 0  # the .md rows are outside the *.txt envelope → untouched
    got = {
        (f.rel_path, f.kind)
        for u in svc.inventory(owner_user_id="user-1").users
        for h in u.hosts for p in h.projects for d in p.dirs for f in d.files
    }
    assert got == {("notes.md", "doc"), ("guide.md", "doc"), ("log.txt", "doc")}


def test_replace_sweep_within_pattern_still_removes_deleted_same_pattern_file():
    # The envelope must NOT neuter the legitimate sweep: a "*.md" re-run that drops one .md
    # still soft-deletes that .md (it's within the *.md envelope) — catalog stays in sync.
    svc = _service()
    svc.report(_req_kind("4090", "retrival", [("a.md", "doc"), ("b.md", "doc")]),
               owner_user_id="user-1")
    md = IndexReportRequest(
        host="4090", project="retrival", root_dir="/repo", replace=True, patterns=["*.md"],
        files=[IndexFileRecord(rel_path="a.md", kind="doc",
                               collection="journal_chunks", chunk_count=2)])
    r = svc.report(md, owner_user_id="user-1")
    assert r.removed == 1  # b.md is within *.md envelope and absent → swept
    got = {f.rel_path for u in svc.inventory(owner_user_id="user-1").users
           for h in u.hosts for p in h.projects for d in p.dirs for f in d.files}
    assert got == {"a.md"}


def test_replace_sweep_respects_depth_envelope():
    # A shallow (max_depth=0) doc re-run must not sweep deeper catalog rows it never scanned.
    svc = _service()
    svc.report(_req_kind("4090", "retrival", [("a.md", "doc"), ("sub/deep.md", "doc")]),
               owner_user_id="user-1")
    shallow = IndexReportRequest(
        host="4090", project="retrival", root_dir="/repo", replace=True,
        patterns=["*.md"], max_depth=0,
        files=[IndexFileRecord(rel_path="a.md", kind="doc",
                               collection="journal_chunks", chunk_count=2)])
    r = svc.report(shallow, owner_user_id="user-1")
    assert r.removed == 0  # sub/deep.md is depth 1 > 0 → outside envelope → kept
    got = {f.rel_path for u in svc.inventory(owner_user_id="user-1").users
           for h in u.hosts for p in h.projects for d in p.dirs for f in d.files}
    assert got == {"a.md", "sub/deep.md"}


def test_inventory_tree_groups_and_separates_projects_and_dirs():
    svc = _service()
    svc.report(_req("4090", "retrival", ["core/db/models.py", "core/db/x.py", "main.py"]),
               owner_user_id="user-1")
    svc.report(_req("kss", "other-proj", ["src/index.ts"]), owner_user_id="user-1")

    inv = svc.inventory(owner_user_id="user-1")
    assert inv.totals["hosts"] == 2
    assert inv.totals["projects"] == 2
    assert inv.totals["files"] == 4

    hosts = {h.host: h for u in inv.users for h in u.hosts}
    retrival = {p.project: p for p in hosts["4090"].projects}["retrival"]
    dirs = {d.dir for d in retrival.dirs}
    assert dirs == {"core/db", "."}


def test_owner_isolation_and_summary():
    svc = _service()
    svc.report(_req("4090", "retrival", ["a.py", "b.py"], chunks=5), owner_user_id="user-1")
    svc.report(_req("4090", "retrival", ["z.py"], chunks=2), owner_user_id="user-2")

    inv_all = svc.inventory()
    assert inv_all.totals["users"] == 2

    inv_one = svc.inventory(owner_user_id="user-1")
    assert inv_one.totals["files"] == 2
    assert inv_one.totals["chunks"] == 10

    owners = {o.owner_user_id: o for o in svc.owners()}
    assert owners["user-1"].file_count == 2
    assert owners["user-1"].chunk_count == 10
    assert owners["user-2"].file_count == 1


def test_scope_overlap_blocks_child_when_parent_has_file_under_it():
    # A parent scope that ACTUALLY indexed a file under the child subtree makes the child
    # redundant → ancestor_block. This is the O(all files) check the guard used to do
    # client-side over a 9MB inventory download; now the server answers it in KB.
    svc = _service()
    svc.report(_req("h1", "/data/erp", ["results/t/a.txt", "main.py"], root_dir="/data/erp"),
               owner_user_id="u1")
    out = svc.scope_overlap(owner_user_id="u1", project="/data/erp/results/t", host="h1", kind="code")
    assert [c.project for c in out.ancestor_blocks] == ["/data/erp"]
    assert out.child_warnings == []


def test_scope_overlap_no_block_when_parent_has_no_file_under_child():
    # Parent exists but indexed nothing under the child subtree → no coverage overlap,
    # so indexing the child is allowed (no block, no warning).
    svc = _service()
    svc.report(_req("h1", "/data/erp", ["src/main.py"], root_dir="/data/erp"),
               owner_user_id="u1")
    out = svc.scope_overlap(owner_user_id="u1", project="/data/erp/results/t", host="h1", kind="code")
    assert out.ancestor_blocks == []
    assert out.child_warnings == []


class TestIndexingAGitignoredSubtreeUnderAnIndexedParent:
    """Indexing a child scope to pick up what the parent's gitignore filter skipped.

    The parent was indexed with respect_gitignore=True, so an ignored subtree
    (``data/``, ``node_modules/``, a generated corpus) never entered the catalog.
    Re-indexing that subtree on its own with respect_gitignore=False is the only way
    to reach those files, and it genuinely ADDS coverage the parent does not have —
    which is precisely the condition the ancestor block is supposed to test for.
    """

    def test_a_wholly_ignored_subtree_is_not_blocked(self):
        # The parent skipped every file under data/, so it holds no coverage there and
        # the existing "did the ancestor index anything under the child" test already
        # lets this through. This is the common case (node_modules, build output).
        svc = _service()
        svc.report(_req("h1", "/repo/erp", ["src/main.py", "README.md"],
                        root_dir="/repo/erp"), owner_user_id="u1")
        out = svc.scope_overlap(owner_user_id="u1", project="/repo/erp/data",
                                host="h1", kind="code", respect_gitignore=False)
        assert out.ancestor_blocks == []
        assert out.child_warnings == []

    def test_a_partly_ignored_subtree_is_not_blocked_when_the_child_widens_the_filter(self):
        # data/ holds a tracked README (the parent indexed it) AND an ignored corpus
        # (the parent skipped it). The ancestor therefore DOES have a file under the
        # child, so the coverage test alone would refuse — and tell the caller to
        # "delete the parent scope first", trading a whole repo index for one subtree.
        # A child that turns respect_gitignore OFF under a parent that had it ON is
        # reaching for files the parent provably never indexed, so it must proceed.
        svc = _service()
        svc.report(_req("h1", "/repo/erp", ["src/main.py", "data/README.md"],
                        root_dir="/repo/erp"), owner_user_id="u1")
        svc.upsert_policy(IndexScopePolicyUpsert(
            host="h1", project="/repo/erp", root_dir="/repo/erp", kind="code",
            respect_gitignore=True), owner_user_id="u1")

        out = svc.scope_overlap(owner_user_id="u1", project="/repo/erp/data",
                                host="h1", kind="code", respect_gitignore=False)

        assert out.ancestor_blocks == []
        assert [c.project for c in out.ancestor_widenings] == ["/repo/erp"]

    def test_the_same_child_is_still_blocked_when_it_keeps_the_filter_on(self):
        # Without widening the filter the child really is redundant: every file it
        # would index, the parent already has. The block must survive the fix.
        svc = _service()
        svc.report(_req("h1", "/repo/erp", ["src/main.py", "data/README.md"],
                        root_dir="/repo/erp"), owner_user_id="u1")
        svc.upsert_policy(IndexScopePolicyUpsert(
            host="h1", project="/repo/erp", root_dir="/repo/erp", kind="code",
            respect_gitignore=True), owner_user_id="u1")

        out = svc.scope_overlap(owner_user_id="u1", project="/repo/erp/data",
                                host="h1", kind="code", respect_gitignore=True)

        assert [c.project for c in out.ancestor_blocks] == ["/repo/erp"]
        assert out.ancestor_widenings == []

    def test_a_parent_that_already_ignored_nothing_still_blocks(self):
        # The parent was itself indexed with respect_gitignore=False, so it already
        # holds the ignored files. A child widening a filter that was never narrow
        # adds nothing, and the block must stand.
        svc = _service()
        svc.report(_req("h1", "/repo/erp", ["src/main.py", "data/README.md"],
                        root_dir="/repo/erp"), owner_user_id="u1")
        svc.upsert_policy(IndexScopePolicyUpsert(
            host="h1", project="/repo/erp", root_dir="/repo/erp", kind="code",
            respect_gitignore=False), owner_user_id="u1")

        out = svc.scope_overlap(owner_user_id="u1", project="/repo/erp/data",
                                host="h1", kind="code", respect_gitignore=False)

        assert [c.project for c in out.ancestor_blocks] == ["/repo/erp"]
        assert out.ancestor_widenings == []

    def test_an_unstated_filter_keeps_the_pre_existing_block(self):
        # An older MCP client does not send respect_gitignore at all. It must observe
        # exactly the behaviour it shipped against, not a silently relaxed guard.
        svc = _service()
        svc.report(_req("h1", "/repo/erp", ["src/main.py", "data/README.md"],
                        root_dir="/repo/erp"), owner_user_id="u1")
        svc.upsert_policy(IndexScopePolicyUpsert(
            host="h1", project="/repo/erp", root_dir="/repo/erp", kind="code",
            respect_gitignore=True), owner_user_id="u1")

        out = svc.scope_overlap(owner_user_id="u1", project="/repo/erp/data",
                                host="h1", kind="code")

        assert [c.project for c in out.ancestor_blocks] == ["/repo/erp"]

    def test_a_parent_with_no_recorded_policy_falls_back_to_the_indexer_default(self):
        # Scopes indexed before policies were stored have no policy row. The indexer's
        # own default is respect_gitignore=True, so assuming that is what actually ran
        # is the reading that matches the data on disk.
        svc = _service()
        svc.report(_req("h1", "/repo/erp", ["src/main.py", "data/README.md"],
                        root_dir="/repo/erp"), owner_user_id="u1")

        out = svc.scope_overlap(owner_user_id="u1", project="/repo/erp/data",
                                host="h1", kind="code", respect_gitignore=False)

        assert out.ancestor_blocks == []
        assert [c.project for c in out.ancestor_widenings] == ["/repo/erp"]


class TestRefreshingAScopeThatAlreadyExists:
    """A scope that already exists must stay refreshable.

    The guard's job is to stop a REDUNDANT NEW scope from being created. Applying
    it to a scope that already exists cannot undo that scope — it only freezes it
    at whatever it held on the day it was made, while the advice it prints
    ("delete the parent scope first") would destroy a whole repository index to
    unfreeze one directory.
    """

    @staticmethod
    def _parent_and_child(svc, *, child_exists: bool):
        svc.report(_req("h1", "/repo/erp", ["src/main.py", "data/README.md"],
                        root_dir="/repo/erp"), owner_user_id="u1")
        svc.upsert_policy(IndexScopePolicyUpsert(
            host="h1", project="/repo/erp", root_dir="/repo/erp", kind="code",
            respect_gitignore=True), owner_user_id="u1")
        if child_exists:
            svc.report(_req("h1", "/repo/erp/data", ["README.md", "corpus.db"],
                            root_dir="/repo/erp/data"), owner_user_id="u1")

    def test_an_existing_child_scope_is_refreshable_with_the_default_filter(self):
        # The path a plain index_code() re-run takes, and the one an already
        # published client takes because it sends no filter at all.
        svc = _service()
        self._parent_and_child(svc, child_exists=True)
        for filter_state in (True, None):
            out = svc.scope_overlap(owner_user_id="u1", project="/repo/erp/data",
                                    host="h1", kind="code",
                                    respect_gitignore=filter_state)
            assert out.ancestor_blocks == [], filter_state

    def test_refreshing_an_existing_child_scope_does_not_repeat_the_warning(self):
        # Auto-refresh replays the scope's stored respect_gitignore on every tick.
        # A warning about an overlap the user already accepted, once per tick, is
        # noise that trains them to ignore the channel real warnings arrive on.
        svc = _service()
        self._parent_and_child(svc, child_exists=True)
        out = svc.scope_overlap(owner_user_id="u1", project="/repo/erp/data",
                                host="h1", kind="code", respect_gitignore=False)
        assert out.ancestor_blocks == []
        assert out.ancestor_widenings == []

    def test_a_child_scope_that_does_not_exist_yet_is_still_judged(self):
        # The guard must keep doing its actual job: a NEW redundant scope is refused.
        svc = _service()
        self._parent_and_child(svc, child_exists=False)
        out = svc.scope_overlap(owner_user_id="u1", project="/repo/erp/data",
                                host="h1", kind="code", respect_gitignore=True)
        assert [c.project for c in out.ancestor_blocks] == ["/repo/erp"]

    def test_a_windows_case_variant_counts_as_the_same_existing_scope(self):
        # Inventory rows predating canonicalisation carry other NTFS spellings; a
        # refresh typed in canonical case must still be recognised as this scope.
        svc = _service()
        svc.report(_req("h1", "C:/Users/Arc/repo", ["data/README.md"],
                        root_dir="C:/Users/Arc/repo"), owner_user_id="u1")
        svc.report(_req("h1", "C:/Users/Arc/repo/Data", ["README.md", "corpus.db"],
                        root_dir="C:/Users/Arc/repo/Data"), owner_user_id="u1")

        out = svc.scope_overlap(owner_user_id="u1", project="C:/users/arc/repo/data",
                                host="h1", kind="code", respect_gitignore=True)

        assert out.ancestor_blocks == []

    def test_refreshing_a_parent_still_warns_about_its_existing_children(self):
        # Suppressing the ancestor verdict must not swallow the descendant one:
        # re-indexing /repo/erp still says a child scope lives under it.
        svc = _service()
        self._parent_and_child(svc, child_exists=True)
        out = svc.scope_overlap(owner_user_id="u1", project="/repo/erp",
                                host="h1", kind="code", respect_gitignore=True)
        assert out.ancestor_blocks == []
        assert [c.project for c in out.child_warnings] == ["/repo/erp/data"]


def test_scope_overlap_warns_when_indexing_parent_over_existing_child():
    # Indexing a parent over an already-indexed child proceeds, but warns (the parent adds
    # files outside the child).
    svc = _service()
    svc.report(_req("h1", "/data/erp/results/t", ["a.txt"], root_dir="/data/erp/results/t"),
               owner_user_id="u1")
    out = svc.scope_overlap(owner_user_id="u1", project="/data/erp", host="h1", kind="code")
    assert out.ancestor_blocks == []
    assert [c.project for c in out.child_warnings] == ["/data/erp/results/t"]


def test_scope_overlap_isolates_owner_and_kind():
    # Overlap is computed only within the caller's own inventory and the requested kind —
    # another owner's parent, or a doc index, must not block a code child.
    svc = _service()
    svc.report(_req("h1", "/data/erp", ["results/t/a.py"], root_dir="/data/erp"),
               owner_user_id="other")                      # different owner
    doc = IndexReportRequest(host="h1", project="/data/erp", root_dir="/data/erp", replace=True,
        files=[IndexFileRecord(rel_path="results/t/n.md", kind="doc",
                               collection="journal_chunks", chunk_count=1)])
    svc.report(doc, owner_user_id="u1")                    # u1's parent is a DOC index
    out = svc.scope_overlap(owner_user_id="u1", project="/data/erp/results/t", host="h1", kind="code")
    assert out.ancestor_blocks == [] and out.child_warnings == []


def test_scope_overlap_skips_non_path_scope():
    # A non-abspath scope (shouldn't happen post-W164, but be safe) yields no overlap.
    svc = _service()
    svc.report(_req("h1", "/data/erp", ["a.py"], root_dir="/data/erp"), owner_user_id="u1")
    out = svc.scope_overlap(owner_user_id="u1", project="my-project", host="h1", kind="code")
    assert out.ancestor_blocks == [] and out.child_warnings == []


def test_equivalent_windows_project_scopes_include_legacy_case_variants():
    """A canonical Windows query must still include pre-normalisation inventory rows."""
    svc = _service()
    legacy = "E:/JETB/Pythons/Playground/Octopus/AxonHub/LLM"
    svc.repository.report_files(
        owner_user_id="u1", host="desktop", project=legacy, root_dir=legacy,
        files=[{"rel_path": "conversion/compact.go", "kind": "code",
                "collection": "code_chunks", "chunk_count": 1}],
        replace=True,
    )

    aliases = svc.equivalent_project_scopes(
        owner_user_id="u1", project="e:/jetb/pythons/playground/octopus/axonhub/llm")

    assert aliases == ["E:/jetb/pythons/playground/octopus/axonhub/llm", legacy]


def test_scope_overlap_route_returns_conflict_lists():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes import index_catalog as ic_routes

    svc = _service()
    svc.report(_req("h1", "/data/erp", ["results/t/a.txt"], root_dir="/data/erp"),
               owner_user_id="u9")
    ic_routes.get_index_catalog_service = lambda: svc
    app = FastAPI()
    app.include_router(ic_routes.router, prefix="/api/index")
    client = TestClient(app)

    r = client.get("/api/index/scope-overlap", headers={"X-Retrival-User-Id": "u9"},
                   params={"project": "/data/erp/results/t", "host": "h1", "kind": "code"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [c["project"] for c in body["ancestor_blocks"]] == ["/data/erp"]
    assert body["child_warnings"] == []


def test_inventory_route_marks_active_code_scope_preparing(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes import index_catalog as ic_routes

    svc = _service()
    svc.upsert_policy(IndexScopePolicyUpsert(
        host="desktop", project="/repo/sing-box", root_dir="/repo/sing-box", kind="code",
    ), owner_user_id="u9")

    class Queue:
        def inventory_scope_states(self, **_kwargs):
            return {
                ("u9", "desktop", "/repo/sing-box"): {
                    "active_index_job_id": "sing-box-job",
                    "previous_completed_snapshot_available": False,
                },
            }

    monkeypatch.setattr(ic_routes, "get_index_catalog_service", lambda: svc)
    monkeypatch.setattr(ic_routes, "get_existing_code_index_queue", lambda: Queue(), raising=False)
    app = FastAPI()
    app.include_router(ic_routes.router, prefix="/api/index")
    client = TestClient(app)

    response = client.get(
        "/api/index/inventory",
        headers={"X-Retrival-User-Id": "u9"},
        params={"kind": "code"},
    )

    assert response.status_code == 200, response.text
    project = response.json()["users"][0]["hosts"][0]["projects"][0]
    assert project["code_snapshot_state"] == "preparing"
    assert project["code_active_index_job_id"] == "sing-box-job"
    assert project["code_previous_completed_snapshot_available"] is False


def test_report_route_emits_doc_index_audit():
    """W172: the doc-index inventory report (POST /api/index/report — what
    journal_index_dir calls) must emit a faithful index-request audit at receipt,
    kind=doc, so LAN/MCP doc indexing lands in the audit log too."""
    import json
    import logging
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes import index_catalog as ic_routes
    from app.services.index_audit import INDEX_AUDIT_LOGGER_NAME

    svc = _service()
    ic_routes.get_index_catalog_service = lambda: svc
    app = FastAPI()
    app.include_router(ic_routes.router, prefix="/api/index")
    client = TestClient(app)

    lines: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: lines.append(r.getMessage())  # type: ignore[assignment]
    logging.getLogger(INDEX_AUDIT_LOGGER_NAME).addHandler(handler)
    try:
        body = {
            "host": "deskh", "project": "/docs/proj", "root_dir": "/docs/proj",
            "replace": True,
            "files": [{"rel_path": "a.md", "kind": "doc", "collection": "journal_chunks",
                       "chunk_count": 2},
                      {"rel_path": "b.md", "kind": "doc", "collection": "journal_chunks",
                       "chunk_count": 1}],
        }
        r = client.post("/api/index/report", headers={"X-Retrival-User-Id": "docu"}, json=body)
        assert r.status_code == 200, r.text
    finally:
        logging.getLogger(INDEX_AUDIT_LOGGER_NAME).removeHandler(handler)

    recs = [json.loads(x) for x in lines if '"index_request"' in x]
    assert len(recs) == 1, recs
    rec = recs[0]
    assert rec["evt"] == "index_request" and rec["kind"] == "doc"
    assert rec["owner_user_id"] == "docu"
    assert rec["project"] == "/docs/proj" and rec["files"] == 2
    assert rec["replace"] is True
