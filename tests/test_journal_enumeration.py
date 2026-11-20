"""Journal per-call paths must not enumerate whole scopes / owners / collections.

Each test runs the SAME operation through the earlier implementation (the oracle in
``journal_enumeration_oracle``) and the current one on real file-backed SQLite databases,
asserts the observable results are identical, and asserts the SQL work (statements issued,
SQLite VM steps, ORM chunk rows loaded - counted by SQLAlchemy/sqlite3 hooks, not stubs) went
down.  Numbers are printed so ``pytest -s`` doubles as the before/after record.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from app.services.journal import fts
from app.services.journal.chunker import JournalTextChunk
from app.services.journal.repository import JournalRepository
from app.services.journal.schemas import JournalBulkDocItem, JournalSearchRequest
from app.services.journal.service import JournalService
from core.db.models import JournalEntry, JournalProject

from tests.journal_enumeration_oracle import (
    LegacyJournalRepository,
    LegacyJournalService,
    Work,
    build_repository,
)

OWNER = "owner-a"
SCOPE = "/work/proj"
CHILD = "/work/proj/d3"
DOCS = 1500


def _embed(texts, include_sparse):
    return ([[float(len(t)), 1.0] for t in texts],
            [{"1": 1.0} for _ in texts] if include_sparse else None)


class RecordingVectors:
    """The vector boundary: records groups and deletions (the SQL side is what is measured)."""

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.delete_calls = 0

    def upsert_chunk_groups(self, groups):
        return sum(len(group.chunks) for group in groups)

    def upsert_chunks(self, chunks, **kwargs):
        return len(chunks)

    def delete_by_entry_ids(self, entry_ids):
        self.delete_calls += 1
        self.deleted.extend(entry_ids)
        return len(entry_ids)

    def delete_by_entry_id(self, entry_id):
        self.delete_calls += 1
        self.deleted.append(entry_id)
        return 1


def _body(title: str, version: int = 0) -> str:
    return f"# {title}\n\nversion {version} of {title} " + "lorem ipsum dolor " * 20


def _seed(repo, *, scope, titles, owner=OWNER, version=0, body_of=None):
    body_of = body_of or (lambda title: _body(title, version))
    documents = [{
        "title": title, "body": body_of(title), "template_key": title,
        "chunks": [JournalTextChunk(chunk_id="t", entry_id="t", chunk_index=i, level=2,
                                    content=f"{body_of(title)} part {i}") for i in range(3)],
    } for title in titles]
    _, _, rows = repo.create_doc_entries_bulk(
        project_slug=scope, owner_user_id=owner, client="mcp", documents=documents)
    repo.mark_index_results_bulk([row["entry_id"] for row in rows], success=True)
    return rows


def _live(repo) -> set[tuple[str, str, str]]:
    with repo.session() as session:
        rows = (session.query(JournalProject.owner_user_id, JournalProject.slug,
                              JournalEntry.title, JournalEntry.content_hash)
                .join(JournalEntry, JournalEntry.project_id == JournalProject.project_id)
                .filter(JournalEntry.deleted_at.is_(None)).all())
        return {(f"{owner}|{slug}", title, content_hash) for owner, slug, title, content_hash in rows}


def _item(title: str, version: int) -> JournalBulkDocItem:
    body = _body(title, version)
    return JournalBulkDocItem(
        title=title, body=body, content_hash=hashlib.sha256(body.encode()).hexdigest(),
        size_bytes=len(body.encode()))


def _titles(count: int) -> list[str]:
    return [f"d{n % 20}/f{n}.md" for n in range(count)]


def _populate_nested_owner(repo) -> None:
    """A big scope, a nested child holding some of the SAME physical files, a foreign owner
    and an unrelated sibling scope reusing the same basenames."""
    titles = _titles(DOCS)
    _seed(repo, scope=SCOPE, titles=titles)
    _seed(repo, scope=CHILD, titles=[t.split("/", 1)[1] for t in titles if t.startswith("d3/")])
    _seed(repo, scope=SCOPE, titles=titles[:200], owner="owner-b")
    _seed(repo, scope="/work/other", titles=[t.split("/", 1)[1] for t in titles[:300]])
    # A pre-existing stale duplicate revision that the dedup path must still sweep.
    _seed(repo, scope=SCOPE, titles=["d5/f5.md"], version=99)


def _job(service):
    files = [_item(title, version=1) for title in _titles(60)]          # changed
    files += [_item(title, version=0) for title in _titles(130)[100:]]  # unchanged -> deduped
    return service.index_docs_bulk(
        project_slug=SCOPE, owner_user_id=OWNER, files=files, client="mcp")


def _phase_timed(service_cls, work: Work, phases: dict):
    """Record the SQL work of the latest-wins promotion phase inside a whole bulk job."""

    class Timed(service_cls):
        def _supersede_doc_versions(self, winners, **kwargs):
            before = (work.statements, work.vm_steps)
            try:
                return super()._supersede_doc_versions(winners, **kwargs)
            finally:
                phases["supersede"] = (work.statements - before[0], work.vm_steps - before[1])

        def _owner_physical_doc_index(self, **kwargs):
            before = (work.statements, work.vm_steps)
            try:
                return super()._owner_physical_doc_index(**kwargs)
            finally:
                phases["inventory"] = (work.statements - before[0], work.vm_steps - before[1])

        def _physical_doc_key(self, project_slug, title):
            phases["path_resolutions"] = phases.get("path_resolutions", 0) + 1
            return super()._physical_doc_key(project_slug, title)

    return Timed


def _schema_facts(engine) -> str:
    with engine.connect() as conn:
        indexes = sorted(row[0] for row in conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name IN ('journal_entries', 'journal_projects', 'journal_chunks')"))
        stat1 = bool(conn.exec_driver_sql(
            "SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'").fetchall())
    return f"sqlite_stat1={'present' if stat1 else 'absent'} {indexes}"


def test_bulk_job_is_equivalent_and_does_far_less_sql(tmp_path: Path):
    old_work, new_work = Work(), Work()
    old_phases: dict = {}
    new_phases: dict = {}
    old_repo, old_engine = build_repository(tmp_path / "old.db", old_work, LegacyJournalRepository)
    new_repo, new_engine = build_repository(tmp_path / "new.db", new_work)
    old_vectors, new_vectors = RecordingVectors(), RecordingVectors()
    old_service = _phase_timed(LegacyJournalService, old_work, old_phases)(
        repository=old_repo, vector_store=old_vectors, embed_fn=_embed)
    new_service = _phase_timed(JournalService, new_work, new_phases)(
        repository=new_repo, vector_store=new_vectors, embed_fn=_embed)
    _populate_nested_owner(old_repo)
    _populate_nested_owner(new_repo)
    assert _live(old_repo) == _live(new_repo)

    with old_work.measure():
        old_result = _job(old_service)
    with new_work.measure():
        new_result = _job(new_service)

    print(f"\nbulk job, {DOCS}-doc scope, 60 changed + 30 unchanged files"
          f"\n  legacy: {old_work}\n  current: {new_work}"
          f"\n  owner inventory (statements, vm_steps): {old_phases['inventory']} -> {new_phases['inventory']}"
          f"\n  stored paths resolved (abspath + scope canonicalisation, tens of us each): "
          f"{old_phases['path_resolutions']} -> {new_phases['path_resolutions']}"
          f"\n  latest-wins promotion (statements, vm_steps): {old_phases['supersede']} -> {new_phases['supersede']}"
          f"\n  vector deletes: {len(old_vectors.deleted)} in {old_vectors.delete_calls} calls (legacy) "
          f"vs {len(new_vectors.deleted)} in {new_vectors.delete_calls} call(s)")
    assert new_result == old_result
    assert _live(new_repo) == _live(old_repo)
    # every changed file (+ its nested copy) retired exactly one old revision, plus the stale dup
    assert len(new_vectors.deleted) == len(old_vectors.deleted) > 60
    assert new_vectors.delete_calls == 1 < old_vectors.delete_calls
    assert len(set(new_vectors.deleted)) == len(new_vectors.deleted)
    with new_repo.session() as session:
        assert fts.is_aligned(session)
    assert new_work.statements < old_work.statements
    # The step count depends on the SQLite planner (version, statistics), so it carries the
    # facts needed to tell a planner difference from a regression.
    assert new_work.vm_steps < old_work.vm_steps * 0.6, (
        f"sqlite {sqlite3.sqlite_version}: current {new_work.vm_steps} vm steps vs legacy "
        f"{old_work.vm_steps} (needs < 0.6x)"
        f"\n  current indexes: {_schema_facts(new_engine)}"
        f"\n  legacy indexes: {_schema_facts(old_engine)}")
    # promotion: one scope read for all files instead of one per file
    assert new_phases["supersede"][0] * 5 < old_phases["supersede"][0]
    assert new_phases["supersede"][1] * 5 < old_phases["supersede"][1]
    # the owner inventory resolves only rows whose basename can match an incoming file
    assert new_phases["path_resolutions"] * 5 < old_phases["path_resolutions"]


def test_windows_case_variant_scopes_supersede_identically(tmp_path: Path):
    """A legacy-cased sibling project for the same Windows scope is swept the same way."""
    results = []
    for name, repo_cls, service_cls in (("old", LegacyJournalRepository, LegacyJournalService),
                                        ("new", JournalRepository, JournalService)):
        work = Work()
        repo, _ = build_repository(tmp_path / f"{name}.db", work, repo_cls)
        service = service_cls(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)
        titles = [f"Docs/File{n}.md" for n in range(40)]
        _seed(repo, scope="E:/repo", titles=titles)
        _seed(repo, scope="e:/Repo", titles=titles[:10], version=7)  # legacy-cased project row
        service.index_docs_bulk(
            project_slug="E:/repo", owner_user_id=OWNER, client="mcp",
            files=[_item(t.lower(), version=2) for t in titles[:20]])
        results.append(_live(repo))
    assert results[0] == results[1]
    assert len(results[0]) == 40 + 0  # one live revision per file across both spellings


def test_owner_inventory_reads_only_related_scopes(tmp_path: Path):
    work = Work()
    repo, _ = build_repository(tmp_path / "j.db", work, LegacyJournalRepository)
    new_service = JournalService(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)
    old_service = LegacyJournalService(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)
    _populate_nested_owner(repo)
    # Windows case variant + ancestor/descendant of the incoming scope, plus lookalike siblings.
    _seed(repo, scope="/work/proj-backup", titles=[t for t in _titles(200)])
    _seed(repo, scope="/work", titles=[f"proj/{t}" for t in _titles(50)])

    titles = _titles(60)
    with work.measure():
        new = new_service._owner_physical_doc_index(
            owner_user_id=OWNER, project_slug=SCOPE, titles=titles)
    new_stats = repr(work)
    with work.measure():
        old = old_service._owner_physical_doc_index(
            owner_user_id=OWNER, project_slug=SCOPE, titles=titles)
    print(f"\nowner physical-path inventory\n  legacy: {work}\n  current: {new_stats}")
    wanted = {new_service._physical_doc_key(SCOPE, title) for title in titles}
    assert wanted
    for physical in wanted:
        assert sorted(new.get(physical, [])) == sorted(old.get(physical, [])), physical
    assert any(len(new.get(physical, [])) > 1 for physical in wanted)  # real fan-out exists


def test_record_fan_out_finds_the_same_targets(tmp_path: Path):
    repo, _ = build_repository(tmp_path / "j.db", Work(), LegacyJournalRepository)
    _populate_nested_owner(repo)
    new_service = JournalService(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)
    old_service = LegacyJournalService(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)
    title = "d3/f23.md"
    for service in (new_service, old_service):
        related = service._related_scope_doc_files(
            owner_user_id=OWNER, project_slug=SCOPE, titles=[title])
        from app.services.scope_paths import file_abspath
        this_abs = file_abspath(SCOPE, title)
        targets = {(s, t) for s, t in related if s != SCOPE and file_abspath(s, t) == this_abs}
        assert targets == {(CHILD, "f23.md")}


def test_reconcile_reads_the_scope_once_and_sweeps_the_same_docs(tmp_path: Path):
    old_work, new_work = Work(), Work()
    old_repo, _ = build_repository(tmp_path / "old.db", old_work, LegacyJournalRepository)
    new_repo, _ = build_repository(tmp_path / "new.db", new_work)
    for repo in (old_repo, new_repo):
        _seed(repo, scope=SCOPE, titles=_titles(400))
    old_service = LegacyJournalService(repository=old_repo, vector_store=RecordingVectors(), embed_fn=_embed)
    new_service = JournalService(repository=new_repo, vector_store=RecordingVectors(), embed_fn=_embed)

    keep = _titles(400)[:300]
    # nothing stale: only the reads remain, so the statement count IS the read count
    with old_work.measure():
        old_noop = old_service.reconcile_doc_scope(
            project_slug=SCOPE, live_titles=_titles(400), owner_user_id=OWNER)
    with new_work.measure():
        new_noop = new_service.reconcile_doc_scope(
            project_slug=SCOPE, live_titles=_titles(400), owner_user_id=OWNER)
    print(f"\nreconcile, nothing stale\n  legacy: {old_work}\n  current: {new_work}")
    assert new_noop.removed_entries == old_noop.removed_entries == 0
    assert new_work.statements < old_work.statements
    assert new_work.vm_steps < old_work.vm_steps

    old_swept = old_service.reconcile_doc_scope(
        project_slug=SCOPE, live_titles=keep, owner_user_id=OWNER, max_depth=1, patterns=["*.md"])
    new_swept = new_service.reconcile_doc_scope(
        project_slug=SCOPE, live_titles=keep, owner_user_id=OWNER, max_depth=1, patterns=["*.md"])
    assert (new_swept.removed_entries, new_swept.deleted_vectors, new_swept.skipped) == (
        old_swept.removed_entries, old_swept.deleted_vectors, old_swept.skipped) == (100, 100, False)
    assert _live(new_repo) == _live(old_repo)


GLOB_TITLES = [
    "a.md", "notes/a.md", "notes/b.md", "notes/deep/a.md", "other/notes/a.md", "sub/a.md",
    "sub/a.mdx", "sub/a.md.bak", "docs/x/y/z.md", "docs/readme.md", "src/util.py",
    "src/util_extra.py", "src/utilXextra.py", "src/deep/util.py", "x/y/z", "x/y/z/w.txt",
    "NOTES/A.MD", "Notes/Mixed.md", "é.md", "dir/é.md", "données/rapport.md", "a.md/inner.txt",
    "100%/p_q.md", "p_q.md", "pXq.md", ".hidden/c.md", "notes/.hid.md", "a.mdx",
    # legacy / odd spellings that the path identity normalises
    "\\notes\\legacy.md", "/lead/slash.md", "back\\slash\\a.md", "dbl//slash.md", "./dot.md",
    "mid/./dot.md", "trail/.", "KELVIN-\u212a.md", "straße.md",
]
GLOB_PATTERNS = [
    "a.md", "notes/a.md", "notes/", "notes", "NOTES/A.MD", "docs/**/*.md", "docs/*",
    "*.md", "**/z.md", "src/", "src/util*", "src/util_*", "[ab].md", "sub/a.md/", "é.md",
    "données/", "!a.md", "#c", "a.md ", "x/y/z", "legacy.md", "notes/legacy.md", "lead/slash.md",
    "slash.md", "back/slash/a.md", "dbl/slash.md", "dot.md", "mid/dot.md", "100%/p_q.md",
    "p_q.md", "p?q.md", "kelvin-k.md", "STRASSE.md", "straße.md", "nothing/here.md",
    "notes/**", "**/a.md", ".hidden/", "dir/", "src/util.py",
]


@pytest.mark.parametrize("scope", ["/gl", "C:/gl"])
def test_path_glob_prefilter_never_changes_the_matched_set(tmp_path: Path, scope: str):
    """Differential over odd stored spellings: SQL narrowing is a superset, so the ids the
    search is filtered to are exactly the full-listing ones for every pattern."""
    work = Work()
    repo, _ = build_repository(tmp_path / "j.db", work, LegacyJournalRepository)
    new_service = JournalService(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)
    old_service = LegacyJournalService(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)
    _seed(repo, scope=scope, titles=GLOB_TITLES,
          body_of=lambda t: f"body of {t!r}")
    _seed(repo, scope=scope, titles=["other.md"], owner="owner-b")
    narrowed = 0
    for pattern in GLOB_PATTERNS:
        request = JournalSearchRequest(
            project_slug=scope, query="q", mode="keyword", path_glob=pattern, owner_user_id=OWNER)
        old = old_service._resolve_path_glob_ids(request)
        new = new_service._resolve_path_glob_ids(request)
        assert sorted(new.ids) == sorted(old.ids), pattern
        if not old.ids:
            assert (new.scope_doc_count, new.sample_title) == (old.scope_doc_count, old.sample_title)
            assert (new_service._path_glob_empty_notice(request, new)
                    == old_service._path_glob_empty_notice(request, old)), pattern
        from app.services.journal.repository import path_glob_title_prefilter
        if path_glob_title_prefilter(request.path_glob, case_sensitive=not scope.startswith("C:")) is not None:
            narrowed += 1
    assert narrowed >= 20  # the fixture really exercises the SQL narrowing


def test_path_glob_search_lists_only_candidates(tmp_path: Path):
    old_work, new_work = Work(), Work()
    repo, _ = build_repository(tmp_path / "j.db", new_work, LegacyJournalRepository)
    _seed(repo, scope=SCOPE, titles=_titles(DOCS))
    new_service = JournalService(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)
    old_service = LegacyJournalService(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)
    for pattern in ("d7/", "d7/f27.md", "d7/f*.md"):
        request = JournalSearchRequest(project_slug=SCOPE, query="q", mode="keyword",
                                       path_glob=pattern, owner_user_id=OWNER)
        with new_work.measure():
            new = new_service._resolve_path_glob_ids(request)
        new_stats = repr(new_work)
        with new_work.measure():
            old = old_service._resolve_path_glob_ids(request)
        pairs = repo.scoped_live_doc_id_titles(
            project_slug=SCOPE, owner_user_id=OWNER, path_glob=request.path_glob)
        print(f"\npath_glob {pattern!r}: {len(new.ids)} ids; docs handed to path_matches "
              f"{DOCS} -> {len(pairs)}\n  legacy SQL: {new_work}\n  current SQL: {new_stats}")
        assert sorted(new.ids) == sorted(old.ids) and new.ids
        assert len(pairs) < DOCS / 10


def test_entry_listings_do_not_load_chunk_rows_and_keep_the_same_responses(tmp_path: Path):
    work = Work()
    repo, _ = build_repository(tmp_path / "j.db", work)
    rows = _seed(repo, scope=SCOPE, titles=_titles(120))
    service = JournalService(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)

    def reference_list(**kwargs):
        entries, total = repo.list_entries(**kwargs)
        return [service._entry_response(e, include_body=False, include_chunks=False)
                for e in entries], total

    with work.measure():
        reference, reference_total = reference_list(
            project_slug=[SCOPE], owner_user_id=OWNER, limit=50, offset=10)
    legacy_chunks = work.chunk_rows
    with work.measure():
        listed = service.list_entries(
            project_slug=SCOPE, owner_user_id=OWNER, limit=50, offset=10)
    print(f"\nlist_entries(limit=50): chunk rows loaded {legacy_chunks} -> {work.chunk_rows}")
    assert [e.model_dump() for e in listed.entries] == [e.model_dump() for e in reference]
    assert listed.total == reference_total
    assert legacy_chunks == 150 and work.chunk_rows == 0
    assert all(e.chunk_count == 3 for e in listed.entries)

    ids = [row["entry_id"] for row in rows[:10]][::-1]
    with work.measure():
        eager = repo.get_entries_by_ids(ids, owner_user_id=OWNER)
    eager_chunks = work.chunk_rows
    with work.measure():
        light = repo.get_entries_by_ids(ids, owner_user_id=OWNER, load_chunks=False, load_body=False)
    print(f"get_entries_by_ids(10): chunk rows loaded {eager_chunks} -> {work.chunk_rows}")
    assert list(light) == list(eager) == ids
    assert eager_chunks == 30 and work.chunk_rows == 0
    for entry_id in ids:
        assert (service._entry_response(light[entry_id], include_body=False, include_chunks=False).model_dump()
                == service._entry_response(eager[entry_id], include_body=False, include_chunks=False).model_dump())
    with pytest.raises(Exception):  # a deferred body is not silently empty
        _ = light[ids[0]].body


ODD_SCOPE_TITLES = GLOB_TITLES + ["notes/a.md ", "UPPER/Case.MD", "upper/case.md", "ß/x.md", "SS/x.md"]


@pytest.mark.parametrize("scope", ["/odd", "C:/odd"])
def test_basename_prefilter_never_loses_a_row_the_exact_check_would_keep(tmp_path: Path, scope: str):
    """The SQL basename narrowing behind the supersede and fan-out lookups is lossless over
    legacy spellings (backslashes, leading slash, // and . segments, trailing slash, non-ASCII,
    case variants): per incoming title the narrowed result equals the brute-force one."""
    from app.services.journal.repository import path_identity, title_basename_key
    from app.services.scope_paths import is_windows_scope

    repo, _ = build_repository(tmp_path / "j.db", Work(), LegacyJournalRepository)
    case_sensitive = not is_windows_scope(scope)
    for index, title in enumerate(ODD_SCOPE_TITLES):
        _seed(repo, scope=scope, titles=[title], body_of=lambda t, i=index: f"body {i} {t!r}")
    _seed(repo, scope=scope, titles=ODD_SCOPE_TITLES[:5], owner="owner-b")
    everything = repo.legacy_all_live_doc_files(owner_user_id=OWNER)
    assert len(everything) == len(ODD_SCOPE_TITLES)
    for incoming in ODD_SCOPE_TITLES + ["a.md", "A.MD", "straße.md", "x.md", "notes/legacy.md"]:
        key = title_basename_key(incoming)
        expected_files = sorted((slug, title) for slug, title in everything
                                if title_basename_key(title) == key)
        assert sorted(repo.live_doc_files_in_scopes(
            owner_user_id=OWNER, project_slug=[scope], titles=[incoming])) == expected_files, incoming
        identity = path_identity(incoming, case_sensitive)
        if identity is None:
            continue
        brute = sorted(repo.legacy_superseded_version_ids(
            project_slug=[scope], source_kind="doc", title=incoming, owner_user_id=OWNER,
            exclude_entry_id="", case_sensitive=case_sensitive))
        assert sorted(repo.superseded_version_ids(
            project_slug=[scope], source_kind="doc", title=incoming, owner_user_id=OWNER,
            exclude_entry_id="", case_sensitive=case_sensitive)) == brute, incoming


def test_owner_inventory_matches_the_full_scan_with_windows_variants_and_legacy_titles(tmp_path: Path):
    repo, _ = build_repository(tmp_path / "j.db", Work(), LegacyJournalRepository)
    new_service = JournalService(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)
    old_service = LegacyJournalService(repository=repo, vector_store=RecordingVectors(), embed_fn=_embed)
    clean = ["notes/a.md", "notes/b.md", "notes/deep/a.md", "docs/readme.md", "Docs/ReadMe.md",
             "\\notes\\legacy.md", "/lead/slash.md", "back\\slash\\a.md"]
    _seed(repo, scope="C:/odd", titles=clean)
    _seed(repo, scope="c:/ODD", titles=["notes/a.md", "docs/README.md"], body_of=lambda t: f"variant {t}")
    _seed(repo, scope="C:/odd/notes", titles=["a.md", "b.md", "deep/a.md", "legacy.md"])
    _seed(repo, scope="C:/odd/notes/deep", titles=["a.md"])
    _seed(repo, scope="C:/odd-backup", titles=["notes/a.md"])  # lookalike sibling: never related
    _seed(repo, scope="D:/odd", titles=["notes/a.md"])
    incoming = ["notes/a.md", "notes/b.md", "notes/deep/a.md", "docs/readme.md", "notes/legacy.md",
                "lead/slash.md", "back/slash/a.md"]
    new = new_service._owner_physical_doc_index(
        owner_user_id=OWNER, project_slug="C:/odd", titles=incoming)
    old = old_service._owner_physical_doc_index(
        owner_user_id=OWNER, project_slug="C:/odd", titles=incoming)
    checked = 0
    for title in incoming:
        physical = new_service._physical_doc_key("C:/odd", title)
        assert sorted(new.get(physical, [])) == sorted(old.get(physical, [])), title
        checked += len(old.get(physical, []))
    assert checked >= 10, checked
    assert all(slug.casefold() != "d:/odd" and "backup" not in slug.casefold()
               for entries in new.values() for slug, _ in entries)
