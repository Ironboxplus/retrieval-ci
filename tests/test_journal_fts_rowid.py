"""journal_fts is keyed by rowid = journal_entries.id (real SQLite, no fakes).

`entry_id` is an UNINDEXED FTS5 column, so deleting by it scans every body; rowid is the
only indexed key of an FTS5 table. These tests pin the keying, the delete plan, the
one-time realignment of tables written with auto rowids, and its idempotence.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.services.journal import fts
from app.services.journal.chunker import JournalTextChunk
from app.services.journal.repository import JournalRepository, new_entry_id
from app.services.journal.schemas import JournalRecordRequest
from app.services.journal.service import JournalService
from core.db.database import Base
from core.db.models import JournalEntry

# The statement the code used before rowid keying; kept only to prove its plan and cost.
LEGACY_DELETE = f"DELETE FROM {fts.TABLE} WHERE entry_id = :entry_id"
LEGACY_INSERT = (
    f"INSERT INTO {fts.TABLE} "
    "(entry_id, owner_user_id, project_slug, source_kind, title, summary, body, tags) "
    "VALUES (:entry_id, :owner, :project, :source, :title, :summary, :body, :tags)"
)


def _repo(tmp_path) -> tuple[JournalRepository, object]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'journal.db'}", connect_args={"check_same_thread": False})
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

    return JournalRepository(session_factory=session_factory), engine


def _fts_rows(engine) -> list[tuple[int, str]]:
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(
            text(f"SELECT rowid, entry_id FROM {fts.TABLE} ORDER BY rowid"))]


def _entry_ids(engine) -> dict[str, int]:
    with engine.connect() as conn:
        return {r[1]: r[0] for r in conn.execute(
            text("SELECT id, entry_id FROM journal_entries"))}


def _plan(engine, sql: str, params: dict) -> str:
    with engine.connect() as conn:
        return " | ".join(r[3] for r in conn.execute(text("EXPLAIN QUERY PLAN " + sql), params))


def _doc_template(title: str, body: str) -> dict:
    return {"title": title, "body": body, "template_key": title, "chunks": [
        JournalTextChunk(chunk_id="tmpl", entry_id="tmpl", chunk_index=0, level=2, content=body)]}


def _create(repo, body: str, *, title: str | None = None, slug: str = "p", owner=None):
    entry, _, _ = repo.create_entry(
        project_slug=slug, client="codex", source_kind="memory", title=title, summary=None,
        body=body, tags=["t1", "t2"], metadata={}, owner_user_id=owner)
    return entry


def test_delete_plan_is_not_the_entry_id_scan(tmp_path):
    repo, engine = _repo(tmp_path)
    _create(repo, "plan probe")
    legacy = _plan(engine, LEGACY_DELETE, {"entry_id": "x"})
    current = _plan(engine, fts.DELETE_BY_ROWID, {"rowid": 1})
    print(f"\nlegacy plan : {legacy}\ncurrent plan: {current}")
    assert legacy == "SCAN journal_fts VIRTUAL TABLE INDEX 0:"
    assert current != legacy
    # rowid equality is planned as a point lookup (idxStr "="), not a scan of all rows.
    assert current == "SCAN journal_fts VIRTUAL TABLE INDEX 0:="


def test_single_and_bulk_creates_key_fts_rowid_to_entry_id(tmp_path):
    repo, engine = _repo(tmp_path)
    singles = [_create(repo, f"single body {i}") for i in range(3)]
    # Leave a gap in the id sequence so rowid == id cannot hold by coincidence of counting.
    repo.soft_delete_entry(singles[1].entry_id)
    _, _, results = repo.create_doc_entries_bulk(
        project_slug="p", owner_user_id=None, client="mcp",
        documents=[_doc_template(f"d{i}.md", f"bulk body {i}") for i in range(4)])
    ids = _entry_ids(engine)
    rows = _fts_rows(engine)
    assert rows, "expected FTS rows"
    assert all(ids[entry_id] == rowid for rowid, entry_id in rows)
    assert {entry_id for _, entry_id in rows} == (
        {singles[0].entry_id, singles[2].entry_id} | {r["entry_id"] for r in results})


def test_soft_delete_paths_remove_exactly_the_right_rows(tmp_path):
    repo, engine = _repo(tmp_path)
    entries = [_create(repo, f"shared needle item{i}", title=f"t{i}") for i in range(6)]
    repo.soft_delete_entry(entries[0].entry_id)
    removed = repo.soft_delete_entries_bulk(
        [entries[1].entry_id, entries[2].entry_id, "journal_entry_missing"], reason="test")
    assert set(removed) == {entries[1].entry_id, entries[2].entry_id}
    assert {e for _, e in _fts_rows(engine)} == {e.entry_id for e in entries[3:]}
    hits, _ = repo.keyword_search(query="needle", limit=20)
    assert {e.entry_id for e, _ in hits} == {e.entry_id for e in entries[3:]}
    # Untouched entries still match by their own distinct term.
    assert [e.entry_id for e, _ in repo.keyword_search(query="item4")[0]] == [entries[4].entry_id]


def test_latest_wins_supersede_drops_only_the_old_version(tmp_path):
    repo, engine = _repo(tmp_path)

    class _NoVectors:
        def upsert_chunks(self, chunks, **kwargs):
            return len(chunks)

        def delete_by_entry_id(self, entry_id):
            return 0

    def embed(texts, include_sparse):
        return ([[1.0, 0.0, 0.0] for _ in texts],
                [{"1": 1.0} for _ in texts] if include_sparse else None)

    service = JournalService(repository=repo, vector_store=_NoVectors(), embed_fn=embed)

    def doc(title, body):
        return service.record(JournalRecordRequest(
            project_slug="retrival", client="mcp", source_kind="doc", title=title,
            body=body, owner_user_id="user-1", dedupe_policy="return_existing"))

    other = doc("other.md", "zeta unrelated sibling")
    old = doc("plan.md", "alpha original revision")
    new = doc("plan.md", "beta replacement revision")
    assert new.entry.entry_id != old.entry.entry_id
    live = {e for _, e in _fts_rows(engine)}
    assert live == {other.entry.entry_id, new.entry.entry_id}
    assert repo.keyword_search(query="alpha", owner_user_id="user-1")[0] == []
    assert [e.entry_id for e, _ in repo.keyword_search(query="beta")[0]] == [new.entry.entry_id]
    assert [e.entry_id for e, _ in repo.keyword_search(query="zeta")[0]] == [other.entry.entry_id]


def _build_legacy_layout(repo, engine) -> list[str]:
    """Write the table exactly as the pre-rowid code did: auto rowids, delete by entry_id."""
    with repo.session_factory() as session:
        fts.ensure_fts(session)
        from core.db.models import JournalProject
        project = JournalProject(slug="p", owner_user_id="u1")
        other = JournalProject(slug="q", owner_user_id=None)
        session.add_all([project, other])
        session.flush()
        spec = [  # (project, title, body) - distinct term counts so bm25 never ties
            (project, "a", "kiwi " * 5 + "filler words here"),
            (project, "b", "kiwi " * 3 + "filler words here and more filler"),
            (other, "c", "kiwi mango " + "filler " * 20),
            (project, "d", "mango mango mango kiwi"),
            (other, "e", "mango " * 2 + "kiwi " * 2 + "tail"),
            (project, "f", "kiwi " * 9),
            (project, "g", "mango " + "kiwi " * 4 + "noise noise"),
        ]
        entries = []
        for owner_project, title, body in spec:
            entry = JournalEntry(
                entry_id=new_entry_id(), project_id=owner_project.project_id, client="mcp",
                source_kind="doc", title=title, body=body, tags_json=["x", "y"],
                metadata_json={}, content_hash=title)
            session.add(entry)
            entries.append(entry)
        session.flush()
        projects = {project.project_id: project, other.project_id: other}

        def legacy_index(entry):
            owner_project = projects[entry.project_id]
            session.execute(text(LEGACY_INSERT), {
                "entry_id": entry.entry_id, "owner": owner_project.owner_user_id or "",
                "project": owner_project.slug, "source": entry.source_kind,
                "title": entry.title or "", "summary": "", "body": entry.body, "tags": "x y"})

        for entry in reversed(entries):  # insertion order != id order -> misaligned rowids
            legacy_index(entry)
        # A deleted entry leaves a gap, as the old per-id delete did.
        session.execute(text(LEGACY_DELETE), {"entry_id": entries[2].entry_id})
        entries[2].deleted_at = entries[2].created_at
        # ...and a soft-deleted entry whose FTS row was never removed (stale row).
        entries[4].deleted_at = entries[4].created_at
        session.flush()
        return [e.entry_id for e in entries]


QUERIES = [
    ("kiwi", {}), ("mango", {}), ("kiwi mango", {}), ("filler", {}), ("noise tail", {}),
    ("kiwi", {"owner_user_id": "u1"}), ("kiwi", {"project_slug": "q"}),
]


def _search_all(repo):
    return [[e.entry_id for e, _ in repo.keyword_search(query=q, limit=20, **kw)[0]]
            for q, kw in QUERIES]


def test_legacy_layout_is_realigned_once_without_changing_search_results(tmp_path, caplog):
    repo, engine = _repo(tmp_path)
    entry_ids = _build_legacy_layout(repo, engine)
    ids = _entry_ids(engine)
    legacy_rows = _fts_rows(engine)
    assert any(ids[e] != rowid for rowid, e in legacy_rows), "fixture must be misaligned"
    assert len(legacy_rows) == 6  # 7 entries - gap(c); stale(e) still present

    # A reader whose guard is pre-satisfied sees the legacy table as production does today.
    legacy_reader = JournalRepository(session_factory=repo.session_factory)
    legacy_reader._fts_aligned = True
    before = _search_all(legacy_reader)
    assert before[0], "queries must hit something"

    # A brand-new repository runs the guard on its first session.
    fresh = JournalRepository(session_factory=repo.session_factory)
    with caplog.at_level(logging.INFO, logger="journal.fts"):
        after = _search_all(fresh)
    realign_lines = [r.getMessage() for r in caplog.records if "realigned" in r.getMessage()]
    print(*realign_lines, sep=chr(10))
    assert len(realign_lines) == 1 and "rebuilt 5 live entries" in realign_lines[0]
    assert after == before

    rows = _fts_rows(engine)
    live = {e: i for e, i in ids.items() if e not in (entry_ids[2], entry_ids[4])}
    assert rows == sorted((i, e) for e, i in live.items())
    with fresh.session() as session:
        assert fts.is_aligned(session)

    # Second run is a no-op: nothing rebuilt, rows byte-for-byte the same, results the same.
    caplog.clear()
    again = JournalRepository(session_factory=repo.session_factory)
    with caplog.at_level(logging.INFO, logger="journal.fts"):
        assert _search_all(again) == before
    messages = [r.getMessage() for r in caplog.records]
    assert not any("realigned" in m for m in messages)
    assert any("no rebuild" in m for m in messages)
    assert _fts_rows(engine) == rows


def test_rebuilt_row_content_matches_what_index_entry_writes(tmp_path):
    repo, engine = _repo(tmp_path)
    entry = _create(repo, "content parity body", title="Parity", slug="proj", owner="u9")
    with engine.connect() as conn:
        written = [tuple(r) for r in conn.execute(text(
            f"SELECT rowid, entry_id, owner_user_id, project_slug, source_kind, title, "
            f"summary, body, tags FROM {fts.TABLE}"))]
    assert repo.reindex_fts() == 1
    with engine.connect() as conn:
        rebuilt = [tuple(r) for r in conn.execute(text(
            f"SELECT rowid, entry_id, owner_user_id, project_slug, source_kind, title, "
            f"summary, body, tags FROM {fts.TABLE}"))]
    assert rebuilt == written == [(
        entry.id, entry.entry_id, "u9", "proj", "memory", "Parity", "", "content parity body",
        "t1 t2")]


def test_bulk_create_cost_old_vs_new_delete_statement(tmp_path):
    """Report numbers only: the same DB, the legacy delete vs the rowid delete."""
    repo, engine = _repo(tmp_path)
    filler = "lorem ipsum dolor sit amet consectetur adipiscing " * 80  # ~4 KB
    with repo.session_factory() as session:
        fts.ensure_fts(session)
        project = repo.get_or_create_project(session, slug="seed")
        for i in range(3000):
            entry = JournalEntry(
                entry_id=new_entry_id(), project_id=project.project_id, client="mcp",
                source_kind="doc", title=f"seed{i}", body=filler, tags_json=[],
                metadata_json={}, content_hash=str(i))
            session.add(entry)
            session.flush()
            fts.index_entry(session, rowid=entry.id, entry_id=entry.entry_id,
                            owner_user_id=None, project_slug="seed", source_kind="doc",
                            title=entry.title, summary=None, body=filler, tags=[])
    with repo.session_factory() as session:
        assert fts.is_aligned(session)

    docs = [_doc_template(f"new{i}.md", f"fresh content {i} " + filler) for i in range(200)]
    started = time.monotonic()
    repo.create_doc_entries_bulk(project_slug="bulk", owner_user_id=None, client="mcp",
                                 documents=docs)
    bulk_new = time.monotonic() - started

    def timed(sql, params_list):
        t0 = time.monotonic()
        with engine.begin() as conn:
            for params in params_list:
                conn.execute(text(sql), params)
        return time.monotonic() - t0

    legacy = timed(LEGACY_DELETE, [{"entry_id": f"nope{i}"} for i in range(200)])
    current = timed(fts.DELETE_BY_ROWID, [{"rowid": 10_000_000 + i} for i in range(200)])
    print(f"\nFTS rows={len(_fts_rows(engine))}: bulk create of 200 docs (new)={bulk_new:.2f}s; "
          f"200 legacy deletes={legacy:.2f}s; 200 rowid deletes={current:.3f}s")
    assert current < legacy
