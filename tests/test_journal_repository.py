"""TDD for ACE journal SQL fact store."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from app.services.journal.repository import JournalRepository


def _repo():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
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

    return JournalRepository(session_factory=session_factory)


def test_create_entry_creates_project_and_append_only_entry():
    repo = _repo()

    entry, project, deduped = repo.create_entry(
        project_slug="retrival",
        client="codex",
        source_kind="session_summary",
        title="Journal design",
        summary="Storage decision",
        body="Use SQL as fact source and Qdrant as rebuildable index.",
        tags=["journal", "storage"],
        metadata={"branch": "main"},
    )

    assert deduped is False
    assert project.slug == "retrival"
    assert entry.entry_id.startswith("journal_entry_")
    assert entry.project_id == project.project_id
    assert entry.content_hash


def test_return_existing_dedupe_policy_reuses_same_entry():
    repo = _repo()
    kwargs = dict(
        project_slug="retrival",
        client="codex",
        source_kind="memory",
        title="Same",
        summary=None,
        body="same body",
        tags=[],
        metadata={},
        dedupe_policy="return_existing",
    )

    first, _, first_deduped = repo.create_entry(**kwargs)
    second, _, second_deduped = repo.create_entry(**kwargs)

    assert first_deduped is False
    assert second_deduped is True
    assert second.entry_id == first.entry_id


def test_replace_chunks_and_get_entry_returns_chunks_in_order():
    repo = _repo()
    entry, _, _ = repo.create_entry(
        project_slug="retrival",
        client="codex",
        source_kind="memory",
        title=None,
        summary=None,
        body="body",
        tags=[],
        metadata={},
    )

    repo.replace_chunks(entry.entry_id, [
        {"chunk_id": "chunk_b", "chunk_index": 1, "level": 2, "heading": "Body", "content": "second"},
        {"chunk_id": "chunk_a", "chunk_index": 0, "level": 1, "heading": "Summary", "content": "first"},
    ])
    loaded = repo.get_entry(entry.entry_id)

    assert loaded is not None
    assert [chunk.chunk_index for chunk in loaded.chunks] == [0, 1]
    assert [chunk.content for chunk in loaded.chunks] == ["first", "second"]


def test_soft_delete_hides_entry_from_default_reads():
    repo = _repo()
    entry, _, _ = repo.create_entry(
        project_slug="retrival",
        client="codex",
        source_kind="memory",
        title=None,
        summary=None,
        body="body",
        tags=[],
        metadata={},
    )

    assert repo.soft_delete_entry(entry.entry_id, reason="test cleanup") is True

    assert repo.get_entry(entry.entry_id) is None
    deleted = repo.get_entry(entry.entry_id, include_deleted=True)
    assert deleted is not None
    assert deleted.deleted_reason == "test cleanup"


def test_keyword_search_finds_entry_by_literal_term():
    repo = _repo()
    repo.create_entry(
        project_slug="retrival", client="codex", source_kind="memory",
        title="Observability sprint", summary=None,
        body="Wired Prometheus metrics and Loki relay for gateway W130.",
        tags=["observability"], metadata={},
    )
    repo.create_entry(
        project_slug="retrival", client="codex", source_kind="memory",
        title="Unrelated", summary=None, body="Downloaded CVPR PDFs.",
        tags=[], metadata={},
    )

    rows, total = repo.keyword_search(query="prometheus", project_slug="retrival")

    assert total == 1
    (entry, score), = rows
    assert "Prometheus" in entry.body
    assert score is not None


def test_keyword_search_is_any_token_not_all():
    # A natural-language query must not require EVERY token to be present (FTS5's
    # default AND made the tool's default keyword mode silently return nothing for
    # sentence queries). OR semantics: an entry matching some terms still ranks.
    repo = _repo()
    repo.create_entry(
        project_slug="retrival", client="codex", source_kind="memory",
        title="Observability sprint", summary=None,
        body="Wired Prometheus metrics and Loki relay for gateway W130.",
        tags=["observability"], metadata={},
    )
    # "dashboards"/"grafana" are absent; only "prometheus"/"metrics" hit → still found.
    rows, total = repo.keyword_search(
        query="prometheus metrics dashboards grafana", project_slug="retrival")
    assert total == 1, "any-token (OR) keyword match should still find the entry"
    assert "Prometheus" in rows[0][0].body


def test_keyword_search_prefix_and_owner_isolation():
    repo = _repo()
    repo.create_entry(
        project_slug="retrival", client="codex", source_kind="memory",
        title=None, summary=None, body="observability metrics", tags=[], metadata={},
        owner_user_id="user-1",
    )
    repo.create_entry(
        project_slug="retrival", client="codex", source_kind="memory",
        title=None, summary=None, body="observability dashboards", tags=[], metadata={},
        owner_user_id="user-2",
    )

    # Prefix match ("observ" → "observability") + cross-user vs scoped.
    all_rows, all_total = repo.keyword_search(query="observ")
    scoped_rows, scoped_total = repo.keyword_search(query="observ", owner_user_id="user-1")

    assert all_total == 2
    assert scoped_total == 1
    assert scoped_rows[0][0].body == "observability metrics"


def test_keyword_search_excludes_soft_deleted():
    repo = _repo()
    entry, _, _ = repo.create_entry(
        project_slug="retrival", client="codex", source_kind="memory",
        title=None, summary=None, body="ephemeral note about qdrant", tags=[], metadata={},
    )
    assert repo.keyword_search(query="qdrant")[1] == 1

    repo.soft_delete_entry(entry.entry_id)

    assert repo.keyword_search(query="qdrant")[1] == 0


def test_reindex_fts_backfills_existing_entries():
    repo = _repo()
    entry, _, _ = repo.create_entry(
        project_slug="retrival", client="codex", source_kind="memory",
        title=None, summary=None, body="reindex target term", tags=[], metadata={},
    )
    # Simulate a legacy row that predates FTS indexing.
    from app.services.journal import fts
    with repo.session() as session:
        fts.ensure_fts(session)
        fts.delete_entry(session, entry.id)
    assert repo.keyword_search(query="reindex")[1] == 0

    assert repo.reindex_fts() == 1
    assert repo.keyword_search(query="reindex")[1] == 1


def test_same_project_slug_is_isolated_by_owner_user_id():
    repo = _repo()
    first, first_project, _ = repo.create_entry(
        project_slug="retrival",
        client="codex",
        source_kind="memory",
        title="User one",
        summary=None,
        body="private to user one",
        tags=[],
        metadata={},
        owner_user_id="user-1",
    )
    second, second_project, _ = repo.create_entry(
        project_slug="retrival",
        client="codex",
        source_kind="memory",
        title="User two",
        summary=None,
        body="private to user two",
        tags=[],
        metadata={},
        owner_user_id="user-2",
    )

    assert first_project.project_id != second_project.project_id
    assert repo.get_entry(first.entry_id, owner_user_id="user-1") is not None
    assert repo.get_entry(first.entry_id, owner_user_id="user-2") is None

    user_one_rows, user_one_total = repo.list_entries(project_slug="retrival", owner_user_id="user-1")
    user_two_rows, user_two_total = repo.list_entries(project_slug="retrival", owner_user_id="user-2")

    assert user_one_total == 1
    assert user_one_rows[0].entry_id == first.entry_id
    assert user_two_total == 1
    assert user_two_rows[0].entry_id == second.entry_id
