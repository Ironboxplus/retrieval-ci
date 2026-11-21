"""Journal search of a subdirectory nobody indexed falls back to its indexed parent.

Real SQLite (journal tables, FTS5 and the production ``index_catalog`` schema share one
database) and a real Qdrant scratch collection. Only DOC entries are narrowed to the
subdirectory: the parent's memory entries and ``.conversations/`` segments are project
history with no file path and keep taking part.
"""
from __future__ import annotations

import time
import uuid
import warnings

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.services.journal.repository import JournalRepository  # noqa: E402
from app.services.journal.schemas import JournalRecordRequest, JournalSearchRequest  # noqa: E402
from app.services.journal.service import JournalService  # noqa: E402
from app.services.journal.vector_store import JournalVectorStore  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402
from tests.catalog_sqlite import CatalogDb  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
OWNER = "journal-subdir-owner"
DIM = 8
CONVERSATION = ".conversations/claude/2026-11-01_abcd/001.md"
DOCS = ["docs/a.md", "docs/sub/b.md", "src/c.md", "documents/d.md", CONVERSATION]


def _embed(texts, include_sparse):
    return [[1.0] + [0.0] * (DIM - 1) for _ in texts], None


@pytest.fixture()
def env():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=30)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    from config import settings

    name = f"journal_subdir_scope_{uuid.uuid4().hex[:8]}_test"
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": params} if settings.embedding.supports_sparse else params)
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = name
    inner.client = client

    catalog = CatalogDb()
    service = JournalService(
        repository=JournalRepository(session_factory=catalog.session),
        vector_store=JournalVectorStore(collection_name=name, store=inner),
        embed_fn=_embed, catalog=catalog.service)

    class Env:
        pass

    state = Env()
    state.service, state.catalog = service, catalog
    try:
        yield state
    finally:
        for attempt in range(3):
            try:
                client.delete_collection(name)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    warnings.warn(f"could not drop scratch collection {name}: {exc}")
                time.sleep(0.5 * (attempt + 1))


def _doc(env, scope, title, body="shared marker text"):
    env.service.record(JournalRecordRequest(
        project_slug=scope, client="claude", source_kind="doc", title=title, body=body,
        owner_user_id=OWNER, dedupe_policy="return_existing"))
    env.catalog.add_files(scope, [title], owner=OWNER, kind="doc")


def _memory(env, scope, body="shared marker lesson"):
    env.service.record(JournalRecordRequest(
        project_slug=scope, client="codex", source_kind="memory", body=body,
        owner_user_id=OWNER))


def _seed_parent(env, scope="/proj"):
    for title in DOCS:
        _doc(env, scope, title)
    _memory(env, scope)


def _search(env, scope, mode, **extra):
    return env.service.search(JournalSearchRequest(
        project_slug=scope, query="marker", mode=mode, top_k=50, owner_user_id=OWNER,
        dedup_by_entry=True, **extra))


def _labels(response):
    return sorted(hit.entry.title or f"<{hit.entry.source_kind}>" for hit in response.hits)


MODES = ["keyword", "dense"]


@pytest.mark.parametrize("mode", MODES)
def test_subdirectory_keeps_its_docs_the_parents_memory_and_conversations(env, mode):
    _seed_parent(env)
    response = _search(env, "/proj/docs", mode)
    assert _labels(response) == sorted(
        ["<memory>", CONVERSATION, "docs/a.md", "docs/sub/b.md"])
    assert response.project_slug == "/proj/docs"
    assert (response.resolved_scope, response.resolved_path_prefix) == ("/proj", "docs")


@pytest.mark.parametrize("mode", MODES)
def test_docs_outside_the_subdirectory_never_appear(env, mode):
    _seed_parent(env)
    titles = _labels(_search(env, "/proj/docs", mode))
    assert "src/c.md" not in titles and "documents/d.md" not in titles


@pytest.mark.parametrize("mode", MODES)
def test_parent_scope_search_is_unchanged(env, mode):
    _seed_parent(env)
    response = _search(env, "/proj", mode)
    assert len(response.hits) == len(DOCS) + 1
    assert response.resolved_scope is None and response.resolved_path_prefix is None


@pytest.mark.parametrize("mode", MODES)
def test_path_glob_narrows_to_matching_docs_relative_to_the_subdirectory(env, mode):
    _seed_parent(env)
    response = _search(env, "/proj/docs", mode, path_glob="sub/**")
    assert _labels(response) == ["docs/sub/b.md"]
    assert _labels(_search(env, "/proj/docs", mode, path_glob="*.md")) == [
        "docs/a.md", "docs/sub/b.md"]


@pytest.mark.parametrize("mode", MODES)
def test_source_kind_doc_still_applies_inside_the_subdirectory(env, mode):
    _seed_parent(env)
    response = _search(env, "/proj/docs", mode, source_kind="doc")
    assert _labels(response) == sorted([CONVERSATION, "docs/a.md", "docs/sub/b.md"])


@pytest.mark.parametrize("mode", MODES)
def test_subdirectory_without_indexed_docs_explains_itself_and_keeps_memory(env, mode):
    _seed_parent(env)
    response = _search(env, "/proj/node_modules", mode)
    assert _labels(response) == sorted(["<memory>", CONVERSATION])
    assert "/proj" in response.notice and "node_modules" in response.notice
    assert "journal_index_dir" in response.notice


@pytest.mark.parametrize("mode", MODES)
def test_subdirectory_with_memory_of_its_own_is_not_redirected(env, mode):
    _seed_parent(env)
    _memory(env, "/proj/docs", body="shared marker child lesson")
    response = _search(env, "/proj/docs", mode)
    assert _labels(response) == ["<memory>"]
    assert response.resolved_scope is None


@pytest.mark.parametrize("mode", MODES)
def test_subdirectory_with_its_own_docs_uses_them_only(env, mode):
    _seed_parent(env)
    _doc(env, "/proj/docs", "own.md")
    assert _labels(_search(env, "/proj/docs", mode)) == ["own.md"]


@pytest.mark.parametrize("mode", MODES)
def test_unindexed_scope_without_ancestor_returns_nothing_quietly(env, mode):
    _seed_parent(env)
    response = _search(env, "/elsewhere/docs", mode)
    assert response.hits == [] and response.resolved_scope is None


@pytest.mark.parametrize("mode", MODES)
def test_windows_subdirectory_is_case_insensitive(env, mode):
    scope = "E:/proj"
    for title in ["Docs/A.md", "docs/sub/b.md", "Src/c.md"]:
        _doc(env, scope, title)
    response = _search(env, "e:/PROJ/DOCS", mode)
    assert _labels(response) == ["Docs/A.md", "docs/sub/b.md"]
    assert (response.resolved_scope, response.resolved_path_prefix) == ("E:/proj", "docs")


def test_another_owners_parent_index_is_not_used(env):
    _seed_parent(env)
    other = JournalSearchRequest(
        project_slug="/proj/docs", query="marker", mode="keyword", top_k=50,
        owner_user_id="somebody-else")
    assert env.service.search(other).hits == []
