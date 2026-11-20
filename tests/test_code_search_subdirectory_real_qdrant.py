"""Searching a subdirectory nobody indexed falls back to its indexed parent directory.

Real SQLite (the production ``index_catalog`` schema) and a real Qdrant scratch collection
on loopback, driven through the same route functions the HTTP handlers call. The parent
scope is indexed once; the child is only ever named in the request.
"""
from __future__ import annotations

import time
import uuid
import warnings

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.api.routes import code_kb as routes  # noqa: E402
from app.services.code_kb.schemas import (  # noqa: E402
    CodeChunkIn, CodeFileIn, CodeIndexRequest, CodeSearchRequest,
    FederatedCodeSearchRequest,
)
from app.services.code_kb.service import CodeKBService  # noqa: E402
from app.services.code_kb.vector_store import CodeVectorStore  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402
from tests.catalog_sqlite import CatalogDb  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
OWNER = "subdir-owner"
DIM = 16
NEEDLE = "needle"

PARENT_FILES = [
    "app/api/routes.py", "app/api/deep/handler.py", "app/main.py", "application/other.py",
    "core/db.py", "docs/guide.md", "README.md",
]


def _embed(texts):
    rows = []
    for text in texts:
        row = [0.0] * DIM
        row[sum(map(ord, text)) % DIM] = 1.0
        rows.append(row)
    return rows


@pytest.fixture()
def env(monkeypatch):
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=30)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    from config import settings

    name = f"code_subdir_scope_{uuid.uuid4().hex[:8]}_test"
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": params} if settings.embedding.supports_sparse else params)
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = name
    inner.client = client
    vectors = CodeVectorStore(collection_name=name, store=inner)
    vectors.ensure_search_payload_indexes()
    service = CodeKBService(embed_fn=_embed, vector_store=vectors)

    catalog = CatalogDb()
    # The production singletons and the vector store open the default repository, which
    # reads ``core.db.database.SessionLocal`` at call time: point it at the test catalog.
    monkeypatch.setattr("core.db.database.SessionLocal", catalog._session_local)
    monkeypatch.setattr(routes, "get_code_kb_service", lambda: service)
    monkeypatch.setattr(routes, "get_existing_code_index_queue", lambda: None)
    monkeypatch.setattr(routes, "get_index_catalog_service", lambda: catalog.service)

    class Env:
        pass

    state = Env()
    state.service, state.catalog, state.client, state.name = service, catalog, client, name
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


def _index(state, project, files):
    state.service.index(CodeIndexRequest(
        project=project, host="host", root_dir=project,
        files=[CodeFileIn(rel_path=rel, language="python", chunks=[CodeChunkIn(
            start_line=0, end_line=1, label="f", content=NEEDLE, code=NEEDLE)])
            for rel in files]), owner_user_id=OWNER)
    state.catalog.add_files(project, files, owner=OWNER)


def _search(project, **extra):
    request = CodeSearchRequest(query=NEEDLE, project=project, mode="dense", top_k=50, **extra)
    return routes._run_private_code_search(request, OWNER)


def _paths(response):
    return sorted(hit.path for hit in response.hits)


def test_subdirectory_of_an_indexed_parent_finds_only_its_own_files(env):
    _index(env, "/repo", PARENT_FILES)
    response = _search("/repo/app")
    assert _paths(response) == ["app/api/deep/handler.py", "app/api/routes.py", "app/main.py"]
    assert {hit.project for hit in response.hits} == {"/repo"}
    assert (response.resolved_scope, response.resolved_path_prefix) == ("/repo", "app")


def test_parent_search_is_unchanged_and_reports_no_resolution(env):
    _index(env, "/repo", PARENT_FILES)
    response = _search("/repo")
    assert len(response.hits) == len(PARENT_FILES)
    assert response.resolved_scope is None and response.resolved_path_prefix is None


def test_user_path_glob_is_relative_to_the_requested_subdirectory(env):
    _index(env, "/repo", PARENT_FILES)
    assert _paths(_search("/repo/app", path_glob="api/**")) == [
        "app/api/deep/handler.py", "app/api/routes.py"]
    assert _paths(_search("/repo/app/api", path_glob="deep/")) == ["app/api/deep/handler.py"]
    assert _paths(_search("/repo/app", path_glob="*.py")) == [
        "app/api/deep/handler.py", "app/api/routes.py", "app/main.py"]


def test_subdirectory_with_its_own_index_uses_only_that_index(env):
    _index(env, "/repo", PARENT_FILES)
    _index(env, "/repo/app", ["own/file.py", "main.py"])
    response = _search("/repo/app")
    assert _paths(response) == ["main.py", "own/file.py"]
    assert {hit.project for hit in response.hits} == {"/repo/app"}
    assert response.resolved_scope is None


def test_nearest_indexed_ancestor_is_the_one_searched(env):
    _index(env, "/repo", PARENT_FILES)
    _index(env, "/repo/app", ["api/own.py", "main.py"])
    response = _search("/repo/app/api")
    assert _paths(response) == ["api/own.py"]
    assert (response.resolved_scope, response.resolved_path_prefix) == ("/repo/app", "api")


def test_parent_with_no_files_under_the_subdirectory_explains_itself(env):
    _index(env, "/repo", PARENT_FILES)
    response = _search("/repo/node_modules")
    assert response.hits == []
    assert "/repo" in response.notice and "node_modules" in response.notice
    assert "index_code" in response.notice
    assert (response.resolved_scope, response.resolved_path_prefix) == (
        "/repo", "node_modules")
    assert response.path_glob_matched_files == 0


def test_a_scope_with_no_indexed_ancestor_is_left_alone(env):
    _index(env, "/repo", PARENT_FILES)
    response = _search("/elsewhere/app")
    assert response.hits == [] and response.resolved_scope is None


def test_a_withheld_parent_withholds_the_subdirectory_too(env):
    _index(env, "/repo", PARENT_FILES)
    env.catalog.repository.set_code_scope_visibility(
        owner_user_id=OWNER, host="host", project="/repo", job_id="job-1",
        state="preparing")
    response = _search("/repo/app")
    assert response.hits == []
    assert response.scope_snapshot_state == "preparing"
    assert response.active_index_job_id == "job-1"


def test_windows_subdirectory_resolves_case_insensitively(env):
    files = ["Internal/Core/a.go", "internal/core/b.go", "Drivers/c.go"]
    _index(env, "E:/go/openlist", files)
    response = _search("e:/Go/OpenList/INTERNAL")
    assert _paths(response) == ["Internal/Core/a.go", "internal/core/b.go"]
    assert (response.resolved_scope, response.resolved_path_prefix) == (
        "E:/go/openlist", "internal")


def test_federated_private_leg_resolves_a_subdirectory_scope(env):
    _index(env, "/repo", PARENT_FILES)
    _index(env, "/other", ["app/foreign.py"])
    request = FederatedCodeSearchRequest(
        query=NEEDLE, projects=["/repo/app"], mode="dense", include_public=False, top_k=20)
    response, excluded, searched = routes._run_federated_private_leg(request, OWNER, pool=50)
    assert _paths(response) == [
        "app/api/deep/handler.py", "app/api/routes.py", "app/main.py"]
    assert {hit.project for hit in response.hits} == {"/repo"}
    assert excluded == [] and searched == ["/repo/app"]


def test_federated_mixes_a_plain_scope_with_a_resolved_subdirectory(env):
    _index(env, "/repo", PARENT_FILES)
    _index(env, "/other", ["app/foreign.py", "lib/x.py"])
    request = FederatedCodeSearchRequest(
        query=NEEDLE, projects=["/repo/app/api", "/other"], mode="dense",
        include_public=False, top_k=20)
    response, _excluded, _searched = routes._run_federated_private_leg(
        request, OWNER, pool=50)
    assert sorted((hit.project, hit.path) for hit in response.hits) == [
        ("/other", "app/foreign.py"), ("/other", "lib/x.py"),
        ("/repo", "app/api/deep/handler.py"), ("/repo", "app/api/routes.py")]


def test_federated_withheld_parent_excludes_the_subdirectory_scope(env):
    _index(env, "/repo", PARENT_FILES)
    _index(env, "/other", ["lib/x.py"])
    env.catalog.repository.set_code_scope_visibility(
        owner_user_id=OWNER, host="host", project="/repo", job_id="job-9", state="preparing")
    request = FederatedCodeSearchRequest(
        query=NEEDLE, projects=["/repo/app", "/other"], mode="dense",
        include_public=False, top_k=20)
    response, excluded, _searched = routes._run_federated_private_leg(request, OWNER, pool=50)
    assert [(hit.project, hit.path) for hit in response.hits] == [("/other", "lib/x.py")]
    assert [(item["project"], item["state"], item["active_index_job_id"])
            for item in excluded] == [("/repo", "preparing", "job-9")]
