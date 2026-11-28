"""TDD tests for the pipeline API surface that survives F1-a.

F1-a deletes the legacy /status, /reindex, /collect, /extract routes and their
PipelineService methods (app/services/pipeline_service.py::get_status,
start_reindex, start_collect/_run_collect, start_extract/_run_extract) —
superseded by /api/pipeline/tasks + /api/papers/batch-index. `_run_reindex`
stays: it is the shared background worker thread target for
`start_batch_index` (tests/test_batch_index_api.py mocks it directly), so
deleting it would break the batch-index route the plan keeps.

These tests run against the real FastAPI app (main.app) so a route that is
merely unregistered from the pipeline sub-router but still reachable through
another mount would still be caught.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app
from core.db.database import init_db


@pytest.fixture(autouse=True, scope="module")
def _init():
    init_db()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestDeletedPipelineRoutesAreGone:
    """The four legacy routes must not be reachable on the real app."""

    def test_status_route_gone(self, client):
        # Unlike the POST routes below, an unmatched GET falls through to
        # main.py's `GET /{spa_path:path}` SPA fallback instead of FastAPI's
        # native 404 — it tries to serve the frontend bundle and, since no
        # `frontend/dist` exists in this test environment, answers 503. Either
        # way the request never reaches the deleted pipeline handler.
        r = client.get("/api/pipeline/status")
        assert r.status_code in (404, 405, 503), r.text

    def test_reindex_route_gone(self, client):
        r = client.post("/api/pipeline/reindex", json={})
        assert r.status_code in (404, 405), r.text

    def test_collect_route_gone(self, client):
        r = client.post("/api/pipeline/collect")
        assert r.status_code in (404, 405), r.text

    def test_extract_route_gone(self, client):
        r = client.post("/api/pipeline/extract")
        assert r.status_code in (404, 405), r.text


class TestSurvivingPipelineRoutesStillRegistered:
    """/ingest, /tasks, /tasks/{id} and /api/papers/batch-index stay."""

    def test_ingest_route_registered(self):
        from app.api.routes.pipeline import router
        paths = [r.path for r in router.routes]
        assert "/api/pipeline/ingest" in paths

    def test_tasks_route_registered(self):
        from app.api.routes.pipeline import router
        paths = [r.path for r in router.routes]
        assert "/api/pipeline/tasks" in paths
        assert "/api/pipeline/tasks/{task_id}" in paths

    def test_batch_index_route_registered(self):
        from app.api.routes.pipeline import batch_index_router
        paths = [r.path for r in batch_index_router.routes]
        assert "/api/papers/batch-index" in paths

    def test_tasks_route_reachable(self, client):
        r = client.get("/api/pipeline/tasks")
        assert r.status_code == 200, r.text

    def test_task_status_404_for_unknown_id(self, client):
        r = client.get("/api/pipeline/tasks/does-not-exist")
        assert r.status_code == 404


class TestDeletedPipelineServiceMethods:
    """The deleted methods must not exist on the service anymore."""

    def test_get_status_removed(self):
        from app.services.pipeline_service import PipelineService
        assert not hasattr(PipelineService(), "get_status")

    def test_start_reindex_removed(self):
        from app.services.pipeline_service import PipelineService
        assert not hasattr(PipelineService(), "start_reindex")

    def test_start_collect_removed(self):
        from app.services.pipeline_service import PipelineService
        assert not hasattr(PipelineService(), "start_collect")

    def test_run_collect_removed(self):
        from app.services.pipeline_service import PipelineService
        assert not hasattr(PipelineService(), "_run_collect")

    def test_start_extract_removed(self):
        from app.services.pipeline_service import PipelineService
        assert not hasattr(PipelineService(), "start_extract")

    def test_run_extract_removed(self):
        from app.services.pipeline_service import PipelineService
        assert not hasattr(PipelineService(), "_run_extract")


class TestSurvivingPipelineServiceMethods:
    """get_task_status, list_tasks, start_batch_index and their _run_reindex
    dependency must remain — start_batch_index reuses it as its worker."""

    def test_get_task_status_present(self):
        from app.services.pipeline_service import PipelineService
        assert hasattr(PipelineService(), "get_task_status")

    def test_list_tasks_present(self):
        from app.services.pipeline_service import PipelineService
        assert hasattr(PipelineService(), "list_tasks")

    def test_start_batch_index_present(self):
        from app.services.pipeline_service import PipelineService
        assert hasattr(PipelineService(), "start_batch_index")

    def test_run_reindex_present(self):
        from app.services.pipeline_service import PipelineService
        assert hasattr(PipelineService(), "_run_reindex")


class TestReindexRequestModelRemoved:
    def test_reindex_request_not_importable(self):
        import app.api.routes.pipeline as mod
        assert not hasattr(mod, "ReindexRequest")
