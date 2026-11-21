"""Bundle import over HTTP: declare, upload in resumable parts, complete, poll the job.

Real git, real SQLite source registry, real job service. The index step reads the
published mirror with the real GitMirrorSourceReader; embedding and Qdrant writes
are covered by the index-service tests and the live import on .88.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.services.public_code.access_profiles import GitAccessProfileResolver
from app.services.public_code.bundle_import import PublicCodeBundleImportService
from app.services.public_code.control_service import PublicCodeControlService
from app.services.public_code.git_mirror import GitMirrorSourceReader
from app.services.public_code.jobs import PublicCodeJobService
from tests.test_public_code_bundle_import import (
    UPSTREAM,
    _bundle,
    _commit,
    _git,
    _operator_clone,
    _repository,
)

ADMIN = {"X-Retrival-User-Id": "gateway-admin", "X-Retrival-User-Tier": "admin"}


@dataclass(frozen=True)
class _Indexed:
    source_id: str
    revision: str
    status: str
    file_count: int
    chunk_count: int


class _MirrorReadingIndex:
    """Stands in for embedding: proves the registered source's mirror is indexable."""

    def __init__(self, repository, mirror_root: Path):
        self.repository = repository
        self.mirror_root = mirror_root
        self.calls = []

    def index_source(self, source_id, progress_cb=None, *, refresh_mirror=True,
                     private_owner_user_id=None, private_project=None):
        self.calls.append((source_id, refresh_mirror))
        self.private_scope = (private_owner_user_id, private_project)
        source = self.repository.select(source_ids=(source_id,))[0]
        manifest = GitMirrorSourceReader(mirror_root=self.mirror_root).build_manifest(source)
        return _Indexed(source_id, manifest.revision_id, "completed", len(manifest.files), 0)


@pytest.fixture
def api(tmp_path, monkeypatch):
    from app.api.routes import public_code

    repository = _repository()
    profiles = GitAccessProfileResolver(default_proxy_url=None, environ={})
    control = PublicCodeControlService(
        repository=repository, update_checks=None, source_validator=profiles.validate_source)
    index = _MirrorReadingIndex(repository, tmp_path / "mirrors")
    jobs = PublicCodeJobService(
        source_registry=repository, control_service=control, index_service=index)
    imports = PublicCodeBundleImportService(
        import_root=tmp_path / "imports", mirror_root=tmp_path / "mirrors",
        repository=repository, max_bundle_bytes=10 * 1024**2, free_space_margin_bytes=0)
    monkeypatch.setattr(public_code, "get_public_code_access_profiles", lambda: profiles)
    monkeypatch.setattr(public_code, "get_public_code_control_service", lambda: control)
    monkeypatch.setattr(public_code, "get_public_code_job_service", lambda: jobs)
    monkeypatch.setattr(public_code, "get_public_code_import_service", lambda: imports)
    app = FastAPI()
    app.include_router(public_code.router, prefix="/api/public-code")
    yield TestClient(app), repository, index
    jobs.shutdown()


def _declare(client, payload: bytes, head: str, **overrides):
    body = {
        "source_id": "demo", "domain": "machine-learning",
        "relative_path": "machine-learning/demo", "upstream_url": UPSTREAM,
        "head_commit": head, "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    body.update(overrides)
    return client.post("/api/public-code/imports", headers=ADMIN, json=body)


def _put(client, import_id: str, offset: int, data: bytes):
    return client.put(f"/api/public-code/imports/{import_id}/data", params={"offset": offset},
                      headers={**ADMIN, "Content-Type": "application/octet-stream"}, content=data)


def _wait_job(client, job_id: str) -> dict:
    deadline = time.time() + 30
    while time.time() < deadline:
        job = client.get(f"/api/public-code/jobs/{job_id}", headers=ADMIN).json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {job}")


def test_import_reuses_only_the_authenticated_creators_canonical_private_scope(api, tmp_path):
    client, repository, index = api
    clone = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    payload = _bundle(clone, tmp_path / "demo.bundle")
    head = _git("rev-parse", "HEAD", cwd=clone)
    created = _declare(client, payload, head, private_project="C:\\Repos\\Demo").json()
    assert created["private_project"] == "C:/repos/demo"
    import_id = created["import_id"]
    _put(client, import_id, 0, payload)
    # Another admin may complete an upload, but cannot change its private owner.
    receipt = client.post(f"/api/public-code/imports/{import_id}/complete", headers={
        "X-Retrival-User-Id": "other-admin", "X-Retrival-User-Tier": "admin"})
    assert receipt.status_code == 202, receipt.text
    assert _wait_job(client, receipt.json()["job"]["job_id"])["status"] == "completed"
    assert index.private_scope == ("gateway-admin", "C:/repos/demo")


@pytest.mark.parametrize("scope", ["", "relative/repo", "C:repo"])
def test_import_rejects_nonabsolute_private_scope(api, scope):
    client, _, _ = api
    response = _declare(client, b"bundle", "a" * 40, private_project=scope)
    assert response.status_code == 422


def test_import_cannot_declare_another_private_owner(api):
    client, _, _ = api
    response = _declare(client, b"bundle", "a" * 40,
                        private_project="C:/repos/demo", private_owner_user_id="victim")
    assert response.status_code == 422


def test_bundle_upload_registers_the_source_and_indexes_its_commit(api, tmp_path):
    client, repository, index = api
    clone = _operator_clone(tmp_path, {"src/model.py": "def forward(x):\n    return x\n"})
    head = _git("rev-parse", "HEAD", cwd=clone)
    payload = _bundle(clone, tmp_path / "demo.bundle")

    created = _declare(client, payload, head)
    assert created.status_code == 201, created.text
    import_id = created.json()["import_id"]
    assert created.json()["received_bytes"] == 0

    first = _put(client, import_id, 0, payload[:200])
    assert first.status_code == 200 and first.json()["received_bytes"] == 200
    # a retried or out-of-order part is refused with the offset to resume from
    stale = _put(client, import_id, 0, payload[:200])
    assert stale.status_code == 409
    assert "does not match the 200 bytes already received" in stale.json()["detail"]
    assert client.get(f"/api/public-code/imports/{import_id}",
                      headers=ADMIN).json()["received_bytes"] == 200
    assert _put(client, import_id, 200, payload[200:]).json()["received_bytes"] == len(payload)

    completed = client.post(f"/api/public-code/imports/{import_id}/complete", headers=ADMIN)
    assert completed.status_code == 202, completed.text
    assert completed.json()["import"]["status"] == "submitted"
    job = _wait_job(client, completed.json()["job"]["job_id"])

    assert job["status"] == "completed", job
    assert job["operation"] == "import"
    assert job["results"][0]["revision"] == head
    assert index.calls == [("demo", False)]
    registered = repository.get_source("demo")
    assert registered.upstream_url == UPSTREAM
    assert registered.created_by_user_id == "gateway-admin"
    final = client.get(f"/api/public-code/imports/{import_id}", headers=ADMIN).json()
    assert final["status"] == "published" and final["job_id"] == job["job_id"]


def test_second_import_advances_an_existing_source(api, tmp_path):
    client, repository, index = api
    clone = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    for name in ("one", "two"):
        head = (_git("rev-parse", "HEAD", cwd=clone) if name == "one"
                else _commit(clone, "b.py", "y = 2\n"))
        payload = _bundle(clone, tmp_path / f"{name}.bundle")
        import_id = _declare(client, payload, head).json()["import_id"]
        _put(client, import_id, 0, payload)
        receipt = client.post(f"/api/public-code/imports/{import_id}/complete", headers=ADMIN)
        job = _wait_job(client, receipt.json()["job"]["job_id"])
        assert job["status"] == "completed", job["errors"]
        assert job["results"][0]["revision"] == head
    assert client.get(f"/api/public-code/imports/{import_id}",
                      headers=ADMIN).json()["source_exists"] is True


def test_failed_import_reports_the_reason_and_registers_nothing(api, tmp_path):
    client, repository, index = api
    clone = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    payload = _bundle(clone, tmp_path / "demo.bundle")
    import_id = _declare(client, payload, "f" * 40).json()["import_id"]
    _put(client, import_id, 0, payload)

    receipt = client.post(f"/api/public-code/imports/{import_id}/complete", headers=ADMIN)
    job = _wait_job(client, receipt.json()["job"]["job_id"])

    assert job["status"] == "failed"
    assert "not the declared head_commit" in job["errors"]["demo"]
    assert repository.get_source("demo") is None
    assert index.calls == []
    state = client.get(f"/api/public-code/imports/{import_id}", headers=ADMIN).json()
    assert state["status"] == "failed" and "head_commit" in state["error"]
    assert client.delete(f"/api/public-code/imports/{import_id}", headers=ADMIN).status_code == 204
    assert client.get(f"/api/public-code/imports/{import_id}", headers=ADMIN).status_code == 404


def test_incomplete_upload_cannot_be_completed(api, tmp_path):
    client, _, _ = api
    clone = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    payload = _bundle(clone, tmp_path / "demo.bundle")
    import_id = _declare(client, payload, _git("rev-parse", "HEAD", cwd=clone)).json()["import_id"]
    _put(client, import_id, 0, payload[:10])

    response = client.post(f"/api/public-code/imports/{import_id}/complete", headers=ADMIN)

    assert response.status_code == 422
    assert "upload incomplete: 10 of" in response.json()["detail"]


@pytest.mark.parametrize("overrides, status, message", [
    ({"head_commit": "abc"}, 422, "at least 40 characters"),
    ({"upstream_url": "file:///srv/private.git"}, 422, "upstream_url"),
    ({"proxy_profile": "x"}, 422, "Extra inputs are not permitted"),
])
def test_invalid_declarations_are_rejected(api, overrides, status, message):
    client, _, _ = api
    response = _declare(client, b"x" * 10, "a" * 40, **overrides)
    assert response.status_code == status
    assert message in response.text


def test_imports_are_admin_only(api):
    client, _, _ = api
    user = {"X-Retrival-User-Id": "someone", "X-Retrival-User-Tier": "paying"}
    body = {"source_id": "demo", "domain": "machine-learning",
            "relative_path": "machine-learning/demo", "upstream_url": UPSTREAM,
            "head_commit": "a" * 40, "size_bytes": 10, "sha256": "b" * 64}
    assert client.post("/api/public-code/imports", headers=user, json=body).status_code == 403
    assert client.get("/api/public-code/imports/" + "0" * 32, headers=user).status_code == 403
