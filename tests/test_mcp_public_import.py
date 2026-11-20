"""import_public_code_repo: a local clone reaches the server as a bundle, never as vectors.

Real git clones, a real uvicorn HTTP server running the public-code routes with the
real import and job services and a real SQLite registry. The index step reads the
published mirror with the real GitMirrorSourceReader.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
import socket
import threading
import time

from fastapi import FastAPI
import pytest
import uvicorn

from app.api import identity
from app.api.identity import is_private_lan
from app.services.public_code.access_profiles import GitAccessProfileResolver
from app.services.public_code.bundle_import import PublicCodeBundleImportService
from app.services.public_code.control_service import PublicCodeControlService
from app.services.public_code.jobs import PublicCodeJobService
from tests.test_mcp_plugin import _load
from tests.test_public_code_bundle_import import _commit, _git, _operator_clone, _repository
from tests.test_public_code_import_routes import _MirrorReadingIndex
from tests.test_public_code_private_reuse import stores

UPSTREAM = "https://example.invalid/demo/demo.git"
TWINS = ("_https_upstream", "_upstream_key", "_run_git_text", "_inspect_public_clone",
         "_resolve_public_import_source", "_create_public_bundle", "_run_public_import",
         "_format_public_import")


def _published_clone(tmp_path: Path, *, depth: int | None = None) -> Path:
    """An operator's clone: HEAD on origin/main, origin pointing at the public upstream."""
    upstream = _operator_clone(tmp_path, {"src/model.py": "def forward(x):\n    return x\n"})
    _commit(upstream, "src/train.py", "def step():\n    pass\n")
    clone = tmp_path / "work"
    depth_args = ["--depth", str(depth)] if depth else []
    _git("clone", *depth_args, "--branch", "main", upstream.as_uri(), str(clone))
    _git("remote", "set-url", "origin", "git@example.invalid:demo/demo.git", cwd=clone)
    return clone


@pytest.fixture
def live_backend(tmp_path, monkeypatch):
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
    # Admin routes trust only non-loopback private peers on the internal door, so the
    # server listens on this machine's LAN address and that port is the internal door.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect(("10.100.100.88", 9))  # picks the LAN interface; sends nothing
        host = route.getsockname()[0]
    if not is_private_lan(host):
        pytest.skip(f"no private LAN address on this host ({host})")
    with socket.socket() as probe:
        probe.bind((host, 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(identity, "INTERNAL_API_PORT", port)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started
    yield f"http://{host}:{port}", repository, index, tmp_path / "mirrors"
    server.should_exit = True
    thread.join(timeout=10)
    jobs.shutdown()


async def _until_indexed(status_tool, handle: str) -> str:
    deadline = time.time() + 60
    while time.time() < deadline:
        text = await status_tool(job_id=handle)
        if "[completed]" in text or "[failed]" in text:
            return text
        await asyncio.sleep(0.1)
    raise AssertionError(f"import did not finish: {text}")


def test_plugin_uploads_a_shallow_clone_and_the_server_indexes_it(live_backend, tmp_path):
    url, repository, index, mirrors = live_backend
    clone = _published_clone(tmp_path, depth=1)
    head = _git("rev-parse", "HEAD", cwd=clone)
    mod = _load()
    mod.BASE_URL, mod._CLIENT = url, None

    async def scenario():
        receipt = await mod.import_public_code_repo(path=str(clone), domain="machine-learning")
        handle = receipt.split()[3]
        return receipt, handle, await _until_indexed(mod.public_code_job_status, handle)

    receipt, handle, status = asyncio.run(scenario())

    assert "new source demo in domain machine-learning (https://example.invalid/demo/demo.git)" \
        in receipt
    assert f'public_code_job_status(job_id="{handle}")' in receipt
    assert f"Public code import {handle} [submitted] demo @ {head[:12]}" in status
    assert "import 1/1 sources" in status and "[completed]" in status, status
    # the server built the mirror and indexed it; this machine only sent the bundle
    assert index.calls == [("demo", False)]
    registered = repository.get_source("demo")
    assert registered.upstream_url == UPSTREAM  # ssh origin normalized to https
    assert (mirrors / "machine-learning" / "demo.git" / "shallow").read_text().split() == [head]


def test_repository_server_tool_takes_the_same_path(live_backend, tmp_path, monkeypatch):
    import mcp_server.server as server
    from mcp_server.backend import RetrievalClient

    url, repository, index, _ = live_backend
    clone = _published_clone(tmp_path)
    monkeypatch.setattr(server, "_client", RetrievalClient(base_url=url))

    async def scenario():
        receipt = await server.import_public_code_repo(path=str(clone), domain="machine-learning")
        return await _until_indexed(server.public_code_job_status, receipt.split()[3])

    status = asyncio.run(scenario())

    assert "[completed]" in status, status
    assert repository.get_source("demo").domain == "machine-learning"


def test_registered_upstream_is_reused_without_a_domain(live_backend, tmp_path):
    url, repository, index, _ = live_backend
    from app.services.public_code.source_catalog import PublicCodeSource

    repository.create_source(
        PublicCodeSource("demo-registered", "systems", "systems/demo", UPSTREAM.removesuffix(".git")),
        update_policy="manual", update_interval_minutes=None)
    clone = _published_clone(tmp_path, depth=1)
    mod = _load()
    mod.BASE_URL, mod._CLIENT = url, None

    async def scenario():
        receipt = await mod.import_public_code_repo(path=str(clone))
        return receipt, await _until_indexed(mod.public_code_job_status, receipt.split()[3])

    receipt, status = asyncio.run(scenario())

    assert "registered source demo-registered" in receipt
    assert "[completed]" in status, status
    assert index.calls == [("demo-registered", False)]


def test_only_published_clean_clones_are_accepted(tmp_path):
    mod = _load()
    clone = _published_clone(tmp_path)

    (clone / "src" / "model.py").write_text("def forward(x):\n    return -x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="uncommitted changes to tracked files"):
        mod._inspect_public_clone(str(clone))
    _git("checkout", "--", "src/model.py", cwd=clone)
    _git("config", "user.email", "t@example.invalid", cwd=clone)
    _git("config", "user.name", "t", cwd=clone)
    local = _commit(clone, "local.py", "secret = 1\n")
    with pytest.raises(ValueError, match=f"HEAD {local[:12]} of .* is not on any origin branch"):
        mod._inspect_public_clone(str(clone))
    _git("reset", "--hard", "origin/main", cwd=clone)
    # untracked files are not part of HEAD, so they do not block the import
    (clone / "notes.txt").write_text("scratch", encoding="utf-8")
    inspected = mod._inspect_public_clone(str(clone))
    assert inspected["origin"] == "https://example.invalid/demo/demo.git"
    assert inspected["branch"] == "main"
    with pytest.raises(ValueError, match="is not a git repository"):
        mod._inspect_public_clone(str(tmp_path))


def test_new_sources_need_a_domain_and_registered_ones_keep_their_identity():
    mod = _load()
    clone = {"repo": "r", "head": "a" * 40, "origin": "https://github.com/pytorch/pytorch",
             "branch": "main"}
    registered = [{"source_id": "pytorch", "domain": "machine-learning",
                   "relative_path": "machine-learning/pytorch", "default_ref": "HEAD",
                   "upstream_url": "https://github.com/pytorch/pytorch.git"}]

    reused = mod._resolve_public_import_source(clone, registered, domain=None, source_id=None)
    assert (reused["source_id"], reused["new"]) == ("pytorch", False)
    with pytest.raises(ValueError, match="already registered as pytorch"):
        mod._resolve_public_import_source(clone, registered, domain="gpu", source_id=None)
    with pytest.raises(ValueError, match="pass domain to add it .*machine-learning"):
        mod._resolve_public_import_source(
            {**clone, "origin": "https://github.com/x/y.git"}, registered, domain=None,
            source_id=None)
    fresh = mod._resolve_public_import_source(
        {**clone, "origin": "https://github.com/x/y.git"}, registered, domain="gpu",
        source_id=None)
    assert (fresh["source_id"], fresh["relative_path"], fresh["new"]) == ("y", "gpu/y", True)
    pinned = [{**registered[0], "default_ref": "release"}]
    with pytest.raises(ValueError, match="indexes branch release"):
        mod._resolve_public_import_source(clone, pinned, domain=None, source_id=None)


def test_upstream_forms_name_one_repository():
    mod = _load()
    forms = ["git@github.com:pytorch/pytorch.git", "https://github.com/pytorch/pytorch",
             "https://GitHub.com/pytorch/pytorch.git/", "ssh://git@github.com/pytorch/pytorch.git"]
    assert {mod._upstream_key(form) for form in forms} == {"github.com/pytorch/pytorch"}
    assert mod._https_upstream(forms[0]) == "https://github.com/pytorch/pytorch.git"


def test_import_helpers_match_repo_client():
    import mcp_server.backend as backend

    mod = _load()
    for helper in TWINS:
        assert inspect.getsource(getattr(mod, helper)) == inspect.getsource(
            getattr(backend, helper)), f"standalone plugin drifted in {helper}"


def test_import_client_automatically_passes_the_clone_as_private_scope(tmp_path):
    mod = _load()
    clone = _published_clone(tmp_path)
    inspected = mod._inspect_public_clone(str(clone))
    source = mod._resolve_public_import_source(inspected, [], domain="systems", source_id=None)
    bodies = []

    async def create(body):
        bodies.append(body)
        return {"import_id": "fixture"}

    async def put(import_id, offset, data):
        return {"received_bytes": offset + len(data)}

    async def complete(import_id):
        return {"job": {"job_id": "job"}}

    status = {}
    asyncio.run(mod._run_public_import(status, inspected, source, create=create, put=put,
                                       get=None, complete=complete))
    assert status["phase"] == "submitted", status
    assert bodies[0]["private_project"] == inspected["repo"]


def test_http_import_reuses_real_private_vectors_for_a_published_git_clone(
    live_backend, stores, tmp_path, monkeypatch,
):
    from app.api.routes import public_code
    from app.services.public_code.git_mirror import GitMirrorSourceReader
    from app.services.public_code.index_service import PublicCodeIndexService
    from app.services.scope_paths import canonicalize_scope_path
    from core.code_kb.isolated_chunker import IsolatedCodeChunker
    from core.code_kb.vector_records import CodeChunkVectorRecord

    url, repository, _, mirrors = live_backend
    private, public = stores
    clone = _published_clone(tmp_path)
    chunker = IsolatedCodeChunker()
    items = []
    try:
        for path in ("src/model.py", "src/train.py"):
            chunks = chunker(path, (clone / path).read_text(), language="python")
            items.extend(CodeChunkVectorRecord(
                chunk.path, "python", chunk.start_line, chunk.end_line,
                chunk.label, chunk.content, chunk.code, ordinal,
            ) for ordinal, chunk in enumerate(chunks))
        assert items
        private.upsert(
            owner_user_id="internal-admin", project=canonicalize_scope_path(str(clone)),
            host="operator", items=items, dense_vectors=[[0.25, 0.75] for _ in items],
            sparse_vectors=[{"7": 0.6} for _ in items],
        )

        def forbidden_embed(texts):
            raise AssertionError("the published private index should satisfy this import")

        jobs = public_code.get_public_code_job_service()
        monkeypatch.setattr(jobs, "_index_service", PublicCodeIndexService(
            source_registry=repository, repository=repository,
            source_reader=GitMirrorSourceReader(mirror_root=mirrors), vector_store=public,
            chunk_fn=chunker, embed_fn=forbidden_embed, private_vector_store_factory=lambda: private,
        ))
        mod = _load()
        mod.BASE_URL, mod._CLIENT = url, None

        async def scenario():
            receipt = await mod.import_public_code_repo(path=str(clone), domain="systems")
            return await _until_indexed(mod.public_code_job_status, receipt.split()[3])

        status = asyncio.run(scenario())
        assert "[completed]" in status, status
        active = repository.get_source("demo").active_revision
        assert active == _git("rev-parse", "HEAD", cwd=clone)
        assert public.count_revision("demo", active) == len(items)
        job = jobs.list_recent()[0]
        assert job.results[0]["reused_chunk_count"] == len(items)
        assert private.count(owner_user_id="internal-admin", project=canonicalize_scope_path(str(clone))) \
            == len(items)
    finally:
        chunker.close()
