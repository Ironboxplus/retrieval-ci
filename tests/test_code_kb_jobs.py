"""TDD for the async code-KB index job queue.

POST /api/code/index used to embed+upsert synchronously in the request; a large repo
(5k+ chunks) held the HTTP request open for minutes and looked like a hang to MCP
clients. The queue makes indexing async: submit → 202 {job_id}, a single in-process
worker thread serializes the embed work (one warm BGE-M3, no concurrent GPU fights),
and progress is pollable per owner.
"""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from app.services.code_kb.jobs import (
    MAX_JOBS_KEPT,
    CodeIndexJob,
    CodeIndexJobAlreadyEnded,
    CodeIndexJobCommitting,
    CodeIndexJobQueue,
    JobStatus,
)
from app.services.code_kb.schemas import (
    CodeChunkIn,
    CodeFileIn,
    CodeIndexReport,
    CodeIndexRequest,
    CodeIndexResponse,
    CodeSearchRequest,
)
from app.services.code_kb.service import CodeKBService
from app.services.index_catalog.repository import IndexCatalogRepository
from app.services.index_catalog.service import IndexCatalogService
from app.services.index_catalog.schemas import IndexFileRecord, IndexReportResponse

from tests.test_code_kb_api import FakeStore, _embed  # reuse fakes


class _SpyCatalog:
    """Captures IndexCatalogService.report calls from the job worker."""

    def __init__(self):
        self.reports: list[dict] = []
        self.deleted_files: list[dict] = []
        self.live_paths: dict[tuple[str, str], set[str]] = {}

    def report(self, request, *, owner_user_id):
        self.reports.append({"owner": owner_user_id, "host": request.host,
                             "project": request.project, "replace": request.replace,
                             "files": [f.rel_path for f in request.files]})
        key = (owner_user_id, request.project)
        reported = {f.rel_path for f in request.files}
        if request.replace:
            self.live_paths[key] = reported
        else:
            self.live_paths.setdefault(key, set()).update(reported)
        return IndexReportResponse(host=request.host, project=request.project,
                                   owner_user_id=owner_user_id, upserted=len(request.files),
                                   removed=0, total_files=len(request.files))

    def delete_files(self, *, owner_user_id, project, host, kind, rel_paths):
        self.deleted_files.append({
            "owner": owner_user_id, "project": project, "host": host,
            "kind": kind, "rel_paths": list(rel_paths),
        })
        key = (owner_user_id, project)
        self.live_paths.setdefault(key, set()).difference_update(rel_paths)
        return len(rel_paths)

    def delete_project(self, *, owner_user_id, project, kind):
        self.live_paths[(owner_user_id, project)] = set()
        return 1


def _report(paths=("a.py",)):
    return CodeIndexReport(files=[IndexFileRecord(rel_path=p, kind="code",
                                                  collection="code_chunks", language="python",
                                                  chunk_count=1, content_hash=f"h_{p}")
                                  for p in paths], replace=True)


def _req(project="proj", n_chunks=3):
    chunks = [CodeChunkIn(start_line=i, end_line=i + 5, label=f"f{i}",
                          content=f"qdrant upsert vector {i}", code=f"def f{i}(): ...")
              for i in range(n_chunks)]
    return CodeIndexRequest(project=project, host="h1", root_dir="/repo",
                            files=[CodeFileIn(rel_path="a.py", language="python", chunks=chunks)])


@pytest.fixture
def queue():
    svc = CodeKBService(embed_fn=_embed, vector_store=FakeStore())
    q = CodeIndexJobQueue(service=svc, catalog_service=_SpyCatalog())
    yield q
    q.shutdown(timeout=5)


def _wait_done(q, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = q.get(job_id)
        if job and job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {q.get(job_id)}")


def _sqlite_catalog_service() -> IndexCatalogService:
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

    return IndexCatalogService(
        repository=IndexCatalogRepository(session_factory=session_factory))


def test_get_code_index_queue_constructs_one_worker_for_parallel_first_calls(monkeypatch):
    """The lazy singleton must not create competing queues/workers on first use."""
    import app.services.code_kb.jobs as jobs_module

    created: list[object] = []
    created_lock = threading.Lock()
    start = threading.Barrier(12)
    results: list[object] = []
    result_lock = threading.Lock()

    class SlowQueue:
        def __init__(self, **_kwargs):
            # Releases the GIL long enough for the unlocked implementation to let every
            # concurrent caller observe `_QUEUE is None` before any assignment wins.
            time.sleep(0.03)
            with created_lock:
                created.append(self)

    def get_one() -> None:
        start.wait()
        queue = jobs_module.get_code_index_queue()
        with result_lock:
            results.append(queue)

    monkeypatch.setattr(jobs_module, "_QUEUE", None)
    monkeypatch.setattr(jobs_module, "CodeIndexJobQueue", SlowQueue)
    threads = [threading.Thread(target=get_one) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert len(created) == 1
    assert len({id(queue) for queue in results}) == 1


def test_get_code_kb_service_constructs_one_shared_instance_for_parallel_first_calls(monkeypatch):
    """The API getter and a new worker queue must share one warm service instance."""
    import app.services.code_kb.service as service_module
    from app.api.routes import code_kb as code_kb_route

    created: list[object] = []
    created_lock = threading.Lock()
    start = threading.Barrier(12)
    results: list[object] = []
    result_lock = threading.Lock()

    class SlowService:
        def __init__(self):
            # Release the GIL between the old unlocked ``None`` check and assignment.
            time.sleep(0.03)
            with created_lock:
                created.append(self)

    def get_one() -> None:
        start.wait()
        service = service_module.get_code_kb_service()
        with result_lock:
            results.append(service)

    monkeypatch.setattr(service_module, "_SERVICE", None)
    monkeypatch.setattr(service_module, "CodeKBService", SlowService)
    # Other route tests replace this imported dependency directly. Rebind the production
    # API getter here so this concurrency contract tests the real singleton wiring.
    monkeypatch.setattr(
        code_kb_route,
        "get_code_kb_service",
        service_module.get_code_kb_service,
    )
    threads = [threading.Thread(target=get_one) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert len(created) == 1
    assert len({id(service) for service in results}) == 1

    queue = CodeIndexJobQueue(catalog_service=_SpyCatalog())
    try:
        assert queue.service is results[0]
        assert code_kb_route.get_code_kb_service() is results[0]
    finally:
        queue.shutdown(timeout=5)


def test_recovery_marker_outage_keeps_worker_alive_and_search_fail_closed(monkeypatch):
    """A second SQLite outage after a partial write cannot take down the FIFO worker."""
    from app.api.routes import code_kb

    class PartialFailStore(FakeStore):
        fail_once = True

        def upsert(self, **kwargs):
            result = super().upsert(**kwargs)
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("simulated primary vector mutation failure")
            return result

    class RecoveryMarkerOutageCatalog(_SpyCatalog):
        def __init__(self):
            super().__init__()
            self.markers: dict[tuple[str, str], dict[str, object]] = {}
            self.recovery_attempts = 0

        def mark_code_scope_preparing(self, *, owner_user_id, host, project, job_id):
            self.markers[(owner_user_id, project)] = {
                "state": "preparing", "job_id": job_id,
                "previous_completed_snapshot_available": False,
            }

        def mark_code_scope_completed(self, *, owner_user_id, host, project, job_id):
            self.markers[(owner_user_id, project)] = {
                "state": "completed", "job_id": job_id,
                "previous_completed_snapshot_available": True,
            }

        def mark_code_scope_recovery_required(self, **_kwargs):
            self.recovery_attempts += 1
            raise RuntimeError("simulated SQLite recovery-marker outage")

        def code_scope_visibility(self, *, owner_user_id, projects):
            for project in projects:
                marker = self.markers.get((owner_user_id, project))
                if marker is not None:
                    return marker
            return None

    catalog = RecoveryMarkerOutageCatalog()
    service = CodeKBService(embed_fn=_embed, vector_store=PartialFailStore())
    queue = CodeIndexJobQueue(service=service, catalog_service=catalog)
    try:
        failed = _wait_done(
            queue,
            queue.submit(_req(project="partial-marker-outage"), owner_user_id="u1").job_id,
        )

        assert failed.status is JobStatus.FAILED
        assert catalog.recovery_attempts >= 1
        assert queue._worker.is_alive()

        # The initial durable ``preparing`` marker remains, but the in-process unsafe
        # state is stronger while this backend is alive: never send callers to poll the
        # failed receipt and never call the vector search over its partial batch.
        monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
        monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: service)
        monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: queue)
        monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: catalog)
        monkeypatch.setattr(
            service,
            "search",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("partial scope must remain fail-closed")),
        )
        response = asyncio.run(code_kb.search_code(
            CodeSearchRequest(query="qdrant", project="partial-marker-outage"),
            request=object(), user_id="u1",
        ))
        assert response.scope_snapshot_state == "recovery_required"
        assert response.recovery_required is True
        assert response.active_index_job_id is None

        healthy = _wait_done(
            queue,
            queue.submit(_req(project="worker-survived"), owner_user_id="u1").job_id,
        )
        assert healthy.status is JobStatus.COMPLETED
        assert queue._worker.is_alive()
    finally:
        queue.shutdown(timeout=5)


def test_fan_out_recovery_marker_outage_isolated_from_worker_loop():
    """The fan-out recovery callback has the same best-effort SQLite boundary."""
    class RecoveryMarkerOutageCatalog(_SpyCatalog):
        def __init__(self):
            super().__init__()
            self.recovery_attempts = 0

        def mark_code_scope_preparing(self, **_kwargs):
            return None

        def mark_code_scope_completed(self, **_kwargs):
            return None

        def mark_code_scope_recovery_required(self, **_kwargs):
            self.recovery_attempts += 1
            raise RuntimeError("simulated SQLite fan-out recovery-marker outage")

    class FanOutFailureService:
        def index(
            self,
            request,
            *,
            owner_user_id,
            before_mutation_cb,
            before_fan_out_target_mutation_cb,
            fan_out_target_mutation_failed_cb,
            after_mutation_cb,
            **_kwargs,
        ):
            before_mutation_cb()
            before_fan_out_target_mutation_cb("fan-out-target")
            fan_out_target_mutation_failed_cb("fan-out-target")
            after_mutation_cb()
            return CodeIndexResponse(
                project=request.project, host=request.host, files=len(request.files), indexed=1,
            )

    catalog = RecoveryMarkerOutageCatalog()
    queue = CodeIndexJobQueue(service=FanOutFailureService(), catalog_service=catalog)
    try:
        first = _wait_done(
            queue,
            queue.submit(_req(project="fan-out-source"), owner_user_id="u1").job_id,
        )

        assert first.status is JobStatus.COMPLETED
        assert catalog.recovery_attempts == 1
        assert queue._worker.is_alive()
        state = queue.scope_search_state(owner_user_id="u1", projects=["fan-out-target"])
        assert state is not None and state.recovery_required is True

        second = _wait_done(
            queue,
            queue.submit(_req(project="fan-out-next"), owner_user_id="u1").job_id,
        )
        assert second.status is JobStatus.COMPLETED
    finally:
        queue.shutdown(timeout=5)


def test_submit_returns_queued_job_and_completes(queue):
    job = queue.submit(_req(), owner_user_id="u1")
    assert job.job_id and job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
    assert job.chunks_total == 3
    done = _wait_done(queue, job.job_id)
    assert done.status is JobStatus.COMPLETED
    assert done.indexed == 3
    assert done.progress == pytest.approx(1.0)
    assert done.finished_at is not None


def test_submit_reuses_only_identical_active_request(queue):
    gate = threading.Event()
    original = queue.service.index

    def slow(request, *, owner_user_id, progress_cb=None, **callbacks):
        gate.wait(timeout=5)
        return original(
            request, owner_user_id=owner_user_id, progress_cb=progress_cb, **callbacks)

    queue.service.index = slow
    first = queue.submit(_req(project="same"), owner_user_id="u1")
    time.sleep(0.05)
    duplicate = queue.submit(_req(project="same"), owner_user_id="u1")
    newer = queue.submit(_req(project="same", n_chunks=9), owner_user_id="u1")
    other_owner = queue.submit(_req(project="same"), owner_user_id="u2")
    assert duplicate.job_id == first.job_id
    assert newer.job_id != first.job_id
    assert other_owner.job_id != first.job_id
    gate.set()
    assert _wait_done(queue, first.job_id).status is JobStatus.COMPLETED
    assert _wait_done(queue, newer.job_id).status is JobStatus.COMPLETED
    assert _wait_done(queue, other_owner.job_id).status is JobStatus.COMPLETED


def test_submit_after_delete_uses_new_scope_generation_instead_of_reusing_running_job():
    gate = threading.Event()
    entered = threading.Event()

    def slow_embed(texts):
        entered.set()
        assert gate.wait(timeout=5)
        return _embed(texts)

    service = CodeKBService(embed_fn=slow_embed, vector_store=FakeStore())
    queue = CodeIndexJobQueue(service=service, catalog_service=_SpyCatalog())
    try:
        request = _req(project="recreated")
        first = queue.submit(request, owner_user_id="u1")
        assert entered.wait(timeout=2)

        queue.cancel_project(owner_user_id="u1", project="recreated")
        replacement = queue.submit(request, owner_user_id="u1")

        assert replacement.job_id != first.job_id
        assert replacement.scope_generation > first.scope_generation
        gate.set()
        assert _wait_done(queue, first.job_id).status is JobStatus.FAILED
        assert _wait_done(queue, replacement.job_id).status is JobStatus.COMPLETED
    finally:
        gate.set()
        queue.shutdown(timeout=5)


def test_jobs_are_processed_serially_not_concurrently(queue):
    """One worker thread: the GPU embed must never run two jobs at once."""
    seen = []
    lock = threading.Lock()
    orig = queue.service.index

    def spy(request, *, owner_user_id, progress_cb=None, **callbacks):
        with lock:
            seen.append("start")
            assert seen.count("start") - seen.count("end") == 1, "two jobs ran concurrently"
        time.sleep(0.05)
        out = orig(
            request, owner_user_id=owner_user_id, progress_cb=progress_cb, **callbacks)
        with lock:
            seen.append("end")
        return out

    queue.service.index = spy
    ids = [queue.submit(_req(project=f"p{i}"), owner_user_id="u1").job_id for i in range(3)]
    for jid in ids:
        assert _wait_done(queue, jid).status is JobStatus.COMPLETED


def test_progress_reaches_one_and_counts_chunks(queue):
    job = queue.submit(_req(n_chunks=5), owner_user_id="u1")
    done = _wait_done(queue, job.job_id)
    assert done.chunks_done == 5 and done.chunks_total == 5


def test_scope_search_state_marks_previous_snapshot_safe_only_before_write_phase():
    """The route can read a prior completed scope while a refresh embeds, but never once
    the Qdrant mutation has started.  This is the visibility boundary that prevents an
    initial or incremental fan-out from being presented as a completed snapshot."""
    embed_started = threading.Event()
    release_embed = threading.Event()
    write_started = threading.Event()
    release_write = threading.Event()

    class BlockingStore(FakeStore):
        block_writes = False

        def upsert(self, **kwargs):
            if self.block_writes:
                write_started.set()
                assert release_write.wait(5)
            return super().upsert(**kwargs)

    store = BlockingStore()
    service = CodeKBService(embed_fn=_embed, vector_store=store)
    q = CodeIndexJobQueue(service=service, catalog_service=_SpyCatalog())
    try:
        initial = q.submit(_req(project="sing-box"), owner_user_id="u1")
        assert _wait_done(q, initial.job_id).status is JobStatus.COMPLETED

        def slow_embed(texts):
            embed_started.set()
            assert release_embed.wait(5)
            return _embed(texts)

        service.embed_fn = slow_embed
        store.block_writes = True
        refresh = q.submit(_req(project="sing-box"), owner_user_id="u1")
        assert embed_started.wait(2)

        before_write = q.scope_search_state(owner_user_id="u1", projects=["sing-box"])
        assert before_write is not None
        assert before_write.active_job_id == refresh.job_id
        assert before_write.previous_completed_snapshot_available is True
        assert before_write.mutation_started is False

        release_embed.set()
        assert write_started.wait(2)
        during_write = q.scope_search_state(owner_user_id="u1", projects=["sing-box"])
        assert during_write is not None
        assert during_write.previous_completed_snapshot_available is True
        assert during_write.mutation_started is True
    finally:
        release_embed.set()
        release_write.set()
        q.shutdown(timeout=5)


def test_fan_out_target_scope_is_visible_during_its_mutation():
    """A parent job also mutates duplicate child scopes, not only its source project.

    While the target upsert is blocked, a scoped search of that child must see the
    same preparing gate as a search of the parent.  Looking only at ``job.project``
    lets fan-out's partial target write leak through.
    """
    target_write_started = threading.Event()
    release_target_write = threading.Event()

    class BlockingTargetStore(FakeStore):
        block_target = False

        def upsert(self, **kwargs):
            if self.block_target and kwargs["project"] == "/repo/child":
                target_write_started.set()
                assert release_target_write.wait(5)
            return super().upsert(**kwargs)

    store = BlockingTargetStore()
    service = CodeKBService(embed_fn=_embed, vector_store=store)
    q = CodeIndexJobQueue(service=service, catalog_service=_SpyCatalog())
    try:
        child = _req(project="/repo/child", n_chunks=1)
        child.files[0].rel_path = "a.py"
        service.index(child, owner_user_id="u1")

        parent = _req(project="/repo", n_chunks=1)
        parent.files[0].rel_path = "child/a.py"
        parent.files[0].chunks[0].code = "def updated_from_parent(): ..."
        store.block_target = True
        job = q.submit(parent, owner_user_id="u1")
        assert target_write_started.wait(2)

        state = q.scope_search_state(owner_user_id="u1", projects=["/repo/child"])
        assert state is not None
        assert state.active_job_id == job.job_id
        assert state.mutation_started is True
    finally:
        release_target_write.set()
        q.shutdown(timeout=5)


def test_partial_vector_write_failure_stays_unsafe_until_a_successful_recovery():
    """A failed in-place write cannot disappear from the visibility gate.

    The next full successful index is the recovery boundary; until then the vector
    store may contain a mixture of the old and newly-upserted batches.
    """
    class PartialFailStore(FakeStore):
        fail_after_upsert = False

        def upsert(self, **kwargs):
            result = super().upsert(**kwargs)
            if self.fail_after_upsert:
                raise RuntimeError("simulated partial qdrant write")
            return result

    store = PartialFailStore()
    service = CodeKBService(embed_fn=_embed, vector_store=store)
    q = CodeIndexJobQueue(service=service, catalog_service=_SpyCatalog())
    try:
        store.fail_after_upsert = True
        failed = _wait_done(q, q.submit(_req(project="unsafe-write"), owner_user_id="u1").job_id)
        assert failed.status is JobStatus.FAILED
        assert store.list_paths(owner_user_id="u1", project="unsafe-write") == {"a.py"}

        unsafe = q.scope_search_state(owner_user_id="u1", projects=["unsafe-write"])
        assert unsafe is not None
        assert unsafe.active_job_id == failed.job_id
        assert unsafe.mutation_started is True

        store.fail_after_upsert = False
        recovered = _wait_done(
            q, q.submit(_req(project="unsafe-write"), owner_user_id="u1").job_id)
        assert recovered.status is JobStatus.COMPLETED
        assert q.scope_search_state(owner_user_id="u1", projects=["unsafe-write"]) is None
    finally:
        q.shutdown(timeout=5)


def test_partial_write_marker_survives_queue_and_service_restart(monkeypatch):
    """The durable marker, not `_QUEUE`, protects a scope after a worker crash/restart."""
    from app.api.routes import code_kb

    class PartialFailStore(FakeStore):
        fail_after_upsert = False

        def upsert(self, **kwargs):
            result = super().upsert(**kwargs)
            if self.fail_after_upsert:
                raise RuntimeError("simulated process crash after partial batch")
            return result

    store = PartialFailStore()
    catalog = _sqlite_catalog_service()
    first_service = CodeKBService(embed_fn=_embed, vector_store=store)
    first_queue = CodeIndexJobQueue(service=first_service, catalog_service=catalog)
    restarted_queue = None
    try:
        store.fail_after_upsert = True
        failed = _wait_done(
            first_queue,
            first_queue.submit(_req(project="restart-unsafe"), owner_user_id="owner-a").job_id,
        )
        assert failed.status is JobStatus.FAILED
        assert catalog.code_scope_visibility(
            owner_user_id="owner-a", projects=["restart-unsafe"])["state"] == "recovery_required"
        first_queue.shutdown(timeout=5)

        # New backend process: new service + new empty queue, while Qdrant (the fake
        # store) and SQLite remain.  The route must not call vector search.
        restarted_service = CodeKBService(embed_fn=_embed, vector_store=store)
        restarted_queue = CodeIndexJobQueue(
            service=restarted_service,
            catalog_service=IndexCatalogService(repository=catalog.repository),
        )
        monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "owner-a")
        monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: restarted_service)
        monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: restarted_queue)
        monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: restarted_queue.catalog_service)

        response = asyncio.run(code_kb.search_code(
            body=CodeSearchRequest(query="qdrant", project="restart-unsafe"),
            request=object(), user_id="owner-a",
        ))

        assert response.scope_snapshot_state == "recovery_required"
        assert response.recovery_required is True
        assert response.total == 0
        assert response.active_index_job_id is None
        assert "full recovery index" in (response.notice or "").lower()
        assert "poll" not in (response.notice or "").lower()

        store.fail_after_upsert = False
        recovered = _wait_done(
            restarted_queue,
            restarted_queue.submit(_req(project="restart-unsafe"), owner_user_id="owner-a").job_id,
        )
        assert recovered.status is JobStatus.COMPLETED
        assert catalog.code_scope_visibility(
            owner_user_id="owner-a", projects=["restart-unsafe"])["state"] == "completed"

        recovered_response = asyncio.run(code_kb.search_code(
            body=CodeSearchRequest(query="qdrant", project="restart-unsafe"),
            request=object(), user_id="owner-a",
        ))
        assert recovered_response.scope_snapshot_state == "completed"
        assert recovered_response.total > 0
    finally:
        if restarted_queue is not None:
            restarted_queue.shutdown(timeout=5)
        else:
            first_queue.shutdown(timeout=5)


def test_failure_is_captured_not_swallowed(queue):
    def boom(texts):
        raise RuntimeError("embedder down")
    queue.service.embed_fn = boom
    job = queue.submit(_req(), owner_user_id="u1")
    done = _wait_done(queue, job.job_id)
    assert done.status is JobStatus.FAILED
    assert "embedder down" in (done.error or "")


def test_list_jobs_scoped_to_owner(queue):
    a = queue.submit(_req(project="pa"), owner_user_id="alice")
    b = queue.submit(_req(project="pb"), owner_user_id="bob")
    _wait_done(queue, a.job_id); _wait_done(queue, b.job_id)
    mine = queue.list_jobs(owner_user_id="alice")
    assert [j.job_id for j in mine] == [a.job_id], "must only see own jobs"
    assert queue.get(b.job_id, owner_user_id="alice") is None, "cross-owner get must miss"


def test_queue_position_reported_for_waiting_jobs(queue):
    # Block the worker with a slow job, then queue two more; they must report positions.
    gate = threading.Event()
    orig = queue.service.index

    def slow(request, *, owner_user_id, progress_cb=None, **callbacks):
        gate.wait(timeout=5)
        return orig(
            request, owner_user_id=owner_user_id, progress_cb=progress_cb, **callbacks)

    queue.service.index = slow
    j1 = queue.submit(_req(project="p1"), owner_user_id="u1")
    time.sleep(0.05)  # let worker pick j1 up
    j2 = queue.submit(_req(project="p2"), owner_user_id="u1")
    j3 = queue.submit(_req(project="p3"), owner_user_id="u1")
    assert queue.get(j2.job_id).queue_position == 1
    assert queue.get(j3.job_id).queue_position == 2
    gate.set()
    for j in (j1, j2, j3):
        _wait_done(queue, j.job_id)
    assert queue.get(j3.job_id).queue_position == 0


def test_job_writes_catalog_on_completion():
    # True-async: the client bundles the inventory records into the submit; the worker
    # writes the catalog ONLY when the job completes — so index_code can return the job_id
    # immediately without blocking to report.
    spy = _SpyCatalog()
    svc = CodeKBService(embed_fn=_embed, vector_store=FakeStore())
    q = CodeIndexJobQueue(service=svc, catalog_service=spy)
    try:
        req = _req()
        req.report = _report(paths=("a.py",))
        job = q.submit(req, owner_user_id="u1")
        _wait_done(q, job.job_id)
        assert len(spy.reports) == 1
        r = spy.reports[0]
        assert r["owner"] == "u1" and r["project"] == "proj" and r["replace"] is True
        assert r["files"] == ["a.py"]
    finally:
        q.shutdown(timeout=5)


def test_job_without_report_does_not_write_catalog():
    spy = _SpyCatalog()
    svc = CodeKBService(embed_fn=_embed, vector_store=FakeStore())
    q = CodeIndexJobQueue(service=svc, catalog_service=spy)
    try:
        job = q.submit(_req(), owner_user_id="u1")  # no .report
        _wait_done(q, job.job_id)
        assert spy.reports == []
    finally:
        q.shutdown(timeout=5)


def test_failed_job_does_not_write_catalog():
    # The catalog must never claim files landed when the embed failed (W156 hazard).
    spy = _SpyCatalog()
    svc = CodeKBService(embed_fn=_embed, vector_store=FakeStore())
    q = CodeIndexJobQueue(service=svc, catalog_service=spy)
    try:
        def boom(texts):
            raise RuntimeError("embedder down")
        q.service.embed_fn = boom
        req = _req()
        req.report = _report()
        job = q.submit(req, owner_user_id="u1")
        done = _wait_done(q, job.job_id)
        assert done.status is JobStatus.FAILED
        assert spy.reports == [], "no catalog write on a failed index"
    finally:
        q.shutdown(timeout=5)


def test_failed_vector_deletion_does_not_publish_catalog_removal():
    class FailingDeleteStore(FakeStore):
        fail_delete = False

        def delete_point_ids(self, point_ids):
            if self.fail_delete:
                raise RuntimeError("qdrant delete failed")
            return super().delete_point_ids(point_ids)

    store = FailingDeleteStore()
    catalog = _SpyCatalog()
    service = CodeKBService(embed_fn=_embed, vector_store=store)
    initial = _req(project="delete-failure", n_chunks=1)
    initial.files[0].rel_path = "gone.py"
    service.index(initial, owner_user_id="u1")
    catalog.live_paths[("u1", "delete-failure")] = {"gone.py"}
    store.fail_delete = True
    queue = CodeIndexJobQueue(service=service, catalog_service=catalog)
    try:
        request = CodeIndexRequest(
            project="delete-failure", host="h1", root_dir="/repo",
            incremental=True, files=[], removed_paths=["gone.py"],
            report=CodeIndexReport(files=[], replace=False),
        )
        done = _wait_done(queue, queue.submit(request, owner_user_id="u1").job_id)

        assert done.status is JobStatus.FAILED
        assert catalog.reports == []
        assert catalog.deleted_files == []
        assert catalog.live_paths[("u1", "delete-failure")] == {"gone.py"}
    finally:
        queue.shutdown(timeout=5)


def test_catalog_write_failure_fails_job_instead_of_reporting_completion():
    class BoomCatalog(_SpyCatalog):
        def report(self, request, *, owner_user_id):
            raise RuntimeError("catalog down")
    svc = CodeKBService(embed_fn=_embed, vector_store=FakeStore())
    q = CodeIndexJobQueue(service=svc, catalog_service=BoomCatalog())
    try:
        req = _req()
        req.report = _report()
        job = q.submit(req, owner_user_id="u1")
        done = _wait_done(q, job.job_id)
        assert done.status is JobStatus.FAILED
        assert "catalog down" in (done.error or "")
    finally:
        q.shutdown(timeout=5)


def test_transient_catalog_failure_is_retried_before_job_completion(monkeypatch):
    from app.services.code_kb import jobs as jobs_module

    class RecoveringCatalog(_SpyCatalog):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def report(self, request, *, owner_user_id):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("catalog temporarily busy")
            return super().report(request, owner_user_id=owner_user_id)

    monkeypatch.setattr(jobs_module.time, "sleep", lambda _seconds: None)
    catalog = RecoveringCatalog()
    service = CodeKBService(embed_fn=_embed, vector_store=FakeStore())
    queue = CodeIndexJobQueue(service=service, catalog_service=catalog)
    try:
        request = _req(project="catalog-retry")
        request.report = _report()
        done = _wait_done(
            queue, queue.submit(request, owner_user_id="u1").job_id)

        assert done.status is JobStatus.COMPLETED
        assert catalog.attempts == 2
        assert catalog.live_paths[("u1", "catalog-retry")] == {"a.py"}
    finally:
        queue.shutdown(timeout=5)


def test_catalog_publish_failure_keeps_scope_unsafe_after_vector_commit(monkeypatch):
    class ToggleCatalog(_SpyCatalog):
        fail_report = False

        def report(self, request, *, owner_user_id):
            if self.fail_report:
                raise RuntimeError("catalog busy")
            return super().report(request, owner_user_id=owner_user_id)

    from app.services.code_kb import jobs as jobs_module

    monkeypatch.setattr(jobs_module.time, "sleep", lambda _seconds: None)
    catalog = ToggleCatalog()
    store = FakeStore()
    svc = CodeKBService(embed_fn=_embed, vector_store=store)
    q = CodeIndexJobQueue(service=svc, catalog_service=catalog)
    try:
        old = _req(project="split-brain")
        old.files[0].rel_path = "old.py"
        old.report = _report(paths=("old.py",))
        assert _wait_done(q, q.submit(old, owner_user_id="u1").job_id).status is JobStatus.COMPLETED
        assert catalog.live_paths[("u1", "split-brain")] == {"old.py"}

        catalog.fail_report = True
        new = _req(project="split-brain")
        new.files[0].rel_path = "new.py"
        new.report = _report(paths=("new.py",))
        done = _wait_done(q, q.submit(new, owner_user_id="u1").job_id)

        assert done.status is JobStatus.FAILED
        assert store.list_paths(owner_user_id="u1", project="split-brain") == {"new.py"}
        assert catalog.live_paths[("u1", "split-brain")] == {"old.py"}
        unsafe = q.scope_search_state(owner_user_id="u1", projects=["split-brain"])
        assert unsafe is not None
        assert unsafe.active_job_id == done.job_id
        assert unsafe.mutation_started is True
        assert q.inventory_scope_states(owner_user_id="u1")[("u1", "h1", "split-brain")][
            "active_index_job_id"] == done.job_id
    finally:
        q.shutdown(timeout=5)


def test_delete_started_during_catalog_publish_wins_without_resurrection():
    report_entered = threading.Event()
    release_report = threading.Event()

    class BlockingCatalog(_SpyCatalog):
        def report(self, request, *, owner_user_id):
            report_entered.set()
            assert release_report.wait(5)
            return super().report(request, owner_user_id=owner_user_id)

    catalog = BlockingCatalog()
    store = FakeStore()
    svc = CodeKBService(embed_fn=_embed, vector_store=store)
    q = CodeIndexJobQueue(service=svc, catalog_service=catalog)
    try:
        req = _req(project="delete-race")
        req.report = _report(paths=("a.py",))
        job = q.submit(req, owner_user_id="u1")
        assert report_entered.wait(2)

        delete_done = threading.Event()

        def delete_scope():
            q.cancel_project(owner_user_id="u1", project="delete-race")
            with svc.scope_snapshot(owner_user_id="u1"):
                svc.delete_project("delete-race", owner_user_id="u1")
                catalog.delete_project(
                    owner_user_id="u1", project="delete-race", kind="code")
            delete_done.set()

        delete_thread = threading.Thread(target=delete_scope)
        delete_thread.start()
        time.sleep(0.05)
        release_report.set()
        delete_thread.join(timeout=5)
        done = _wait_done(q, job.job_id)

        assert delete_done.is_set()
        assert done.status in {JobStatus.COMPLETED, JobStatus.FAILED}
        assert store.count(owner_user_id="u1", project="delete-race") == 0
        assert catalog.live_paths[("u1", "delete-race")] == set()
    finally:
        release_report.set()
        q.shutdown(timeout=5)


def test_project_delete_fences_running_job_and_prevents_resurrection():
    gate = threading.Event()
    entered = threading.Event()

    def slow_embed(texts):
        entered.set()
        gate.wait(timeout=5)
        return _embed(texts)

    spy = _SpyCatalog()
    store = FakeStore()
    svc = CodeKBService(embed_fn=slow_embed, vector_store=store)
    q = CodeIndexJobQueue(service=svc, catalog_service=spy)
    try:
        req = _req(project="doomed")
        req.report = _report()
        job = q.submit(req, owner_user_id="u1")
        assert entered.wait(timeout=2)
        q.cancel_project(owner_user_id="u1", project="doomed")
        deleted = threading.Thread(
            target=svc.delete_project, args=("doomed",), kwargs={"owner_user_id": "u1"})
        deleted.start()
        gate.set()
        deleted.join(timeout=5)

        done = _wait_done(q, job.job_id)
        assert done.status is JobStatus.FAILED
        assert "superseded" in (done.error or "")
        assert store.count(owner_user_id="u1", project="doomed") == 0
        assert spy.reports == []
    finally:
        gate.set()
        q.shutdown(timeout=5)


def test_client_job_id_remains_queryable_and_aliases_reused_active_job():
    release_embed = threading.Event()
    embed_started = threading.Event()

    def slow_embed(texts):
        embed_started.set()
        release_embed.wait(timeout=5)
        return _embed(texts)

    store = FakeStore()
    q = CodeIndexJobQueue(
        service=CodeKBService(embed_fn=slow_embed, vector_store=store),
        catalog_service=_SpyCatalog(),
    )
    try:
        first_request = _req(project="durable-receipt")
        first_request.client_job_id = "a" * 32
        first = q.submit(first_request, owner_user_id="u1")
        assert first.job_id == "a" * 32
        assert q.get("a" * 32, owner_user_id="u1") is not None
        assert embed_started.wait(timeout=2)

        reused_request = _req(project="durable-receipt")
        reused_request.client_job_id = "b" * 32
        reused = q.submit(reused_request, owner_user_id="u1")
        assert reused.job_id == "b" * 32
        assert q._job_aliases["b" * 32] == "a" * 32
        alias_view = q.get("b" * 32, owner_user_id="u1")
        assert alias_view is not None and alias_view.job_id == "b" * 32
    finally:
        release_embed.set()
        q.shutdown(timeout=5)


def test_delete_waits_for_cross_scope_fan_out_commit_boundary():
    class BlockingFanOutStore(FakeStore):
        def __init__(self):
            super().__init__()
            self.block_fan_out = False
            self.fan_out_entered = threading.Event()
            self.release_fan_out = threading.Event()

        def list_point_ids(self, *, owner_user_id, project, path):
            # Block on the per-TARGET read, which happens inside the write fence
            # immediately before that target's upsert — i.e. genuinely mid-fan-out.
            # Blocking `list_owner_files` instead would no longer test this: that
            # enumeration is a single pre-read hoisted out of the batch loop (it
            # scans every scope the owner has, so it must not run per batch), and a
            # delete overlapping a pure read is not what the invariant is about.
            if self.block_fan_out and project != "/repo":
                self.block_fan_out = False
                self.fan_out_entered.set()
                assert self.release_fan_out.wait(5)
            return {
                id(point) for point in self.points
                if point["owner_user_id"] == owner_user_id
                and point["project"] == project and point["path"] == path
            }

        def delete_point_ids(self, point_ids):
            before = len(self.points)
            self.points = [point for point in self.points if id(point) not in point_ids]
            return before - len(self.points)

    store = BlockingFanOutStore()
    catalog = _SpyCatalog()
    service = CodeKBService(embed_fn=_embed, vector_store=store)
    queue = CodeIndexJobQueue(service=service, catalog_service=catalog)
    try:
        child = _req(project="/repo/child", n_chunks=1)
        child.files[0].rel_path = "a.py"
        child.files[0].chunks[0].content = "qdrant OLD_CHILD"
        child.files[0].chunks[0].code = "def marker(): return 'OLD_CHILD'"
        service.index(child, owner_user_id="u1")

        parent = _req(project="/repo", n_chunks=1)
        parent.files[0].rel_path = "child/a.py"
        parent.files[0].chunks[0].content = "qdrant NEW_PARENT"
        parent.files[0].chunks[0].code = "def marker(): return 'NEW_PARENT'"
        parent.report = _report(paths=("child/a.py",))
        store.block_fan_out = True
        job = queue.submit(parent, owner_user_id="u1")
        assert store.fan_out_entered.wait(2)

        delete_done = threading.Event()

        def delete_parent():
            queue.cancel_project(owner_user_id="u1", project="/repo")
            service.delete_project("/repo", owner_user_id="u1")
            delete_done.set()

        delete_thread = threading.Thread(target=delete_parent)
        delete_thread.start()
        assert not delete_done.wait(0.1), "delete crossed a half-completed fan-out"
        store.release_fan_out.set()
        delete_thread.join(timeout=5)
        done = _wait_done(queue, job.job_id)

        # The index transaction committed before the waiting delete acquired the
        # scope fence. Reporting COMPLETED is important: a watcher must not retry
        # work that really landed, while the later delete remains authoritative.
        assert done.status is JobStatus.COMPLETED
        assert delete_done.is_set()
        assert not [
            point for point in store.points
            if point["owner_user_id"] == "u1" and point["project"] == "/repo"
        ]
        child_points = [
            point for point in store.points
            if point["owner_user_id"] == "u1" and point["project"] == "/repo/child"
        ]
        assert child_points and "NEW_PARENT" in child_points[0]["code"]
    finally:
        store.release_fan_out.set()
        queue.shutdown(timeout=5)


def test_empty_enumeration_completes_without_wiping(queue):
    """The sync path's guard (W136-adjacent) must survive queueing: 0 chunks → no delete."""
    queue.service.index(_req(n_chunks=2), owner_user_id="u1")
    empty = CodeIndexRequest(project="proj", host="h1", root_dir="/repo", files=[])
    job = queue.submit(empty, owner_user_id="u1")
    done = _wait_done(queue, job.job_id)
    assert done.status is JobStatus.COMPLETED and done.indexed == 0
    assert queue.service.vector_store.count() == 2, "empty submit must not wipe the index"


def test_cancel_queued_job_marks_cancelled_and_worker_never_runs_it():
    gate = threading.Event()
    store = FakeStore()
    service = CodeKBService(embed_fn=_embed, vector_store=store)
    original = service.index

    def slow(request, *, owner_user_id, progress_cb=None, **callbacks):
        gate.wait(timeout=5)
        return original(
            request, owner_user_id=owner_user_id, progress_cb=progress_cb, **callbacks)

    service.index = slow
    q = CodeIndexJobQueue(service=service, catalog_service=_SpyCatalog())
    try:
        blocker = q.submit(_req(project="p-block"), owner_user_id="u1")
        deadline = time.time() + 2
        while q.get(blocker.job_id).status is not JobStatus.RUNNING and time.time() < deadline:
            time.sleep(0.01)
        assert q.get(blocker.job_id).status is JobStatus.RUNNING

        queued = q.submit(_req(project="p-queued"), owner_user_id="u1")
        assert q.get(queued.job_id).status is JobStatus.QUEUED

        cancelled = q.cancel(queued.job_id, owner_user_id="u1")
        assert cancelled.status is JobStatus.CANCELLED
        assert cancelled.finished_at is not None

        gate.set()
        assert _wait_done(q, blocker.job_id).status is JobStatus.COMPLETED
        still = q.get(queued.job_id)
        assert still.status is JobStatus.CANCELLED
        assert still.chunks_done == 0
        assert not any(p["project"] == "p-queued" for p in store.points), (
            "a cancelled-while-queued job must never have run at all")
    finally:
        gate.set()
        q.shutdown(timeout=5)


def test_cancel_running_job_before_mutation_stops_with_nothing_written():
    embed_started = threading.Event()
    release_embed = threading.Event()

    def slow_embed(texts):
        embed_started.set()
        assert release_embed.wait(5)
        return _embed(texts)

    store = FakeStore()
    service = CodeKBService(embed_fn=slow_embed, vector_store=store)
    q = CodeIndexJobQueue(service=service, catalog_service=_SpyCatalog())
    try:
        job = q.submit(_req(project="cancel-before-write"), owner_user_id="u1")
        assert embed_started.wait(2)

        cancelled = q.cancel(job.job_id, owner_user_id="u1")
        # Not yet transitioned — the worker settles it asynchronously once the
        # blocked embed call returns and the mutation fence observes the flag.
        assert cancelled.status is JobStatus.RUNNING

        release_embed.set()
        done = _wait_done(q, job.job_id)
        assert done.status is JobStatus.CANCELLED
        assert done.error is None
        assert store.count(owner_user_id="u1", project="cancel-before-write") == 0
        assert q.scope_search_state(
            owner_user_id="u1", projects=["cancel-before-write"]) is None
    finally:
        release_embed.set()
        q.shutdown(timeout=5)


def test_cancel_running_job_after_mutation_started_returns_committing():
    write_started = threading.Event()
    release_write = threading.Event()

    class BlockingStore(FakeStore):
        def upsert(self, **kwargs):
            write_started.set()
            assert release_write.wait(5)
            return super().upsert(**kwargs)

    store = BlockingStore()
    service = CodeKBService(embed_fn=_embed, vector_store=store)
    q = CodeIndexJobQueue(service=service, catalog_service=_SpyCatalog())
    try:
        job = q.submit(_req(project="cancel-after-write"), owner_user_id="u1")
        assert write_started.wait(2)
        assert q.get(job.job_id).mutation_started is True

        with pytest.raises(CodeIndexJobCommitting):
            q.cancel(job.job_id, owner_user_id="u1")

        release_write.set()
        done = _wait_done(q, job.job_id)
        assert done.status is JobStatus.COMPLETED
        assert store.count(owner_user_id="u1", project="cancel-after-write") == 3
    finally:
        release_write.set()
        q.shutdown(timeout=5)


def test_cancel_during_preparing_marker_write_reports_committing_not_a_false_cancel():
    """Regression: cancel_requested was checked under the lock, but the lock was then
    released for the SQLite preparing-marker write — a cancel() landing in that gap
    saw mutation_started=False, set the flag, and returned 200 while the job kept
    writing and completed anyway (the API reported a cancel that never happened).

    commit_claimed must close that window atomically: set in the SAME locked section
    as the cancel_requested check, before the lock is released for the marker write.
    """
    entered_marker_write = threading.Event()
    release_marker_write = threading.Event()

    class GatedCatalogService(IndexCatalogService):
        def mark_code_scope_preparing(self, **kwargs):
            entered_marker_write.set()
            assert release_marker_write.wait(5), "test did not release the marker write"
            return super().mark_code_scope_preparing(**kwargs)

    catalog = GatedCatalogService(repository=_sqlite_catalog_service().repository)
    service = CodeKBService(embed_fn=_embed, vector_store=FakeStore())
    q = CodeIndexJobQueue(service=service, catalog_service=catalog)
    try:
        job = q.submit(_req(project="commit-claimed-race"), owner_user_id="u1")
        assert entered_marker_write.wait(2)

        # mutation_started is still False here (only the marker write, not yet the
        # Qdrant upsert, is under way) — the old check would have accepted this as a
        # real cancel.
        assert q.get(job.job_id).mutation_started is False
        with pytest.raises(CodeIndexJobCommitting):
            q.cancel(job.job_id, owner_user_id="u1")

        release_marker_write.set()
        done = _wait_done(q, job.job_id)
        assert done.status is JobStatus.COMPLETED, (
            "cancel() correctly refused, so the job must run to completion, not land "
            "as cancelled after having actually written"
        )
        assert done.indexed == 3
    finally:
        release_marker_write.set()
        q.shutdown(timeout=5)


def test_preparing_marker_write_failure_aborts_cleanly_without_recovery_required():
    """A SQLite failure writing the FIRST preparing marker means nothing was ever
    written to Qdrant. commit_claimed must not be conflated with mutation_started:
    the job fails cleanly, with no unsafe/recovery-required scope, exactly like
    before commit_claimed was introduced."""
    class BoomOnPreparingCatalog(_SpyCatalog):
        def mark_code_scope_preparing(self, **_kwargs):
            raise RuntimeError("simulated SQLite outage before the first write")

    store = FakeStore()
    service = CodeKBService(embed_fn=_embed, vector_store=store)
    q = CodeIndexJobQueue(service=service, catalog_service=BoomOnPreparingCatalog())
    try:
        job = q.submit(_req(project="preparing-marker-outage"), owner_user_id="u1")
        done = _wait_done(q, job.job_id)

        assert done.status is JobStatus.FAILED
        assert done.mutation_started is False
        assert not any(p["project"] == "preparing-marker-outage" for p in store.points), (
            "nothing may have reached Qdrant when the very first marker write failed"
        )
        assert q.scope_search_state(
            owner_user_id="u1", projects=["preparing-marker-outage"]) is None, (
            "a marker-write failure before any Qdrant write must not mark the scope unsafe"
        )
    finally:
        q.shutdown(timeout=5)


def test_trim_history_evicts_cancelled_jobs_past_the_cap(queue):
    """Before this fix, _trim_history_locked's eligible-for-eviction set was only
    COMPLETED/FAILED — a scope that kept getting cancelled-while-queued grew the
    in-process job history forever instead of being trimmed like any other
    terminal job."""
    with queue._lock:
        for i in range(MAX_JOBS_KEPT + 5):
            job = CodeIndexJob(
                job_id=f"cancelled-{i}", owner_user_id="u1", project=f"p{i}",
                host="h1", files_total=0, chunks_total=0,
                status=JobStatus.CANCELLED, finished_at=time.time(),
            )
            queue._jobs[job.job_id] = job
        queue._trim_history_locked()

    assert len(queue._jobs) <= MAX_JOBS_KEPT


def test_cancel_already_finished_job_raises_already_ended(queue):
    done = _wait_done(queue, queue.submit(_req(), owner_user_id="u1").job_id)
    assert done.status is JobStatus.COMPLETED
    with pytest.raises(CodeIndexJobAlreadyEnded):
        queue.cancel(done.job_id, owner_user_id="u1")


def test_cancel_unknown_or_other_owners_job_raises_keyerror(queue):
    with pytest.raises(KeyError):
        queue.cancel("does-not-exist", owner_user_id="u1")

    theirs = queue.submit(_req(project="theirs"), owner_user_id="u2")
    _wait_done(queue, theirs.job_id)
    with pytest.raises(KeyError):
        queue.cancel(theirs.job_id, owner_user_id="u1")


def test_cancel_before_write_then_next_job_for_same_scope_completes_and_is_searchable():
    """Regression: a cancelled job must not poison the scope for the next real job."""
    embed_started = threading.Event()
    release_embed = threading.Event()
    first_call = True

    def slow_embed(texts):
        nonlocal first_call
        if first_call:
            first_call = False
            embed_started.set()
            assert release_embed.wait(5)
        return _embed(texts)

    store = FakeStore()
    service = CodeKBService(embed_fn=slow_embed, vector_store=store)
    q = CodeIndexJobQueue(service=service, catalog_service=_SpyCatalog())
    try:
        cancelled_job = q.submit(_req(project="recover-after-cancel"), owner_user_id="u1")
        assert embed_started.wait(2)
        q.cancel(cancelled_job.job_id, owner_user_id="u1")
        release_embed.set()
        cancelled_done = _wait_done(q, cancelled_job.job_id)
        assert cancelled_done.status is JobStatus.CANCELLED

        next_job = q.submit(_req(project="recover-after-cancel"), owner_user_id="u1")
        next_done = _wait_done(q, next_job.job_id)
        assert next_done.status is JobStatus.COMPLETED
        assert next_done.indexed == 3
        assert q.scope_search_state(
            owner_user_id="u1", projects=["recover-after-cancel"]) is None

        search_resp = service.search(
            CodeSearchRequest(query="qdrant upsert vector", project="recover-after-cancel"),
            owner_user_id="u1",
        )
        assert search_resp.total > 0
    finally:
        release_embed.set()
        q.shutdown(timeout=5)


def test_watcher_job_soft_deletes_only_explicit_catalog_paths():
    store = FakeStore()
    catalog = _SpyCatalog()
    catalog.live_paths[("u1", "proj")] = {"a.py", "b.py"}
    service = CodeKBService(embed_fn=_embed, vector_store=store)
    service.index(_req(n_chunks=1), owner_user_id="u1")
    queue = CodeIndexJobQueue(service=service, catalog_service=catalog)
    try:
        request = CodeIndexRequest(
            project="proj", host="h1", root_dir="/repo", incremental=True,
            files=[], removed_paths=["b.py"],
            report=CodeIndexReport(files=[], replace=False),
        )
        done = _wait_done(queue, queue.submit(request, owner_user_id="u1").job_id)

        assert done.status is JobStatus.COMPLETED
        assert catalog.deleted_files == [{
            "owner": "u1", "project": "proj", "host": "h1",
            "kind": "code", "rel_paths": ["b.py"],
        }]
        assert catalog.live_paths[("u1", "proj")] == {"a.py"}
    finally:
        queue.shutdown(timeout=5)
