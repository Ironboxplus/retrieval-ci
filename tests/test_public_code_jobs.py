"""Public-code control jobs return immediately and isolate long-running work."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import threading
import time

from app.services.public_code.index_service import PublicCodeIndexProgress
from app.services.public_code.jobs import PublicCodeJobService, PublicCodeJobStatus
from app.services.public_code.source_catalog import PublicCodeSource, PublicCodeSourceCatalog
from app.services.public_code.update_checks import PublicCodeUpdateCheck


def catalog():
    return PublicCodeSourceCatalog(
        catalog_id="fixture",
        sources=(
            PublicCodeSource("ffmpeg", "multimedia", "multimedia/ffmpeg", "https://example.invalid/ffmpeg.git"),
            PublicCodeSource("pion", "multimedia", "multimedia/pion", "https://example.invalid/pion.git"),
        ),
    )


class ControlService:
    def __init__(self, gate=None):
        self.gate = gate

    def check_updates(self, *, source_ids, domains, progress_cb=None):
        if self.gate is not None:
            self.gate.wait(timeout=5)
        checks = tuple(
            PublicCodeUpdateCheck(
                source_id=source_id,
                domain="multimedia",
                local_revision="old",
                remote_revision="new",
                status="update_available",
            )
            for source_id in source_ids
        )
        if progress_cb is not None:
            for completed, check in enumerate(checks, start=1):
                progress_cb(check, completed, len(checks))
        return checks


@dataclass(frozen=True)
class IndexResult:
    source_id: str
    revision: str
    status: str = "completed"
    file_count: int = 1
    chunk_count: int = 2


class IndexService:
    def __init__(self, *, fail_source=None, gate=None, activity=None):
        self.fail_source = fail_source
        self.gate = gate
        self.activity = activity

    def index_source(self, source_id, progress_cb=None, *, refresh_mirror=True):
        self.refresh_calls = getattr(self, "refresh_calls", []) + [refresh_mirror]
        if self.activity is not None:
            with self.activity["lock"]:
                self.activity["active"] += 1
                self.activity["max"] = max(self.activity["max"], self.activity["active"])
        try:
            if self.gate is not None:
                self.gate.wait(timeout=5)
            if progress_cb is not None:
                progress_cb(PublicCodeIndexProgress(
                    source_id=source_id,
                    revision=f"{source_id}-revision",
                    phase="chunking",
                    files_done=1,
                    files_total=2,
                    chunks_done=3,
                    current_path="src/example.c",
                    message="chunked source file",
                ))
            if source_id == self.fail_source:
                raise RuntimeError(f"cannot index {source_id}")
            return IndexResult(source_id=source_id, revision=f"{source_id}-revision")
        finally:
            if self.activity is not None:
                with self.activity["lock"]:
                    self.activity["active"] -= 1


@dataclass
class JobStore:
    snapshots: dict[str, dict] = field(default_factory=dict)

    def save_job_snapshot(self, snapshot, *, max_retained_jobs):
        self.snapshots[snapshot["job_id"]] = dict(snapshot)

    def get_job_snapshot(self, job_id):
        snapshot = self.snapshots.get(job_id)
        return dict(snapshot) if snapshot is not None else None

    def list_job_snapshots(self, *, limit):
        return tuple(list(self.snapshots.values())[-limit:])

    def fail_incomplete_jobs(self, *, reason):
        for job_id, snapshot in list(self.snapshots.items()):
            if snapshot["status"] not in {"queued", "running"}:
                continue
            updated = dict(snapshot)
            updated.update({
                "status": "failed",
                "phase": "failed",
                "message": reason,
                "errors": {"job": reason},
                "finished_at": time.time(),
            })
            self.snapshots[job_id] = updated


def wait_done(service, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = service.get(job_id)
        if job and job.status in {PublicCodeJobStatus.COMPLETED, PublicCodeJobStatus.FAILED}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job did not finish: {service.get(job_id)}")


def test_update_check_submission_returns_before_background_work_finishes():
    gate = threading.Event()
    jobs = PublicCodeJobService(
        source_registry=catalog(),
        control_service=ControlService(gate),
        index_service=IndexService(),
    )
    try:
        receipt = jobs.submit_update_check(source_ids=("ffmpeg",))
        assert receipt.job_id
        assert receipt.status in {PublicCodeJobStatus.QUEUED, PublicCodeJobStatus.RUNNING}
        assert jobs.get(receipt.job_id).completed_sources == 0

        gate.set()
        completed = wait_done(jobs, receipt.job_id)
        assert completed.status is PublicCodeJobStatus.COMPLETED
        assert completed.completed_sources == 1
        assert completed.results[0]["status"] == "update_available"
        assert completed.phase == "completed"
        assert completed.source_progress["ffmpeg"]["phase"] == "completed"
    finally:
        gate.set()
        jobs.shutdown()


def test_index_job_continues_other_sources_and_reports_partial_failure():
    jobs = PublicCodeJobService(
        source_registry=catalog(),
        control_service=ControlService(),
        index_service=IndexService(fail_source="ffmpeg"),
    )
    try:
        receipt = jobs.submit_index(domains=("multimedia",))
        completed = wait_done(jobs, receipt.job_id)

        assert completed.status is PublicCodeJobStatus.FAILED
        assert completed.completed_sources == 2
        assert [result["source_id"] for result in completed.results] == ["pion"]
        assert completed.errors == {"ffmpeg": "cannot index ffmpeg"}
        assert completed.source_progress["ffmpeg"]["phase"] == "failed"
        assert completed.source_progress["pion"]["chunks_done"] == 3
    finally:
        jobs.shutdown()


def test_index_jobs_are_serialized_for_the_shared_gpu_embedder():
    gate = threading.Event()
    activity = {"lock": threading.Lock(), "active": 0, "max": 0}
    jobs = PublicCodeJobService(
        source_registry=catalog(),
        control_service=ControlService(),
        index_service=IndexService(gate=gate, activity=activity),
    )
    try:
        first = jobs.submit_index(source_ids=("ffmpeg",))
        second = jobs.submit_index(source_ids=("pion",))
        time.sleep(0.05)
        assert activity["max"] == 1

        gate.set()
        assert wait_done(jobs, first.job_id).status is PublicCodeJobStatus.COMPLETED
        assert wait_done(jobs, second.job_id).status is PublicCodeJobStatus.COMPLETED
        assert activity["max"] == 1
    finally:
        gate.set()
        jobs.shutdown()


def test_index_job_exposes_current_file_and_chunk_progress_before_completion():
    gate = threading.Event()

    class ObservableIndexService(IndexService):
        def index_source(self, source_id, progress_cb=None):
            progress_cb(PublicCodeIndexProgress(
                source_id=source_id,
                revision="commit-a",
                phase="chunking",
                files_done=4,
                files_total=10,
                chunks_done=42,
                current_path="libavcodec/decoder.c",
                message="chunking source files",
            ))
            gate.wait(timeout=5)
            return IndexResult(source_id=source_id, revision="commit-a")

    jobs = PublicCodeJobService(
        source_registry=catalog(),
        control_service=ControlService(),
        index_service=ObservableIndexService(),
    )
    try:
        receipt = jobs.submit_index(source_ids=("ffmpeg",))
        deadline = time.time() + 2
        running = None
        while time.time() < deadline:
            running = jobs.get(receipt.job_id)
            if running and running.current_path:
                break
            time.sleep(0.01)

        assert running.phase == "chunking"
        assert running.current_source_id == "ffmpeg"
        assert running.current_path == "libavcodec/decoder.c"
        assert running.files_done == 4
        assert running.files_total == 10
        assert running.chunks_done == 42
        assert running.progress == 0.4
    finally:
        gate.set()
        wait_done(jobs, receipt.job_id)
        jobs.shutdown()


def test_recent_jobs_are_returned_newest_first_for_page_reload_observability():
    gate = threading.Event()
    jobs = PublicCodeJobService(
        source_registry=catalog(),
        control_service=ControlService(),
        index_service=IndexService(gate=gate),
    )
    try:
        first = jobs.submit_index(source_ids=("ffmpeg",))
        second = jobs.submit_update_check(source_ids=("pion",))

        recent = jobs.list_recent(limit=10)

        assert [job.job_id for job in recent[:2]] == [second.job_id, first.job_id]
    finally:
        gate.set()
        wait_done(jobs, first.job_id)
        jobs.shutdown()


def test_completed_job_history_is_bounded_without_dropping_active_jobs():
    jobs = PublicCodeJobService(
        source_registry=catalog(),
        control_service=ControlService(),
        index_service=IndexService(),
        max_retained_jobs=2,
    )
    try:
        receipts = []
        for _ in range(3):
            receipt = jobs.submit_update_check(source_ids=("ffmpeg",))
            receipts.append(receipt)
            wait_done(jobs, receipt.job_id)

        recent = jobs.list_recent(limit=10)

        assert len(recent) == 2
        assert receipts[0].job_id not in {job.job_id for job in recent}
    finally:
        jobs.shutdown()


def test_persisted_running_job_becomes_queryable_interrupted_failure_after_restart():
    gate = threading.Event()
    store = JobStore()
    first_process = PublicCodeJobService(
        source_registry=catalog(),
        control_service=ControlService(),
        index_service=IndexService(gate=gate),
        job_store=store,
        persistence_interval_seconds=0.001,
    )
    receipt = first_process.submit_index(source_ids=("ffmpeg",))
    deadline = time.time() + 2
    while time.time() < deadline and store.snapshots[receipt.job_id]["status"] == "queued":
        time.sleep(0.01)

    restarted_process = PublicCodeJobService(
        source_registry=catalog(),
        control_service=ControlService(),
        index_service=IndexService(),
        job_store=store,
    )
    try:
        recovered = restarted_process.get(receipt.job_id)

        assert recovered is not None
        assert recovered.status is PublicCodeJobStatus.FAILED
        assert "backend restarted" in recovered.errors["job"]
    finally:
        gate.set()
        first_process.shutdown()
        restarted_process.shutdown()


def test_import_job_publishes_the_mirror_then_indexes_without_fetching_upstream():
    order = []
    index = IndexService()
    jobs = PublicCodeJobService(
        source_registry=catalog(), control_service=ControlService(), index_service=index)
    try:
        def prepare(progress):
            progress("building a bare mirror from the bundle")
            order.append("prepare")

        receipt = jobs.submit_import(source_id="pion", prepare=prepare)
        assert receipt.operation == "import"
        job = wait_done(jobs, receipt.job_id)
    finally:
        jobs.shutdown()

    assert job.status is PublicCodeJobStatus.COMPLETED
    assert order == ["prepare"]
    # the bundle already wrote the revision; fetching upstream is what it replaces
    assert index.refresh_calls == [False]
    assert job.results[0]["source_id"] == "pion"


def test_failed_import_is_reported_and_never_indexed():
    index = IndexService()
    jobs = PublicCodeJobService(
        source_registry=catalog(), control_service=ControlService(), index_service=index)
    try:
        def prepare(progress):
            raise ValueError("sha256 mismatch: declared a, received b")

        job = wait_done(jobs, jobs.submit_import(source_id="pion", prepare=prepare).job_id)
    finally:
        jobs.shutdown()

    assert job.status is PublicCodeJobStatus.FAILED
    assert job.errors == {"pion": "sha256 mismatch: declared a, received b"}
    assert job.source_progress["pion"]["phase"] == "failed"
    assert not hasattr(index, "refresh_calls")


def test_a_running_import_counts_as_active_indexing():
    gate = threading.Event()
    jobs = PublicCodeJobService(
        source_registry=catalog(), control_service=ControlService(), index_service=IndexService())
    try:
        receipt = jobs.submit_import(source_id="pion", prepare=lambda progress: gate.wait(5))
        assert jobs.has_active_index("pion")
        gate.set()
        wait_done(jobs, receipt.job_id)
        assert not jobs.has_active_index("pion")
    finally:
        gate.set()
        jobs.shutdown()
