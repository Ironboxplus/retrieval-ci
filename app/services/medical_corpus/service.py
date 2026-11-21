"""Medical PDF download as a durable ``TaskStore`` / ``PipelineOrchestrator`` job."""
from __future__ import annotations

import os
import shutil
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from filelock import FileLock, Timeout

from app.services.pipeline import GateResult, PipelineOrchestrator, Progress, StageSpec, TaskStore
from app.services.pipeline.runner_registry import cancel_run, register_runner
from app.services.pipeline.stage_lock import StageLockSet
from pipeline.medical_corpus.pdf_fetch import DEFAULT_MAX_BYTES_PER_SECOND, DownloadRateLimiter
from pipeline.medical_corpus.prepare import (
    DEFAULT_MIN_FREE_BYTES,
    PrepareConfig,
    _load_jsonl,
    require_absolute_medical_data_root,
    run_download_pass,
)
from pipeline.medical_corpus.executor import (  # noqa: F401 - the service's public names
    MEDICAL_ACTIONS_EXECUTOR,
    MEDICAL_PRODUCTION_DATA_ROOT,
    designated_medical_executor,
    require_designated_medical_cli_execution,
    require_designated_medical_download_host,
)
from pipeline.medical_corpus.proxy import DEFAULT_PROXY_URL, MedicalProxyConfig, load_medical_proxy_from_settings

# A normal durable Task type.  It deliberately appears in the existing Tasks
# list/detail surface rather than owning a second control-plane UI.
MEDICAL_DOWNLOAD_PIPELINE = "medical_corpus_download"


@dataclass
class MedicalDownloadContext:
    task_id: int
    store: TaskStore
    data_root: Path
    proxy: MedicalProxyConfig
    download_limit: int
    allow_cf_bypass: bool
    min_free_bytes: int
    rate_limiter: DownloadRateLimiter
    abort_event: threading.Event

    @property
    def config(self) -> PrepareConfig:
        return PrepareConfig(
            data_root=self.data_root,
            execute_download=True,
            download_limit=self.download_limit,
            allow_cf_bypass=self.allow_cf_bypass,
            proxy=self.proxy,
            max_bytes_per_second=DEFAULT_MAX_BYTES_PER_SECOND,
            min_free_bytes=self.min_free_bytes,
        )


def _manifest_progress(ctx: MedicalDownloadContext) -> Progress:
    records = _load_jsonl(ctx.config.manifest_path)
    if not records:
        return Progress(0, 0, "medical manifest is empty")
    state = ctx.store.get_state(ctx.task_id)
    target_ids = set(state.get("target_paper_ids") or [])
    target = [row for row in records if not target_ids or row.paper_id in target_ids]
    # Failed/blocked are processed facts but never inflate success progress;
    # gate below changes the parent status to failed for that batch.
    verified = sum(row.download_status in {"downloaded", "skipped"} for row in target)
    failed = sum(row.download_status in {"failed", "blocked"} for row in target)
    remaining = max(0, len(records) - len(target))
    return Progress(
        verified,
        len(target),
        f"verified={verified}/{len(target)} failed_or_blocked={failed} remaining_manifest_pending={remaining}",
    )


def _persist_target_set(
    ctx: MedicalDownloadContext,
    *,
    records=None,
) -> None:
    """Freeze this task's batch before progress/poller can observe it."""
    state = ctx.store.get_state(ctx.task_id)
    if state.get("target_paper_ids") is not None:
        return
    if records is None:
        records = _load_jsonl(ctx.config.manifest_path)
    candidates = [row for row in records if row.download_status not in {"downloaded", "skipped"}]
    budget = ctx.download_limit if ctx.download_limit > 0 else len(candidates)
    target = candidates[:budget]
    ctx.store.update_state(
        ctx.task_id,
        target_paper_ids=[row.paper_id for row in target],
        target_total=len(target),
        remaining_manifest_pending=max(0, len(candidates) - len(target)),
    )


def _medical_download_gate(ctx: MedicalDownloadContext) -> GateResult:
    records = _load_jsonl(ctx.config.manifest_path)
    state = ctx.store.get_state(ctx.task_id)
    summary = state.get("summary") or {}
    if summary.get("stopped_low_disk") or summary.get("disk_guard_reason"):
        return GateResult(False, f"medical download stopped by disk guard: {summary.get('disk_guard_reason') or 'low disk'}")
    target_ids = set(state.get("target_paper_ids") or [])
    target = [row for row in records if not target_ids or row.paper_id in target_ids]
    bad = [row.paper_id for row in target if row.download_status == "downloaded" and not row.pdf_sha256]
    if bad:
        return GateResult(False, f"downloaded records missing sha256: {bad[:3]}")
    failed = [row.paper_id for row in target if row.download_status in {"failed", "blocked"}]
    if failed:
        return GateResult(False, f"medical download has failed/blocked records: {failed[:3]}")
    pending = [row.paper_id for row in target if row.download_status in {"pending", "planned"}]
    if pending:
        return GateResult(False, f"medical download left pending records: {pending[:3]}")
    return GateResult(True)


def build_medical_download_pipeline(ctx: MedicalDownloadContext) -> list[StageSpec]:
    def run(c: MedicalDownloadContext, sink) -> None:
        _persist_target_set(c)
        sink(
            "medical PDF download: proxy=http://127.0.0.1:10808 rate=2.00 MB/s "
            "(2000000 B/s)"
        )

        last_flush_at = 0.0
        last_flush_bytes = 0
        aggregate_bytes = int(c.store.get_state(c.task_id).get("aggregate_bytes") or 0)
        current_paper_id: str | None = None
        current_paper_base = aggregate_bytes

        def on_progress(record, current: int, total: int | None) -> None:
            nonlocal last_flush_at, last_flush_bytes, aggregate_bytes, current_paper_id, current_paper_base
            now = time.monotonic()
            # 64 KiB chunks at 2 MB/s are ~31 calls/sec.  TaskStore is SQLite
            # truth, but it must not become the bottleneck; a 250 ms / 512 KiB
            # coalescer preserves useful UI progress and force-flushes below at
            # every manifest record checkpoint.
            paper_changed = current_paper_id != record.paper_id
            if paper_changed:
                current_paper_id = record.paper_id
                current_paper_base = aggregate_bytes
            aggregate_bytes = max(aggregate_bytes, current_paper_base + current)
            if not paper_changed and current - last_flush_bytes < 512 * 1024 and now - last_flush_at < 0.25:
                return
            c.store.update_state(
                c.task_id,
                current_paper_id=record.paper_id,
                aggregate_bytes=aggregate_bytes,
                current_file_bytes=current,
                current_file_total=total,
                max_bytes_per_second=DEFAULT_MAX_BYTES_PER_SECOND,
                proxy=c.proxy.http_proxy,
            )
            c.store.heartbeat(c.task_id)
            last_flush_at = now
            last_flush_bytes = current

        def on_checkpoint(record, attempted: int, budget: int) -> None:
            nonlocal aggregate_bytes
            if current_paper_id == record.paper_id and record.pdf_bytes:
                aggregate_bytes = max(aggregate_bytes, current_paper_base + int(record.pdf_bytes))
            progress = _manifest_progress(c)
            c.store.update_progress(c.task_id, "download", progress.done, progress.total, progress.message)
            # The generic single-stage PipelineOrchestrator contract intentionally
            # mirrors its artifact probe to the parent Task.  Force the checkpoint
            # update too, so normal Tasks table polling advances immediately rather
            # than waiting for the two-second probe sampler.
            c.store.set_task_progress(c.task_id, progress.done, progress.total)
            c.store.update_state(
                c.task_id,
                current_paper_id=None,
                aggregate_bytes=aggregate_bytes,
                current_file_bytes=0,
                current_file_total=None,
                attempted=attempted,
                budget=budget,
                last_paper_id=record.paper_id,
                last_download_status=record.download_status,
            )
            sink(f"checkpoint paper={record.paper_id} status={record.download_status} {progress.done}/{progress.total}")

        last_cancel_db_check = 0.0

        def should_abort() -> bool:
            nonlocal last_cancel_db_check
            if c.abort_event.is_set():
                return True
            now = time.monotonic()
            # A second backend process receives generic Tasks cancellation with
            # a different in-memory Event.  Poll the durable source of truth at
            # most four times per second; once observed, set this local Event
            # so downloader and orchestrator agree on the cancelled outcome.
            if now - last_cancel_db_check < 0.25:
                return False
            last_cancel_db_check = now
            task = c.store.get_task(c.task_id)
            if task is not None and task.get("status") == "cancelled":
                c.abort_event.set()
                return True
            return False

        summary = run_download_pass(
            c.config,
            progress_callback=on_progress,
            checkpoint_callback=on_checkpoint,
            rate_limiter=c.rate_limiter,
            should_cancel=should_abort,
            abort_event=c.abort_event,
        )
        c.store.update_state(c.task_id, current_paper_id=None, summary=summary)

    return [StageSpec(
        key="download",
        title="Download medical OA PDFs via .88 proxy",
        run=run,
        probe=_manifest_progress,
        gate=_medical_download_gate,
        poll_interval_s=2.0,
        # A 0-limit task can be a legitimate explicit re-run after new manifest
        # rows arrive; the download pass itself skips verified artifacts.
        skip_when_satisfied=False,
        resource_key="medical-pdf-net-2mbps",
        parent_progress_from_probe=True,
    )]


class MedicalDownloadPipelineRunner:
    """Launch/resume DB-backed medical download pipelines, one global bandwidth lane."""

    task_type = MEDICAL_DOWNLOAD_PIPELINE

    def __init__(
        self,
        *,
        store: TaskStore | None = None,
        data_root: Path | None = None,
        proxy: MedicalProxyConfig | None = None,
        execution_guard: Callable[[], None] = require_designated_medical_download_host,
        stage_locks: StageLockSet | None = None,
        enforce_designated_proxy: bool = True,
        enforce_designated_root: bool | None = None,
    ) -> None:
        self.store = store or TaskStore()
        configured_root = data_root or Path(
            os.environ.get("MEDICAL_DATA_ROOT", "/mnt/lsb/retrival/medical"))
        self.data_root = require_absolute_medical_data_root(configured_root)
        self.proxy = proxy or load_medical_proxy_from_settings()
        self.execution_guard = execution_guard
        self.enforce_designated_proxy = enforce_designated_proxy
        self.enforce_designated_root = (
            execution_guard is require_designated_medical_download_host
            if enforce_designated_root is None else enforce_designated_root
        )
        self.stage_locks = stage_locks or StageLockSet()
        # One limiter is shared by every task in this backend process.  The
        # StageLockSet serializes this process; artifact-root FileLocks below
        # extend the ownership boundary across a second backend process during
        # restart/recovery without letting both consume the proxy lane.
        self.rate_limiter = DownloadRateLimiter(DEFAULT_MAX_BYTES_PER_SECOND)
        self._active: set[int] = set()
        self._abort_events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()

    def start_download(self, *, download_limit: int, allow_cf_bypass: bool, wait: bool = False) -> dict:
        if download_limit < 0:
            raise ValueError("download_limit must be >= 0")
        # This lock covers the only unsafe gap: checking for a running durable
        # task, allocating its row, and handing it to the worker must be one
        # root-wide transaction even when two backend processes receive POSTs.
        start_lease_path = self.data_root / "manifest" / "medical-download-start.lock"
        start_lease_path.parent.mkdir(parents=True, exist_ok=True)
        start_lease = FileLock(str(start_lease_path))
        with start_lease.acquire(timeout=10):
            for task in self.store.list_running(MEDICAL_DOWNLOAD_PIPELINE):
                return {"status": "conflict", "task_id": task["id"], "error": "a medical download task is already running"}
            free_bytes = self._preflight_execution()
            # Read and construct *every* JSONL record before allocating a
            # running Task.  The former create -> running -> target-set order
            # left a permanent conflict row if malformed JSONL raised here,
            # before any worker had been launched to mark it failed.
            manifest_records = self._validated_manifest_for_start()
            task_id = self.store.create_task(
                task_type=MEDICAL_DOWNLOAD_PIPELINE,
                params={
                    "data_root": str(self.data_root),
                    "download_limit": download_limit,
                    "allow_cf_bypass": allow_cf_bypass,
                    "proxy": DEFAULT_PROXY_URL,
                    "max_bytes_per_second": DEFAULT_MAX_BYTES_PER_SECOND,
                    # Persist the floor with the task.  A restart must not
                    # silently reinterpret an in-flight batch under a later
                    # process environment/default.
                    "min_free_bytes": DEFAULT_MIN_FREE_BYTES,
                },
                total=1,
            )
            # Make the durable conflict truth visible before releasing the
            # root-wide start lease.  The orchestrator treats running as
            # idempotent and will not create a second task on another process.
            self.store.set_task_status(task_id, "running")
            self.store.update_state(
                task_id,
                disk_free_bytes=free_bytes,
                reserved_bytes=DEFAULT_MIN_FREE_BYTES,
                disk_guard_reason=None,
            )
            with self._lock:
                self._abort_events[task_id] = threading.Event()
            _persist_target_set(self._context_for(task_id), records=manifest_records)
            if not wait:
                self._launch(task_id, wait=False)
        if wait:
            self._launch(task_id, wait=True)
        return {"status": "started", "task_id": task_id}

    def resume_running(self, *, wait: bool = False) -> list[int]:
        resumed: list[int] = []
        for task in self.store.list_running(MEDICAL_DOWNLOAD_PIPELINE):
            with self._lock:
                if task["id"] in self._active:
                    continue
            try:
                task_root = require_absolute_medical_data_root(
                    Path((task.get("params") or {})["data_root"])
                )
                if task_root != self.data_root:
                    raise RuntimeError(
                        f"data-root mismatch persisted={task_root} configured={self.data_root}"
                    )
            except Exception as exc:
                reason = f"resume preflight failed: {type(exc).__name__}: {exc}"
                self.store.set_task_status(task["id"], "failed", error=reason)
                self.store.update_state(task["id"], resume_preflight_error=reason)
                continue
            # A process restart invalidates every process-local assumption.  Do
            # not trust the earlier start-time check: the resumed executor must
            # prove its host, fixed proxy and artifact mount capacity again.
            try:
                min_free_bytes = self._task_min_free_bytes(task.get("params") or {})
                free_bytes = self._preflight_execution(
                    data_root=task_root,
                    min_free_bytes=min_free_bytes,
                )
            except Exception as exc:
                reason = f"resume preflight failed: {type(exc).__name__}: {exc}"
                self.store.set_task_status(task["id"], "failed", error=reason)
                self.store.update_state(task["id"], resume_preflight_error=reason)
                continue
            self.store.update_state(
                task["id"], resumed_after_restart=True,
                disk_free_bytes=free_bytes,
                reserved_bytes=min_free_bytes,
                disk_guard_reason=None,
            )
            self._launch(task["id"], wait=wait)
            resumed.append(task["id"])
        return resumed

    # --- TaskRunner (runner_registry) entry points -------------------------
    def start(self, params: dict) -> dict:
        return self.start_download(
            download_limit=int(params.get("download_limit") or 0),
            allow_cf_bypass=bool(params.get("allow_cf_bypass", True)),
        )

    def cancel(self, task_id: int) -> bool:
        with self._lock:
            abort = self._abort_events.get(task_id)
        return cancel_run(self.store, task_id, abort)

    def retry(self, task_id: int) -> dict:
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"task {task_id} not found")
        if task["status"] not in ("failed", "cancelled"):
            raise ValueError(f"Cannot retry task in status: {task['status']}")
        if task["retry_count"] >= task["max_retries"]:
            raise ValueError(f"Task has exceeded max retries ({task['max_retries']})")
        if self.retry_download(task_id):
            return {"status": "started", "task_id": task_id}
        return {"status": "conflict", "task_id": task_id,
                "error": "Medical download retry could not be scheduled"}

    def cancel_download(self, task_id: int) -> bool:
        """Signal the running stage; the generic Tasks route remains the API."""
        with self._lock:
            event = self._abort_events.get(task_id)
        if event is None:
            return False
        event.set()
        return True

    def retry_download(self, task_id: int) -> bool:
        task = self.store.get_task(task_id)
        if task is None:
            return False
        params = task.get("params") or {}
        task_root = require_absolute_medical_data_root(
            Path(params.get("data_root", self.data_root))
        )
        # Retry is a fresh execution, not permission to redirect an existing
        # durable task onto whatever artifact mount this process happens to
        # have.  Keep exactly the same fence as startup recovery *before* a
        # TaskStore reset can make a worker write the old root.
        if task_root != self.data_root:
            raise RuntimeError(
                f"data-root mismatch persisted={task_root} configured={self.data_root}"
            )
        self._preflight_execution(
            data_root=task_root,
            min_free_bytes=self._task_min_free_bytes(params),
        )
        with self._lock:
            if task_id in self._active:
                return False
        if not self.store.reset_pipeline_for_retry(task_id):
            return False
        with self._lock:
            self._abort_events[task_id] = threading.Event()
        self._launch(task_id, wait=False)
        return True

    def _validate_proxy(self) -> None:
        if not self.enforce_designated_proxy:
            return
        if self.proxy.http_proxy != DEFAULT_PROXY_URL or self.proxy.https_proxy != DEFAULT_PROXY_URL:
            raise RuntimeError(
                f"medical PDF jobs require compute .88 proxy {DEFAULT_PROXY_URL}; got {self.proxy.http_proxy!r}"
            )

    def _validated_manifest_for_start(self):
        """Fully parse the immutable admission snapshot before creating a Task.

        JSONL manifest replacement is atomic, so this either returns one
        complete snapshot used to calculate the initial target set or fails
        without writing a durable running/conflict row.  A concurrent later
        replacement is handled by the worker's ordinary pipeline failure path;
        it cannot recreate this pre-launch orphan state.
        """
        path = PrepareConfig(data_root=self.data_root).manifest_path
        try:
            records = _load_jsonl(path)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"medical manifest validation failed at {path}: {type(exc).__name__}: {exc}"
            ) from exc
        if not records:
            raise RuntimeError(f"medical manifest validation failed at {path}: no records")
        return records

    def _preflight_execution(
        self,
        *,
        data_root: Path | None = None,
        min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    ) -> int:
        """Recheck all non-persistent execution prerequisites immediately before launch."""
        self.execution_guard()
        root = require_absolute_medical_data_root(data_root or self.data_root)
        if self.enforce_designated_root:
            if root != MEDICAL_PRODUCTION_DATA_ROOT:
                raise RuntimeError(
                    "medical production executor requires MEDICAL_DATA_ROOT=/mnt/lsb/retrival/medical"
                )
            resolved = {address[4][0] for address in socket.getaddrinfo(socket.gethostname(), None)}
            if "10.100.100.88" not in resolved:
                raise RuntimeError("medical production executor host does not resolve to 10.100.100.88")
        self._validate_proxy()
        return self._validate_disk_space(data_root=root, min_free_bytes=min_free_bytes)

    def _validate_disk_space(
        self,
        *,
        data_root: Path | None = None,
        min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    ) -> int:
        floor = min_free_bytes
        root = require_absolute_medical_data_root(data_root or self.data_root)
        try:
            free_bytes = shutil.disk_usage(root).free
        except OSError as exc:
            raise RuntimeError(f"medical data mount is unavailable: {root}: {exc}") from exc
        if free_bytes < floor:
            raise RuntimeError(
                f"medical data mount free={free_bytes} below reserved floor={floor}; refusing to start"
            )
        return free_bytes

    @staticmethod
    def _task_min_free_bytes(params: dict) -> int:
        """Read a durable floor once, rejecting corrupted task metadata safely."""
        try:
            floor = int(params.get("min_free_bytes", DEFAULT_MIN_FREE_BYTES))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("medical task min_free_bytes must be an integer") from exc
        if floor < 0:
            raise RuntimeError("medical task min_free_bytes must be >= 0")
        return floor

    def _context_for(self, task_id: int) -> MedicalDownloadContext:
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"task {task_id} not found")
        params = task["params"]
        return MedicalDownloadContext(
            task_id=task_id,
            store=self.store,
            data_root=Path(params["data_root"]),
            proxy=self.proxy,
            download_limit=int(params.get("download_limit") or 0),
            allow_cf_bypass=bool(params.get("allow_cf_bypass", True)),
            min_free_bytes=self._task_min_free_bytes(params),
            rate_limiter=self.rate_limiter,
            abort_event=self._abort_events.setdefault(task_id, threading.Event()),
        )

    def _launch(self, task_id: int, *, wait: bool) -> None:
        # Register before starting the thread so a Tasks API cancellation sent
        # immediately after the 202 receipt cannot race past the worker.
        with self._lock:
            self._abort_events.setdefault(task_id, threading.Event())

        def target() -> None:
            with self._lock:
                self._active.add(task_id)
                self._abort_events.setdefault(task_id, threading.Event())
            try:
                ctx = self._context_for(task_id)
                task_lease_path = ctx.config.manifest_path.parent / f"medical-download-task-{task_id}.lock"
                task_lease_path.parent.mkdir(parents=True, exist_ok=True)
                task_lease = FileLock(str(task_lease_path))
                try:
                    with task_lease.acquire(timeout=0):
                        PipelineOrchestrator(
                            task_id,
                            build_medical_download_pipeline(ctx),
                            ctx,
                            self.store,
                            abort=ctx.abort_event,
                            locks=self.stage_locks,
                        ).run()
                except Timeout:
                    # A concurrent backend already owns the exact durable task.
                    # It will keep TaskStore updated; this process must stand
                    # down rather than turning a duplicate recovery into failure.
                    self.store.emit_event(
                        task_id, None, "medical task already owned by another executor; standing down",
                        level="warning", event_type="pipeline.resume.standby",
                    )
            finally:
                with self._lock:
                    self._active.discard(task_id)
                    self._abort_events.pop(task_id, None)

        if wait:
            target()
        else:
            threading.Thread(target=target, daemon=True, name=f"medical-download-{task_id}").start()


_RUNNER: MedicalDownloadPipelineRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_medical_download_runner() -> MedicalDownloadPipelineRunner:
    global _RUNNER
    if _RUNNER is None:
        with _RUNNER_LOCK:
            if _RUNNER is None:
                _RUNNER = MedicalDownloadPipelineRunner()
    return _RUNNER


register_runner(MEDICAL_DOWNLOAD_PIPELINE, get_medical_download_runner)
