"""PipelineRunner — create, launch, and resume ingest pipelines.

`start_ingest()` creates a parent `ingest_pipeline` Task and drives the generic
orchestrator over the registry's stage list in a daemon thread. `resume_running()`
re-attaches pipelines left `running` by a prior (killed) process — W61 resilience;
the orchestrator's resume-skip makes already-completed stages no-ops.

Everything is injectable (store / pipeline_builder / context_factory) so the
runner can be unit-tested offline with fake stages. The module exposes a process
singleton via `get_pipeline_runner()` so the API route and the startup resume
hook share one in-process active-set (no double launch).
"""
from __future__ import annotations

import threading
from typing import Callable, List, Optional

from loguru import logger

from app.services.pipeline.context import StageContext
from app.services.pipeline.orchestrator import PipelineOrchestrator
from app.services.pipeline.registry import STAGE_KEYS, build_ingest_pipeline
from app.services.pipeline.stage_lock import StageLockSet
from app.services.pipeline.store import TaskStore

INGEST_PIPELINE = "ingest_pipeline"


class PipelineRunner:
    def __init__(
        self,
        *,
        store: Optional[TaskStore] = None,
        pipeline_builder: Callable[[StageContext], list] = build_ingest_pipeline,
        context_factory: Optional[Callable[..., StageContext]] = None,
        stage_locks: Optional[StageLockSet] = None,
    ):
        self.store = store or TaskStore()
        self._build = pipeline_builder
        self._ctx_factory = context_factory or self._default_ctx
        self._active: set[int] = set()
        self._lock = threading.Lock()
        # Process-wide lock set shared by every PipelineOrchestrator launched
        # from this runner. This is the mechanism that lets multiple concurrent
        # ingest_pipeline tasks pipeline through stages without double-booking
        # the underlying resource. See app/services/pipeline/stage_lock.py.
        self.stage_locks = stage_locks or StageLockSet()

    @staticmethod
    def _default_ctx(
        *, venue, year, store, task_id, concurrency, world_size,
        ssh_target=None, remote_repo=None, ranks_per_gpu=None,
        execution_mode=None,
    ) -> StageContext:
        # StageContext fields ssh_target/remote_repo/ranks_per_gpu/execution_mode
        # have safe defaults; we only override when the caller passed something
        # explicit so existing tests that pre-date these kwargs still see the
        # same defaults.
        kw = dict(
            venue=venue, year=year, store=store, task_id=task_id,
            concurrency=concurrency, world_size=world_size,
        )
        if ssh_target is not None:
            kw["ssh_target"] = ssh_target
        if remote_repo is not None:
            kw["remote_repo"] = remote_repo
        if ranks_per_gpu is not None:
            kw["ranks_per_gpu"] = ranks_per_gpu
        if execution_mode is not None:
            kw["execution_mode"] = execution_mode
        return StageContext(**kw)

    def start_ingest(
        self, venue: str, year: int, *,
        ssh_target: Optional[str] = None,
        remote_repo: Optional[str] = None,
        concurrency: int = 8, world_size: int = 8,
        ranks_per_gpu: Optional[int] = None,
        execution_mode: Optional[str] = None,
        start_stage: Optional[str] = None,
        chunk_id_before: Optional[int] = None,
        shard_start: Optional[int] = None,
        shard_end: Optional[int] = None,
        wait: bool = False,
    ) -> dict:
        """Start a pipeline for one venue+year on a chosen remote target.

        Multiple ingest_pipeline tasks may run concurrently — they serialize
        per-stage on the shared StageLockSet (one task at a time per resource
        key: gpu-{target}, sql, qdrant, net), so safety is enforced at the
        stage level instead of by a global one-pipeline-at-a-time gate.

        The ONLY guard at submission time is: don't start a SECOND pipeline
        for the EXACT same (venue, year) — that would just queue useless
        duplicate work. A second POST for an in-flight (venue, year) returns
        a conflict pointing at the existing task.

        `ssh_target` controls WHERE extract/embed run. When None we keep the
        StageContext default ('kss'); the HTTP API requires it to be set so
        callers can't trip over the default and collide with someone else's
        job on a shared box.
        """
        if start_stage is not None and start_stage not in STAGE_KEYS:
            raise ValueError(
                f"start_stage must be one of {STAGE_KEYS}, got {start_stage!r}"
            )
        self._validate_external_stage_state(
            start_stage=start_stage,
            chunk_id_before=chunk_id_before,
            shard_start=shard_start,
            shard_end=shard_end,
        )

        for r in self.store.list_running(INGEST_PIPELINE):
            params_r = r.get("params") or {}
            if params_r.get("venue") == venue and params_r.get("year") == year:
                logger.warning(
                    f"pipeline.start.duplicate existing={r['id']} "
                    f"venue={venue} year={year}"
                )
                return {
                    "status": "conflict",
                    "task_id": r["id"],
                    "error": f"an ingest pipeline (#{r['id']}) is already running "
                             f"for {venue} {year}",
                }

        params: dict = {
            "venue": venue, "year": year,
            "concurrency": concurrency, "world_size": world_size,
        }
        if ssh_target is not None:
            params["ssh_target"] = ssh_target
        if remote_repo is not None:
            params["remote_repo"] = remote_repo
        if ranks_per_gpu is not None:
            params["ranks_per_gpu"] = ranks_per_gpu
        if execution_mode is not None:
            params["execution_mode"] = execution_mode
        if start_stage is not None:
            params["start_stage"] = start_stage
        if chunk_id_before is not None:
            params["chunk_id_before"] = chunk_id_before
        if shard_start is not None:
            params["shard_start"] = shard_start
        if shard_end is not None:
            params["shard_end"] = shard_end

        task_id = self.store.create_task(
            task_type=INGEST_PIPELINE, params=params, total=len(STAGE_KEYS),
        )
        logger.info(
            f"pipeline.start task={task_id} venue={venue} year={year} "
            f"ssh_target={ssh_target!r} remote_repo={remote_repo!r} "
            f"ranks_per_gpu={ranks_per_gpu!r} start_stage={start_stage!r}"
        )
        self._launch(
            task_id, venue, year, concurrency, world_size,
            ssh_target=ssh_target, remote_repo=remote_repo,
            ranks_per_gpu=ranks_per_gpu, execution_mode=execution_mode,
            start_stage=start_stage,
            chunk_id_before=chunk_id_before,
            shard_start=shard_start,
            shard_end=shard_end,
            wait=wait,
        )
        return {"status": "started", "task_id": task_id}

    def resume_running(self, *, wait: bool = False) -> List[int]:
        """Relaunch orchestrators for running ingest pipelines (startup hook).

        Ordering: tasks with an in-flight subtask (any `started_at` set on a
        non-completed stage) launch FIRST. They likely already hold a remote
        worker (marker tmux, embed parquet) — letting them grab their resource
        lock first prevents fresh peers from racing in, running preflight, and
        failing on a GPU the in-flight task is using. W78.
        """
        resumed: List[int] = []
        try:
            running = self.store.list_running(INGEST_PIPELINE)
        except Exception:
            logger.exception("pipeline.resume.query_failed")
            return resumed

        def _has_inflight(t: dict) -> bool:
            for key in STAGE_KEYS:
                try:
                    s = self.store.get_subtask(t["id"], key)
                except Exception:
                    s = None
                if s and s.get("started_at") and s.get("status") not in ("completed", "failed", "cancelled"):
                    return True
            return False

        running.sort(key=lambda t: (0 if _has_inflight(t) else 1, t["id"]))

        # Hand-off pause: launch in-flight tasks first, sleep briefly so
        # their orchestrator threads reach lock.acquire() before fresh peers
        # are spawned. Without this, even sorted threads race on acquire()
        # and a fresh peer can win, run preflight against the GPU the
        # in-flight task is using, and fail. W78 race repro.
        import time as _time

        for t in running:
            tid = t["id"]
            with self._lock:
                if tid in self._active:  # already being driven in this process
                    continue
            params = t.get("params") or {}
            venue, year = params.get("venue"), params.get("year")
            if not venue or year is None:
                logger.warning(f"pipeline.resume.skip task={tid} missing venue/year")
                continue
            logger.info(f"pipeline.resume task={tid} venue={venue} year={year}")
            self._launch(
                tid, venue, year,
                params.get("concurrency", 8), params.get("world_size", 8),
                ssh_target=params.get("ssh_target"),
                remote_repo=params.get("remote_repo"),
                ranks_per_gpu=params.get("ranks_per_gpu"),
                execution_mode=params.get("execution_mode"),
                start_stage=params.get("start_stage"),
                chunk_id_before=params.get("chunk_id_before"),
                shard_start=params.get("shard_start"),
                shard_end=params.get("shard_end"),
                wait=wait,
            )
            resumed.append(tid)
            # Brief pause after launching an in-flight task so its
            # orchestrator thread reaches lock.acquire() before the next
            # peer is spawned. No effect in tests (wait=True is synchronous).
            if not wait and _has_inflight(t):
                _time.sleep(0.3)
        return resumed

    @staticmethod
    def _validate_external_stage_state(
        *,
        start_stage: Optional[str],
        chunk_id_before: Optional[int],
        shard_start: Optional[int],
        shard_end: Optional[int],
    ) -> None:
        if start_stage == "embed":
            if chunk_id_before is None or chunk_id_before < 0:
                raise ValueError(
                    "chunk_id_before >= 0 is required when start_stage='embed'"
                )
            if shard_start is not None or shard_end is not None:
                raise ValueError(
                    "shard_start/shard_end are only valid when start_stage='ingest'"
                )
            return
        if start_stage == "ingest":
            if shard_start is None or shard_end is None:
                raise ValueError(
                    "shard_start and shard_end are required when start_stage='ingest'"
                )
            if shard_start < 0 or shard_end <= shard_start:
                raise ValueError(
                    "ingest shard range must satisfy 0 <= shard_start < shard_end"
                )
            if chunk_id_before is not None:
                raise ValueError(
                    "chunk_id_before is only valid when start_stage='embed'"
                )
            return
        if any(value is not None for value in (
            chunk_id_before,
            shard_start,
            shard_end,
        )):
            raise ValueError(
                "external cursors are only valid for start_stage='embed' or 'ingest'"
            )

    def _seed_external_stage_state(
        self,
        ctx: StageContext,
        *,
        start_stage: Optional[str],
        chunk_id_before: Optional[int],
        shard_start: Optional[int],
        shard_end: Optional[int],
    ) -> None:
        self._validate_external_stage_state(
            start_stage=start_stage,
            chunk_id_before=chunk_id_before,
            shard_start=shard_start,
            shard_end=shard_end,
        )
        if start_stage == "embed":
            ctx.state_set(chunk_id_before=chunk_id_before)
        elif start_stage == "ingest":
            ctx.state_set(shard_start=shard_start, shard_end=shard_end)

    def _complete_external_stage_prefix(
        self,
        task_id: int,
        specs: list,
        start_stage: Optional[str],
    ) -> None:
        """Mark stages before ``start_stage`` as externally supplied.

        The task keeps the canonical five-stage shape consumed by the existing
        Tasks UI and startup resume path. Only durable subtask rows are changed;
        the orchestrator still drives every stage and skips these completed
        prefixes through its normal resume rule.
        """
        if start_stage is None:
            return
        keys = [spec.key for spec in specs]
        if start_stage not in keys:
            raise ValueError(
                f"start_stage {start_stage!r} is absent from pipeline stages {keys}"
            )
        for order, spec in enumerate(specs[: keys.index(start_stage)]):
            self.store.ensure_subtask(
                task_id,
                spec.key,
                subtask_type="pipeline_stage",
                name=spec.title,
                order=order,
                total=1,
            )
            current = self.store.get_subtask(task_id, spec.key)
            if current is not None and current.get("status") == "completed":
                continue
            message = (
                f"externally supplied before pipeline start_stage={start_stage}"
            )
            self.store.update_progress(task_id, spec.key, 1, 1, message)
            self.store.complete_subtask(
                task_id,
                spec.key,
                message=message,
                result={"source": "external", "start_stage": start_stage},
            )
            self.store.emit_event(
                task_id,
                spec.key,
                message,
                event_type="stage.external_completed",
            )

    def _launch(
        self, task_id: int, venue: str, year: int,
        concurrency: int, world_size: int, *,
        ssh_target: Optional[str] = None,
        remote_repo: Optional[str] = None,
        ranks_per_gpu: Optional[int] = None,
        execution_mode: Optional[str] = None,
        start_stage: Optional[str] = None,
        chunk_id_before: Optional[int] = None,
        shard_start: Optional[int] = None,
        shard_end: Optional[int] = None,
        wait: bool = False,
    ) -> None:
        def target():
            with self._lock:
                self._active.add(task_id)
            try:
                ctx = self._ctx_factory(
                    venue=venue, year=year, store=self.store, task_id=task_id,
                    concurrency=concurrency, world_size=world_size,
                    ssh_target=ssh_target, remote_repo=remote_repo,
                    ranks_per_gpu=ranks_per_gpu, execution_mode=execution_mode,
                )
                self._seed_external_stage_state(
                    ctx,
                    start_stage=start_stage,
                    chunk_id_before=chunk_id_before,
                    shard_start=shard_start,
                    shard_end=shard_end,
                )
                specs = self._build(ctx)
                self._complete_external_stage_prefix(
                    task_id, specs, start_stage,
                )
                PipelineOrchestrator(
                    task_id, specs, ctx, self.store,
                    locks=self.stage_locks,
                ).run()
            except Exception as e:
                logger.exception(f"pipeline.crash task={task_id}: {e}")
                try:
                    self.store.set_task_status(task_id, "failed", error=f"orchestrator crash: {e}")
                except Exception:
                    logger.exception("pipeline.crash.status_write_failed")
            finally:
                with self._lock:
                    self._active.discard(task_id)

        if wait:
            target()
        else:
            threading.Thread(target=target, daemon=True, name=f"pipeline-{task_id}").start()


_runner: Optional[PipelineRunner] = None
_runner_lock = threading.Lock()


def get_pipeline_runner() -> PipelineRunner:
    """Process-wide singleton shared by the API route and the startup resume hook."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = PipelineRunner()
    return _runner
