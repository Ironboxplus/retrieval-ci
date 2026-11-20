"""PipelineRunner — create, launch, and resume ingest pipelines.

`start_ingest()` creates a parent `ingest_pipeline` Task and drives the generic
orchestrator over the registry's stage list in a daemon thread. `resume_running()`
re-attaches pipelines left `running` by a prior (killed) process — W61 resilience;
the orchestrator's resume-skip makes already-completed stages no-ops.

`cancel()` (CancellableRunner) flips the task to `cancelled` atomically and sets
the run's abort Event, which the orchestrator hands to every stage context so
run_local() kills the stage's subprocess. The runner registers itself in
runner_registry, which is how the Tasks API, the scheduler and the startup hook
reach it without naming this class.

Everything is injectable (store / pipeline_builder / context_factory) so the
runner can be unit-tested offline with fake stages. The module exposes a process
singleton via `get_pipeline_runner()` so the API route and the startup resume
hook share one in-process active-set (no double launch).
"""
from __future__ import annotations

import threading
from typing import Callable, List, Optional

from loguru import logger

from app.services.pipeline.arxiv_stages import (
    ARXIV_VENUE,
    ArxivStageContext,
    is_arxiv_venue,
    maybe_arxiv_context,
    validate_slice_key,
    validate_source_option,
)
from pipeline.paper_crawler.arxiv_corpus import DELTA_KEY
from app.services.pipeline.context import StageContext
from app.services.pipeline.download_batch import (
    DOWNLOAD_BATCH_VENUE,
    is_download_batch_venue,
    maybe_download_batch_context,
    parse_download_targets,
)
from app.services.pipeline.orchestrator import PipelineOrchestrator
from app.services.pipeline.registry import STAGE_KEYS, build_ingest_pipeline
from app.services.pipeline.runner_registry import CancellableRunner, register_runner
from app.services.pipeline.stage_lock import StageLockSet
from app.services.pipeline.stages import HOST_NEVER_DOWNLOADS, validate_receive_options
from app.services.pipeline.store import TaskStore

INGEST_PIPELINE = "ingest_pipeline"

# Stages whose output is a durable corpus artifact (PDFs, Markdown, SQL rows) that
# exists before the task does, so an external start can be checked against it.
# embed and ingest keep theirs in the task's own vectors directory.
_EXTERNAL_VERIFIABLE_STAGES = ("download", "extract", "preindex")


class PipelineRunner(CancellableRunner):
    task_type = INGEST_PIPELINE

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
        self._aborts: dict[int, threading.Event] = {}
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
        execution_mode=None, options=None,
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
        if options:
            kw["options"] = options
        arxiv_ctx = maybe_arxiv_context(**kw)
        if arxiv_ctx is not None:
            return arxiv_ctx
        batch_ctx = maybe_download_batch_context(**kw)
        if batch_ctx is not None:
            return batch_ctx
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
        options: Optional[dict] = None,
        origin: Optional[dict] = None,
        wait: bool = False,
    ) -> dict:
        """Start a pipeline for one venue+year on a chosen remote target.

        `options` are per-run parameters handed to the stage context
        (StageContext.options); `origin` records who started the task
        ({schedule_id, trigger, scheduled_for, attempt}). Both are stored in the
        task's params, so a restart or retry resumes with the same values.

        Multiple ingest_pipeline tasks may run concurrently — they serialize
        per-stage on the shared StageLockSet (one task at a time per resource
        key: gpu-{target}, sql, qdrant, net), so safety is enforced at the
        stage level instead of by a global one-pipeline-at-a-time gate.

        The ONLY guard at submission time is: don't start a SECOND pipeline
        for the EXACT same (venue, year) — that would just queue useless
        duplicate work. A second POST for an in-flight (venue, year) returns
        a conflict pointing at the existing task.

        `execution_mode` controls WHERE extract/embed run; None means the
        StageContext default, "local". `ssh_target` names the box only for the
        "host" mode (a registry host: embed, and extract when it has a marker
        block) and the retired "remote" topology.
        """
        if is_arxiv_venue(venue):
            venue = ARXIV_VENUE  # one spelling, so the duplicate-run guard sees it
            validate_slice_key(year)
            validate_source_option(options)  # this host does not download
        elif is_download_batch_venue(venue):
            venue = DOWNLOAD_BATCH_VENUE
            if year < 1:
                raise ValueError(f"download batch number must be >= 1, got {year}")
            parse_download_targets((options or {}).get("targets"))  # before a task exists
            if start_stage not in (None, "download"):
                raise ValueError("a download batch has only the download stage")
            # The batch code stays for a host that downloads; this one only receives.
            raise ValueError(f"{HOST_NEVER_DOWNLOADS} (a Papers-Download batch is not created here)")
        conference = not (is_arxiv_venue(venue) or is_download_batch_venue(venue))
        validate_receive_options(options, execution_mode, conference=conference)
        if execution_mode == "host":
            from app.services.pipeline.stages import load_embed_host, receive_mode
            host = load_embed_host(ssh_target or "")  # unknown host -> ValueError before a task exists
            if conference and host.marker is None and not receive_mode(options, "extract_from"):
                raise ValueError(
                    f"host {host.name!r} has no 'marker' block in the registry, so it cannot run "
                    "extract; add one, or receive the Markdown with options.extract_from='push'")
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
        if start_stage is not None:
            # A placeholder context: probes only read the corpus and SQL, so the
            # start can be refused before a task row exists.
            probe_ctx = self._ctx_factory(
                venue=venue, year=year, store=None, task_id=0,
                concurrency=concurrency, world_size=world_size,
                ssh_target=ssh_target, remote_repo=remote_repo,
                ranks_per_gpu=ranks_per_gpu, execution_mode=execution_mode,
                options=options,
            )
            self._verify_external_prefix(probe_ctx, self._build(probe_ctx), start_stage)

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
        if options:
            params["options"] = options
        if origin:
            params["origin"] = origin

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
            options=options,
            wait=wait,
        )
        return {"status": "started", "task_id": task_id}

    def start(self, params: dict) -> dict:
        """TaskRunner entry: `params` = {venue, year, options?, origin?, ...start_ingest kwargs}."""
        kwargs = dict(params)
        venue, year = kwargs.pop("venue"), kwargs.pop("year")
        return self.start_ingest(venue, year, **kwargs)

    def retry(self, task_id: int) -> dict:
        return self.retry_failed(task_id)

    def start_arxiv_batches(self, months: List[int], **ingest_kw) -> dict:
        """One ingest_pipeline task per arXiv batch of the given months.

        All tasks start together; StageLockSet streams them through the
        stages (batch N embeds while N+1 converts and N+2 downloads). Keys come
        from what this host knows (arxiv_stages.discover_batch_keys), so run it
        after the downloading host has pushed its first progress file. Running
        it again later only adds new batches: batches with a completed ingest
        are skipped, in-flight ones answer conflict (the per-key guard)."""
        from app.services.pipeline.arxiv_stages import discover_batch_keys
        from pipeline.paper_crawler.arxiv_corpus import harvest_root

        done = {t["params"].get("year") for t in self.store.list_completed(INGEST_PIPELINE)
                if t["params"].get("venue") == ARXIV_VENUE}
        keys = discover_batch_keys(harvest_root(), months)
        skipped = [k for k in keys if k in done]
        keys = [k for k in keys if k not in done]
        started, conflicts = [], []
        for key in keys:
            result = self.start_ingest(ARXIV_VENUE, key, **ingest_kw)
            (started if result.get("status") == "started" else conflicts).append(
                {"year": key, "task_id": result.get("task_id")})
        logger.info(f"pipeline.arxiv months={months} batches={len(keys)} skipped={len(skipped)} "
                    f"started={len(started)} conflicts={len(conflicts)}")
        return {"batches": len(keys), "started": started, "conflict": conflicts,
                "skipped_completed": skipped}

    def start_arxiv_delta(self, options: Optional[dict] = None, origin: Optional[dict] = None,
                          **ingest_kw) -> dict:
        """The daily arXiv update: one task for DELTA_KEY (the 2026+ papers that are new
        or changed against SQL). At most one runs at a time: the per-(venue, year)
        guard of start_ingest answers `conflict` with the running task."""
        return self.start_ingest(ARXIV_VENUE, DELTA_KEY, options=options, origin=origin, **ingest_kw)

    def start_download_batch(
        self, targets: List[str], *,
        min_interval_seconds: Optional[float] = None,
        host_cooldown_seconds: Optional[float] = None,
        collect: bool = False,
        wait: bool = False,
    ) -> dict:
        """One task that downloads the PDFs of several manifest slices as a single
        host-alternating queue (see download_batch). The batch number is the next
        free one, so each call is its own task; the slices are ingested later by
        their own (venue, year) pipelines. `collect` first collects the slices that
        have no manifest rows on this host."""
        options: dict = {"targets": [f"{c}:{y}" for c, y in parse_download_targets(targets)]}
        if min_interval_seconds is not None:
            options["min_interval_seconds"] = float(min_interval_seconds)
        if host_cooldown_seconds is not None:
            options["host_cooldown_seconds"] = float(host_cooldown_seconds)
        if collect:
            options["collect"] = True
        used = [
            t["params"].get("year") or 0
            for t in self.store.list_running(INGEST_PIPELINE) + self.store.list_completed(INGEST_PIPELINE)
            if t["params"].get("venue") == DOWNLOAD_BATCH_VENUE
        ]
        return self.start_ingest(DOWNLOAD_BATCH_VENUE, max(used, default=0) + 1, options=options, wait=wait)

    def retry_failed(self, task_id: int, *, wait: bool = False) -> dict:
        """Drive a failed or cancelled ingest pipeline again, in place.

        The task keeps its id and per-task state (chunk range, shard anchor);
        stages whose probe is already satisfied skip, so work resumes at the
        stage that stopped. Same per-(venue, year) guard as start_ingest."""
        task = self.store.get_task(task_id)
        if task is None or task.get("task_type") != INGEST_PIPELINE:
            raise ValueError(f"task {task_id} is not an {INGEST_PIPELINE} task")
        if task.get("status") not in ("failed", "cancelled"):
            raise ValueError(f"task {task_id} is {task.get('status')}, only failed/cancelled can be retried")
        # reset_pipeline_for_retry below counts the attempt; this is its ceiling.
        if task["retry_count"] >= task["max_retries"]:
            raise ValueError(f"task {task_id} has exceeded max retries ({task['max_retries']})")
        with self._lock:
            if task_id in self._active:
                raise ValueError(f"task {task_id} is still being driven in this process")
        params = task.get("params") or {}
        venue, year = params.get("venue"), params.get("year")
        for r in self.store.list_running(INGEST_PIPELINE):
            p = r.get("params") or {}
            if p.get("venue") == venue and p.get("year") == year:
                return {"status": "conflict", "task_id": r["id"],
                        "error": f"an ingest pipeline (#{r['id']}) is already running for {venue} {year}"}
        self.store.reset_pipeline_for_retry(task_id)
        logger.info(f"pipeline.retry task={task_id} venue={venue} year={year}")
        self._launch(
            task_id, venue, year,
            params.get("concurrency", 8), params.get("world_size", 8),
            ssh_target=params.get("ssh_target"),
            remote_repo=params.get("remote_repo"),
            ranks_per_gpu=params.get("ranks_per_gpu"),
            execution_mode=params.get("execution_mode"),
            start_stage=params.get("start_stage"),
            chunk_id_before=params.get("chunk_id_before"),
            shard_start=params.get("shard_start"),
            shard_end=params.get("shard_end"),
            options=params.get("options"),
            wait=wait,
        )
        return {"status": "started", "task_id": task_id}

    def resume_running(self, *, wait: bool = False) -> List[int]:
        """Relaunch orchestrators for running ingest pipelines (startup hook).

        Tasks launch in id order. W78 (an in-flight task, with a surviving
        remote worker, must reach its resource before fresh peers run preflight
        against the GPU it is using) is handled by the lock itself: the
        orchestrator passes a priority and StageLockSet grants a resuming stage
        first, whatever the launch order.
        """
        resumed: List[int] = []
        try:
            running = self.store.list_running(INGEST_PIPELINE)
        except Exception:
            logger.exception("pipeline.resume.query_failed")
            return resumed

        running.sort(key=lambda t: t["id"])

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
                options=params.get("options"),
                wait=wait,
            )
            resumed.append(tid)
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

    @staticmethod
    def _probes_external_prefix(ctx: StageContext) -> bool:
        """arXiv batches have their own stage semantics; only conference slices are probed."""
        return not isinstance(ctx, ArxivStageContext)

    def _verify_external_prefix(self, ctx: StageContext, specs: list, start_stage: str) -> None:
        """Refuse an external start whose last verifiable earlier stage produced nothing.

        The stage right before ``start_stage`` is the one whose output the new run
        consumes, so its probe must show something. embed and ingest keep their
        output in the task's own directory and cannot be probed before the task
        exists; for those the nearest earlier stage in
        ``_EXTERNAL_VERIFIABLE_STAGES`` is judged instead. Earlier stages may
        legitimately be empty (Markdown shipped in without its PDFs)."""
        keys = [spec.key for spec in specs]
        if start_stage not in keys:
            raise ValueError(
                f"start_stage {start_stage!r} is absent from pipeline stages {keys}"
            )
        if not self._probes_external_prefix(ctx):
            return
        prefix = specs[: keys.index(start_stage)]
        checkable = [s for s in prefix if s.key in _EXTERNAL_VERIFIABLE_STAGES]
        if not checkable:
            return
        spec = checkable[-1]
        progress = spec.probe(ctx)
        if progress.done <= 0:
            raise ValueError(
                f"start_stage={start_stage!r} needs the output of {spec.key} "
                f"({ctx.venue} {ctx.year}) but found {progress.done}/{progress.total}"
                + (f": {progress.message}" if progress.message else "")
            )

    def _complete_external_stage_prefix(
        self,
        task_id: int,
        specs: list,
        start_stage: Optional[str],
        ctx: StageContext,
    ) -> None:
        """Mark stages before ``start_stage`` as externally supplied.

        The task keeps the canonical five-stage shape consumed by the existing
        Tasks UI and startup resume path. Only durable subtask rows are changed;
        the orchestrator still drives every stage and skips these completed
        prefixes through its normal resume rule. Each stage that can be probed
        records the done/total its probe sees; the others record 1/1.

        The last verifiable stage of the prefix is the one whose output
        ``start_stage`` consumes, so it must also pass its own gate: a download
        at 57% is not "supplied", it is a stage that has work left. That stage
        and the prefix specs after it are left pending for the orchestrator to
        run. Earlier stages may legitimately be empty and are not judged.
        """
        if start_stage is None:
            return
        keys = [spec.key for spec in specs]
        if start_stage not in keys:
            raise ValueError(
                f"start_stage {start_stage!r} is absent from pipeline stages {keys}"
            )
        probes = self._probes_external_prefix(ctx)
        prefix = specs[: keys.index(start_stage)]
        supplied_until = len(prefix)
        verifiable = [i for i, s in enumerate(prefix) if s.key in _EXTERNAL_VERIFIABLE_STAGES] if probes else []
        if verifiable:
            index = verifiable[-1]
            judged = prefix[index]
            current = self.store.get_subtask(task_id, judged.key)
            # A resumed task keeps what it already completed; only a stage still open is judged.
            if judged.gate is not None and not (current is not None and current.get("status") == "completed"):
                verdict = judged.gate(ctx)
                if not verdict.ok:
                    supplied_until = index
                    self.store.ensure_subtask(
                        task_id, judged.key, subtask_type="pipeline_stage", name=judged.title, order=index,
                    )
                    self.store.emit_event(
                        task_id,
                        judged.key,
                        f"externally supplied output fails the gate ({verdict.reason}); the stage runs",
                        level="warning",
                        event_type="stage.external_gate_failed",
                    )
        for order, spec in enumerate(prefix[:supplied_until]):
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
            done, total, seen = 1, 1, ""
            if probes and spec.key in _EXTERNAL_VERIFIABLE_STAGES:
                progress = spec.probe(ctx)
                done, total, seen = progress.done, progress.total, f" ({progress.done}/{progress.total})"
            message = (
                f"externally supplied before pipeline start_stage={start_stage}{seen}"
            )
            # complete_subtask lifts progress to total, so the real numbers go in after it.
            self.store.complete_subtask(
                task_id,
                spec.key,
                message=message,
                result={"source": "external", "start_stage": start_stage},
            )
            self.store.update_progress(task_id, spec.key, done, total, message)
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
        options: Optional[dict] = None,
        wait: bool = False,
    ) -> None:
        # Registered before the thread exists, so a cancel racing the launch
        # still finds the Event and the orchestrator starts already aborted.
        abort = self._register_abort(task_id)

        def target():
            with self._lock:
                self._active.add(task_id)
            try:
                ctx = self._ctx_factory(
                    venue=venue, year=year, store=self.store, task_id=task_id,
                    concurrency=concurrency, world_size=world_size,
                    ssh_target=ssh_target, remote_repo=remote_repo,
                    ranks_per_gpu=ranks_per_gpu, execution_mode=execution_mode,
                    options=options,
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
                    task_id, specs, start_stage, ctx,
                )
                PipelineOrchestrator(
                    task_id, specs, ctx, self.store,
                    abort=abort, locks=self.stage_locks,
                ).run()
            except Exception as e:
                logger.exception(f"pipeline.crash task={task_id}: {e}")
                self._fail_crashed(task_id, e)
            finally:
                with self._lock:
                    self._active.discard(task_id)
                self._drop_abort(task_id)

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


register_runner(INGEST_PIPELINE, get_pipeline_runner)
