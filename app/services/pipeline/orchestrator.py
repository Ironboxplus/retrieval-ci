"""PipelineOrchestrator — the generic, stage-agnostic engine.

One loop drives an ordered list of StageSpec callbacks. It contains ZERO
stage-specific branching: it never knows whether a stage downloads PDFs or runs
a remote GPU embed. Per-stage behaviour lives entirely in the callbacks.

Lifecycle per stage:
  1. resume      — if the subtask is already `completed`, skip (crash recovery).
  2. satisfied   — else probe(); if already complete (artifacts present), mark
                   done without running (e.g. CVPR PDFs already on disk).
  3. lock        — acquire spec.resource_key in the shared StageLockSet so
                   concurrent ingest_pipeline tasks serialize per-resource.
                   Multiple tasks pipeline through stages in parallel without
                   double-booking SQL / Qdrant / a remote GPU box.
  4. preflight   — optional safety check BEFORE run (e.g. GPU health). Fail→halt.
  5. run         — blocking; a background poller updates progress meanwhile.
  6. gate        — optional acceptance check AFTER run (e.g. MD sample). Fail→halt.
  7. complete    — mark subtask done; release the lock; advance.

Any halt fails the parent task and stops the pipeline. The lock is released
on every exit path (try/finally inside `locks.hold`).

Every task-status write is a compare-and-set (`TaskStore.compare_and_set_status`),
never a read-then-write: the cancel path (PipelineRunner.cancel, the Tasks API)
can commit `cancelled` at any moment, and a plain write here would turn it back
into `running`/`completed`. The orchestrator only moves any-but-cancelled -> running
and running -> terminal; losing that race means somebody else already decided the
final state, and it is reported as-is. Resumability is built on
DB state + artifact probes, so a backend restart (W61) re-attaches cleanly: a
fresh orchestrator over the same task skips completed stages and resumes — the
in-memory lock is rebuilt fresh, and re-acquired naturally on the resumed run.
"""
from __future__ import annotations

import threading
import time
from typing import Any, List, Optional

from loguru import logger

from app.services.pipeline.spec import GateResult, Progress, StageSpec
from app.services.pipeline.stage_lock import StageLockSet
from app.services.pipeline.store import TaskStore

# A preflight that fails for a condition that clears by itself (GateResult.transient: no GPU has the
# memory free because another process holds it) is a wait, not a failure. The stage keeps its place
# at the head of the resource queue and asks again (pause doubling from the initial to the maximum)
# while the tasks behind it wait, still in grant order; only after the deadline does it fail, with
# the preflight's own message. 2026-11-16: one held card failed 18 queued ingest tasks in a second.
PREFLIGHT_WAIT_INITIAL_S = 60.0
PREFLIGHT_WAIT_MAX_S = 600.0
PREFLIGHT_WAIT_DEADLINE_S = 6 * 3600.0


class _Poller:
    """Background progress sampler: probe() → update_progress every interval."""

    def __init__(self, store: TaskStore, task_id: int, spec: StageSpec, ctx: Any,
                 abort: threading.Event, *, update_parent_from_probe: bool = False):
        self._stop = threading.Event()
        self._store = store
        self._task_id = task_id
        self._spec = spec
        self._ctx = ctx
        self._abort = abort
        self._update_parent_from_probe = update_parent_from_probe
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"poll-{spec.key}")

    def start(self) -> "_Poller":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set() and not self._abort.is_set():
            try:
                pr = self._spec.probe(self._ctx)
                self._store.update_progress(self._task_id, self._spec.key, pr.done, pr.total, pr.message)
                if self._update_parent_from_probe:
                    self._store.set_task_progress(self._task_id, pr.done, pr.total)
                self._store.heartbeat(self._task_id)
            except Exception:  # poller must never crash the stage
                logger.exception(f"stage.poll.error stage={self._spec.key}")
            self._stop.wait(self._spec.poll_interval_s)


class PipelineOrchestrator:
    def __init__(
        self,
        task_id: int,
        specs: List[StageSpec],
        ctx: Any,
        store: TaskStore,
        *,
        abort: Optional[threading.Event] = None,
        locks: Optional[StageLockSet] = None,
    ):
        self.task_id = task_id
        self.specs = list(specs)
        self.ctx = ctx
        self.store = store
        self.abort = abort or threading.Event()
        # Stage callbacks reach the Event through their ctx (run_local(abort=c.abort)),
        # so the ctx must carry the SAME Event the runner sets on cancel. ctx is
        # opaque to the orchestrator; contexts without an `abort` slot opt out.
        if hasattr(ctx, "abort"):
            ctx.abort = self.abort
        # When None, every stage runs free-for-all (no inter-task mutex). Tests
        # that don't care about concurrency get this default; production passes
        # a shared StageLockSet via PipelineRunner so concurrent ingest_pipeline
        # tasks serialize per-stage via spec.resource_key.
        self.locks = locks or StageLockSet()

    def run(self) -> str:
        """Drive all stages. Returns the final task status string."""
        store = self.store
        total = len(self.specs)
        keys = [s.key for s in self.specs]
        logger.info(f"pipeline.start task={self.task_id} stages={keys}")
        parent_probe_spec = next(
            (spec for spec in self.specs if spec.parent_progress_from_probe), None)
        if parent_probe_spec is not None and total != 1:
            raise ValueError("parent_progress_from_probe requires a single-stage pipeline")
        # A generic Tasks cancellation can be committed by another backend
        # process before this orchestrator gets CPU.  Never resurrect it: the
        # claim is one conditional UPDATE, so a cancel landing between a read and
        # a write cannot be lost.
        # Any status but `cancelled` may be (re)driven: pending is a fresh start,
        # running a resume, failed/timeout/completed a re-run of the same task.
        if not store.compare_and_set_status(
            self.task_id, expect={"pending", "running", "completed", "failed", "timeout"}, new="running",
        ):
            store.emit_event(self.task_id, None, "pipeline already cancelled before start", level="warning", event_type="pipeline.cancelled")
            return "cancelled"
        if parent_probe_spec is None:
            store.set_task_progress(self.task_id, 0, total)
        else:
            initial = self._safe_probe(parent_probe_spec)
            store.set_task_progress(self.task_id, initial.done, initial.total)
        store.emit_event(self.task_id, None, f"pipeline started: {' -> '.join(keys)}", event_type="pipeline.started")

        # Ensure every stage subtask exists (idempotent; resume reuses rows).
        for i, spec in enumerate(self.specs):
            store.ensure_subtask(self.task_id, spec.key, subtask_type="pipeline_stage", name=spec.title, order=i)

        completed = 0
        for spec in self.specs:
            if self.abort.is_set():
                self._finish("cancelled")
                store.emit_event(self.task_id, None, "pipeline cancelled", level="warning", event_type="pipeline.cancelled")
                logger.warning(f"pipeline.cancelled task={self.task_id}")
                return "cancelled"

            outcome = self._run_stage(spec)
            if outcome in ("completed", "skipped"):
                completed += 1
                if parent_probe_spec is None:
                    store.set_task_progress(self.task_id, completed, total)
                continue

            if outcome == "cancelled":
                self._finish("cancelled")
                store.emit_event(self.task_id, None, "pipeline cancelled", level="warning", event_type="pipeline.cancelled")
                logger.warning(f"pipeline.cancelled task={self.task_id} stage={spec.key}")
                return "cancelled"

            # halted
            reason = f"stage '{spec.key}' failed"
            final = self._finish("failed", error=reason)
            if final != "failed":
                return final
            store.emit_event(self.task_id, None, reason, level="error", event_type="pipeline.failed")
            logger.error(f"pipeline.failed task={self.task_id} stage={spec.key}")
            return "failed"

        # Cancellation may race in after the final stage returns.  It is a
        # user-visible terminal choice, never something a late completion may
        # overwrite.
        if self.abort.is_set():
            self._finish("cancelled")
            store.emit_event(self.task_id, None, "pipeline cancelled", level="warning", event_type="pipeline.cancelled")
            logger.warning(f"pipeline.cancelled task={self.task_id} after final stage")
            return "cancelled"
        final = self._finish("completed")
        if final != "completed":
            return final
        if parent_probe_spec is None:
            store.set_task_progress(self.task_id, total, total)
        else:
            final = self._safe_probe(parent_probe_spec)
            store.set_task_progress(self.task_id, final.done, final.total)
        store.emit_event(self.task_id, None, "pipeline completed", event_type="pipeline.completed")
        logger.info(f"pipeline.completed task={self.task_id}")
        return "completed"

    def _finish(self, status: str, *, error: Optional[str] = None) -> str:
        """Write the task's terminal status if it is still `running`.

        Returns the status that is now in effect: `status` when this call won,
        otherwise whatever the winner (normally the cancel path) wrote.
        """
        if self.store.compare_and_set_status(self.task_id, expect={"running"}, new=status, error=error):
            return status
        actual = (self.store.get_task(self.task_id) or {}).get("status") or status
        logger.info(f"pipeline.finish.lost task={self.task_id} wanted={status} actual={actual}")
        return actual

    def _run_stage(self, spec: StageSpec) -> str:
        """Run one stage. Returns 'completed' | 'skipped' | 'failed'."""
        store = self.store
        key = spec.key

        # 1. resume: subtask already done in a previous (crashed) run
        sub = store.get_subtask(self.task_id, key)
        if sub and sub["status"] == "completed":
            logger.info(f"stage.skip.resume task={self.task_id} stage={key}")
            return "skipped"

        # 2. already satisfied: artifacts already present → no work to do.
        # Stages with SHARED cross-venue artifacts (embed/ingest write to one
        # vectors/ dir) opt out via skip_when_satisfied=False — their probe would
        # count OTHER venues' shards as "done" and wrongly skip real work (W72).
        pr = self._safe_probe(spec)
        if spec.parent_progress_from_probe:
            store.set_task_progress(self.task_id, pr.done, pr.total)
        if spec.skip_when_satisfied and pr.is_complete:
            store.update_progress(self.task_id, key, pr.done, pr.total, pr.message)
            store.complete_subtask(self.task_id, key, message="already satisfied")
            store.emit_event(
                self.task_id, key,
                f"{spec.title}: already satisfied ({pr.done}/{pr.total})",
                event_type="stage.skipped",
            )
            logger.info(f"stage.skip.satisfied task={self.task_id} stage={key} {pr.done}/{pr.total}")
            return "skipped"

        # 3. acquire the cross-task resource mutex. Concurrent ingest_pipeline
        # tasks (different venue+year) serialize on spec.resource_key so a
        # downstream stage of task A can run in parallel with an upstream stage
        # of task B (different keys), but two tasks at the SAME stage queue up.
        # Empty resource_key = no serialization (legacy / tests).
        #
        # Grant order is (resumed first, then oldest task): a stage resuming work
        # that may still be running on the resource (W78: a surviving remote
        # worker) goes ahead of everything, then the lowest task id, so a task's
        # embed is never starved by younger tasks' extracts.
        #
        # Resume guard: if the subtask was previously started but never
        # completed, a remote worker (marker / embed tmux) is likely still
        # alive — its own resource use will then make preflight fail (it
        # would refuse the GPUs WE are already running on). Detect this and
        # skip preflight, letting run() re-attach via probe poll. See W67.
        is_resume_attempt = bool(sub and sub.get("started_at"))
        priority = (0 if is_resume_attempt else 1, self.task_id)
        store.emit_event(
            self.task_id, key,
            f"{spec.title}: queued for resource '{spec.resource_key}'"
            if spec.resource_key else f"{spec.title}: queued",
            event_type="stage.queued",
        )
        with self.locks.hold(
            spec.resource_key, self.abort, owner=self.task_id, priority=priority,
        ) as acquired:
            if not acquired:
                # Abort raised while waiting → caller cancelled; halt cleanly.
                store.cancel_subtask(self.task_id, key, "cancelled while waiting for resource")
                self._finish("cancelled")
                store.emit_event(
                    self.task_id, key,
                    "cancelled while waiting for resource",
                    level="warning", event_type="stage.cancelled",
                )
                logger.warning(f"stage.cancelled.queued task={self.task_id} stage={key}")
                return "cancelled"
            if spec.resource_key:
                store.emit_event(
                    self.task_id, key,
                    f"{spec.title}: acquired resource '{spec.resource_key}'",
                    event_type="stage.resource.acquired",
                )

            # 4. preflight safety check (before doing any work, INSIDE the lock —
            # otherwise we'd race against a peer pipeline grabbing the resource
            # between our preflight and our run). Skipped on a resume attempt
            # (see the resume guard above).
            if spec.preflight is not None and not is_resume_attempt:
                g = spec.preflight(self.ctx)
                if not g.ok and g.transient:
                    g, cancelled = self._wait_out_transient_preflight(spec, g)
                    if cancelled:
                        store.cancel_subtask(self.task_id, key, "cancelled while waiting for resource")
                        self._finish("cancelled")
                        store.emit_event(
                            self.task_id, key, "cancelled while waiting for resource",
                            level="warning", event_type="stage.cancelled",
                        )
                        logger.warning(f"stage.cancelled.queued task={self.task_id} stage={key}")
                        return "cancelled"
                if not g.ok:
                    store.fail_subtask(self.task_id, key, f"preflight: {g.reason}")
                    store.emit_event(self.task_id, key, f"preflight failed: {g.reason}", level="error", event_type="stage.preflight.failed")
                    logger.error(f"stage.preflight.failed task={self.task_id} stage={key}: {g.reason}")
                    return "failed"
            elif is_resume_attempt:
                store.emit_event(
                    self.task_id, key,
                    "skipping preflight on resume (remote worker likely alive)",
                    event_type="stage.preflight.skipped",
                )
                logger.info(f"stage.preflight.skipped task={self.task_id} stage={key} (resume)")

            # 5. run (blocking), with a background progress poller
            store.start_subtask(self.task_id, key, message="running", stage=key)
            store.update_progress(self.task_id, key, pr.done, pr.total, pr.message)
            store.emit_event(self.task_id, key, f"{spec.title}: started", event_type="stage.started")
            logger.info(f"stage.start task={self.task_id} stage={key}")

            def sink(message: str, level: str = "info") -> None:
                store.emit_event(self.task_id, key, message, level=level)

            poller = None
            if spec.poll_interval_s and spec.poll_interval_s > 0:
                poller = _Poller(
                    store, self.task_id, spec, self.ctx, self.abort,
                    update_parent_from_probe=spec.parent_progress_from_probe,
                ).start()
            try:
                spec.run(self.ctx, sink)
            except Exception as e:
                if self.abort.is_set():
                    store.cancel_subtask(self.task_id, key, "cancelled while stage was running")
                    store.emit_event(self.task_id, key, "cancelled while stage was running", level="warning", event_type="stage.cancelled")
                    return "cancelled"
                logger.exception(f"stage.run.error task={self.task_id} stage={key}: {e}")
                store.fail_subtask(self.task_id, key, f"{type(e).__name__}: {e}")
                store.emit_event(self.task_id, key, f"run failed: {e}", level="error", event_type="stage.failed")
                return "failed"
            finally:
                if poller is not None:
                    poller.stop()

            # A cancellation that lands while the blocking stage returns must
            # win before post-run probe/gate can promote this task to complete.
            if self.abort.is_set():
                store.cancel_subtask(self.task_id, key, "cancelled while stage was completing")
                store.emit_event(
                    self.task_id, key, "cancelled while stage was completing",
                    level="warning", event_type="stage.cancelled",
                )
                return "cancelled"

            # post-run probe so final progress reflects reality
            pr2 = self._safe_probe(spec)
            store.update_progress(self.task_id, key, pr2.done, pr2.total, pr2.message)
            if spec.parent_progress_from_probe:
                store.set_task_progress(self.task_id, pr2.done, pr2.total)

            # 6. acceptance gate (after run)
            note = ""
            if spec.gate is not None:
                g = spec.gate(self.ctx)
                if not g.ok:
                    store.fail_subtask(self.task_id, key, f"gate: {g.reason}")
                    store.emit_event(self.task_id, key, f"gate failed: {g.reason}", level="error", event_type="stage.gate.failed")
                    logger.error(f"stage.gate.failed task={self.task_id} stage={key}: {g.reason}")
                    return "failed"
                note = g.reason  # a gate that passes may still say what it let through

            # 7. done
            store.complete_subtask(self.task_id, key, message=f"completed: {note}" if note else "completed")
            store.emit_event(self.task_id, key, f"{spec.title}: completed" + (f" ({note})" if note else ""), event_type="stage.completed")
            logger.info(f"stage.done task={self.task_id} stage={key}")
            return "completed"

    def _wait_out_transient_preflight(self, spec: StageSpec, gate: GateResult) -> tuple[GateResult, bool]:
        """Ask a preflight again, while the stage holds its resource, as long as it fails with a
        transient condition (``GateResult.transient``): the tasks queued behind this one could not
        pass the same check, so they wait in grant order instead of failing one after another.
        Returns (the last verdict, whether the pipeline was cancelled while waiting). After
        ``PREFLIGHT_WAIT_DEADLINE_S`` the last verdict stands and the stage fails with its message."""
        started = time.monotonic()
        delay = PREFLIGHT_WAIT_INITIAL_S
        while not gate.ok and gate.transient:
            remaining = PREFLIGHT_WAIT_DEADLINE_S - (time.monotonic() - started)
            if remaining <= 0:
                break
            pause = min(delay, remaining)
            self.store.emit_event(
                self.task_id, spec.key,
                f"{spec.title}: waiting, asking again in {pause:.0f}s: {gate.reason}",
                level="warning", event_type="stage.preflight.waiting",
            )
            logger.warning(f"stage.preflight.waiting task={self.task_id} stage={spec.key} "
                           f"retry_in={pause:.0f}s: {gate.reason}")
            if self.abort.wait(pause):
                return gate, True
            delay = min(delay * 2, PREFLIGHT_WAIT_MAX_S)
            gate = spec.preflight(self.ctx)
        return gate, False

    def _safe_probe(self, spec: StageSpec) -> Progress:
        try:
            return spec.probe(self.ctx)
        except Exception:
            logger.exception(f"stage.probe.error stage={spec.key}")
            return Progress(0, 0)
