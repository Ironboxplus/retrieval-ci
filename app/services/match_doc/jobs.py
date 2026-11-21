"""Durable exact match-paper jobs backed by the existing Tasks tables/UI.

Four properties this file has to provide that the shared Tasks plumbing does not,
because every other task type is short, CPU-cheap and internal-only:

* **Atomic transitions.** `TaskStore.set_task_status` reads then writes, so a
  cancel landing between the two is overwritten and lost. Every transition here
  goes through `compare_and_set_status`, and the final result through
  `publish_result_if_running`, so a lost race is a no-op rather than a cancelled
  job reported as `completed`.
* **A cross-process lease.** An in-process set of active ids cannot see a second
  backend process; both would `resume_active()` the same row and run the whole
  GPU job twice. Ownership is taken with `claim_task` and refreshed on every
  progress tick; losing the lease stops the worker.
* **An end-to-end deadline.** The match is minutes of GPU work. Without a budget
  a wedged job holds the execution lock forever. The deadline is checked at the
  same points as cancellation and reports the phase it died in.
* **An owner.** `task_id` is a sequential integer, so on the public surface it is
  otherwise trivial to poll or cancel someone else's job.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

from loguru import logger

from app.services.match_doc.schemas import MatchDocRequest
from app.services.match_doc.service import get_match_doc_service
from app.services.pipeline.store import TaskStore
from core.retrieval.reranker import RerankCancelledError
from core.storage.qdrant_store import QdrantRetrievalCancelledError


MATCH_PAPER_JOB = "match_paper"

# One id per backend process. `claim_task` treats a row whose heartbeat is older
# than the lease as abandoned, so a crashed process's work is picked up by the
# next one instead of being stranded in `running` forever.
WORKER_ID = f"match-{os.getpid()}-{uuid.uuid4().hex[:8]}"
LEASE_SECONDS = 120
DEFAULT_DEADLINE_SECONDS = 1800


class MatchPaperDeadlineExceeded(RuntimeError):
    """The job ran past its end-to-end budget. Carries the phase it died in."""


class MatchPaperJobRunner:
    """Run one exact match job at a time and resume durable active rows."""

    def __init__(
        self,
        search_pipeline,
        *,
        store: TaskStore | None = None,
        worker_id: str = WORKER_ID,
        lease_seconds: int = LEASE_SECONDS,
        default_deadline_seconds: int = DEFAULT_DEADLINE_SECONDS,
    ) -> None:
        self._pipeline = search_pipeline
        self.store = store or TaskStore()
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._default_deadline_seconds = default_deadline_seconds
        self._active: set[int] = set()
        self._active_lock = threading.Lock()
        self._execution_lock = threading.Lock()

    def start(
        self,
        request: MatchDocRequest,
        *,
        wait: bool = False,
        owner_user_id: str | None = None,
        deadline_seconds: int | None = None,
    ) -> dict[str, Any]:
        if request.input_type != "paper_id":
            raise ValueError("durable match-paper jobs require input_type=paper_id")
        params = request.model_dump(mode="json")
        # `or` would fold a caller-supplied 0 into the default — an explicitly
        # requested "expire immediately" budget must not become 30 minutes.
        params["deadline_seconds"] = int(
            self._default_deadline_seconds if deadline_seconds is None else deadline_seconds
        )
        task_id = self.store.create_task(
            task_type=MATCH_PAPER_JOB,
            params=params,
            total=100,
            owner_user_id=owner_user_id,
        )
        self.store.emit_event(
            task_id,
            None,
            f"Queued exact match for {request.paper_id}",
            event_type="match_paper.queued",
            stage="match_paper.queued",
        )
        self._launch(task_id, wait=wait)
        return {"status": "pending", "task_id": task_id}

    def cancel(self, task_id: int) -> bool:
        """Request cancellation. True if this call performed the transition.

        The worker notices at its next cancel check; it never publishes a result
        afterwards because publication is itself conditional on still being
        `running` under our worker id.
        """
        return self.store.compare_and_set_status(
            task_id, expect={"pending", "running"}, new="cancelled",
        )

    def resume_active(self, *, wait: bool = False) -> list[int]:
        resumed: list[int] = []
        for task in self.store.list_active(MATCH_PAPER_JOB):
            task_id = int(task["id"])
            with self._active_lock:
                if task_id in self._active:
                    continue
            # Do not emit "resuming" before we own it — another process may hold a
            # live lease on this row, in which case _run claims nothing and exits.
            self._launch(task_id, wait=wait)
            resumed.append(task_id)
        return resumed

    def _launch(self, task_id: int, *, wait: bool) -> None:
        with self._active_lock:
            if task_id in self._active:
                return
            self._active.add(task_id)

        def target() -> None:
            try:
                # Search/rerank models and Qdrant are process-level shared
                # resources.  Serialize long match jobs so submitting several
                # papers cannot starve ordinary search or multiply GPU work.
                with self._execution_lock:
                    self._run(task_id)
            finally:
                with self._active_lock:
                    self._active.discard(task_id)

        if wait:
            target()
        else:
            threading.Thread(
                target=target,
                daemon=True,
                name=f"match-paper-{task_id}",
            ).start()

    def _run(self, task_id: int) -> None:
        snapshot = self.store.get_task(task_id)
        if snapshot is None or snapshot.get("status") not in {"pending", "running"}:
            return

        # Claim BEFORE the status transition: losing the claim means another live
        # process owns this row, and we must not touch its status at all.
        if not self.store.claim_task(
            task_id, self._worker_id, lease_seconds=self._lease_seconds,
        ):
            logger.info(
                "match_paper.job.not_claimed task={} worker={}", task_id, self._worker_id,
            )
            return

        # Conditional start. If a cancel landed after the snapshot above, this
        # fails and the job never runs — the bug the old unconditional
        # set_task_status("running") had.
        if not self.store.compare_and_set_status(
            task_id, expect={"pending", "running"}, new="running",
        ):
            self.store.release_claim(task_id, self._worker_id)
            return

        params = dict(snapshot.get("params") or {})
        stored_deadline = params.pop("deadline_seconds", None)
        deadline_seconds = int(
            self._default_deadline_seconds if stored_deadline is None else stored_deadline
        )
        request = MatchDocRequest.model_validate(params)
        expires_at = time.monotonic() + deadline_seconds

        self.store.set_task_progress(task_id, 1, 100)
        self.store.emit_event(
            task_id,
            None,
            f"Exact match worker started (worker={self._worker_id}, "
            f"deadline={deadline_seconds}s)",
            event_type="match_paper.started",
            stage="match_paper.running",
        )

        last_progress = -1
        last_phase = "starting"
        completed_in_phase = (0, 0)
        # Why a flag and not an exception: should_cancel runs deep inside Qdrant and
        # FlagEmbedding call stacks. A novel exception type would have to survive
        # every broad `except Exception` between here and there, and one that
        # swallowed it would turn a deadline into a silent empty result. Returning
        # True instead rides the cancellation path that is already threaded through
        # those layers; the flag is what lets _run label the outcome `timeout`
        # rather than `cancelled`.
        timed_out: dict[str, str] = {}

        def should_cancel() -> bool:
            if time.monotonic() > expires_at and not timed_out:
                timed_out["reason"] = (
                    f"exceeded {deadline_seconds}s budget during phase "
                    f"{last_phase!r} at {completed_in_phase[0]}/{completed_in_phase[1]}"
                )
            if timed_out:
                return True
            if not self.store.renew_lease(task_id, self._worker_id):
                # Someone else claimed this row (our lease expired while we were
                # stalled). Stop rather than race the new owner's writes.
                logger.warning("match_paper.job.lease_lost task={}", task_id)
                return True
            current = self.store.get_task(task_id)
            return current is None or current.get("status") == "cancelled"

        def on_progress(event: dict[str, Any]) -> None:
            nonlocal last_progress, last_phase, completed_in_phase
            phase = str(event.get("phase") or "running")
            completed = max(0, int(event.get("completed") or 0))
            total = max(0, int(event.get("total") or 0))
            ratio = min(1.0, completed / total) if total else 0.0
            if phase == "prepared":
                progress = 5
            elif phase == "embedding":
                progress = 5 + int(10 * ratio)
            elif phase == "qdrant":
                progress = 15 + int(55 * ratio)
            elif phase == "rerank":
                progress = 70 + int(25 * ratio)
            elif phase == "aggregate":
                progress = 98
            else:
                progress = max(last_progress, 1)
            self.store.set_task_progress(task_id, progress, 100)
            self.store.renew_lease(task_id, self._worker_id)
            if progress != last_progress or phase != last_phase:
                self.store.emit_event(
                    task_id,
                    None,
                    f"{phase} {completed}/{total}",
                    event_type="match_paper.progress",
                    stage=f"match_paper.{phase}",
                )
                last_progress = progress
                last_phase = phase
            completed_in_phase = (completed, total)

        try:
            response = get_match_doc_service(self._pipeline).match(
                request,
                progress_callback=on_progress,
                should_cancel=should_cancel,
            )
            if timed_out:
                raise MatchPaperDeadlineExceeded(timed_out["reason"])
            result = response.model_dump(mode="json")
            # Conditional publication: if a cancel or a lease takeover happened
            # while we were aggregating, this is a no-op and the job stays
            # cancelled instead of being relabelled `completed` with a result.
            if self.store.publish_result_if_running(task_id, result, self._worker_id):
                self.store.emit_event(
                    task_id,
                    None,
                    f"Exact match completed with {len(response.papers)} papers",
                    event_type="match_paper.completed",
                    stage="match_paper.completed",
                )
            else:
                self.store.emit_event(
                    task_id,
                    None,
                    "Result discarded: task was cancelled or reclaimed before publication",
                    level="warning",
                    event_type="match_paper.discarded",
                    stage="match_paper.discarded",
                )
        except MatchPaperDeadlineExceeded as exc:
            self.store.compare_and_set_status(
                task_id, expect={"pending", "running"}, new="timeout", error=str(exc),
            )
            self.store.emit_event(
                task_id,
                None,
                str(exc),
                level="error",
                event_type="match_paper.timeout",
                stage=f"match_paper.{last_phase}",
            )
        except (QdrantRetrievalCancelledError, RerankCancelledError) as exc:
            # Both layers raise these when should_cancel() returns True, so a
            # deadline arrives here too — the flag is the only thing that tells the
            # two apart, and a timeout must not be reported as a user cancellation.
            if timed_out:
                reason = timed_out["reason"]
                self.store.compare_and_set_status(
                    task_id, expect={"pending", "running"}, new="timeout", error=reason,
                )
                self.store.emit_event(
                    task_id, None, reason, level="error",
                    event_type="match_paper.timeout",
                    stage=f"match_paper.{last_phase}",
                )
                return
            self.store.compare_and_set_status(
                task_id, expect={"pending", "running"}, new="cancelled",
            )
            self.store.emit_event(
                task_id,
                None,
                f"Exact match cancelled without publishing a partial result ({exc})",
                level="warning",
                event_type="match_paper.cancelled",
                stage="match_paper.cancelled",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.store.compare_and_set_status(
                task_id, expect={"pending", "running"}, new="failed", error=error,
            )
            self.store.emit_event(
                task_id,
                None,
                error,
                level="error",
                event_type="match_paper.failed",
                stage="match_paper.failed",
            )
            logger.exception("match_paper.job.failed task={}", task_id)
        finally:
            self.store.release_claim(task_id, self._worker_id)


_RUNNER: MatchPaperJobRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_match_paper_job_runner(search_pipeline=None) -> MatchPaperJobRunner:
    global _RUNNER
    if _RUNNER is None:
        if search_pipeline is None:
            raise RuntimeError("match-paper job runner is not initialized")
        with _RUNNER_LOCK:
            if _RUNNER is None:
                _RUNNER = MatchPaperJobRunner(search_pipeline)
    return _RUNNER
