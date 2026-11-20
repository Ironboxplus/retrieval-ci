"""Stage contract — a pipeline stage is just a handful of callbacks.

This module is pure data + types: NO database, NO subprocess, NO stage-specific
knowledge. The orchestrator drives any `StageSpec` without knowing whether it is
a download, a remote GPU embed, or a Qdrant ingest — all variation lives in the
callbacks. That is the whole point: the backend core stays generic, complexity
is pushed into small per-stage callbacks (see app/services/pipeline/stages.py).

Callback surface (everything a stage needs, nothing more):
  run(ctx, sink)   — do the work, blocking. Spawn a local subprocess, ssh-launch
                     a KSS worker, or run a multi-step sequence. Raise on hard
                     failure. Use sink(msg, level=...) to emit log events.
  probe(ctx)       — count output artifacts → Progress(done, total). Pure read,
                     decoupled from HOW the work runs; also used to skip a stage
                     that is already satisfied (artifacts present) and to resume.
  gate(ctx)        — automated acceptance check AFTER run() returns. Default ok.
  preflight(ctx)   — automated safety check BEFORE run() (e.g. GPU health).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Progress:
    """Artifact-count progress for a stage. `message` surfaces as a log line."""

    done: int
    total: int
    message: str = ""

    @property
    def is_complete(self) -> bool:
        # total must be known (>0); a 0-total probe means "unknown", never complete.
        return self.total > 0 and self.done >= self.total

    @property
    def percent(self) -> float:
        return round(self.done / self.total * 100, 1) if self.total else 0.0


@dataclass
class GateResult:
    """Verdict from a preflight / gate check. `reason` is shown on failure."""

    ok: bool
    reason: str = ""


# Duck-typed: ctx is whatever StageContext the orchestrator forwards; sink is
# the event emitter the orchestrator injects per stage.
EventSink = Callable[..., None]
RunFn = Callable[[Any, EventSink], None]
ProbeFn = Callable[[Any], Progress]
GateFn = Callable[[Any], GateResult]


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage, expressed entirely as callbacks."""

    key: str
    title: str
    run: RunFn
    probe: ProbeFn
    gate: Optional[GateFn] = None
    preflight: Optional[GateFn] = None
    poll_interval_s: float = 15.0  # background probe cadence; <=0 disables the poller
    # When True (default) the orchestrator may satisfied-skip this stage if its
    # probe reports already-complete. Set False for stages whose artifacts live
    # in a SHARED, cross-venue directory: embed/ingest write to ONE vectors/ dir
    # that accumulates every venue's shards, so a count-based "done" check there
    # wrongly skips real work (e.g. counts 111 prior-venue shards as this run's).
    # Those stages must always run; resume is handled by their OWN internal
    # idempotency (dump --min-chunk-id watermark + W34 parquet-exists skip +
    # ingest_done.txt shard tracking), not by the orchestrator's skip. W72.
    skip_when_satisfied: bool = True
    # Resource key for cross-pipeline mutex. Concurrent ingest_pipeline tasks
    # serialize on this key (StageLockSet), so parallel tasks pipeline through
    # stages without double-booking the underlying resource. Empty = no lock
    # (free-for-all). Typical:
    #   download → "net"
    #   extract  → f"gpu-{ssh_target}"
    #   preindex → "sql"
    #   embed    → f"gpu-{ssh_target}"   (shares lock with extract on same box)
    #   ingest   → "qdrant"
    # See app/services/pipeline/stage_lock.py for the design rationale.
    resource_key: str = ""
    # A one-stage artifact pipeline may make its parent Task's ordinary
    # ``progress/total`` mirror this stage's probe.  This keeps the existing
    # Tasks table meaningful for long single-stage work (for example a download
    # over N manifest artifacts) without changing the default stage-count
    # roll-up used by multi-stage ingest pipelines.
    parent_progress_from_probe: bool = False
