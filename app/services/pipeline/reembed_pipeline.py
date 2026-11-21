"""``reembed_pipeline``: repair every vector written before the embedding profile, as a Task.

One task per named scope (``reembed_scopes.SCOPES``). It is an ordinary pipeline Task: the
orchestrator drives its stages in order, each stage holds the resource lock it needs (``qdrant``,
the embed GPU), and it shows on the task page, can be cancelled, retried and resumed
after a restart like the ingest pipelines.

Stage order (and why): the points that are really truncated are repaired first; stamping only
adds the fingerprint to vectors that were already right, so it comes last.

1. ``select``  [none]         scroll the scope's stale points (read only; hours on papers/arxiv, so it
                              must not block ingest); write chunk shards, stamp ids, oversized list
                              (``:rebuild`` scopes: scan SQLite for too-long chunks and re-chunk the
                              Markdown instead, pausing while the host is short of memory)
2. ``embed``   [gpu-<host>]   the embed host encodes the chunk shards (``embed-service``: this backend's
                              embedding service through its shared embedder); skipped when there are none
3. ``write``   [qdrant]       replace dense+sparse (journal and papers/arxiv ``:rebuild``: upsert the rebuilt
                              points, replace the SQL chunk rows, delete the old points; see reembed_papers)
4. ``resplit`` [qdrant]       code points longer than the model limit are cut into line pieces
5. ``sample``  [none]         stamp candidates only: their stored vectors must equal a fresh encode
6. ``stamp``   [none]         set ``embed_fp`` on the stamp candidates (refused unless ``sample`` passed);
                              payload-only and throttled by ``pause_s``, so it runs beside ingest
7. ``verify``                 gate: nothing left to do, and a random sample equals a fresh encode
"""
from __future__ import annotations

import threading
from dataclasses import replace
from typing import Callable, List, Optional

from loguru import logger

from app.services.pipeline.orchestrator import PipelineOrchestrator
from app.services.pipeline.reembed_context import ReembedContext
from app.services.pipeline.reembed_scopes import collection_name, get_scope
from app.services.pipeline.reembed_select import run_select, select_gate, select_probe
from app.services.pipeline.reembed_verify import run_sample, sample_gate, sample_probe, verify_gate
from app.services.pipeline.reembed_write import (
    resplit_probe, run_resplit, run_stamp, run_write, stamp_probe, write_gate, write_probe,
)
from app.services.pipeline.runner_registry import CancellableRunner, register_runner
from app.services.pipeline.spec import GateResult, Progress, StageSpec
from app.services.pipeline.stage_lock import StageLockSet
from app.services.pipeline.stages import load_embed_host, make_embed_stage_host, make_embed_stage_local
from app.services.pipeline.store import TaskStore

__all__ = ["REEMBED_PIPELINE", "ReembedContext", "ReembedPipelineRunner", "build_reembed_pipeline",
           "get_reembed_runner"]

REEMBED_PIPELINE = "reembed_pipeline"
DEFAULT_SSH_TARGET = "arc99"
LOCAL_TARGET = "local"          # ssh_target that means "this backend's embedding service" (execution_mode "local")
STAGE_KEYS = ["select", "embed", "write", "resplit", "sample", "stamp", "verify"]


def _use_selected_shards(c: ReembedContext, sink, host) -> None:
    """The embed stage's ``dump``: the chunk shards already exist, so only their range is handed over."""
    start, end = int(c.state_get("select_shard_start", 0) or 0), int(c.state_get("select_shard_end", 0) or 0)
    c.state_set(dump_start_shard=start, shard_start=start, shard_end=end)
    sink(f"embed: embedding the select stage's shards [{start}, {end}) on {host.name if host else 'the embedding service'}")


def _no_chunk_shards(c: ReembedContext) -> bool:
    return (c.state_get("select_complete") is True
            and c.state_get("select_shard_end") == c.state_get("select_shard_start"))


def _unless_no_chunk_shards(spec: StageSpec) -> StageSpec:
    """``spec`` that does nothing, and passes its checks, when the select stage wrote no chunk shard
    (a scope that only stamps, or one with nothing stale): no GPU is needed for it."""
    def run(ctx, sink):
        if _no_chunk_shards(ctx):
            sink(f"{spec.key}: no chunk shards were selected, nothing to embed")
            return
        spec.run(ctx, sink)

    def check(fn):
        return lambda ctx: GateResult(True) if _no_chunk_shards(ctx) else fn(ctx)

    return replace(spec, run=run, preflight=check(spec.preflight), gate=check(spec.gate))


def make_reembed_embed_stage(ctx: ReembedContext) -> StageSpec:
    """The embed stage for ``ctx.execution_mode``: ``host`` = the registered ssh embed host
    (resource ``gpu-<host>``), ``local`` = this backend's shared embedder, a client of its vLLM
    embedding service (resource ``embed-service``: it holds no card, so it does not block an extract).
    Both take the select stage's shards through the same ``dump``; only where the shards are encoded
    differs, so two tasks on two GPUs overlap while their ``qdrant`` stages queue."""
    if ctx.execution_mode == "local":
        return make_embed_stage_local(
            ctx, dump=_use_selected_shards, title=f"Embed {ctx.scope_name} (vLLM embedding service)",
            encode=ctx.embed_fn)
    if ctx.execution_mode == "host":
        return make_embed_stage_host(
            ctx, dump=_use_selected_shards, title=f"Embed {ctx.scope_name} (BGE-M3 on {ctx.ssh_target})")
    raise ValueError(f"execution_mode must be 'host' or 'local', not {ctx.execution_mode!r}")


def build_reembed_pipeline(ctx: ReembedContext) -> List[StageSpec]:
    title = ctx.scope_name
    embed = make_reembed_embed_stage(ctx)
    specs = [
        # select and stamp take no resource lock: select only reads, and a stamp is a throttled payload
        # write, so neither may hold ``qdrant`` for hours against ingest or another re-embed's write.
        StageSpec(key="select", title=f"Select stale {title} points", run=run_select, probe=select_probe,
                  gate=select_gate, skip_when_satisfied=False, poll_interval_s=15.0),
        _unless_no_chunk_shards(embed),
        StageSpec(key="write", title=f"Write {title} vectors to Qdrant", run=run_write, probe=write_probe,
                  gate=write_gate, skip_when_satisfied=False, resource_key="qdrant", poll_interval_s=15.0),
        StageSpec(key="resplit", title=f"Re-split oversized {title} points", run=run_resplit, probe=resplit_probe,
                  skip_when_satisfied=False, resource_key="qdrant", poll_interval_s=0),
        StageSpec(key="sample", title=f"Sample check of {title} stamp candidates", run=run_sample,
                  probe=sample_probe, gate=sample_gate, skip_when_satisfied=False, poll_interval_s=0),
        StageSpec(key="stamp", title=f"Stamp {title} points whose vectors are unchanged", run=run_stamp,
                  probe=stamp_probe, skip_when_satisfied=False, poll_interval_s=15.0),
        StageSpec(key="verify", title=f"Verify {title}", run=lambda c, sink: None,
                  probe=lambda c: Progress(0, 0), gate=verify_gate, skip_when_satisfied=False, poll_interval_s=0),
    ]
    assert [s.key for s in specs] == STAGE_KEYS
    return specs


class ReembedPipelineRunner(CancellableRunner):
    task_type = REEMBED_PIPELINE

    def __init__(
        self,
        *,
        store: Optional[TaskStore] = None,
        pipeline_builder: Callable[[ReembedContext], list] = build_reembed_pipeline,
        stage_locks: Optional[StageLockSet] = None,
    ):
        self.store = store or TaskStore()
        self._build = pipeline_builder
        self._active: set[int] = set()
        self._aborts: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self.stage_locks = stage_locks or StageLockSet()

    def start_reembed(self, collection: str, scope: str, ssh_target: str = DEFAULT_SSH_TARGET, *,
                      wait: bool = False) -> dict:
        """Start the preset ``scope`` on the embed host ``ssh_target``, or through this backend's vLLM
        embedding service when ``ssh_target`` is ``"local"``.

        ``collection`` must be the family the scope names (``papers:uploads`` belongs to ``papers``): a
        request cannot aim a preset at another collection. A second start for a scope that is already
        running answers ``conflict`` with the running task."""
        preset = get_scope(scope)
        if collection != preset.family:
            raise ValueError(f"scope {scope!r} does not belong to collection {collection!r} "
                             f"(it works on {preset.family!r})")
        if ssh_target != LOCAL_TARGET:
            load_embed_host(ssh_target)                  # an unknown host is refused before a task exists
        for running in self.store.list_running(REEMBED_PIPELINE):
            if (running.get("params") or {}).get("scope") == scope:
                return {"status": "conflict", "task_id": running["id"],
                        "error": f"a re-embed task (#{running['id']}) is already running for {scope}"}
        params = {"collection": collection, "scope": scope, "ssh_target": ssh_target}
        task_id = self.store.create_task(task_type=REEMBED_PIPELINE, params=params, total=len(STAGE_KEYS))
        logger.info(f"reembed.start task={task_id} scope={scope} ssh_target={ssh_target}")
        self.store.set_task_status(task_id, "running")
        self._launch(task_id, params, wait=wait)
        return {"status": "started", "task_id": task_id}

    def start(self, params: dict) -> dict:
        """TaskRunner entry: ``params`` = {collection, scope, ssh_target?}."""
        return self.start_reembed(params["collection"], params["scope"],
                                  params.get("ssh_target") or DEFAULT_SSH_TARGET)

    def retry(self, task_id: int) -> dict:
        task = self.store.get_task(task_id)
        if task is None or task.get("task_type") != REEMBED_PIPELINE:
            raise ValueError(f"task {task_id} is not a {REEMBED_PIPELINE} task")
        if task.get("status") not in ("failed", "cancelled"):
            raise ValueError(f"task {task_id} is {task.get('status')}, only failed/cancelled can be retried")
        if task["retry_count"] >= task["max_retries"]:
            raise ValueError(f"task {task_id} has exceeded max retries ({task['max_retries']})")
        with self._lock:
            if task_id in self._active:
                raise ValueError(f"task {task_id} is still being driven in this process")
        params = task.get("params") or {}
        for running in self.store.list_running(REEMBED_PIPELINE):
            if (running.get("params") or {}).get("scope") == params.get("scope"):
                return {"status": "conflict", "task_id": running["id"],
                        "error": f"a re-embed task (#{running['id']}) is already running for {params.get('scope')}"}
        self.store.reset_pipeline_for_retry(task_id)
        logger.info(f"reembed.retry task={task_id} scope={params.get('scope')}")
        self._launch(task_id, params, wait=False)
        return {"status": "started", "task_id": task_id}

    def resume_running(self, *, wait: bool = False) -> list[int]:
        """Relaunch the tasks a previous process left ``running`` (startup hook). The stages resume
        from their own checkpoints in the task state and files."""
        resumed: list[int] = []
        try:
            running = self.store.list_running(REEMBED_PIPELINE)
        except Exception:
            logger.exception("reembed.resume.query_failed")
            return resumed
        for task in running:
            with self._lock:
                if task["id"] in self._active:
                    continue
            logger.info(f"reembed.resume task={task['id']} scope={(task.get('params') or {}).get('scope')}")
            self._launch(task["id"], task.get("params") or {}, wait=wait)
            resumed.append(task["id"])
        return resumed

    @staticmethod
    def _context(store: TaskStore, task_id: int, params: dict) -> ReembedContext:
        scope = params["scope"]
        ssh_target = params.get("ssh_target") or DEFAULT_SSH_TARGET
        return ReembedContext(
            venue=scope, year=0, store=store, task_id=task_id,
            execution_mode="local" if ssh_target == LOCAL_TARGET else "host",
            ssh_target=ssh_target, scope_name=scope,
            collection=collection_name(get_scope(scope)))

    def _launch(self, task_id: int, params: dict, *, wait: bool) -> None:
        # Registered before the thread exists, so a cancel racing the launch still finds the Event.
        abort = self._register_abort(task_id)

        def target():
            with self._lock:
                self._active.add(task_id)
            try:
                ctx = self._context(self.store, task_id, params)
                PipelineOrchestrator(task_id, self._build(ctx), ctx, self.store,
                                     abort=abort, locks=self.stage_locks).run()
            except Exception as exc:
                logger.exception(f"reembed.crash task={task_id}: {exc}")
                self._fail_crashed(task_id, exc)
            finally:
                with self._lock:
                    self._active.discard(task_id)
                self._drop_abort(task_id)

        if wait:
            target()
        else:
            threading.Thread(target=target, daemon=True, name=f"reembed-{task_id}").start()


_runner: Optional[ReembedPipelineRunner] = None
_runner_lock = threading.Lock()


def get_reembed_runner() -> ReembedPipelineRunner:
    """Process-wide singleton. It shares the ingest runner's StageLockSet, so a re-embed stage on
    ``qdrant`` or the embed host's GPU waits for an ingest stage on the same resource."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                from app.services.pipeline.service import get_pipeline_runner
                _runner = ReembedPipelineRunner(stage_locks=get_pipeline_runner().stage_locks)
    return _runner


register_runner(REEMBED_PIPELINE, get_reembed_runner)
