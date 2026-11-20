"""Extract on an ssh host: batched PDF round trips through a registered worker.

Real PDFs, a real worker program (tests/marker_stand_in_worker.py, marker_worker.py's
command-line contract, reading each PDF and writing its Markdown), a real registry
file, real batching/retry/cancel logic, the real orchestrator and preindex script,
a real SQLite task table and data database.

The one stand-in is the network hop, as in test_slice_bundle_runtime: no sshd is
reachable from the test machine, so the stage runs tests/local_ssh_transport.py in
place of the ssh and scp executables (same argument vector, a real shell for the
remote command, real file copies); the host's GPU probe (nvidia-smi over ssh) is
the other thing not exercised, so the tests that run the stage hand it the card
the preflight would have chosen.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from app.services.pipeline import stages
from app.services.pipeline.context import StageContext
from app.services.pipeline.orchestrator import PipelineOrchestrator
from app.services.pipeline.registry import build_ingest_pipeline
from app.services.pipeline.service import PipelineRunner
from app.services.pipeline.spec import GateResult
from core.db.models import Chunk as DBChunk, Paper as DBPaper
from tests.test_host_turn_download import tiny_pdf
from tests.test_pipeline_external_md_runtime import CORPUS_NAME, REPO, _record, _task_store

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
HELPER = TESTS / "local_ssh_transport.py"
WORKER = TESTS / "marker_stand_in_worker.py"
SSH = [sys.executable, str(HELPER), "ssh"]
SCP = [sys.executable, str(HELPER), "scp"]
PAPERS = {"cvpr-2026-aa": "Predictive Placement", "cvpr-2026-bb": "Migrating Instead Of Killing"}


def _entry(tmp_path: Path, **marker) -> dict:
    py = Path(sys.executable).as_posix()
    return {
        "ssh_target": "test-host", "python": py, "worker": WORKER.as_posix(), "model_dir": "unused",
        "workdir": (tmp_path / "hostwork").as_posix(), "preflight_wait_s": 0, "shard_attempts": 2,
        "marker": {"python": py, "worker": WORKER.as_posix(), "batch_size": 2,
                   "env": {"MARKER_TEST_TAG": "from-registry"}, "args": ["--mode", "fast"], **marker},
    }


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A registry file with one marker-capable host and one without a marker block."""
    entry = _entry(tmp_path)
    plain = {k: v for k, v in entry.items() if k != "marker"}
    path = tmp_path / "embed_hosts.json"
    path.write_text(json.dumps({"test-host": entry, "plain-host": {**plain, "ssh_target": "plain-host"}}),
                    encoding="utf-8")
    monkeypatch.setenv("PIPELINE_EMBED_HOSTS", str(path))
    return path


def _ctx(tmp_path: Path, n_pdfs: int, *, host: str = "test-host", store=None, options=None) -> StageContext:
    corpus = tmp_path / "corpus"
    ctx = StageContext(venue="CVPR", year=2026, store=store, task_id=0, corpus_root=corpus,
                       execution_mode="host", ssh_target=host, options=options or {})
    ctx.pdf_dir.mkdir(parents=True)
    for i in range(n_pdfs):
        (ctx.pdf_dir / f"cvpr-2026-{i:02d}.pdf").write_bytes(tiny_pdf())
    return ctx


class Sink(list):
    def __call__(self, message, level="info"):
        self.append(message)


def _with_gpu(ctx, store, tmp_path):
    """The ctx as the stage sees it once a task exists and the preflight has chosen card 1."""
    task_id = store.create_task(task_type="ingest_pipeline", params={})
    ctx = dataclasses.replace(ctx, store=store, task_id=task_id)
    ctx.state_set(gpu_ids="1")
    return ctx


# ------------------------------------------------------------------- registry
def test_registry_marker_block_is_optional_typed_and_validated(registry, tmp_path):
    host = stages.load_embed_host("test-host")
    assert host.marker.batch_size == 2 and host.marker.args == ("--mode", "fast")
    assert host.marker.env == {"MARKER_TEST_TAG": "from-registry"}
    assert host.marker.extract_min_free_mb == 16384  # the default floor
    assert stages.load_embed_host("plain-host").marker is None
    with pytest.raises(ValueError, match="unknown embed host"):
        stages.load_embed_host("nobody")

    for bad in ({"python": 'py"; rm -rf /'}, {"worker": "C:\\x\\w.py"}, {"env": {"A": "`id`"}}, {"args": ["$HOME"]}):
        broken = json.loads(registry.read_text(encoding="utf-8"))
        broken["test-host"]["marker"].update(bad)
        registry.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="embed host registry"):
            stages.load_embed_host("test-host")


INHERITED_FROM_BACKEND = {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "MALLOC_ARENA_MAX", "HF_HUB_OFFLINE",
                          "TRANSFORMERS_OFFLINE"}


def test_deployed_registry_runs_arc99_extract_in_wsl_with_the_same_converter_as_88(monkeypatch):
    """arc99's 3090 converts inside WSL through the Windows->WSL launcher; everything after the launcher's own
    arguments is .88's converter configuration, so the two machines cannot drift apart, and the ingest embed
    stays on this backend's embedding service."""
    monkeypatch.delenv("PIPELINE_EMBED_HOSTS", raising=False)
    host = stages.load_embed_host("arc99")
    assert host.model_dir and host.worker.endswith("embed_worker.py")  # re-embed still embeds on the host
    assert host.ingest_embed == "local"
    marker = host.marker
    assert marker is not None and marker.python == host.python
    assert marker.worker.endswith("/scripts/remote_pvl/wsl_worker.py")
    assert marker.args[0:8:2] == ("--wsl-distro", "--wsl-cwd", "--wsl-python", "--wsl-worker")
    assert marker.args[8] == "--wsl-stop-on-eof"  # a cancelled extract must reach the worker in WSL
    local = json.loads((REPO / "deploy" / "compute88" / "local_extractor.json").read_text(encoding="utf-8"))
    tail = len(local["args"]) + 9
    assert marker.args[9:tail] == tuple(local["args"])
    # After .88's arguments only scheduling knobs may follow (how many PDFs share the pipeline), never anything
    # that changes what the models read: the 3090 idles at ~31% with .88's 4 PDFs at a time (2026-11-20).
    extra = marker.args[tail:]
    assert len(extra) % 2 == 0 and {flag for flag in extra[0::2]} <= {"--batch-pdfs"}
    # KATEX_NODE and CUDA_HOME name what .88 has system-wide (node 22; CUDA 13.2 as /usr/local/cuda, which FlashInfer
    # compiles its sampling kernel with) and WSL keeps beside its own versions.
    # .88's worker also inherits the backend unit's thread, allocator and offline settings; arc99's has to be given
    # them (without MALLOC_ARENA_MAX the worker held 6-7 GB on 16 cores against 2.5 GB on .88, 2026-11-20).
    unit = (REPO / "deploy" / "compute88" / "arc-retrival-backend.service").read_text(encoding="utf-8")
    inherited = {k: v for k, v in (line.split("=", 2)[1:] for line in unit.splitlines()
                                   if line.startswith("Environment=")) if k in INHERITED_FROM_BACKEND}
    assert set(inherited) == INHERITED_FROM_BACKEND
    assert {k: marker.env.get(k) for k in inherited} == inherited
    assert {k: v for k, v in marker.env.items()
            if k not in ("KATEX_NODE", "CUDA_HOME", "WSLENV", *INHERITED_FROM_BACKEND)} == local["env"]
    assert marker.env["CUDA_HOME"] == "/usr/local/cuda-13.2"
    # Every variable reaches the Linux side: WSLENV names exactly the others, as plain strings (/u).
    assert set(marker.env["WSLENV"].split(":")) == {f"{k}/u" for k in marker.env if k != "WSLENV"}
    # 120 PDFs per round trip: each one starts both model servers again (~7 min of a 34 min, 40-paper batch).
    assert (marker.extract_min_free_mb, marker.batch_size, marker.ranks_per_gpu) == (14336, 120, 1)


def test_worker_command_line_is_parameterized_by_the_registry():
    plain = stages.MarkerHost(python="py", worker="w.py")
    assert stages.build_marker_host_cmd(plain, rin="in", rout="out", gpu=3) == \
        "py w.py --input in --output out --gpu 3"
    custom = stages.MarkerHost(python="py", worker="other_tool.py", args=("--mode", "fast", "--disable-ocr"),
                               env={"A": "1", "B": "two"})
    cmd = stages.build_marker_host_cmd(custom, rin="in", rout="out", gpu=0)
    assert cmd.startswith('py -c "import os,subprocess,sys; os.environ.update(')
    assert "'A': '1'" in cmd and "'B': 'two'" in cmd
    assert "['py', 'other_tool.py', '--input', 'in', '--output', 'out', '--gpu', '0', " \
           "'--mode', 'fast', '--disable-ocr']" in cmd


# ---------------------------------------------------------------------- stage
def test_host_extract_converts_in_batches_and_leaves_nothing_behind(registry, tmp_path):
    store = _task_store(tmp_path / "tasks.db")
    ctx = _with_gpu(_ctx(tmp_path, 5), store, tmp_path)
    spec = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: None, ssh=SSH, scp=SCP)
    assert (spec.key, spec.resource_key) == ("extract", "gpu-test-host")
    assert spec.skip_when_satisfied and spec.preflight is not None
    sink = Sink()

    spec.run(ctx, sink)

    mds = sorted(ctx.local_md_dir.glob("*.md"))
    assert [m.name for m in mds] == [f"cvpr-2026-{i:02d}.md" for i in range(5)]
    body = mds[0].read_text(encoding="utf-8")
    # The registry's args, env and the preflight's card reached the worker.
    assert "- mode: fast" in body and "- tag: from-registry" in body and "- gpu: 1" in body
    assert any("3 batch(es)" in m for m in sink) and any("batch 3/3 done" in m for m in sink)
    progress = spec.probe(ctx)
    assert (progress.done, progress.total) == (5, 5) and spec.gate(ctx).ok
    host_work = tmp_path / "hostwork"
    assert not (host_work / "marker").exists() or not any((host_work / "marker").iterdir())
    assert not [p for p in ctx.local_md_dir.parent.iterdir() if p.name.startswith(".incoming")]
    # A worker that writes no sidecars leaves none locally either.
    assert not (ctx.local_md_dir / "_quality").exists() and not (ctx.local_md_dir / "_pages").exists()

    # A second run converts nothing: the Markdown is already there.
    before = {m.name: m.stat().st_mtime_ns for m in mds}
    again = Sink()
    spec.run(ctx, again)
    assert any("0 PDFs to convert" in m for m in again)
    assert {m.name: m.stat().st_mtime_ns for m in ctx.local_md_dir.glob("*.md")} == before


def test_a_worker_that_fails_some_pdfs_keeps_the_rest_and_the_gate_judges(registry, tmp_path):
    broken = json.loads(registry.read_text(encoding="utf-8"))
    broken["test-host"]["marker"]["args"] = ["--fail-stem", "cvpr-2026-03"]
    registry.write_text(json.dumps(broken), encoding="utf-8")
    store = _task_store(tmp_path / "tasks.db")
    ctx = _with_gpu(_ctx(tmp_path, 6), store, tmp_path)
    spec = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: None, ssh=SSH, scp=SCP)
    sink = Sink()

    spec.run(ctx, sink)  # one PDF cannot be converted; the stage still finishes its work

    assert sorted(m.stem for m in ctx.local_md_dir.glob("*.md")) == [
        f"cvpr-2026-{i:02d}" for i in (0, 1, 2, 4, 5)]
    assert any("cvpr-2026-03.pdf" in m and "produced no Markdown" in m for m in sink)
    assert any("retrying" in m for m in sink)  # the batch was retried for it (2 attempts)
    verdict = spec.gate(ctx)
    assert not verdict.ok and "only 5/6 MD" in verdict.reason  # 83% < 95%: the gate stops the task


# ------------------------------------------------- the worker's sidecars (2026-11-20)
def _set_worker_args(registry: Path, *args: str) -> None:
    edited = json.loads(registry.read_text(encoding="utf-8"))
    edited["test-host"]["marker"]["args"] = list(args)
    registry.write_text(json.dumps(edited), encoding="utf-8")


def _quality(ctx: StageContext) -> dict:
    return {r.stem: json.loads(r.read_text(encoding="utf-8")) for r in (ctx.local_md_dir / "_quality").glob("*.json")}


def test_the_batch_brings_back_the_workers_reports_and_kept_pages(registry, tmp_path):
    """The host work dir is deleted after every batch, so a report or kept page left there is lost: the gate would
    count every refusal as a miss and the refused paper would be converted again from scratch."""
    seen = tmp_path / "seen.log"
    _set_worker_args(registry, "--sidecars", "--refuse-stem", "cvpr-2026-03", "--seen-log", seen.as_posix())
    store = _task_store(tmp_path / "tasks.db")
    ctx = _with_gpu(_ctx(tmp_path, 4), store, tmp_path)
    spec = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: None, ssh=SSH, scp=SCP)

    spec.run(ctx, Sink())

    reports = _quality(ctx)  # one per paper, and not the worker's own run file
    assert sorted(reports) == [f"cvpr-2026-{i:02d}" for i in range(4)]
    assert reports["cvpr-2026-03"]["status"] == "failed" and reports["cvpr-2026-00"]["status"] == "ok"
    assert [p.name for p in (ctx.local_md_dir / "_pages").iterdir()] == ["cvpr-2026-03.json.gz"]
    assert not (ctx.local_md_dir / "cvpr-2026-03.md").exists()
    assert not seen.exists()  # nothing was kept before this run, so nothing went up
    verdict = spec.gate(ctx)  # 1 of 4 refused is within the cap: the refusal is accounted for, not a miss
    assert verdict.ok and "1/4 refused" in verdict.reason
    assert not (tmp_path / "hostwork" / "marker").exists() or not any((tmp_path / "hostwork" / "marker").iterdir())
    assert not [p for p in ctx.local_md_dir.parent.iterdir() if p.name.startswith(".incoming")]


def test_a_later_run_judges_the_kept_pages_again_and_drops_them_once_the_paper_converts(registry, tmp_path):
    seen = tmp_path / "seen.log"
    first = ("--sidecars", "--refuse-stem", "cvpr-2026-01", "--seen-log", seen.as_posix())
    _set_worker_args(registry, *first)
    store = _task_store(tmp_path / "tasks.db")
    ctx = _with_gpu(_ctx(tmp_path, 2), store, tmp_path)
    spec = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: None, ssh=SSH, scp=SCP)
    spec.run(ctx, Sink())
    kept = ctx.local_md_dir / "_pages" / "cvpr-2026-01.json.gz"
    assert kept.is_file() and not (ctx.local_md_dir / "cvpr-2026-01.md").exists()
    seen.unlink(missing_ok=True)

    _set_worker_args(registry, *first, "--accept-kept")  # the checks were fixed in between
    again = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: None, ssh=SSH, scp=SCP)
    sink = Sink()
    again.run(ctx, sink)

    assert any("1 PDFs to convert" in m for m in sink)  # only the refused paper goes up again
    assert seen.read_text(encoding="utf-8").split() == ["cvpr-2026-01"]  # its kept pages went with it
    assert (ctx.local_md_dir / "cvpr-2026-01.md").is_file()
    assert not kept.exists()  # converted: the kept pages are done with, as on the host
    assert _quality(ctx)["cvpr-2026-01"]["status"] == "ok"
    assert again.gate(ctx).ok and again.gate(ctx).reason == ""


def _runs(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def test_a_refused_paper_is_accounted_for_and_not_sent_again_in_the_same_run(registry, tmp_path):
    """A refusal is the checks' verdict on what the model read; sending the paper again re-judges its kept pages,
    may run the fallback model again, and converts it from scratch when no pages were kept (the P0 of 2026-11-19)."""
    runs = tmp_path / "runs.log"
    _set_worker_args(registry, "--sidecars", "--refuse-stem", "cvpr-2026-03", "--run-log", runs.as_posix())
    store = _task_store(tmp_path / "tasks.db")
    ctx = _with_gpu(_ctx(tmp_path, 4), store, tmp_path)
    spec = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: pytest.fail("no retry, so no wait"), ssh=SSH, scp=SCP)
    sink = Sink()

    spec.run(ctx, sink)

    assert _runs(runs) == ["cvpr-2026-00 cvpr-2026-01", "cvpr-2026-02 cvpr-2026-03"]  # one start per batch
    assert any("1 paper(s) refused verification" in m for m in sink)
    assert spec.gate(ctx).ok


def test_a_batch_that_is_entirely_refused_completes_and_the_gate_judges(registry, tmp_path):
    runs = tmp_path / "runs.log"
    _set_worker_args(registry, "--sidecars", "--refuse-stem", "cvpr-2026-00", "--refuse-stem", "cvpr-2026-01",
                     "--run-log", runs.as_posix())
    store = _task_store(tmp_path / "tasks.db")
    ctx = _with_gpu(_ctx(tmp_path, 2), store, tmp_path)
    spec = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: pytest.fail("no retry, so no wait"), ssh=SSH, scp=SCP)
    sink = Sink()

    spec.run(ctx, sink)  # a normal outcome, not "no Markdown after 2 attempts"

    assert _runs(runs) == ["cvpr-2026-00 cvpr-2026-01"]
    assert any("2 paper(s) refused verification" in m for m in sink)
    assert not list(ctx.local_md_dir.glob("*.md"))
    verdict = spec.gate(ctx)  # the gate reads the reports and applies its cap
    assert not verdict.ok and "2 refused is over the 25% cap" in verdict.reason


def test_a_paper_with_neither_markdown_nor_report_is_still_retried(registry, tmp_path):
    runs = tmp_path / "runs.log"
    _set_worker_args(registry, "--sidecars", "--refuse-stem", "cvpr-2026-00", "--fail-stem", "cvpr-2026-01",
                     "--run-log", runs.as_posix())
    store = _task_store(tmp_path / "tasks.db")
    ctx = _with_gpu(_ctx(tmp_path, 2), store, tmp_path)
    spec = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: None, ssh=SSH, scp=SCP)
    sink = Sink()

    spec.run(ctx, sink)

    # The refused paper stays out of the second attempt; the one the worker lost (no Markdown, no report) goes again.
    assert _runs(runs) == ["cvpr-2026-00 cvpr-2026-01", "cvpr-2026-01"]
    assert any("cvpr-2026-01.pdf" in m and "produced no Markdown" in m for m in sink)
    assert sorted(_quality(ctx)) == ["cvpr-2026-00"]


def test_the_worker_runs_with_a_stdin_that_stays_open_for_as_long_as_the_session_lives(registry, tmp_path):
    """The remote end learns that the ssh session ended from EOF on its stdin (plan 2.3b), so the stage must hand the
    worker's ssh call an open pipe; the backend's own stdin is at EOF."""
    stdin_log = tmp_path / "stdin.log"
    _set_worker_args(registry, "--stdin-log", stdin_log.as_posix())
    store = _task_store(tmp_path / "tasks.db")
    ctx = _with_gpu(_ctx(tmp_path, 1), store, tmp_path)

    stages.make_extract_stage_host(ctx, sleep_fn=lambda s: None, ssh=SSH, scp=SCP).run(ctx, Sink())

    assert stdin_log.read_text(encoding="utf-8") == "open"


def test_sidecars_come_back_also_when_the_worker_exits_non_zero(registry, tmp_path):
    _set_worker_args(registry, "--sidecars", "--fail-stem", "cvpr-2026-01")
    store = _task_store(tmp_path / "tasks.db")
    ctx = _with_gpu(_ctx(tmp_path, 4), store, tmp_path)
    spec = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: None, ssh=SSH, scp=SCP)

    spec.run(ctx, Sink())

    assert sorted(_quality(ctx)) == ["cvpr-2026-00", "cvpr-2026-02", "cvpr-2026-03"]
    assert sorted(m.stem for m in ctx.local_md_dir.glob("*.md")) == ["cvpr-2026-00", "cvpr-2026-02", "cvpr-2026-03"]


def test_a_batch_that_converts_nothing_fails_the_stage(registry, tmp_path):
    broken = json.loads(registry.read_text(encoding="utf-8"))
    broken["test-host"]["marker"]["args"] = ["--fail-stem", "cvpr-2026-00", "--fail-stem", "cvpr-2026-01"]
    registry.write_text(json.dumps(broken), encoding="utf-8")
    store = _task_store(tmp_path / "tasks.db")
    ctx = _with_gpu(_ctx(tmp_path, 2), store, tmp_path)
    spec = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: None, ssh=SSH, scp=SCP)

    with pytest.raises(RuntimeError, match="no Markdown after 2 attempts"):
        spec.run(ctx, Sink())
    assert not ctx.local_md_dir.exists() or not list(ctx.local_md_dir.glob("*.md"))


def test_cancellation_is_checked_between_batches(registry, tmp_path):
    store = _task_store(tmp_path / "tasks.db")
    ctx = _with_gpu(_ctx(tmp_path, 6), store, tmp_path)
    spec = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: None, ssh=SSH, scp=SCP)

    def sink(message, level="info"):
        if "batch 1/3 done" in message:
            assert store.compare_and_set_status(ctx.task_id, expect={"pending", "running"}, new="cancelled")

    with pytest.raises(RuntimeError, match="cancelled before batch 2"):
        spec.run(ctx, sink)
    assert len(list(ctx.local_md_dir.glob("*.md"))) == 2  # batch 1 landed, batch 2 never started

    ctx2 = _with_gpu(_ctx(tmp_path / "other", 2), store, tmp_path)
    ctx2.abort.set()
    with pytest.raises(RuntimeError, match="cancelled before batch 1"):
        stages.make_extract_stage_host(ctx2, ssh=SSH, scp=SCP).run(ctx2, Sink())


def test_a_run_without_the_preflights_gpu_choice_refuses_to_guess(registry, tmp_path):
    store = _task_store(tmp_path / "tasks.db")
    task_id = store.create_task(task_type="ingest_pipeline", params={})
    ctx = dataclasses.replace(_ctx(tmp_path, 1), store=store, task_id=task_id)
    with pytest.raises(RuntimeError, match="no gpu_ids"):
        stages.make_extract_stage_host(ctx, ssh=SSH, scp=SCP).run(ctx, Sink())


def test_preflight_names_its_own_failures(registry, tmp_path):
    from app.services.pipeline.stages import embed_host_preflight

    store = _task_store(tmp_path / "tasks.db")
    no_marker = _ctx(tmp_path, 0, host="plain-host", store=store)
    gate = embed_host_preflight(no_marker, stage="extract")
    assert not gate.ok and "no 'marker' block" in gate.reason
    unknown = _ctx(tmp_path / "x", 0, host="nobody", store=store)
    assert not embed_host_preflight(unknown, stage="extract").ok
    # The extract floor comes from the marker block, and a card that never frees up fails after the wait.
    broken = json.loads(registry.read_text(encoding="utf-8"))
    broken["test-host"]["marker"]["extract_min_free_mb"] = 10 ** 7
    registry.write_text(json.dumps(broken), encoding="utf-8")
    impossible = _ctx(tmp_path / "y", 0, store=store)
    assert ">=10000000 MB free" in embed_host_preflight(impossible, stage="extract", sleep_fn=lambda s: None).reason


# ------------------------------------------------------------ routing + start
def test_registry_routes_extract_and_embed_through_the_for_factories(registry, tmp_path):
    ctx = _ctx(tmp_path, 0)
    specs = {s.key: s for s in build_ingest_pipeline(ctx)}
    assert specs["extract"].resource_key == "gpu-test-host" and "marker on test-host" in specs["extract"].title
    assert specs["embed"].resource_key == "gpu-test-host"  # one lock for the host's one GPU
    local = {s.key: s for s in build_ingest_pipeline(dataclasses.replace(ctx, execution_mode="local"))}
    assert local["extract"].resource_key == "gpu-local" and local["embed"].resource_key == "embed-service"
    pushed = dataclasses.replace(ctx, options={"extract_from": "push"})
    assert stages.make_extract_stage_for(pushed).resource_key == ""  # received, no GPU, no host needed


def _set_ingest_embed(registry: Path, value) -> None:
    edited = json.loads(registry.read_text(encoding="utf-8"))
    edited["test-host"]["ingest_embed"] = value
    registry.write_text(json.dumps(edited), encoding="utf-8")


def test_ingest_embed_places_the_embed_stage_of_a_host_task(registry, tmp_path):
    ctx = _ctx(tmp_path, 0)
    assert stages.load_embed_host("test-host").ingest_embed == "host"  # the default: today's behaviour
    assert stages.make_embed_stage_for(ctx).resource_key == "gpu-test-host"

    _set_ingest_embed(registry, "local")
    assert stages.load_embed_host("test-host").ingest_embed == "local"
    embed = stages.make_embed_stage_for(ctx)  # converted on the host, embedded by this backend's service
    assert embed.resource_key == "embed-service" and "vLLM embedding service" in embed.title
    specs = {s.key: s for s in build_ingest_pipeline(ctx)}
    assert specs["extract"].resource_key == "gpu-test-host" and specs["embed"].resource_key == "embed-service"
    # Re-embed names the host it embeds on and still does (it calls the host stage directly).
    assert stages.make_embed_stage_host(ctx).resource_key == "gpu-test-host"

    for bad in ("elsewhere", True, None, ""):
        _set_ingest_embed(registry, bad)
        with pytest.raises(ValueError, match="ingest_embed"):
            stages.load_embed_host("test-host")
    with pytest.raises(ValueError, match="ingest_embed"):  # refused before a task exists
        PipelineRunner(store=_task_store(tmp_path / "tasks.db")).start_ingest(
            "CVPR", 2026, execution_mode="host", ssh_target="test-host")


def test_start_ingest_accepts_host_mode_for_conferences_and_keeps_the_unknown_host_422(registry, tmp_path):
    store = _task_store(tmp_path / "tasks.db")
    runner = PipelineRunner(store=store)
    with pytest.raises(ValueError, match="unknown embed host"):
        runner.start_ingest("CVPR", 2026, execution_mode="host", ssh_target="nobody")
    with pytest.raises(ValueError, match="no 'marker' block"):
        runner.start_ingest("CVPR", 2026, execution_mode="host", ssh_target="plain-host")
    assert store.list_running("ingest_pipeline") == []  # both refused before a task existed


# ------------------------------------------------------- the whole thing, to SQL
def test_host_extract_then_preindex_through_the_orchestrator(registry, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    (data_dir / "db").mkdir(parents=True)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    corpus = data_dir / "papers" / CORPUS_NAME
    (corpus / "manifests").mkdir(parents=True)
    (corpus / "manifests" / "paper_manifest.json").write_text(
        json.dumps([_record(pid, title) for pid, title in PAPERS.items()]), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-c", "from core.db.database import engine, Base; import core.db.models; "
                               "Base.metadata.create_all(engine)"],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    engine = create_engine(f"sqlite:///{data_dir / 'db' / 'retrieval.db'}", connect_args={"check_same_thread": False})
    local_session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def data_factory():
        session = local_session()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    store = _task_store(tmp_path / "tasks.db")
    task_id = store.create_task(task_type="ingest_pipeline", params={"venue": "CVPR", "year": 2026})
    ctx = StageContext(venue="CVPR", year=2026, store=store, task_id=task_id, corpus_root=corpus,
                       execution_mode="host", ssh_target="test-host")
    ctx.pdf_dir.mkdir(parents=True)
    for pid in PAPERS:
        (ctx.pdf_dir / f"{pid}.pdf").write_bytes(tiny_pdf())

    host_stage = stages.make_extract_stage_host(ctx, sleep_fn=lambda s: None, ssh=SSH, scp=SCP)
    # The preflight's only job beyond the GPU probe is choosing the card; nvidia-smi over ssh is not
    # reachable here, so this stand-in chooses it.
    host_stage = dataclasses.replace(
        host_stage, preflight=lambda c: (c.state_set(gpu_ids="0"), GateResult(True))[1])
    specs = [host_stage, stages.make_preindex_stage(ctx, session_factory=data_factory)]

    final = PipelineOrchestrator(task_id, specs, ctx, store).run()

    assert final == "completed"
    assert store.get_subtask(task_id, "extract")["status"] == "completed"
    assert len(list(ctx.local_md_dir.glob("*.md"))) == 2
    with data_factory() as db:
        assert {r[0] for r in db.query(DBPaper.paper_id)} == set(PAPERS)
        assert db.query(func.count(DBChunk.id)).scalar() > len(PAPERS)
    manifest = json.loads((corpus / "manifests" / "paper_manifest.json").read_text(encoding="utf-8"))
    assert all(r["markdown_path"].endswith(f"{r['paper_id']}.md") for r in manifest)
    engine.dispose()
