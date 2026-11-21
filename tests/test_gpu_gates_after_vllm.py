"""The local stages' GPU gates after the models moved into the vLLM services.

Since the backend loads no model, the local embed stage puts nothing on the card: it needs the
embedding service to answer, not free GPU memory, and it must not hold the card's lock against the
extract. The extract's floor is the one figure of the registered extractor, which is what
``scripts/remote_pvl/pvl_worker.py`` needs to start beside the resident services.

Real objects throughout: the live embedding service (RETRIEVAL_TEST_VLLM_EMBED_URL, skipped when
unset, tests/vllm_support.py), a real HTTP server for the wrong-service cases, the repository's own
deploy files, and ``pvl_worker`` loaded by path. The only stand-in is the hardware probe
(``gpu_health``), which on the extract side records what it was asked.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from app.services.pipeline import stages as st
from app.services.pipeline.context import StageContext
from tests.reembed_support import new_store
from tests.test_vllm_backend import ModelsServer
from tests.vllm_support import EMBED_ENV, service_url

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy" / "compute88"
CARD_MIB = 12282                       # RTX 4070 SUPER on .88


@pytest.fixture
def ctx(tmp_path):
    store, _ = new_store()
    task_id = store.create_task(task_type="ingest_pipeline")
    return StageContext(venue="IEEE S&P", year=2026, store=store, task_id=task_id,
                        project_root=tmp_path, corpus_root=tmp_path / "corpus", execution_mode="local")


@pytest.fixture
def embedding_service_at(monkeypatch):
    """Point the process-wide embedder (what the stage's preflight asks) at a URL for one test."""
    from config import settings
    from core.indexing import embedders

    def point(url: str) -> None:
        monkeypatch.setattr(settings.embedding, "service_url", url)
        embedders.reset_embedder_cache()

    yield point
    embedders.reset_embedder_cache()


@pytest.fixture
def no_gpu_probe(monkeypatch):
    """A tripwire: any GPU query by the embed stage fails the test (a GPU-less box must pass)."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"the embed stage asked about the GPU: {args} {kwargs}")

    monkeypatch.setattr(st, "gpu_health", refuse)
    monkeypatch.setattr(st, "_local_idle_gpus", refuse)


# ------------------------------------------------------------------------- the local embed stage
def test_the_local_embed_stage_is_the_service_client_and_no_longer_shares_the_extract_gpu_lock(ctx):
    spec = st.make_embed_stage_local(ctx)
    assert "vLLM embedding service" in spec.title and "BGE-M3" not in spec.title
    assert spec.resource_key == "embed-service"
    assert st.make_extract_stage_local(ctx).resource_key == "gpu-local"
    assert spec.resource_key != st.make_extract_stage_local(ctx).resource_key


def test_the_embed_preflight_passes_against_the_live_service_without_any_gpu_query(
        ctx, embedding_service_at, no_gpu_probe):
    embedding_service_at(service_url(EMBED_ENV))
    gate = st.make_embed_stage_local(ctx).preflight(ctx)
    assert gate.ok, gate.reason
    assert ctx.state_get("gpu_ids") in (None, "")


def test_the_embed_preflight_fails_when_the_service_is_not_reachable(ctx, embedding_service_at, no_gpu_probe):
    embedding_service_at("http://127.0.0.1:9")                      # port 9 refuses connections
    gate = st.make_embed_stage_local(ctx).preflight(ctx)
    assert not gate.ok
    assert "embedding service not reachable at http://127.0.0.1:9" in gate.reason


def test_the_embed_preflight_fails_with_the_error_text_of_a_service_serving_something_else(
        ctx, embedding_service_at, no_gpu_probe):
    server = ModelsServer("/models/BAAI/bge-m3-0000000000000000000000000000000000000000").start()
    try:
        embedding_service_at(server.url)
        gate = st.make_embed_stage_local(ctx).preflight(ctx)
    finally:
        server.stop()
    assert not gate.ok
    assert "revision" in gate.reason and server.url in gate.reason


def test_the_embed_stage_has_no_gpu_floor_left():
    assert not hasattr(st, "EMBED_LOCAL_MIN_FREE_MB")
    assert not hasattr(st, "local_min_free_mb")


# --------------------------------------------------------------------------- the local extract floor
def _registry(tmp_path, floor: int) -> Path:
    path = tmp_path / "extractor.json"
    path.write_text(json.dumps({"python": Path(sys.executable).as_posix(),
                                "worker": "scripts/remote_pvl/pvl_worker.py",
                                "extract_min_free_mb": floor}), encoding="utf-8")
    return path


def test_the_extract_floor_comes_from_the_registered_extractor_only(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(_registry(tmp_path, 6657)))
    monkeypatch.setenv("PIPELINE_LOCAL_MIN_FREE_MB", "1")
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACT_MIN_FREE_MB", "2")
    monkeypatch.setenv("PIPELINE_LOCAL_EMBED_MIN_FREE_MB", "3")
    assert st.extract_local_min_free_mb() == 6657


def test_the_extract_preflight_asks_for_the_registered_floor_and_ignores_a_ratio_variable(
        ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(_registry(tmp_path, 6657)))
    monkeypatch.setenv("PIPELINE_LOCAL_MIN_FREE_MB", "1")
    monkeypatch.setenv("PIPELINE_LOCAL_MIN_FREE_RATIO", "0.99")
    monkeypatch.delenv("PIPELINE_LOCAL_GPU_IDS", raising=False)
    asked = {}
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: asked.update(target=target, **kw) or [0])

    gate = st.extract_local_preflight(ctx)

    assert gate.ok and ctx.state_get("gpu_ids") == "0"
    assert asked["target"] is None and asked["min_free_mb"] == 6657
    assert asked.get("min_free_ratio", 0.0) == 0.0                  # no idle-ratio gate on the shared card


def test_the_extract_preflight_message_states_the_floor_and_no_ratio(ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(_registry(tmp_path, 6657)))
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: [])
    gate = st.extract_local_preflight(ctx)
    assert not gate.ok and "6657MB free" in gate.reason and "85%" not in gate.reason


# ------------------------------------------------------- the deployed figure equals what the worker needs
def _pvl_worker():
    path = REPO / "scripts" / "remote_pvl" / "pvl_worker.py"
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("pvl_worker_floor", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


def test_the_deployed_extract_floor_is_the_least_the_pvl_worker_can_start_with(monkeypatch):
    worker = _pvl_worker()
    least = worker.LAYOUT_RESERVE_MIB + worker.MEMORY_MARGIN_MIB + math.ceil(worker.MIN_SERVER_FRACTION * CARD_MIB)
    assert least == 6657
    # the worker really starts at that figure and refuses one MiB less
    assert worker.server_memory_fraction(least, CARD_MIB) >= worker.MIN_SERVER_FRACTION
    with pytest.raises(RuntimeError):
        worker.server_memory_fraction(least - 1, CARD_MIB)
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(DEPLOY / "local_extractor.json"))
    assert st.load_local_extractor().extract_min_free_mb == least
