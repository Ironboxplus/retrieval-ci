"""papers:uploads end to end with the real model: the first live-proof shape of the pipeline.

The orchestrator drives the real stages over a scratch Qdrant collection: select scrolls real
stale points, the embed stage ships the shards to the registered host (this machine, through the
ssh/scp shim) where the real embed_worker encodes them through the BGE-M3 vLLM service
(RETRIEVAL_TEST_VLLM_EMBED_URL, tests/vllm_support.py; skipped without it), write replaces the
vectors, and verify re-encodes a sample in-process through the same service and compares it with
what Qdrant now stores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from app.services.pipeline import stages as st
from app.services.pipeline.orchestrator import PipelineOrchestrator
from app.services.pipeline.reembed_pipeline import ReembedContext, build_reembed_pipeline
from core.indexing.embedding_profile import current_embed_fp, get_profile, load_tokenizer
from core.indexing.embedders import vllm_client
from tests.embedding_model_support import bge_m3_snapshot, real_text
from tests.reembed_support import (
    create_scratch_collection, drop_collection, install_local_host, new_store, qdrant_client, stored_point,
    write_host_registry,
)
from tests.vllm_support import EMBED_ENV, service_url

WORKER = Path(__file__).resolve().parents[1] / "scripts" / "remote_embed" / "embed_worker.py"


@pytest.fixture()
def real(tmp_path, monkeypatch):
    from config import settings

    url = service_url(EMBED_ENV)
    snapshot = bge_m3_snapshot()
    monkeypatch.setenv("BGE_M3_PATH", snapshot)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("EMBEDDING__SERVICE_URL", url)               # the host worker subprocess reads it
    monkeypatch.setattr(settings.embedding, "service_url", url)
    registry = tmp_path / "embed_hosts.json"
    write_host_registry(registry, workdir=tmp_path / "hostwork", python=sys.executable, worker=WORKER,
                        model_dir=snapshot, shard_size=4)
    monkeypatch.setenv("PIPELINE_EMBED_HOSTS", str(registry))
    install_local_host(monkeypatch, st, [])
    monkeypatch.setattr(st, "gpu_health", lambda *a, **k: [0])        # the card is this machine's; no ssh probe

    profile = get_profile()
    yield tmp_path, lambda texts: vllm_client.encode(texts, profile, base_url=url, priority=5)


def test_papers_uploads_through_the_orchestrator_with_the_real_model(real):
    tmp_path, encode = real
    client = qdrant_client()
    name = create_scratch_collection(client, "e2e_uploads", keyword_indexes=("source_type", "embed_fp", "doc_id"))
    try:
        tokenizer = load_tokenizer(get_profile())
        texts = [real_text(tokenizer, n)[len(real_text(tokenizer, 20)) * (i % 3):] or "x"
                 for i, n in enumerate((60, 200, 520, 900, 1500, 2600, 4000, 300, 80))]
        client.upsert(name, points=[stored_point(i + 1, {"source_type": "upload", "doc_id": "d", "content": t})
                                    for i, t in enumerate(texts)], wait=True)
        store, _ = new_store()
        task_id = store.create_task(task_type="reembed_pipeline")
        ctx = ReembedContext(venue="papers:uploads", year=0, store=store, task_id=task_id, project_root=tmp_path,
                             execution_mode="host", ssh_target="shim-host", scope_name="papers:uploads",
                             collection=name, client_factory=lambda: client,
                             embed_fn=lambda batch: encode(list(batch)), pause_s=0.0)

        status = PipelineOrchestrator(task_id, build_reembed_pipeline(ctx), ctx, store).run()

        assert status == "completed", [store.get_subtask(task_id, k) for k in ("select", "embed", "write", "verify")]
        dense, _ = encode(texts)
        points = {p.id: p for p in client.retrieve(name, ids=list(range(1, len(texts) + 1)),
                                                   with_vectors=True, with_payload=True)}
        for i, text in enumerate(texts):
            point = points[i + 1]
            assert point.payload["embed_fp"] == current_embed_fp() and point.payload["content"] == text
            stored = np.asarray(point.vector["dense"], dtype=np.float32)
            assert float(stored @ dense[i]) >= 0.999
        note = store.get_subtask(task_id, "verify")["message"]
        assert "sample" in note and "cosine" in note
    finally:
        drop_collection(client, name)
