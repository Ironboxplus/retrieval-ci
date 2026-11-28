"""Real runtime contracts for externally supplied ingest pipeline stages."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.routes.pipeline import IngestPipelineRequest
from app.services.pipeline import stages
from app.services.pipeline.context import StageContext
from app.services.pipeline.registry import build_ingest_pipeline
from app.services.pipeline.service import INGEST_PIPELINE, PipelineRunner
from app.services.pipeline.store import TaskStore
from core.db import models  # noqa: F401 - register every ORM table
from core.db.database import Base


def _real_task_store(db_path) -> TaskStore:
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    local_session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = local_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return TaskStore(session_factory)


def test_external_preindex_start_completes_real_production_prefix(tmp_path):
    store = _real_task_store(tmp_path / "pipeline.db")
    task_id = store.create_task(
        task_type=INGEST_PIPELINE,
        params={"venue": "ACL", "year": 2026, "start_stage": "preindex"},
        total=5,
    )
    ctx = StageContext(
        venue="ACL",
        year=2026,
        store=store,
        task_id=task_id,
        project_root=tmp_path,
        corpus_root=tmp_path / "corpus",
        execution_mode="local",
    )
    (tmp_path / "corpus" / "manifests").mkdir(parents=True)
    (tmp_path / "corpus" / "manifests" / "paper_manifest.json").write_text(
        json.dumps([{"paper_id": f"acl-2026-{i}", "conference": "ACL", "year": 2026} for i in range(4)]),
        encoding="utf-8")
    ctx.pdf_dir.mkdir(parents=True)
    for i in range(2):
        (ctx.pdf_dir / f"acl-2026-{i}.pdf").write_bytes(b"%PDF-1.4")
    ctx.local_md_dir.mkdir(parents=True)
    for i in range(2):
        # Past extract's gate (a sample must be > 200 bytes with a header): the stage right before
        # start_stage is judged by its gate now, not only probed.
        (ctx.local_md_dir / f"acl-2026-{i}.md").write_text("# t\n\n" + "A sentence of body text. " * 12, encoding="utf-8")
    specs = build_ingest_pipeline(ctx)
    runner = PipelineRunner(store=store)

    runner._complete_external_stage_prefix(task_id, specs, "preindex", ctx)
    runner._complete_external_stage_prefix(task_id, specs, "preindex", ctx)

    # What the probes saw, not an asserted 1/1.
    for key, (done, total) in {"download": (2, 4), "extract": (2, 2)}.items():
        subtask = store.get_subtask(task_id, key)
        assert subtask is not None
        assert subtask["status"] == "completed"
        assert (subtask["progress"], subtask["total"]) == (done, total)
        assert "externally supplied" in subtask["message"]
    assert store.get_subtask(task_id, "preindex") is None


def test_ingest_request_accepts_only_existing_stage_names():
    request = IngestPipelineRequest(
        venue="ACL",
        year=2026,
        ssh_target="4090-88",
        execution_mode="local",
        start_stage="preindex",
    )
    assert request.start_stage == "preindex"

    with pytest.raises(ValidationError):
        IngestPipelineRequest(
            venue="ACL",
            year=2026,
            ssh_target="4090-88",
            execution_mode="local",
            start_stage="store_sql",
        )


def test_external_embed_start_requires_the_preindex_chunk_watermark():
    with pytest.raises(ValidationError, match="chunk_id_before"):
        IngestPipelineRequest(
            venue="ACL",
            year=2026,
            ssh_target="4090-88",
            execution_mode="local",
            start_stage="embed",
        )

    request = IngestPipelineRequest(
        venue="ACL",
        year=2026,
        ssh_target="4090-88",
        execution_mode="local",
        start_stage="embed",
        chunk_id_before=14_685_957,
    )
    assert request.chunk_id_before == 14_685_957


def test_external_ingest_start_requires_a_bounded_shard_range():
    with pytest.raises(ValidationError, match="shard_start"):
        IngestPipelineRequest(
            venue="ACL",
            year=2026,
            ssh_target="4090-88",
            execution_mode="local",
            start_stage="ingest",
        )

    request = IngestPipelineRequest(
        venue="ACL",
        year=2026,
        ssh_target="4090-88",
        execution_mode="local",
        start_stage="ingest",
        shard_start=12,
        shard_end=15,
    )
    assert (request.shard_start, request.shard_end) == (12, 15)


def test_external_resume_state_is_persisted_in_the_real_task_store(tmp_path):
    store = _real_task_store(tmp_path / "resume-state.db")
    task_id = store.create_task(
        task_type=INGEST_PIPELINE,
        params={"venue": "ACL", "year": 2026, "start_stage": "embed"},
        total=5,
    )
    ctx = StageContext(
        venue="ACL",
        year=2026,
        store=store,
        task_id=task_id,
        project_root=tmp_path,
        corpus_root=tmp_path / "corpus",
        execution_mode="local",
    )

    PipelineRunner(store=store)._seed_external_stage_state(
        ctx,
        start_stage="embed",
        chunk_id_before=14_685_957,
        shard_start=None,
        shard_end=None,
    )

    assert store.get_state(task_id) == {"chunk_id_before": 14_685_957}


def test_qdrant_preflight_uses_configured_url_and_real_http_server():
    seen_paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib callback name
            seen_paths.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    old_url = os.environ.get("QDRANT_URL")
    os.environ["QDRANT_URL"] = f"http://127.0.0.1:{server.server_port}"
    try:
        assert stages._qdrant_reachable() is True
        assert seen_paths == ["/collections/papers"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if old_url is None:
            os.environ.pop("QDRANT_URL", None)
        else:
            os.environ["QDRANT_URL"] = old_url


def test_retry_failed_redrives_in_place_from_the_failed_stage(tmp_path):
    """A pipeline that failed at a preflight is re-driven by retry_failed: the
    satisfied stage skips, task state survives, the failed stage runs."""
    from app.services.pipeline.spec import GateResult, Progress, StageSpec

    store = _real_task_store(tmp_path / "pipeline.db")
    runs = {"download": 0, "extract": 0}
    allow = tmp_path / "gpu-free"

    def builder(ctx):
        def done_probe(key):
            return lambda c: Progress(done=int((tmp_path / f"{key}.done").exists()), total=1)

        def run(key):
            def _run(c, sink):
                runs[key] += 1
                if key == "download":
                    c.state_set(chunk_id_after=42)
                (tmp_path / f"{key}.done").write_text("x")
            return _run

        return [
            StageSpec(key="download", title="first", run=run("download"), probe=done_probe("download"),
                      poll_interval_s=0),
            StageSpec(key="extract", title="second", run=run("extract"), probe=done_probe("extract"),
                      poll_interval_s=0,
                      preflight=lambda c: GateResult(allow.exists(), "gpu busy")),
        ]

    runner = PipelineRunner(store=store, pipeline_builder=builder)
    task_id = runner.start_ingest("TESTCONF", 2026, wait=True)["task_id"]
    assert store.get_task(task_id)["status"] == "failed"
    assert runs == {"download": 1, "extract": 0}

    with pytest.raises(ValueError):
        runner.retry_failed(task_id + 1000)
    allow.write_text("x")
    assert runner.retry_failed(task_id, wait=True) == {"status": "started", "task_id": task_id}
    task = store.get_task(task_id)
    assert task["status"] == "completed"
    assert runs == {"download": 1, "extract": 1}  # the satisfied stage skipped
    assert task["params"]["state"]["chunk_id_after"] == 42
    with pytest.raises(ValueError, match="only failed/cancelled"):
        runner.retry_failed(task_id)
