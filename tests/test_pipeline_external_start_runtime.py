"""Real runtime contracts for externally supplied ingest pipeline stages."""

from __future__ import annotations

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
    specs = build_ingest_pipeline(ctx)
    runner = PipelineRunner(store=store)

    runner._complete_external_stage_prefix(task_id, specs, "preindex")
    runner._complete_external_stage_prefix(task_id, specs, "preindex")

    for key in ("download", "extract"):
        subtask = store.get_subtask(task_id, key)
        assert subtask is not None
        assert subtask["status"] == "completed"
        assert subtask["progress"] == 1
        assert subtask["total"] == 1
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
