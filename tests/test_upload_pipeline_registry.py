"""TDD: build_upload_pipeline assembles [extract, index] in order."""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from core.db import models  # noqa: F401
from app.services.pipeline.store import TaskStore
from app.services.pipeline.upload_context import UploadStageContext
from app.services.pipeline.registry import build_upload_pipeline, UPLOAD_STAGE_KEYS


@pytest.fixture
def store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        s = Local()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    return TaskStore(factory)


def test_build_upload_pipeline_order(store):
    ctx = UploadStageContext(doc_id=1, file_path="/data/uploads/a.pdf", store=store, task_id=1)
    specs = build_upload_pipeline(ctx)
    assert [s.key for s in specs] == ["extract", "index"]
    assert UPLOAD_STAGE_KEYS == ["extract", "index"]
