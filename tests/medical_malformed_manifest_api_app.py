"""Real-uvicorn fixture for malformed medical-manifest admission tests.

The app configures the actual singleton runner with an isolated SQLite-backed
TaskStore and temporary artifact root.  It deliberately bypasses only the .88
host/proxy deployment guards: no worker should be launched by this fixture, so
the test exercises admission and durable task semantics rather than networking.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from app.api.routes.pipeline import router as pipeline_router
from app.api.routes.tasks import router as tasks_router
import app.services.medical_corpus.service as medical_service
from app.services.medical_corpus.service import MedicalDownloadPipelineRunner
from app.services.pipeline import TaskStore
from core.db.database import init_db
from pipeline.medical_corpus.proxy import MedicalProxyConfig


init_db()
data_root = Path(os.environ["MEDICAL_DATA_ROOT"])
medical_service._RUNNER = MedicalDownloadPipelineRunner(
    store=TaskStore(),
    data_root=data_root,
    proxy=MedicalProxyConfig(
        http_proxy="http://127.0.0.1:9",
        https_proxy="http://127.0.0.1:9",
    ),
    execution_guard=lambda: None,
    enforce_designated_proxy=False,
)

app = FastAPI()
app.include_router(tasks_router, prefix="/api/tasks")
app.include_router(pipeline_router)
