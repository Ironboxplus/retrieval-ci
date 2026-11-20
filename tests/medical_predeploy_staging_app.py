"""Isolated .88 medical predeploy app; never used by the production unit.

It deliberately loads the real medical runner and existing Tasks/Pipeline API
routes without importing ``main.py``.  That avoids loading retrieval models on
GPU0 during a network/durable-task deployment gate.
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
from pipeline.medical_corpus.proxy import DEFAULT_PROXY_URL, MedicalProxyConfig


data_root = Path(os.environ["MEDICAL_DATA_ROOT"]).resolve()
if not data_root.name.startswith("medical-predeploy-047-"):
    raise RuntimeError("staging medical root must be an isolated medical-predeploy-047-* directory")
if os.environ.get("MEDICAL_DOWNLOAD_HOST") != "10.100.100.88":
    raise RuntimeError("staging medical host must explicitly identify compute .88")

init_db()
medical_service._RUNNER = MedicalDownloadPipelineRunner(
    store=TaskStore(),
    data_root=data_root,
    proxy=MedicalProxyConfig(http_proxy=DEFAULT_PROXY_URL, https_proxy=DEFAULT_PROXY_URL),
    execution_guard=lambda: None,
    enforce_designated_proxy=True,
    enforce_designated_root=False,
)

app = FastAPI()
app.include_router(tasks_router, prefix="/api/tasks")
app.include_router(pipeline_router)


@app.on_event("startup")
def resume_staging_medical_tasks() -> None:
    """Mirror the production startup recovery without importing GPU-backed main."""
    medical_service.get_medical_download_runner().resume_running(wait=False)
