"""Minimal real-uvicorn app used only by medical Task API black-box tests."""
import threading
import time

from fastapi import FastAPI

from app.api.routes.pipeline import router as pipeline_router
from app.api.routes.tasks import router as tasks_router
from app.services.medical_corpus import MEDICAL_DOWNLOAD_PIPELINE
from app.services.pipeline import TaskStore
from core.db.database import init_db

app = FastAPI()
init_db()
store = TaskStore()
seed_task_id = store.create_task(
    task_type=MEDICAL_DOWNLOAD_PIPELINE,
    params={
        "max_bytes_per_second": 2_000_000,
        "proxy": "http://127.0.0.1:10808",
    },
    total=2,
)
store.set_task_status(seed_task_id, "running")
store.ensure_subtask(
    seed_task_id,
    "download",
    subtask_type="pipeline_stage",
    name="Download medical OA PDFs via .88 proxy",
    total=2,
)
store.start_subtask(seed_task_id, "download", message="manifest=0/2", stage="download")
store.update_progress(seed_task_id, "download", 0, 2, "manifest=0/2")
store.update_state(
    seed_task_id,
    current_paper_id="med-pmc-api-visible",
    aggregate_bytes=1_250_000,
    current_file_bytes=250_000,
    current_file_total=2_000_000,
    max_bytes_per_second=2_000_000,
    proxy="http://127.0.0.1:10808",
    disk_free_bytes=900_000_000_000,
    reserved_bytes=20_000_000_000,
    disk_guard_reason=None,
)


def _advance_visible_pipeline_progress() -> None:
    """Keep the isolated browser fixture running long enough to observe polling."""
    time.sleep(8)
    store.set_task_progress(seed_task_id, 1, 2)
    store.update_progress(seed_task_id, "download", 1, 2, "manifest=1/2")


threading.Thread(target=_advance_visible_pipeline_progress, daemon=True).start()
app.include_router(tasks_router, prefix="/api/tasks")
app.include_router(pipeline_router)
