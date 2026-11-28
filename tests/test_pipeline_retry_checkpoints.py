"""Retry must resume durable stages rather than re-importing an old arXiv snapshot."""
from app.services.pipeline.store import TaskStore
from tests.sqlite_task_db import make_session_factory


def test_retry_keeps_completed_upstream_stages_and_reopens_unfinished_stages(tmp_path):
    store = TaskStore(make_session_factory(tmp_path / "retry.db"))
    task = store.create_task(task_type="ingest_pipeline", params={"venue": "arXiv", "year": 0})
    for i, key in enumerate(("download", "extract", "preindex", "embed", "ingest", "cleanup")):
        store.ensure_subtask(task, key, subtask_type="pipeline_stage", name=key, order=i)
    for key in ("download", "extract", "preindex"):
        store.start_subtask(task, key)
        store.update_progress(task, key, 12, 12)
        store.complete_subtask(task, key)
    store.start_subtask(task, "embed")
    store.set_task_status(task, "cancelled")
    before = store.get_subtask(task, "preindex")

    store.reset_pipeline_for_retry(task)

    assert store.get_subtask(task, "preindex") == before
    assert all(store.get_subtask(task, k)["status"] == "completed"
               for k in ("download", "extract", "preindex"))
    assert all(store.get_subtask(task, k)["status"] == "pending"
               for k in ("embed", "ingest", "cleanup"))
    assert store.get_task(task)["retry_count"] == 1
