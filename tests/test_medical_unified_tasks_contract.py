"""Regression contract: medical downloads use the generic pipeline Tasks surface."""
from __future__ import annotations

from pathlib import Path


def test_unified_tasks_page_has_no_medical_task_type_branch() -> None:
    """Medical observability belongs to generic pipeline_stage rows, never a UI fork."""
    source = (Path(__file__).parents[1] / "frontend" / "src" / "pages" / "Tasks.tsx").read_text(
        encoding="utf-8")
    assert "medical_corpus_download" not in source
    assert "Medical download observability" not in source


def test_pipeline_orchestrator_exposes_probe_artifact_counts_on_parent_and_stage() -> None:
    """A one-stage artifact pipeline can expose 0/N -> x/N -> N/N without UI logic.

    This is deliberately a real TaskStore-backed test: the generic orchestrator
    creates the normal ``pipeline_stage`` row and updates exactly the fields read
    by the existing Tasks table and Steps view.
    """
    from contextlib import contextmanager

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.services.pipeline import PipelineOrchestrator, Progress, StageSpec, TaskStore
    from core.db.database import Base

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    store = TaskStore(session_factory)
    task_id = store.create_task(task_type="artifact_download", total=1)
    state = {"done": 0, "samples": []}

    def probe(_ctx):
        return Progress(state["done"], 4, f"manifest={state['done']}/4")

    def run(_ctx, _sink):
        state["initial"] = (
            store.get_task(task_id)["progress"], store.get_task(task_id)["total"],
            store.get_subtask(task_id, "download")["progress"],
            store.get_subtask(task_id, "download")["total"],
        )
        for done in (1, 2, 3, 4):
            state["done"] = done
            # Equivalent to a download checkpoint: the stage's standard probe
            # emits the artifact count rather than a UI-only fractional formula.
            current = probe(None)
            store.update_progress(task_id, "download", current.done, current.total, current.message)
            store.set_task_progress(task_id, current.done, current.total)
            state["samples"].append((
                store.get_task(task_id)["progress"],
                store.get_task(task_id)["total"],
                store.get_subtask(task_id, "download")["progress"],
                store.get_subtask(task_id, "download")["total"],
            ))

    spec = StageSpec(
        key="download", title="Download artifacts", run=run, probe=probe,
        poll_interval_s=0, parent_progress_from_probe=True,
    )
    assert PipelineOrchestrator(task_id, [spec], object(), store).run() == "completed"

    # The new generic contract must update the parent, not only its subtask.
    assert state["initial"] == (0, 4, 0, 4)
    assert state["samples"][0] == (1, 4, 1, 4)
    task = store.get_task(task_id)
    subtask = store.get_subtask(task_id, "download")
    assert task is not None and (task["progress"], task["total"]) == (4, 4)
    assert subtask is not None and (subtask["progress"], subtask["total"]) == (4, 4)
