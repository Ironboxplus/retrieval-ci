import unittest
from pathlib import Path

from core.db import get_db_session
from core.db.models import Task
from app.services.task_runtime import (
    record_task_event,
    record_task_resource_sample,
    upsert_task_subtask,
    upsert_workflow_edge,
)
from app.tasks.base import TASKS_DIR, run_with_context


def build_missing_task_id() -> str:
    candidate = 987654321
    with get_db_session() as db:
        while db.query(Task.id).filter(Task.id == candidate).first() is not None:
            candidate += 1
    return str(candidate)


class TaskRuntimeGuardTests(unittest.TestCase):
    def test_runtime_writers_ignore_missing_parent_task(self):
        missing_task_id = int(build_missing_task_id())

        self.assertIsNone(
            record_task_event(
                missing_task_id,
                event_type="unit.test",
                message="missing task should be ignored",
            )
        )
        self.assertIsNone(
            upsert_task_subtask(
                missing_task_id,
                subtask_key="missing-task-subtask",
                subtask_type="unit-test",
                status="pending",
            )
        )
        self.assertIsNone(
            record_task_resource_sample(
                missing_task_id,
                phase="unit-test",
            )
        )
        self.assertIsNone(
            upsert_workflow_edge(
                missing_task_id,
                edge_key="missing-task-edge",
                from_key="source",
                to_key="target",
            )
        )

    def test_run_with_context_skips_orphaned_numeric_task(self):
        missing_task_id = build_missing_task_id()
        task_file = TASKS_DIR / f"{missing_task_id}.json"
        if task_file.exists():
            task_file.unlink()

        calls = []

        @run_with_context("unit-test")
        def should_not_run(ctx):
            calls.append(ctx.task_id)
            return {"success": True}

        result = should_not_run(missing_task_id)

        self.assertEqual({"status": "orphaned", "task_id": missing_task_id}, result)
        self.assertEqual([], calls)
        self.assertFalse(task_file.exists(), f"orphaned task should not create {task_file}")


if __name__ == "__main__":
    unittest.main()
