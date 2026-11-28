"""TDD for the ICML 2026 external task watcher lifecycle."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "watch_icml2026_web_task",
    REPO_ROOT / "scripts" / "collect" / "watch_icml2026_web_task.py",
)
watcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watcher
SPEC.loader.exec_module(watcher)  # type: ignore[union-attr]


def _task(task_id: int, source: str | None):
    return SimpleNamespace(id=task_id, params_json={"source": source} if source else {})


def test_select_canonical_task_reuses_oldest_even_after_completion():
    rows = [
        _task(288, watcher.WATCHER_STATE_KEY),
        _task(109, watcher.WATCHER_STATE_KEY),
        _task(108, "another-source"),
    ]

    assert watcher.select_canonical_task_id(rows) == 109


def test_select_canonical_task_returns_none_without_matching_source():
    assert watcher.select_canonical_task_id([_task(108, "another-source")]) is None


def test_all_stages_completed_requires_every_stage():
    complete = [SimpleNamespace(status="completed") for _ in watcher.STAGES]
    incomplete = [*complete[:-1], SimpleNamespace(status="running")]

    assert watcher.all_stages_completed(complete) is True
    assert watcher.all_stages_completed(incomplete) is False
