"""Background duplicate-code analysis after code indexing: trigger, throttle, algorithm,
result, Tasks rows. Real Qdrant, real SQLite, the real CodeIndexJobQueue/CodeKBService.

The flow under test is what production does: the MCP client submits an index job, the
job writes vectors and the catalog report, and its completion tells the scheduler. The
scheduler decides from the catalog watermark whether anything changed, throttles per
scope, scans only the changed files, and records a summary.
"""
from __future__ import annotations

import re
import time
from datetime import datetime

import pytest

from app.services.code_kb import analysis as analysis_module
from app.services.code_kb import duplicates
from app.services.code_kb.analysis import (
    TASK_TYPE, CodeAnalysisScheduler, build_task_runner)
from app.services.code_kb.analysis_store import CodeAnalysisStore
from app.services.index_catalog.schemas import IndexReportRequest
from app.services.pipeline.runner_registry import TaskNotCancellable
from app.services.scope_paths import canonicalize_scope_path
from core.code_kb.vector_records import CodeChunkVectorRecord
from core.db.models import Task
from tests.code_analysis_support import (
    E, OWNER, PROJECT, SlowQueryClient, env, make_request, vector_store)  # noqa: F401


@pytest.fixture()
def sched(env, monkeypatch):
    made: list[CodeAnalysisScheduler] = []

    def build(**kwargs) -> CodeAnalysisScheduler:
        kwargs.setdefault("min_interval_seconds", 0.0)
        scheduler = CodeAnalysisScheduler(
            service=env.service, catalog_service=env.catalog,
            store=CodeAnalysisStore(env.session_factory), tasks=env.tasks, **kwargs)
        env.queue.analysis = scheduler
        # The Tasks runner reaches the process scheduler, as it does in production.
        monkeypatch.setattr(analysis_module, "_SCHEDULER", scheduler)
        made.append(scheduler)
        return scheduler

    yield build
    for scheduler in made:
        scheduler.shutdown()


def _store(env) -> CodeAnalysisStore:
    return CodeAnalysisStore(env.session_factory)


def _tasks(env) -> list[dict]:
    with env.task_store._sf() as session:
        ids = [t.id for t in session.query(Task).filter(
            Task.task_type == TASK_TYPE).order_by(Task.id)]
    return [env.task_store.get_task(task_id) for task_id in ids]


def _index(env, scheduler, project, files=None, **kwargs):
    """Index, then wait until the completion has reached the scheduler and it is idle."""
    seen = scheduler.notifications
    job = env.index(project, files, **kwargs)
    deadline = time.time() + 20
    while scheduler.notifications <= seen and time.time() < deadline:
        time.sleep(0.01)
    assert scheduler.notifications > seen, "the index job never notified the scheduler"
    assert scheduler.wait_idle(40), "the scheduler did not go idle"
    return job


class _Sink:
    """Stands in for "no scheduler attached yet" while still letting a test see that the
    job's completion notification has been delivered (it is sent just after COMPLETED)."""

    def __init__(self):
        self.notifications = 0

    def notify(self, owner_user_id, project):
        self.notifications += 1


def _index_before_any_scheduler(env, project, files):
    sink = _Sink()
    env.queue.analysis = sink
    env.index(project, files)
    deadline = time.time() + 20
    while sink.notifications < 1 and time.time() < deadline:
        time.sleep(0.01)
    assert sink.notifications == 1
    env.queue.analysis = None


def _write_with_catalog(env, project, files):
    """What an index job does to the scope, without going through the queue."""
    request = make_request(project, files, [])
    env.service.index(request, owner_user_id=OWNER)
    env.catalog.report(IndexReportRequest(
        host="host", project=project, root_dir=project, files=request.report.files,
        replace=False), owner_user_id=OWNER)


# ---------------------------------------------------------------------------- results

def test_the_first_analysis_says_found_not_new_and_stores_the_pair(env, sched):
    scheduler = sched()
    _index(env, scheduler, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})

    state = _store(env).get_state(OWNER, PROJECT)
    assert state.version == 1 and state.delivered_version == 0
    text = state.summary["text"]
    assert "found" in text.lower() and "new" not in text.lower(), text
    assert re.search(r"app/alpha\.py:\d+-\d+ <-> app/beta\.py:\d+-\d+", text), text
    assert "2 shared chunks" in text and "1.00" in text, text
    assert [(p.path_a, p.path_b) for p in _store(env).all_pairs(OWNER, PROJECT)] == [
        ("app/alpha.py", "app/beta.py")]


def test_a_later_index_reports_new_pairs_and_a_fix_reports_them_gone(env, sched):
    scheduler = sched()
    _index(env, scheduler, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})

    _index(env, scheduler, PROJECT, {"app/gamma.py": [0, 1]})
    state = _store(env).get_state(OWNER, PROJECT)
    assert state.version == 2
    assert "new" in state.summary["text"].lower()
    assert "2" in state.summary["text"]
    assert {(p.path_a, p.path_b) for p in _store(env).all_pairs(OWNER, PROJECT)} == {
        ("app/alpha.py", "app/beta.py"), ("app/alpha.py", "app/gamma.py"),
        ("app/beta.py", "app/gamma.py")}

    # gamma is rewritten into something unrelated: both of its pairs are gone.
    _index(env, scheduler, PROJECT, {"app/gamma.py": [4, 5]})
    state = _store(env).get_state(OWNER, PROJECT)
    assert state.version == 3
    assert "no longer" in state.summary["text"]
    assert [(p.path_a, p.path_b) for p in _store(env).all_pairs(OWNER, PROJECT)] == [
        ("app/alpha.py", "app/beta.py")]


def test_a_pair_that_survives_an_edit_is_neither_new_nor_gone(env, sched):
    scheduler = sched()
    _index(env, scheduler, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    before = _store(env).get_state(OWNER, PROJECT)

    # alpha gains a chunk but still shares the same two with beta: nothing to report.
    _index(env, scheduler, PROJECT, {"app/alpha.py": [0, 1, 2]})
    after = _store(env).get_state(OWNER, PROJECT)
    assert after.version == before.version
    assert after.summary == before.summary
    assert after.watermark > before.watermark, "the change was still consumed"
    assert len(_tasks(env)) == 2


def test_removing_a_file_clears_its_pairs_and_reports_them_gone(env, sched):
    scheduler = sched()
    _index(env, scheduler, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    _index(env, scheduler, PROJECT, removed=["app/beta.py"])

    assert _store(env).all_pairs(OWNER, PROJECT) == []
    state = _store(env).get_state(OWNER, PROJECT)
    assert state.version == 2 and "no longer" in state.summary["text"]


def test_excluded_directories_never_become_findings(env, sched):
    scheduler = sched()
    _index(env, scheduler, PROJECT, {
        "app/alpha.py": [0, 1], "storage/dump.py": [0, 1], "node_modules/x/y.js": [0, 1]})
    assert _store(env).all_pairs(OWNER, PROJECT) == []
    assert _store(env).get_state(OWNER, PROJECT).version == 0


def test_the_summary_lists_three_pairs_with_locations_and_points_to_the_tasks_page(env, sched):
    scheduler = sched()
    _index(env, scheduler, PROJECT, {f"app/f{i}.py": [0, 1] for i in range(4)})  # 6 pairs

    text = _store(env).get_state(OWNER, PROJECT).summary["text"]
    task_id = _tasks(env)[0]["id"]
    listed = [line for line in text.splitlines() if "<->" in line]
    assert len(listed) == 3, text
    for line in listed:
        assert re.search(r"app/f\d\.py:\d+-\d+ <-> app/f\d\.py:\d+-\d+", line), line
        assert "2 shared chunks" in line and "1.00" in line, line
    assert f"3 more on Tasks page #{task_id}" in text, text
    # The Tasks row carries every pair, not just the three the summary names.
    result = _tasks(env)[0]["result"]
    assert len(result["new_pairs"]) == 6


def test_a_change_over_the_chunk_cap_is_summarised_as_too_large_and_not_scanned(env, sched):
    scheduler = sched(max_chunks=3)
    _index(env, scheduler, PROJECT, {"app/big.py": [0, 1, 2, 3], "app/twin.py": [0, 1, 2, 3]})

    state = _store(env).get_state(OWNER, PROJECT)
    assert state.version == 1
    assert "too large" in state.summary["text"] and "3" in state.summary["text"]
    assert _store(env).all_pairs(OWNER, PROJECT) == []
    assert _tasks(env)[0]["result"]["status"] == "too_large"
    # The change was consumed: it is not retried on every later index.
    _index(env, scheduler, PROJECT, {"app/other.py": [6]})
    assert state.watermark is not None
    assert len(_tasks(env)) == 2


# ------------------------------------------------------------------- scope spellings

WIN = canonicalize_scope_path("E:/Work/Repo")
WIN_LEGACY = "E:/Work/Repo"


def _seed_legacy_spelling(env, files):
    """The same repo under an older case spelling of its scope, as old clients wrote it."""
    items, vectors, records = [], [], []
    for path, basis in files.items():
        for ordinal, b in enumerate(basis):
            start = ordinal * 10 + 1
            items.append(CodeChunkVectorRecord(
                rel_path=path, language="python", start_line=start, end_line=start + 9,
                label=f"chunk{ordinal}", content=f"basis {b}", code=f"basis {b}",
                chunk_ordinal=ordinal))
            vectors.append(E[b])
        records.append({"rel_path": path, "kind": "code", "chunk_count": len(basis),
                        "content_hash": f"legacy-{path}-{basis}"})
    env.vector_store.upsert(owner_user_id=OWNER, project=WIN_LEGACY, host="host",
                            items=items, dense_vectors=vectors)
    env.catalog.repository.report_files(
        owner_user_id=OWNER, host="host", project=WIN_LEGACY, root_dir=WIN_LEGACY,
        files=records, replace=False)


def test_scope_spellings_count_once_and_a_change_under_either_is_noticed(env, sched):
    assert WIN != WIN_LEGACY
    scheduler = sched()
    layout = {"app/alpha.py": [0, 1], "app/beta.py": [0, 1], "app/gamma.py": [0, 2]}
    _seed_legacy_spelling(env, layout)
    _index(env, scheduler, WIN, layout)

    pairs = _store(env).all_pairs(OWNER, WIN)
    assert [(p.path_a, p.path_b, p.shared_chunks) for p in pairs] == [
        ("app/alpha.py", "app/beta.py", 2)], (
        "two spellings of one repo must not double the shared chunks or promote gamma's "
        "single shared chunk past the minimum")
    assert _store(env).get_state(OWNER, WIN_LEGACY) is None

    # A change recorded only in the legacy spelling's catalog rows is still this scope's.
    _seed_legacy_spelling(env, {"app/delta.py": [0, 1]})
    scheduler.notify(OWNER, WIN)
    assert scheduler.wait_idle(40)
    assert ("app/alpha.py", "app/delta.py") in {
        (p.path_a, p.path_b) for p in _store(env).all_pairs(OWNER, WIN)}


# ------------------------------------------------------------------------- triggering

def test_an_index_that_changes_nothing_does_not_run_an_analysis(env, sched):
    scheduler = sched()
    files = {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]}
    _index(env, scheduler, PROJECT, files)
    assert len(_tasks(env)) == 1

    _index(env, scheduler, PROJECT, files)  # same content, same hashes
    assert len(_tasks(env)) == 1, "no content change, so no analysis and no Tasks row"
    assert _store(env).get_state(OWNER, PROJECT).version == 1


def test_changes_to_documents_do_not_trigger_a_code_analysis(env, sched):
    scheduler = sched()
    env.catalog.repository.report_files(
        owner_user_id=OWNER, host="host", project=PROJECT, root_dir=PROJECT,
        files=[{"rel_path": "notes.md", "kind": "doc", "chunk_count": 1,
                "content_hash": "h"}], replace=False)
    scheduler.notify(OWNER, PROJECT)
    assert scheduler.wait_idle(10)
    assert _tasks(env) == []


def test_a_scope_that_is_still_preparing_is_not_analysed(env, sched):
    _index_before_any_scheduler(
        env, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    env.catalog.mark_code_scope_preparing(
        owner_user_id=OWNER, host="host", project=PROJECT, job_id="writer")
    scheduler = sched()
    scheduler.notify(OWNER, PROJECT)
    assert scheduler.wait_idle(10)
    assert _tasks(env) == [] and _store(env).get_state(OWNER, PROJECT) is None

    env.catalog.mark_code_scope_completed(
        owner_user_id=OWNER, host="host", project=PROJECT, job_id="writer")
    scheduler.notify(OWNER, PROJECT)
    assert scheduler.wait_idle(30)
    assert len(_tasks(env)) == 1 and _store(env).get_state(OWNER, PROJECT).version == 1


def test_nothing_is_analysed_at_startup_and_the_next_completion_covers_the_backlog(
        env, sched):
    _index_before_any_scheduler(
        env, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    scheduler = sched()
    time.sleep(0.5)
    assert scheduler.wait_idle(5)
    assert _tasks(env) == [], "a new scheduler must not scan anything by itself"

    _index(env, scheduler, PROJECT, {"app/unrelated.py": [5]})
    assert [(p.path_a, p.path_b) for p in _store(env).all_pairs(OWNER, PROJECT)] == [
        ("app/alpha.py", "app/beta.py")], "the backlog is covered by the watermark"


def test_index_runs_inside_the_interval_coalesce_into_one_later_run(env, sched):
    scheduler = sched(min_interval_seconds=2.0)
    _index(env, scheduler, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    assert len(_tasks(env)) == 1

    # Two more index runs, both well inside the interval.
    seen = scheduler.notifications
    env.index(PROJECT, {"app/gamma.py": [0, 1]})
    env.index(PROJECT, {"app/delta.py": [0, 1]})
    deadline = time.time() + 10
    while scheduler.notifications < seen + 2 and time.time() < deadline:
        time.sleep(0.01)
    assert len(_tasks(env)) == 1, "inside the interval nothing runs yet"

    assert scheduler.wait_idle(30)
    rows = _tasks(env)
    assert len(rows) == 2, "both index runs share one analysis"
    assert rows[1]["result"]["changed_files"] == 2
    gap = rows[1]["started_at"] - rows[0]["started_at"]
    assert gap >= 1.8, f"two analyses of one scope started {gap:.2f}s apart"
    assert {(p.path_a, p.path_b) for p in _store(env).all_pairs(OWNER, PROJECT)} >= {
        ("app/alpha.py", "app/gamma.py"), ("app/alpha.py", "app/delta.py"),
        ("app/delta.py", "app/gamma.py")}


def test_the_interval_survives_a_restart(env, sched):
    first = sched(min_interval_seconds=30.0)
    _index(env, first, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    assert len(_tasks(env)) == 1
    first.shutdown()

    second = sched(min_interval_seconds=30.0)  # a restarted backend
    env.index(PROJECT, {"app/gamma.py": [0, 1]})
    time.sleep(0.8)
    assert len(_tasks(env)) == 1, "the stored last run still throttles the scope"
    assert second.notifications >= 1


# ------------------------------------------------------------ concurrent writes, cancel

def test_a_write_during_the_scan_discards_the_result_and_the_scan_is_redone(
        env, sched, monkeypatch):
    monkeypatch.setattr(duplicates, "_QUERY_BATCH", 1)
    scheduler = sched(min_interval_seconds=0.3)
    _index(env, scheduler, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})

    fired = []

    def write_during_scan(query_number: int) -> None:
        if not fired:
            fired.append(query_number)
            _write_with_catalog(env, PROJECT, {"app/gamma.py": [0, 1]})

    env.vector_store.store.client = SlowQueryClient(
        env.vector_store.store.client, on_query=write_during_scan)
    _index(env, scheduler, PROJECT, {"app/alpha.py": [0, 1, 4]})

    rows = _tasks(env)
    assert rows[1]["result"]["status"] == "discarded", rows[1]["result"]
    assert rows[-1]["result"]["status"] == "analyzed"
    pairs = {(p.path_a, p.path_b) for p in _store(env).all_pairs(OWNER, PROJECT)}
    assert pairs == {("app/alpha.py", "app/beta.py"), ("app/alpha.py", "app/gamma.py"),
                     ("app/beta.py", "app/gamma.py")}


def test_a_queued_analysis_can_be_cancelled_but_a_running_one_cannot(env, sched):
    _index_before_any_scheduler(
        env, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    _index_before_any_scheduler(
        env, "/second", {"app/gamma.py": [0, 1], "app/delta.py": [0, 1]})
    scheduler = sched()
    runner = build_task_runner(env.task_store)
    env.vector_store.store.client = SlowQueryClient(
        env.vector_store.store.client, delay=1.5)

    scheduler.notify(OWNER, PROJECT)
    deadline = time.time() + 10
    while time.time() < deadline and not any(t["status"] == "running" for t in _tasks(env)):
        time.sleep(0.01)
    running = next(t for t in _tasks(env) if t["status"] == "running")
    scheduler.notify(OWNER, "/second")
    deadline = time.time() + 10
    while time.time() < deadline and len(_tasks(env)) < 2:
        time.sleep(0.01)
    queued = next(t for t in _tasks(env) if t["status"] == "pending")

    with pytest.raises(TaskNotCancellable):
        runner.cancel(running["id"])
    assert runner.cancel(queued["id"]) is True
    assert scheduler.wait_idle(40)

    by_id = {t["id"]: t for t in _tasks(env)}
    assert by_id[running["id"]]["status"] == "completed"
    assert by_id[queued["id"]]["status"] == "cancelled"
    assert _store(env).get_state(OWNER, "/second") is None, "a cancelled analysis never ran"


def test_a_failing_scheduler_never_fails_or_delays_the_index_job(env):
    class Broken:
        notifications = 0

        def notify(self, owner_user_id, project):
            raise RuntimeError("analysis is down")

    env.queue.analysis = Broken()
    started = time.perf_counter()
    job = env.index(PROJECT, {"app/alpha.py": [0, 1]})
    assert job.status.value == "completed"
    assert time.perf_counter() - started < 10


def test_the_task_row_records_what_the_analysis_did(env, sched):
    scheduler = sched()
    _index(env, scheduler, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    (row,) = _tasks(env)
    assert row["task_type"] == TASK_TYPE == "code_duplicates"
    assert row["status"] == "completed" and row["owner_user_id"] == OWNER
    assert row["params"]["project"] == PROJECT and row["params"]["job_id"]
    result = row["result"]
    assert result["status"] == "analyzed" and result["first_analysis"] is True
    assert result["changed_files"] == 2 and result["removed_files"] == 0
    assert result["scanned_chunks"] == 4  # both files are new, two chunks each
    assert [(p["path_a"], p["path_b"]) for p in result["new_pairs"]] == [
        ("app/alpha.py", "app/beta.py")]
    assert result["new_pairs"][0]["shared_chunks"] == 2
    assert result["summary"] == _store(env).get_state(OWNER, PROJECT).summary["text"]


def test_the_queue_hands_its_own_service_and_catalog_to_the_analysis_factory(env):
    """The analysis must share the writer's CodeKBService: its owner fence and write
    epochs are what keep a scan from straddling an index write."""
    from app.services.code_kb.jobs import CodeIndexJobQueue

    received = []
    marker = object()

    def factory(service, catalog_service):
        received.append((service, catalog_service))
        return marker

    queue = CodeIndexJobQueue(
        service=env.service, catalog_service=env.catalog, analysis_factory=factory)
    try:
        assert received == [(env.service, env.catalog)]
        assert queue.analysis is marker
    finally:
        queue.shutdown(timeout=5)
