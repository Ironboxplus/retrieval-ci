"""A long duplicate scan must not hold the owner's fence for its whole length.

`CodeKBService.scope_snapshot` is the per-owner mutex that every search and every index
write takes. `find_duplicates` used to hold it across the entire scan: measured on `.88`,
scanning this repository (17,839 chunks) blocks that owner's searches and writes for
over two minutes. The scan now takes the fence per query batch only, and detects a
write that landed between batches by comparing a per-scope write epoch, discarding what
it read instead of reporting a mix of old and new chunks.

Real Qdrant; the Qdrant client is wrapped only to slow each query down or to perform
a real index write at an exact point of the scan.
"""
from __future__ import annotations

import threading
import time

import pytest

from app.services.code_kb import duplicates
from app.services.code_kb.duplicates import ScopeChangedDuringScan, scan_changed_files
from app.services.code_kb.schemas import CodeDuplicateScanRequest
from tests.code_analysis_support import (
    OWNER, PROJECT, SlowQueryClient, env, make_request, vector_store)  # noqa: F401


def _slow(env, **kwargs) -> SlowQueryClient:
    client = SlowQueryClient(env.vector_store.store.client, **kwargs)
    env.vector_store.store.client = client
    return client


def _write(env, project, files, owner=OWNER):
    env.service.index(make_request(project, files, []), owner_user_id=owner)


def _timed(run, timing: dict):
    """Record how long the whole scan took, so the fence limit scales with the machine."""
    def go():
        started = time.perf_counter()
        try:
            run()
        finally:
            timing["scan"] = time.perf_counter() - started
    return go


def _wait_limit(timing: dict) -> float:
    """Holding the fence for the whole scan would wait about as long as the scan; one
    batch is a small fraction of it. Half the scan just run separates the two on a fast
    and on a loaded machine alike."""
    return max(1.0, timing["scan"] / 2)


def _fence_wait_while(scan_thread: threading.Thread, env, *, samples: int = 4) -> list[float]:
    """How long this owner's next fence acquisitions (what a search takes) wait, sampled
    while the scan is demonstrably still running."""
    waits = []
    for _ in range(samples):
        started = time.perf_counter()
        with env.service.scope_snapshot(owner_user_id=OWNER):
            pass
        waits.append(time.perf_counter() - started)
        assert scan_thread.is_alive(), "the scan ended before the fence was measured"
        time.sleep(0.05)
    return waits


def test_a_manual_duplicate_scan_does_not_block_the_owners_fence(env, monkeypatch):
    monkeypatch.setattr(duplicates, "_QUERY_BATCH", 1)
    chunks = list(range(8))
    _write(env, PROJECT, {"app/alpha.py": chunks, "app/beta.py": chunks})
    client = _slow(env, delay=0.3)  # 16 batches: a scan of about five seconds

    outcome: dict = {}
    timing: dict = {}
    scan = threading.Thread(
        target=_timed(lambda: outcome.update(response=env.service.find_duplicates(
            CodeDuplicateScanRequest(project=PROJECT), owner_user_id=OWNER)), timing))
    scan.start()
    try:
        while client.queries < 2:
            time.sleep(0.01)
        waits = _fence_wait_while(scan, env)
    finally:
        scan.join(30)

    # One batch holds the fence for ~0.3 s; the whole scan would hold it for ~5 s.
    limit = _wait_limit(timing)
    assert max(waits) < limit, (
        f"owner fence waits while a scan runs: {waits} (limit {limit:.2f}s, scan {timing['scan']:.2f}s)")
    response = outcome["response"]
    assert [(p.path_a, p.path_b) for p in response.pairs] == [("app/alpha.py", "app/beta.py")]
    assert response.pairs[0].shared_chunks == 8


def test_an_incremental_scan_does_not_block_the_owners_fence(env, monkeypatch):
    monkeypatch.setattr(duplicates, "_QUERY_BATCH", 1)
    chunks = list(range(8))
    _write(env, PROJECT, {"app/alpha.py": chunks, "app/beta.py": chunks})
    client = _slow(env, delay=0.3)

    outcome: dict = {}
    timing: dict = {}
    scan = threading.Thread(target=_timed(lambda: outcome.update(result=scan_changed_files(
        env.vector_store, owner_user_id=OWNER, project=PROJECT,
        changed_paths=["app/alpha.py"],
        guard=env.service.scan_guard(owner_user_id=OWNER, projects=[PROJECT]))), timing))
    scan.start()
    try:
        while client.queries < 2:
            time.sleep(0.01)
        waits = _fence_wait_while(scan, env)
    finally:
        scan.join(30)

    limit = _wait_limit(timing)
    assert max(waits) < limit, (
        f"owner fence waits while a scan runs: {waits} (limit {limit:.2f}s, scan {timing['scan']:.2f}s)")
    assert outcome["result"].pairs[0].shared_chunks == 8


def test_a_write_between_batches_makes_the_manual_scan_start_over(env, monkeypatch):
    monkeypatch.setattr(duplicates, "_QUERY_BATCH", 1)
    _write(env, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})

    def write_once(query_number: int) -> None:
        if query_number == 1:
            # A real index write, between the scan's first and second batch.
            _write(env, PROJECT, {"app/gamma.py": [0, 1]})

    client = _slow(env, on_query=write_once)
    response = env.service.find_duplicates(
        CodeDuplicateScanRequest(project=PROJECT), owner_user_id=OWNER)

    pairs = {(p.path_a, p.path_b) for p in response.pairs}
    # The report describes the scope as it is after the write, not a mix.
    assert pairs == {("app/alpha.py", "app/beta.py"), ("app/alpha.py", "app/gamma.py"),
                     ("app/beta.py", "app/gamma.py")}
    assert client.queries > 6, "the scan should have been repeated after the write"
    assert response.notice is None


def test_a_scan_that_keeps_being_overtaken_says_so_instead_of_claiming_a_clean_result(
        env, monkeypatch):
    monkeypatch.setattr(duplicates, "_QUERY_BATCH", 1)
    _write(env, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})
    counter = {"n": 0}

    def write_every_attempt(query_number: int) -> None:
        counter["n"] += 1
        _write(env, PROJECT, {f"app/churn{counter['n']}.py": [5, 6]})

    _slow(env, on_query=write_every_attempt)
    response = env.service.find_duplicates(
        CodeDuplicateScanRequest(project=PROJECT), owner_user_id=OWNER)

    assert response.notice and "while" in response.notice and "index" in response.notice
    assert response.pairs == [] and response.truncated is False


def test_a_write_to_another_scope_of_the_owner_does_not_restart_a_scan(env, monkeypatch):
    monkeypatch.setattr(duplicates, "_QUERY_BATCH", 1)
    _write(env, PROJECT, {"app/alpha.py": [0, 1], "app/beta.py": [0, 1]})

    def write_elsewhere(query_number: int) -> None:
        if query_number == 1:
            _write(env, "/some-other-repo", {"x.py": [2, 3]})

    client = _slow(env, on_query=write_elsewhere)
    response = env.service.find_duplicates(
        CodeDuplicateScanRequest(project=PROJECT), owner_user_id=OWNER)

    assert client.queries == 4, "an unrelated scope's write must not repeat the scan (4 chunks, 1 per batch)"
    assert [(p.path_a, p.path_b) for p in response.pairs] == [("app/alpha.py", "app/beta.py")]


def test_the_guard_notices_a_write_to_its_scope_and_ignores_other_scopes(env):
    _write(env, PROJECT, {"app/alpha.py": [0, 1]})
    guard = env.service.scan_guard(owner_user_id=OWNER, projects=[PROJECT])
    with guard.batch():
        pass
    guard.finish()

    _write(env, "/some-other-repo", {"x.py": [2]})
    guard.finish()  # still fine: another scope

    _write(env, PROJECT, {"app/beta.py": [0, 1]})
    with pytest.raises(ScopeChangedDuringScan):
        with guard.batch():
            pass
    with pytest.raises(ScopeChangedDuringScan):
        guard.finish()


def test_the_guard_notices_a_scope_deletion(env):
    _write(env, PROJECT, {"app/alpha.py": [0, 1]})
    guard = env.service.scan_guard(owner_user_id=OWNER, projects=[PROJECT])
    env.service.delete_project(PROJECT, owner_user_id=OWNER)
    with pytest.raises(ScopeChangedDuringScan):
        guard.finish()


def test_the_guard_notices_a_write_to_a_scope_spelling(env):
    _write(env, "/Repo", {"app/alpha.py": [0, 1]})
    guard = env.service.scan_guard(owner_user_id=OWNER, projects=["/repo", "/Repo"])
    _write(env, "/Repo", {"app/beta.py": [0, 1]})
    with pytest.raises(ScopeChangedDuringScan):
        guard.finish()


def test_the_guard_refuses_a_scope_that_is_no_longer_a_completed_snapshot(env):
    _write(env, PROJECT, {"app/alpha.py": [0, 1]})
    state = {"complete": True}
    guard = env.service.scan_guard(
        owner_user_id=OWNER, projects=[PROJECT], still_complete=lambda: state["complete"])
    guard.finish()
    state["complete"] = False
    with pytest.raises(ScopeChangedDuringScan):
        guard.finish()


def test_an_index_write_waits_for_a_running_batch_but_not_for_the_whole_scan(env, monkeypatch):
    """The other direction: writers are not starved either."""
    monkeypatch.setattr(duplicates, "_QUERY_BATCH", 1)
    chunks = list(range(8))
    _write(env, PROJECT, {"app/alpha.py": chunks, "app/beta.py": chunks})
    client = _slow(env, delay=0.3)

    timing: dict = {}
    scan = threading.Thread(target=_timed(lambda: env.service.find_duplicates(
        CodeDuplicateScanRequest(project=PROJECT), owner_user_id=OWNER), timing))
    scan.start()
    try:
        while client.queries < 2:
            time.sleep(0.01)
        started = time.perf_counter()
        _write(env, "/some-other-repo", {"x.py": [2, 3]})
        waited = time.perf_counter() - started
    finally:
        scan.join(60)
    limit = _wait_limit(timing)
    assert waited < limit, (
        f"an index write waited {waited:.2f}s behind one scan (limit {limit:.2f}s, scan {timing['scan']:.2f}s)")
