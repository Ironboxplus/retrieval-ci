"""A stage that ends by an exception must not leave its processes behind (2026-11-16 incident).

task 1309's extract stage failed because its log line (the sink) raised, but ``run_local`` only
stopped the child when the orchestrator's abort Event was set, so the PaddleOCR-VL worker and its
vLLM server kept the GPU, untracked, and the dead worker then stayed ``<defunct>`` because nobody
``wait()``ed for it. Real child process + real grandchild, nothing faked.
"""
from __future__ import annotations

import sys
import time

import psutil
import pytest

from app.services.pipeline.runners import run_local

# The child starts a grandchild (the "vLLM server" of the worker), reports both pids and idles.
TREE = (
    "import os, subprocess, sys, time\n"
    "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
    "open(sys.argv[1], 'w').write(f'{os.getpid()} {grandchild.pid}')\n"
    "print('tree ready', flush=True)\n"
    "time.sleep(120)\n"
)


def _alive(pid: int) -> bool:
    """Running, not merely a zombie someone forgot to reap."""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait(predicate, timeout, step=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


@pytest.fixture
def pids(tmp_path):
    """pid file of the tree; whatever a failing test leaves running is killed by pid afterwards."""
    pid_file = tmp_path / "tree.pids"
    yield pid_file
    if pid_file.exists():
        for pid in map(int, pid_file.read_text().split()):
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass


def test_a_sink_that_raises_stops_the_whole_process_tree(tmp_path, pids):
    seen = []

    def sink(line, level="info"):
        seen.append(line)
        raise RuntimeError("database is locked")

    with pytest.raises(RuntimeError, match="database is locked"):
        run_local([sys.executable, "-c", TREE, str(pids)], tmp_path, sink)

    assert seen == ["tree ready"]
    child, grandchild = map(int, pids.read_text().split())
    assert _wait(lambda: not _alive(child) and not _alive(grandchild), timeout=15), (
        f"left running after the sink raised: child={_alive(child)} grandchild={_alive(grandchild)}")
    assert not psutil.pid_exists(child), "the child was killed but never reaped (defunct)"


def test_the_original_error_wins_when_the_child_is_already_gone(tmp_path):
    """Stopping a child that exited by itself must not replace the sink's error with its own."""
    def sink(line, level="info"):
        raise ValueError("sink broke")

    with pytest.raises(ValueError, match="sink broke"):
        run_local([sys.executable, "-c", "print('one line')"], tmp_path, sink)


def test_a_normal_failure_still_reports_the_exit_code(tmp_path):
    with pytest.raises(RuntimeError, match=r"command failed \(exit 3\)"):
        run_local([sys.executable, "-c", "import sys; print('x'); sys.exit(3)"], tmp_path, lambda *a, **k: None)
