"""The daemon is a child of the shim that started it and outlives that shim by design.

On POSIX a child that exited but was never wait()ed on is a zombie: ``os.kill(pid, 0)`` still
succeeds for it, so ``retrieval-mcp stop`` reported "authenticated shutdown accepted but daemon
did not exit" while the shim that spawned the daemon was still alive (Linux only; Windows
asks for the exit code).  Real child processes, no mocks.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcp_plugin"))

import retrieval_mcp_daemon as daemon  # noqa: E402


def _exit_quickly() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=os.name != "nt")


def _wait_until_exited_without_reaping(child: subprocess.Popen) -> None:
    """Block until the child has exited, leaving it unreaped (a zombie on POSIX)."""
    if hasattr(os, "waitid"):
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
    else:  # Windows has no zombies: an exited process whose handle is still held
        child.wait()


def test_an_exited_child_the_parent_has_not_reaped_does_not_count_as_alive():
    child = _exit_quickly()
    try:
        _wait_until_exited_without_reaping(child)

        assert daemon._pid_exists(child.pid) is False
        assert daemon._wait_for_exit(child.pid, 1.0) is True
    finally:
        child.wait()


def test_a_running_process_still_counts_as_alive():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert daemon._pid_exists(child.pid) is True
    finally:
        child.kill()
        child.wait()


def test_the_launcher_reaps_the_daemon_it_spawned_when_the_daemon_exits():
    child = _exit_quickly()

    daemon._reap_in_background(child)

    deadline = time.monotonic() + 5
    while child.returncode is None and time.monotonic() < deadline:
        time.sleep(0.02)  # nothing here calls poll() or wait(): only the reaper may set it
    assert child.returncode == 0
