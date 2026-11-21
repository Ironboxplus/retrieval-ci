"""Real configuration contract for the Marker 2 batch worker.

These tests exercise the production worker's actual configuration builder and
the dependency file used to create the isolated conversion environment.  They
do not replace Marker with a fake implementation; full binary/PDF execution is
covered by the post-install smoke gate.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.remote_marker.marker_worker import (
    build_marker_config,
    install_windows_surya_process_cleanup_guard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_marker_worker_defaults_to_balanced_without_disabling_vlm():
    assert build_marker_config(mode="balanced", disable_ocr=False) == {
        "mode": "balanced",
    }


def test_marker_worker_only_disables_vlm_when_explicitly_requested():
    assert build_marker_config(mode="fast", disable_ocr=True) == {
        "mode": "fast",
        "disable_ocr": True,
    }


def test_marker_worker_dependency_contract_is_marker_2():
    requirements = {
        line.strip()
        for line in (PROJECT_ROOT / "requirements-marker.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "marker-pdf==2.0.0" in requirements
    assert "surya-ocr==0.22.1" in requirements
    assert "psutil>=7.2.2,<8" in requirements


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific Surya compatibility")
def test_windows_surya_cleanup_guard_accepts_installed_surya_generation():
    """The shared worker must remain runnable in the retained Marker 1 env."""
    assert isinstance(install_windows_surya_process_cleanup_guard(), bool)


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific Surya cleanup regression")
def test_windows_surya_cleanup_terminates_real_child_without_interrupting_parent():
    """The Surya cleanup hook must not use signal 0 as a Windows liveness probe.

    ``signal.CTRL_C_EVENT`` is numeric zero on Windows.  Surya 0.22.1 used
    ``os.kill(pid, 0)`` after terminating llama-server, which delivered a real
    console interrupt and could terminate the Codex host that launched Marker.
    Exercise the installed cleanup hook against a real child process so this
    remains a process-lifecycle test rather than a mocked call assertion.
    """
    try:
        from surya.inference.backends import spawn
    except ModuleNotFoundError:
        pytest.skip("requires the isolated Marker 2 runtime")

    assert install_windows_surya_process_cleanup_guard() is True

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        spawn._stop_process(child.pid, "marker-v2-cleanup-regression")
        child.wait(timeout=10)
        assert child.returncode is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
