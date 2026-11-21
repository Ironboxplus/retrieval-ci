"""Regression tests for coordinated CUDA allocator maintenance."""
from __future__ import annotations

import sys
import threading
from types import SimpleNamespace


def test_cuda_device_ordinal_is_classified_as_cuda():
    from core.gpu_memory import is_cuda_device

    assert is_cuda_device("cuda") is True
    assert is_cuda_device("cuda:0") is True
    assert is_cuda_device("cuda:7") is True
    assert is_cuda_device("cpu") is False
    assert is_cuda_device("cuda-enabled") is False
    assert is_cuda_device("cuda:ordinal") is False


def test_configure_cuda_memory_fraction_uses_selected_cuda_device(monkeypatch):
    from core import gpu_memory

    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            set_per_process_memory_fraction=lambda fraction, *, device: calls.append(
                (fraction, device),
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert gpu_memory.configure_cuda_memory_fraction("cuda:0", 0.35) is True
    assert calls == [(0.35, 0)]
    assert gpu_memory.configure_cuda_memory_fraction("cpu", 0.35) is False


def test_trim_is_skipped_while_a_model_operation_is_active():
    from core.gpu_memory import GpuAllocatorCoordinator

    coordinator = GpuAllocatorCoordinator()
    entered = threading.Event()
    release = threading.Event()

    def active_operation() -> None:
        with coordinator.activity():
            entered.set()
            assert release.wait(timeout=1)

    worker = threading.Thread(target=active_operation)
    worker.start()
    assert entered.wait(timeout=1)

    trimmed: list[bool] = []
    assert coordinator.trim_if_idle(lambda: trimmed.append(True)) is False
    assert trimmed == []

    release.set()
    worker.join(timeout=1)
    assert not worker.is_alive()


def test_trim_runs_after_all_model_operations_are_idle():
    from core.gpu_memory import GpuAllocatorCoordinator

    coordinator = GpuAllocatorCoordinator()
    trimmed: list[bool] = []

    assert coordinator.trim_if_idle(lambda: trimmed.append(True)) is True
    assert trimmed == [True]


def test_second_trim_is_skipped_while_the_first_trim_is_running():
    from core.gpu_memory import GpuAllocatorCoordinator

    coordinator = GpuAllocatorCoordinator()
    trim_started = threading.Event()
    release = threading.Event()

    def slow_trim() -> None:
        trim_started.set()
        assert release.wait(timeout=1)

    worker = threading.Thread(target=lambda: coordinator.trim_if_idle(slow_trim))
    worker.start()
    assert trim_started.wait(timeout=1)

    assert coordinator.trim_if_idle(lambda: None) is False

    release.set()
    worker.join(timeout=1)
    assert not worker.is_alive()


def test_requested_trim_runs_once_when_the_last_model_operation_becomes_idle():
    """A cache-release request must not be lost just because CUDA is busy now."""
    from core.gpu_memory import GpuAllocatorCoordinator

    coordinator = GpuAllocatorCoordinator()
    entered = threading.Event()
    release = threading.Event()
    trimmed = threading.Event()

    def active_operation() -> None:
        with coordinator.activity():
            entered.set()
            assert release.wait(timeout=1)

    worker = threading.Thread(target=active_operation)
    worker.start()
    assert entered.wait(timeout=1)

    assert coordinator.request_trim(lambda: trimmed.set()) is False
    assert not trimmed.is_set()

    release.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert trimmed.wait(timeout=1)


def test_trim_failure_is_fail_open(monkeypatch):
    """Allocator maintenance must never turn a completed search into a 500."""
    from core import gpu_memory

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: (_ for _ in ()).throw(RuntimeError("maintenance failed")),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert gpu_memory.trim_cuda_cache_and_capture_if_idle("cuda:0") == (False, None)


def test_requested_trim_failure_is_fail_open(monkeypatch):
    """A caller's finally block must be safe even if the lower helper regresses."""
    from core import gpu_memory

    monkeypatch.setattr(
        gpu_memory,
        "trim_cuda_cache_if_idle",
        lambda _device: (_ for _ in ()).throw(RuntimeError("maintenance failed")),
    )

    assert gpu_memory.request_cuda_cache_trim_if_idle("cuda:0") is False
