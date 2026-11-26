"""Safe, process-wide maintenance for PyTorch's CUDA caching allocator.

Model inference may run concurrently, but allocator trimming is only safe when no
participating model is using CUDA. The coordinator therefore never serializes normal
inference; it only blocks new model operations briefly while ``empty_cache`` releases
already-unused blocks.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import threading
from typing import Callable, Iterator


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CudaAllocatorSnapshot:
    """Allocator counters for one CUDA device at one instant."""

    allocated_bytes: int
    reserved_bytes: int
    inactive_split_bytes: int
    num_ooms: int
    num_alloc_retries: int

    @property
    def unused_cache_bytes(self) -> int:
        """Reserved memory that is not backing a live allocation."""
        return max(0, self.reserved_bytes - self.allocated_bytes)


class GpuAllocatorCoordinator:
    """Gate cache trimming without serializing normal CUDA model work."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active_operations = 0
        self._trimming = False
        self._pending_trim: Callable[[], None] | None = None

    @contextmanager
    def activity(self) -> Iterator[None]:
        """Mark a model operation that must not race allocator trimming."""
        with self._condition:
            while self._trimming:
                self._condition.wait()
            self._active_operations += 1
        try:
            yield
        finally:
            scheduled_trim: Callable[[], None] | None = None
            with self._condition:
                self._active_operations -= 1
                scheduled_trim = self._take_pending_trim_locked()
                self._condition.notify_all()
            if scheduled_trim is not None:
                self._run_trim(scheduled_trim)

    def trim_if_idle(self, trim: Callable[[], None]) -> bool:
        """Run ``trim`` only when every participating model operation is idle."""
        with self._condition:
            if self._active_operations or self._trimming:
                return False
            self._trimming = True
        try:
            trim()
            return True
        finally:
            with self._condition:
                self._trimming = False
                self._condition.notify_all()

    def request_trim(self, trim: Callable[[], None]) -> bool:
        """Run now when idle, or retain one release request for the next idle point.

        A release request represents maintenance, not inference. Coalescing it is
        intentional: returning cache once after concurrent CUDA work drains is
        enough, while running every queued callback would only add allocator churn.
        """
        with self._condition:
            if self._active_operations or self._trimming:
                self._pending_trim = trim
                return False
            self._trimming = True
        self._run_trim(trim)
        return True

    def _take_pending_trim_locked(self) -> Callable[[], None] | None:
        if self._active_operations or self._trimming or self._pending_trim is None:
            return None
        trim = self._pending_trim
        self._pending_trim = None
        self._trimming = True
        return trim

    def _run_trim(self, trim: Callable[[], None]) -> None:
        try:
            trim()
        finally:
            scheduled_trim: Callable[[], None] | None = None
            with self._condition:
                self._trimming = False
                scheduled_trim = self._take_pending_trim_locked()
                self._condition.notify_all()
            if scheduled_trim is not None:
                self._run_trim(scheduled_trim)


_GPU_ALLOCATOR_COORDINATOR = GpuAllocatorCoordinator()


def is_cuda_device(device: str) -> bool:
    """Recognise the CUDA spellings accepted by the model constructors."""
    return device == "cuda" or (
        device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
    )


def configure_cuda_memory_fraction(device: str, fraction: float | None) -> bool:
    """Apply an opt-in process-wide allocator ceiling before models load.

    PyTorch exposes this only per process/device, not per model. The caller must
    invoke it before BGE-M3 or the reranker creates CUDA allocations.
    """
    if fraction is None or not is_cuda_device(device):
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        device_index = 0 if device == "cuda" else int(device.removeprefix("cuda:"))
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
        return True
    except (ImportError, RuntimeError):
        return False


@contextmanager
def gpu_model_activity(device: str) -> Iterator[None]:
    """Register CUDA work so a concurrent cache trim cannot interrupt it."""
    if not is_cuda_device(device):
        yield
        return
    with _GPU_ALLOCATOR_COORDINATOR.activity():
        yield


def capture_cuda_allocator_snapshot(device: str) -> CudaAllocatorSnapshot | None:
    """Read counters without creating a CUDA context on CPU deployments."""
    if not is_cuda_device(device):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return _capture_cuda_allocator_snapshot(torch, device)
    except (ImportError, RuntimeError):
        return None


def _capture_cuda_allocator_snapshot(torch, device: str) -> CudaAllocatorSnapshot:
    stats = torch.cuda.memory_stats(device)
    return CudaAllocatorSnapshot(
        allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),
        inactive_split_bytes=int(stats.get("inactive_split_bytes.all.current", 0)),
        num_ooms=int(stats.get("num_ooms", 0)),
        num_alloc_retries=int(stats.get("num_alloc_retries", 0)),
    )


def trim_cuda_cache_and_capture_if_idle(
    device: str,
) -> tuple[bool, CudaAllocatorSnapshot | None]:
    """Trim unused cache and capture its exact post-trim state under the same gate."""
    if not is_cuda_device(device):
        return False, None
    try:
        import torch

        if not torch.cuda.is_available():
            return False, None
    except ImportError:
        return False, None

    snapshot: CudaAllocatorSnapshot | None = None

    succeeded = False

    def trim_and_capture() -> None:
        nonlocal snapshot, succeeded
        try:
            torch.cuda.empty_cache()
            snapshot = _capture_cuda_allocator_snapshot(torch, device)
            succeeded = True
        except Exception:
            # This callback can run later on the thread that releases the last
            # CUDA operation, so it must be self-contained and fail-open there.
            _logger.warning("cuda allocator trim failed", exc_info=True)

    trimmed = _GPU_ALLOCATOR_COORDINATOR.request_trim(trim_and_capture)
    return (trimmed and succeeded), snapshot


def trim_cuda_cache_if_idle(device: str) -> bool:
    """Return unused cached blocks only if no participating CUDA inference is active."""
    trimmed, _snapshot = trim_cuda_cache_and_capture_if_idle(device)
    return trimmed


def request_cuda_cache_trim_if_idle(device: str) -> bool:
    """Request fail-open cache maintenance after one completed CUDA operation."""
    try:
        return trim_cuda_cache_if_idle(device)
    except Exception:
        # Embedder and reranker callers use this from ``finally`` blocks. A
        # maintenance regression must never change a completed result or hide
        # the model exception that caused the cleanup path to run.
        _logger.warning("CUDA allocator maintenance request failed", exc_info=True)
        return False
