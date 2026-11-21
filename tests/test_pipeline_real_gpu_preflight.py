"""Opt-in real-GPU checks for the local pipeline preflight.

These tests intentionally execute the installed CUDA/PyTorch stack and the
real ``nvidia-smi`` binary.  They do not replace either dependency with a fake.
Run on the compute host with ``RETRIVAL_REAL_GPU_TEST=1``.
"""
from __future__ import annotations

import os

import pytest

from app.services.pipeline.runners import gpu_health


pytestmark = pytest.mark.skipif(
    os.environ.get("RETRIVAL_REAL_GPU_TEST") != "1",
    reason="requires an explicitly selected real CUDA compute host",
)


def test_local_gpu_health_reports_the_real_visible_cuda_device() -> None:
    import torch

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    assert 0 < free_bytes <= total_bytes

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first = visible.split(",", 1)[0].strip()
        assert first.isdigit(), "real test requires a numeric physical GPU mapping"
        expected_physical_id = int(first)
    else:
        expected_physical_id = 0

    # One MiB is deliberately below the live free-memory reading: this test
    # validates device discovery even when the host's NVML CLI is unavailable.
    assert expected_physical_id in gpu_health(
        None,
        min_free_mb=1,
        min_free_ratio=0.0,
    )
