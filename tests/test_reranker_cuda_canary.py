"""Contract tests for the no-write `.88` CUDA release verifier."""
from __future__ import annotations

import importlib.util
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def canary_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_reranker_cuda_canary.py"
    spec = importlib.util.spec_from_file_location("reranker_cuda_canary_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_canary_allows_only_the_private_88_compute_door(canary_module):
    assert canary_module.validate_release_target("http://10.100.100.88:18080") == (
        "http://10.100.100.88:18080"
    )
    for unsafe in (
        "http://127.0.0.1:18080",
        "https://10.100.100.88:18080",
        "http://10.100.100.88:8000",
        "http://token@example.invalid:18080",
        "http://example.invalid:18080",
    ):
        with pytest.raises(ValueError):
            canary_module.validate_release_target(unsafe)


def test_score_vector_gate_rejects_short_or_nonfinite_model_output(canary_module):
    assert canary_module._validate_scores([0.1, 0.2], 2, arm="managed") is None
    assert "expected 2 scores" in canary_module._validate_scores([0.1], 2, arm="managed")
    assert "non-finite" in canary_module._validate_scores([0.1, float("nan")], 2, arm="managed")


def test_runtime_gate_requires_shared_gpu_cap_and_offline_local_model(monkeypatch, tmp_path, canary_module):
    monkeypatch.setattr(canary_module, "_local_ipv4_addresses", lambda: {"10.100.100.88"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("BGE_RERANKER_PATH", str(tmp_path))
    monkeypatch.setenv("BGE_M3_PATH", str(tmp_path))

    assert canary_module._validate_runtime(
        memory_fraction=canary_module.RELEASE_MEMORY_FRACTION,
    ) == tmp_path
    with pytest.raises(ValueError, match="memory-fraction"):
        canary_module._validate_runtime(memory_fraction=0.20)
    monkeypatch.delenv("HF_HUB_OFFLINE")
    with pytest.raises(RuntimeError, match="OFFLINE"):
        canary_module._validate_runtime(
            memory_fraction=canary_module.RELEASE_MEMORY_FRACTION,
        )


def test_runtime_gate_rejects_any_host_other_than_88_before_cuda(monkeypatch, tmp_path, canary_module):
    monkeypatch.setattr(canary_module, "_local_ipv4_addresses", lambda: {"10.100.100.111"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("BGE_RERANKER_PATH", str(tmp_path))
    monkeypatch.setenv("BGE_M3_PATH", str(tmp_path))

    with pytest.raises(RuntimeError, match="must run on 10.100.100.88"):
        canary_module._validate_runtime(
            memory_fraction=canary_module.RELEASE_MEMORY_FRACTION,
        )


def test_gpu_preflight_requires_the_expected_88_gpu_uuid_and_headroom(monkeypatch, canary_module):
    result = SimpleNamespace(
        stdout=(
            "0, GPU-4ae88dce-1545-3658-b2b6-f08328c7d366, "
            "00000000:01:00.0, 12282, 2800\n"
        ),
    )
    monkeypatch.setattr(canary_module.subprocess, "run", lambda *_args, **_kwargs: result)

    gpu = canary_module._validate_gpu_preflight(max_canary_peak_mib=2400)

    assert gpu.index == 0
    assert gpu.uuid == canary_module.RELEASE_GPU_UUID
    assert gpu.free_mib == 2800

    result.stdout = (
        "0, GPU-4ae88dce-1545-3658-b2b6-f08328c7d366, "
        "00000000:01:00.0, 12282, 2500\n"
    )
    with pytest.raises(RuntimeError, match="free memory"):
        canary_module._validate_gpu_preflight(max_canary_peak_mib=2400)

    result.stdout = "0, GPU-unexpected, 00000000:01:00.0, 12282, 2800\n"
    with pytest.raises(RuntimeError, match="UUID"):
        canary_module._validate_gpu_preflight(max_canary_peak_mib=2400)


def test_gpu_process_reader_and_peach_guard_are_bound_to_approved_gpu0(monkeypatch, canary_module):
    calls = []
    result = SimpleNamespace(stdout="77, /srv/PeachObj/venv/bin/python, 4000\n")
    monkeypatch.setattr(
        canary_module.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or result,
    )

    processes = canary_module._gpu_processes()

    assert calls[0][:3] == ["nvidia-smi", "-i", "0"]
    assert canary_module._peach_processes(processes) == {77: 4000}


def test_managed_maintenance_requires_post_trim_idle_cache_within_the_release_budget(canary_module):
    mib = 1024 * 1024
    failures = canary_module._maintenance_failures(
        [
            {
                "after": {"allocated_bytes": 100 * mib, "reserved_bytes": 700 * mib},
                "trim_executed": True,
                "post_trim": {"allocated_bytes": 100 * mib, "reserved_bytes": 180 * mib},
            },
        ],
        max_post_trim_idle_cache_mib=64,
    )
    assert failures == ["managed iteration 1: post-trim idle CUDA cache 80.0 MiB exceeds 64 MiB"]


def test_canary_defers_bge_import_until_after_runtime_gpu_preflight():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_reranker_cuda_canary.py"
    module_ast = ast.parse(path.read_text(encoding="utf-8"))
    top_level_imports = [
        node.module
        for node in module_ast.body
        if isinstance(node, ast.ImportFrom)
    ]

    assert "core.indexing.embedders.bge_m3" not in top_level_imports
    assert "from core.indexing.embedders.bge_m3 import BgeM3Embedder" in path.read_text(
        encoding="utf-8",
    )
