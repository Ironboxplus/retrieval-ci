"""deploy/compute88/verify_embed_service.py: the acceptance run of arc-vllm-embed.service on .88.

The log check is tested on the line the real sitecustomize prints (a subprocess imports it, the way
the service's two processes do) and on ``ps`` output in the format the script asks for; the numeric
gate is not re-implemented: the script runs tests/test_vllm_client.py's own test against the service.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FP16_DIR = ROOT / "scripts" / "remote_embed" / "fp16_accumulation"


@pytest.fixture(scope="module")
def verify():
    spec = importlib.util.spec_from_file_location("verify_embed_service", ROOT / "deploy" / "compute88" / "verify_embed_service.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def real_sitecustomize_output() -> str:
    """What one Python process of the service prints: the real sitecustomize, imported at start-up."""
    done = subprocess.run([sys.executable, "-c", "pass"], capture_output=True, text=True, timeout=180,
                          env={**os.environ, "PYTHONPATH": str(FP16_DIR)})
    assert done.returncode == 0, done.stderr[-500:]
    return done.stderr


def test_the_real_sitecustomize_line_is_parsed(verify, real_sitecustomize_output):
    found = verify.fp16_pids(real_sitecustomize_output)
    assert len(found) == 1 and list(found.values()) == [True]


def test_both_processes_printing_true_pass(verify, real_sitecustomize_output):
    line = real_sitecustomize_output.strip().splitlines()[-1]
    api = line.replace(f"pid {next(iter(verify.fp16_pids(line)))}:", "pid 4100:")
    engine = line.replace(f"pid {next(iter(verify.fp16_pids(line)))}:", "pid 4177:")
    log = "\n".join(["INFO vllm starting", api, "INFO loading weights", engine, "INFO Application startup complete."])

    assert verify.judge_fp16_log(log, main_pid=4100, engine_pids=[4177]) == []


def test_a_false_value_a_missing_engine_and_a_missing_api_are_each_named(verify):
    template = "[fp16_accumulation] pid {pid}: torch.backends.cuda.matmul.allow_fp16_accumulation={value}"
    api, engine = template.format(pid=10, value=True), template.format(pid=11, value=True)

    assert verify.judge_fp16_log(api + "\n" + template.format(pid=11, value=False), 10, [11])[0].startswith("pid 11")
    assert any("engine" in p for p in verify.judge_fp16_log(api, 10, [11]))
    assert any("API" in p for p in verify.judge_fp16_log(engine, 10, [11]))
    assert any("no [fp16_accumulation] line" in p for p in verify.judge_fp16_log("INFO nothing", 10, [11]))


def test_a_restart_in_the_log_uses_the_last_value_of_each_pid(verify):
    template = "[fp16_accumulation] pid {pid}: torch.backends.cuda.matmul.allow_fp16_accumulation={value}"
    log = "\n".join([template.format(pid=10, value=False), template.format(pid=10, value=True)])
    assert verify.fp16_pids(log) == {10: True}


def test_engine_processes_are_the_descendants_named_enginecore(verify):
    ps = """\
    1     0 /sbin/init
 4100     1 /home/arc/.venvs/retrival-vllm/bin/python3 /home/arc/.venvs/retrival-vllm/bin/vllm serve /models/bge-m3 --port 18200
 4177  4100 VLLM::EngineCore
 4190  4177 VLLM::EngineCore
 4200     1 VLLM::EngineCore
 4300  4100 /home/arc/.venvs/retrival-vllm/bin/python3 -c from multiprocessing.resource_tracker import main;main(7)
"""
    assert verify.find_engine_pids(ps, 4100) == [4177, 4190]          # 4200 belongs to another service


def test_the_numeric_gate_is_the_client_test_pointed_at_the_service(verify):
    command, env = verify.numeric_gate_command("http://127.0.0.1:18200")
    assert command[-1] == "tests/test_vllm_client.py::test_vectors_match_the_stored_flagembedding_fp16_vectors"
    assert env["EMBEDDING__SERVICE_URL"] == "http://127.0.0.1:18200"
    # and that test exists with that name, so the script cannot point at nothing
    assert "def test_vectors_match_the_stored_flagembedding_fp16_vectors(" in (ROOT / "tests" / "test_vllm_client.py").read_text(encoding="utf-8")


def test_a_skipped_or_failed_gate_run_is_not_a_pass(verify):
    assert verify.numeric_gate_passed(0, "1 passed, 1 warning in 20.1s") is True
    assert verify.numeric_gate_passed(0, "1 skipped in 0.1s") is False       # the service was not running
    assert verify.numeric_gate_passed(1, "1 failed in 20.1s") is False
    assert verify.numeric_gate_passed(0, "") is False
