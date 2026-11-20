#!/usr/bin/env python3
"""Acceptance run of arc-vllm-embed.service on .88, to run on that host after the unit is installed and started.

    python3 deploy/compute88/verify_embed_service.py [--base-url http://127.0.0.1:18200] [--unit arc-vllm-embed.service]

Three checks, each printed with its evidence; the exit code is 0 only if all pass:

1. fp16 accumulation: the service log since the unit's last start holds an
   ``[fp16_accumulation] pid N: ...allow_fp16_accumulation=True`` line from the API process (the unit's
   MainPID) and from at least one EngineCore process under it, and no line with another value. Either
   process missing means PYTHONPATH did not reach it and that process runs with fp32 accumulation.
2. the service serves the profile: ``vllm_client.check_service`` (directory names the profile's revision,
   ``max_model_len`` is the profile's).
3. the numbers: tests/test_vllm_client.py::test_vectors_match_the_stored_flagembedding_fp16_vectors run
   against this service (``EMBEDDING__SERVICE_URL``), i.e. the 48 stored production FlagEmbedding vectors of
   tests/fixtures/bge_m3_flagembedding_reference.json.gz: dense cos mean >= 0.9999 and min >= 0.998. The gate
   is that test's own; nothing is re-implemented here. The tokenizer comes from ``BGE_M3_PATH`` (the unit's
   model directory is used when the variable is unset).

Needs the repository's Python environment (pytest, numpy, httpx, transformers) on the host.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIT = "arc-vllm-embed.service"
DEFAULT_BASE_URL = "http://127.0.0.1:18200"
DEFAULT_TOKENIZER_DIR = "/home/arc/retrival-4090-migration/.cache/modelscope/BAAI/bge-m3"
GATE_TEST = "tests/test_vllm_client.py::test_vectors_match_the_stored_flagembedding_fp16_vectors"
FP16_LINE = re.compile(
    r"\[fp16_accumulation\] pid (\d+): torch\.backends\.cuda\.matmul\.allow_fp16_accumulation=(True|False)")


def fp16_pids(log_text: str) -> dict[int, bool]:
    """The last printed value of every process that printed an ``[fp16_accumulation]`` line."""
    return {int(pid): value == "True" for pid, value in FP16_LINE.findall(log_text)}


def judge_fp16_log(log_text: str, main_pid: int, engine_pids: list[int]) -> list[str]:
    """What is wrong with the log, one sentence each; an empty list is a pass."""
    printed = fp16_pids(log_text)
    problems = [f"pid {pid} printed allow_fp16_accumulation=False" for pid, value in printed.items() if not value]
    if not printed:
        problems.append("no [fp16_accumulation] line in the service log: PYTHONPATH did not reach any process")
    if main_pid not in printed:
        problems.append(f"API process pid {main_pid} printed no [fp16_accumulation] line")
    if not engine_pids:
        problems.append(f"no EngineCore process found under pid {main_pid}")
    elif not any(pid in printed for pid in engine_pids):
        problems.append(f"engine process (pids {engine_pids}) printed no [fp16_accumulation] line")
    return problems


def find_engine_pids(ps_output: str, main_pid: int) -> list[int]:
    """Descendants of ``main_pid`` whose command line is ``VLLM::EngineCore`` (``ps -eo pid=,ppid=,args=``)."""
    rows = {}
    for line in ps_output.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows[int(parts[0])] = (int(parts[1]), parts[2])
    def below(pid: int) -> bool:
        seen = set()
        while pid in rows and pid not in seen:
            seen.add(pid)
            pid = rows[pid][0]
            if pid == main_pid:
                return True
        return False
    return sorted(pid for pid, (_, args) in rows.items() if "EngineCore" in args and below(pid))


def numeric_gate_command(base_url: str) -> tuple[list[str], dict[str, str]]:
    env = {**os.environ, "EMBEDDING__SERVICE_URL": base_url, "HF_HUB_OFFLINE": "1", "NO_PROXY": "localhost,127.0.0.1"}
    if not env.get("BGE_M3_PATH") and Path(DEFAULT_TOKENIZER_DIR).is_dir():
        env["BGE_M3_PATH"] = DEFAULT_TOKENIZER_DIR
    return [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", GATE_TEST], env


def numeric_gate_passed(returncode: int, output: str) -> bool:
    """A skip (the service was not running) is not a pass."""
    return returncode == 0 and " passed" in output and "skipped" not in output


def _run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def _check_fp16(unit: str) -> bool:
    main_pid = int(_run(["systemctl", "--user", "show", "-p", "MainPID", "--value", unit]).strip())
    since = _run(["systemctl", "--user", "show", "-p", "ActiveEnterTimestamp", "--value", unit]).strip()
    if main_pid == 0 or not since:
        print(f"FAIL fp16: {unit} is not running")
        return False
    log = _run(["journalctl", "--user", "-u", unit, "--since", since, "--no-pager", "-o", "cat"])
    engines = find_engine_pids(_run(["ps", "-eo", "pid=,ppid=,args="]), main_pid)
    problems = judge_fp16_log(log, main_pid, engines)
    printed = fp16_pids(log)
    print(f"fp16: API pid {main_pid}, engine pids {engines}, printed {printed}")
    for problem in problems:
        print(f"FAIL fp16: {problem}")
    return not problems


def _check_service(base_url: str) -> bool:
    sys.path.insert(0, str(ROOT))
    from core.indexing.embedders import vllm_client
    from core.indexing.embedding_profile import get_profile

    try:
        entry = vllm_client.check_service(get_profile(), base_url)
    except vllm_client.ServiceError as exc:
        print(f"FAIL service: {exc}")
        return False
    print(f"service: root={entry.get('root')} max_model_len={entry.get('max_model_len')}")
    return True


def _check_numbers(base_url: str) -> bool:
    command, env = numeric_gate_command(base_url)
    done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    output = done.stdout + done.stderr
    passed = numeric_gate_passed(done.returncode, output)
    print(f"numbers: {'PASS' if passed else 'FAIL'}: {GATE_TEST}")
    print("\n".join(output.strip().splitlines()[-15:]))
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default=os.environ.get("EMBEDDING__SERVICE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--unit", default=DEFAULT_UNIT)
    args = parser.parse_args(argv)
    results = {"fp16": _check_fp16(args.unit), "service": _check_service(args.base_url)}
    if results["service"]:
        results["numbers"] = _check_numbers(args.base_url)
    print("RESULT " + " ".join(f"{name}={'ok' if ok else 'FAILED'}" for name, ok in results.items()))
    return 0 if results and all(results.values()) and len(results) == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
