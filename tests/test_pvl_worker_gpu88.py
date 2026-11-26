"""scripts/remote_pvl/pvl_worker.py on the real thing: the RTX 4070 SUPER of .88, a vLLM
server, the PaddleOCR-VL pipeline and the fixtures' PDFs. Nothing is replaced.

Runs only on .88 inside the PaddleOCR-VL environment, with the card idle:

    PVL_GPU88=1 PADDLE_PDX_CACHE_HOME=... PADDLE_PDX_MODEL_SOURCE=modelscope \\
        PVL_PADDLE_MODEL=<dir of PaddleOCR-VL-1.6> PVL_MINERU_MODEL=<dir of MinerU2.5-Pro-2605-1.2B> \\
        /home/arc/.venvs/retrival-pvl/bin/python -m pytest tests/test_pvl_worker_gpu88.py --noconftest -m gpu88

The model servers run in the server venv (``PVL_SERVER_VENV``, default /home/arc/.venvs/retrival-pvl-server).
(``--noconftest``: the project's conftest wants the backend's dependencies, which that
environment does not have.) One worker run takes several minutes: the model server starts
once and is shared by the tests that need a converted paper.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu88,
    pytest.mark.skipif(os.environ.get("PVL_GPU88") != "1", reason="needs the 4070 SUPER of .88 (set PVL_GPU88=1)"),
]

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "pvl"
WORKER = ROOT / "scripts" / "remote_pvl" / "pvl_worker.py"
SERVER_CONFIG = ROOT / "deploy" / "compute88" / "pvl_vllm.yaml"
FALLBACK_CONFIG = ROOT / "deploy" / "compute88" / "mineru_vllm.yaml"
VLLM = Path(os.environ.get("PVL_SERVER_VENV", "/home/arc/.venvs/retrival-pvl-server")) / "bin" / "vllm"
GOOD = ("plain", "table")  # pages the model converts and the checks accept
REFUSED = "formula"        # its first page has a tau where the PDF has a T


def server_command() -> str:
    return (f"{VLLM} serve {os.environ['PVL_PADDLE_MODEL']} --config {{config}} --port {{port}} --no-enable-prefix-caching "
            f"--chat-template {{root}}/deploy/compute88/PaddleOCR-VL-1.6-0.9B.jinja")


def fallback_args() -> list[str]:
    command = (f"{VLLM} serve {os.environ['PVL_MINERU_MODEL']} --config {{config}} --port {{port}} --no-enable-prefix-caching "
               "--logits-processors mineru_vl_utils:MinerULogitsProcessor")
    return ["--fallback-command", command, "--fallback-config", str(FALLBACK_CONFIG),
            "--fallback-model", "MinerU2.5-Pro-2605-1.2B"]


def worker_cmd(input_dir: Path, output_dir: Path, *extra: str) -> list[str]:
    return [sys.executable, str(WORKER), "--input", str(input_dir), "--output", str(output_dir), "--gpu", "0",
            "--recursive", "--server-config", str(SERVER_CONFIG), "--server-command", server_command(),
            "--max-concurrency", "64", "--progress-every", "1", *extra]


def worker_env() -> dict:
    env = dict(os.environ)
    env.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", MALLOC_ARENA_MAX="2")
    return env


RESIDENT_MODELS_MIB = 3500  # the always-on backend models hold ~2.5 GB of the card


def gpu_used_mib() -> int:
    out = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  text=True)
    return int(out.strip())


def processes_with(needle: str) -> list[int]:
    """Pids whose command line holds ``needle`` (read from /proc: no extra dependency)."""
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and int(entry.name) != os.getpid():
            try:
                if needle in (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace"):
                    found.append(int(entry.name))
            except OSError:
                pass
    return found


@pytest.fixture(scope="module")
def first_run(tmp_path_factory):
    """All three fixture PDFs through one worker; the output directory and the exit code."""
    root = tmp_path_factory.mktemp("pvl_gpu88")
    inputs, output = root / "in", root / "out"
    inputs.mkdir()
    for stem in (*GOOD, REFUSED):
        (inputs / f"{stem}.pdf").write_bytes((FIXTURES / f"{stem}.pdf").read_bytes())
    done = subprocess.run(worker_cmd(inputs, output), env=worker_env(), cwd=ROOT, timeout=1500,
                          capture_output=True, text=True)
    (root / "worker.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    return inputs, output, done


def test_good_papers_get_markdown_and_the_refused_one_only_a_report(first_run):
    inputs, output, done = first_run
    assert done.returncode == 1, done.stderr[-2000:]  # a paper was refused: the worker says so
    for stem in GOOD:
        text = (output / f"{stem}.md").read_text(encoding="utf-8")
        assert len(text) > 3000 and "<div" not in text.split("<table", 1)[0]
        report = json.loads((output / "_quality" / f"{stem}.json").read_text(encoding="utf-8"))
        assert report["status"] == "passed" and report["page_counts"]["fail"] == 0
        assert report["engine"]["model"].startswith("PaddleOCR-VL")
    assert not (output / f"{REFUSED}.md").exists()
    report = json.loads((output / "_quality" / f"{REFUSED}.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["failure_reasons"][0].startswith("page 1:")
    failed = [p for p in report["pages"] if p["status"] == "fail"]
    assert failed and failed[0]["retried"] and "blocks" in failed[0]["attempts"][0]
    done_list = (output / "done.rank0.txt").read_text(encoding="utf-8").splitlines()
    assert sorted(Path(line).stem for line in done_list) == sorted(GOOD)


def test_the_run_summary_records_speed_and_peaks(first_run):
    _, output, _ = first_run
    summary = json.loads((output / "_quality" / "_run.rank0.json").read_text(encoding="utf-8"))
    assert summary["pages"] == 9 and summary["pages_per_second"] > 0.05
    assert 4000 < summary["vram_mib"] < 12282  # the card, resident models included
    assert 500 < summary["worker_rss_mib"] < 8000 and 500 < summary["server_rss_mib"] < 8000
    assert summary["server_config"]["gpu-memory-utilization"] <= 0.5
    print("peaks:", {k: summary[k] for k in ("pages_per_second", "vram_mib", "worker_rss_mib", "server_rss_mib")})


def test_a_second_run_skips_what_is_done_and_never_starts_a_server(first_run, tmp_path):
    inputs, output, _ = first_run
    only_done = tmp_path / "in"
    only_done.mkdir()
    for stem in GOOD:
        (only_done / f"{stem}.pdf").write_bytes((inputs / f"{stem}.pdf").read_bytes())
    started = time.monotonic()
    done = subprocess.run(worker_cmd(only_done, output), env=worker_env(), cwd=ROOT, timeout=300,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "worker.done ok=0 fail=0 skipped=2" in done.stderr + done.stdout
    assert time.monotonic() - started < 60  # no model was loaded


def test_kill_minus_nine_on_the_worker_takes_its_model_server_down_with_it(tmp_path):
    inputs, output = tmp_path / "in", tmp_path / "out"
    inputs.mkdir()
    (inputs / "plain.pdf").write_bytes((FIXTURES / "plain.pdf").read_bytes())
    worker = subprocess.Popen(worker_cmd(inputs, output), env=worker_env(), cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    marker = str(output)  # the server's command line names its effective config under the output directory
    try:
        deadline = time.monotonic() + 300
        while not processes_with(marker + "/_quality") and time.monotonic() < deadline and worker.poll() is None:
            time.sleep(1)
        servers = [pid for pid in processes_with(marker) if pid != worker.pid]
        assert servers, "the model server never appeared"
        os.kill(worker.pid, signal.SIGKILL)
        worker.wait(timeout=30)
        deadline = time.monotonic() + 60
        while [pid for pid in servers if Path(f"/proc/{pid}").exists()] and time.monotonic() < deadline:
            time.sleep(1)
        assert not [pid for pid in servers if Path(f"/proc/{pid}").exists()], "the model server outlived its worker"
        assert not processes_with(marker)
        # and its engine process: the memory it held on the card is given back (what stays is the resident models)
        deadline = time.monotonic() + 60
        while gpu_used_mib() > RESIDENT_MODELS_MIB and time.monotonic() < deadline:
            time.sleep(2)
        assert gpu_used_mib() <= RESIDENT_MODELS_MIB, "the model server's engine still holds GPU memory"
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)


def test_a_page_paddleocr_vl_gets_wrong_is_read_by_mineru_and_its_paper_is_written(tmp_path):
    """The formula paper's first page has a tau where the PDF has a T and PaddleOCR-VL writes it twice;
    with the second model configured the same page is re-read by MinerU and the paper stands."""
    inputs, output = tmp_path / "in", tmp_path / "out"
    inputs.mkdir()
    (inputs / f"{REFUSED}.pdf").write_bytes((FIXTURES / f"{REFUSED}.pdf").read_bytes())
    done = subprocess.run(worker_cmd(inputs, output, *fallback_args()), env=worker_env(), cwd=ROOT, timeout=1500,
                          capture_output=True, text=True)
    log = done.stdout + done.stderr
    assert done.returncode == 0, log[-3000:]
    assert "fallback.start" in log and "formula_witness.done" in log
    report = json.loads((output / "_quality" / f"{REFUSED}.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    first = report["pages"][0]
    assert [a["engine"] for a in first["attempts"]] == ["paddleocr_vl", "paddleocr_vl", "mineru"] and first["engine"] == "mineru"
    assert len((output / f"{REFUSED}.md").read_text(encoding="utf-8")) > 3000
    assert not processes_with(str(output)), "a model server outlived the worker"
