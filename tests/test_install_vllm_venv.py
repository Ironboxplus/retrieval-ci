"""deploy/compute88/install_vllm_venv.sh: the vLLM environment of the model services, installed from exact pins.

Real bash runs the real script. What needs a network (the install) is not run here; what must hold
before any download is: a proxy is required and must answer, the retired venv of PaddleOCR-VL's server
is never touched, nothing is created when a check fails. ``--check`` is run against the WSL venv the
pins were taken from, so the lock file and the script's own verification are tested on real data.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "compute88" / "install_vllm_venv.sh"
LOCK = ROOT / "deploy" / "compute88" / "vllm_venv_requirements.txt"
BASH = shutil.which("bash")
WSL = shutil.which("wsl.exe") or shutil.which("wsl")
UNREACHABLE_PROXY = "http://127.0.0.1:9"                 # port 9 refuses connections

needs_bash = pytest.mark.skipif(BASH is None, reason="no bash")


def run(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = {k: v for k, v in os.environ.items() if k.upper() not in ("VLLM_VENV_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
                                                                      "ALL_PROXY", "NO_PROXY")}
    env.update(env_extra or {})
    return subprocess.run([BASH, str(SCRIPT).replace("\\", "/"), *args], capture_output=True, text=True,
                          timeout=120, env=env)


def test_the_lock_file_pins_every_package_exactly_and_names_the_verified_versions():
    lines = [l for l in LOCK.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
    assert len(lines) > 150
    assert all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][^\s=<>~!]*", line) for line in lines), \
        [l for l in lines if not re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][^\s=<>~!]*", l)]
    assert "vllm==0.33.0" in lines and "torch==2.14.0" in lines and "transformers==5.20.2" in lines
    assert not any(l.lower().startswith("bge-m3-sparse-plugin") for l in lines)      # installed from the repo


@needs_bash
def test_a_proxy_is_required_and_nothing_is_created_without_one(tmp_path):
    venv = tmp_path / "venv"
    done = run("--venv", str(venv).replace("\\", "/"))

    assert done.returncode != 0
    assert "proxy" in done.stderr.lower() and "required" in done.stderr.lower()
    assert not venv.exists()


@needs_bash
def test_an_unreachable_proxy_fails_the_run_before_anything_is_created(tmp_path):
    venv = tmp_path / "venv"
    done = run("--venv", str(venv).replace("\\", "/"), "--proxy", UNREACHABLE_PROXY)

    assert done.returncode != 0
    assert UNREACHABLE_PROXY in done.stderr and "not" in done.stderr.lower()
    assert not venv.exists()                                        # no direct connection was tried either


@needs_bash
def test_the_proxy_can_come_from_the_environment_and_is_then_checked_the_same_way(tmp_path):
    venv = tmp_path / "venv"
    done = run("--venv", str(venv).replace("\\", "/"), env_extra={"VLLM_VENV_PROXY": UNREACHABLE_PROXY})

    assert done.returncode != 0 and UNREACHABLE_PROXY in done.stderr and not venv.exists()


@needs_bash
@pytest.mark.parametrize("proxy", ["socks5://127.0.0.1:10808", "127.0.0.1:10808", "ftp://x"])
def test_only_http_proxies_are_accepted(tmp_path, proxy):
    done = run("--venv", str(tmp_path / "venv").replace("\\", "/"), "--proxy", proxy)

    assert done.returncode != 0 and "http" in done.stderr.lower()


@needs_bash
@pytest.mark.parametrize("venv", [
    "/home/arc/.venvs/retrival-pvl-server",
    "/home/arc/.venvs/retrival-pvl-server/",
    "/home/arc/.venvs/retrival-pvl-server/lib/python3.12",
    "/home/arc/.venvs/../.venvs/retrival-pvl-server",
])
def test_the_paddle_server_venv_is_refused_in_every_spelling(venv):
    done = run("--venv", venv, "--proxy", UNREACHABLE_PROXY)

    assert done.returncode != 0
    assert "retrival-pvl-server" in done.stderr and "refus" in done.stderr.lower()
    assert "proxy" not in done.stderr.lower()                       # refused first, before any other check


@needs_bash
def test_check_fails_on_a_venv_that_does_not_exist(tmp_path):
    done = run("--check", "--venv", str(tmp_path / "missing").replace("\\", "/"))

    assert done.returncode != 0 and "missing" in done.stderr


@pytest.mark.skipif(WSL is None, reason="no WSL")
def test_check_passes_on_the_venv_the_pins_were_taken_from():
    """/root/vllm-bench/venv on arc99's WSL is the reference environment: every pin matches and the sparse
    plugin's entry point is registered, with no network and no proxy."""
    drive = ROOT.drive.rstrip(":").lower()
    script = f"/mnt/{drive}/" + SCRIPT.relative_to(ROOT.anchor).as_posix()
    done = subprocess.run([WSL, "-d", "Ubuntu-24.04", "-u", "root", "-e", "bash", script, "--check",
                           "--venv", "/root/vllm-bench/venv"], capture_output=True, text=True, timeout=300)
    out = done.stdout.replace("\x00", "") + done.stderr.replace("\x00", "")

    assert done.returncode == 0, out[-1500:]
    assert "vllm 0.33.0" in out and "bge_m3_sparse_plugin" in out and "all pins match" in out


@pytest.mark.skipif(WSL is None, reason="no WSL")
def test_check_names_a_pin_that_does_not_match(tmp_path):
    """The same venv against a lock with one changed pin: the mismatch is reported and the exit code is not 0."""
    drive = ROOT.drive.rstrip(":").lower()
    script = f"/mnt/{drive}/" + SCRIPT.relative_to(ROOT.anchor).as_posix()
    bent = tmp_path / "bent.txt"
    bent.write_text(LOCK.read_text(encoding="utf-8").replace("vllm==0.33.0", "vllm==0.32.0"), encoding="utf-8")
    bent_wsl = f"/mnt/{drive}/" + bent.relative_to(bent.anchor).as_posix()
    done = subprocess.run([WSL, "-d", "Ubuntu-24.04", "-u", "root", "-e", "bash", script, "--check", "--venv",
                           "/root/vllm-bench/venv", "--requirements", bent_wsl],
                          capture_output=True, text=True, timeout=300)
    out = done.stdout.replace("\x00", "") + done.stderr.replace("\x00", "")

    assert done.returncode != 0 and "vllm" in out and "0.32.0" in out and "0.33.0" in out
