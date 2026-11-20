"""The Windows -> WSL launcher (scripts/remote_pvl/wsl_worker.py) that lets the host extract stage run a worker
inside a WSL distro on a Windows host (arc99's RTX 3090).

The path translation and the command line are pure functions; the end-to-end tests start the real launcher, which
starts the real ``wsl.exe`` (skipped where there is none or the distro is missing) and runs a tiny Linux Python
program as the worker; the signal test starts a real child process (Windows only).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "remote_pvl" / "wsl_worker.py"
GUARD = ROOT / "scripts" / "remote_pvl" / "stdin_guard.py"
DISTRO = os.environ.get("WSL_TEST_DISTRO", "Ubuntu-24.04")


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("wsl_worker_under_test", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_windows_paths_become_mnt_paths(launcher):
    assert launcher.windows_to_wsl("C:/Users/Arc/Prd/x y/out") == "/mnt/c/Users/Arc/Prd/x y/out"
    assert launcher.windows_to_wsl("D:\\data\\batch_0001\\in") == "/mnt/d/data/batch_0001/in"
    assert launcher.windows_to_wsl("c:/a//b") == "/mnt/c/a//b"
    assert launcher.windows_to_wsl("E:/") == "/mnt/e"
    for not_a_drive_path in ("relative/dir", "/already/linux", "//server/share/x", "C:relative", ""):
        with pytest.raises(ValueError, match="absolute Windows drive path"):
            launcher.windows_to_wsl(not_a_drive_path)


def test_the_command_translates_the_contract_arguments_and_passes_the_rest_verbatim(launcher):
    rest = ["--server-command", "/opt/vllm serve m --config {config} --port {port}", "--server-prefix",
            "systemd-run --user --scope -p MemoryHigh=4G --", "--max-concurrency", "64"]
    command = launcher.build_command([
        "--wsl-distro", "Ubuntu-24.04", "--wsl-cwd", "/home/arc/repo", "--wsl-python", "/home/arc/venv/bin/python",
        "--wsl-worker", "scripts/remote_pvl/pvl_worker.py",
        "--input", "C:/work/b1/in", "--output", "C:/work/b1/out", "--gpu", "1", *rest])
    assert command == ["wsl.exe", "-d", "Ubuntu-24.04", "--cd", "/home/arc/repo", "-e", "/home/arc/venv/bin/python",
                       "scripts/remote_pvl/pvl_worker.py", "--input", "/mnt/c/work/b1/in", "--output",
                       "/mnt/c/work/b1/out", "--gpu", "1", *rest]
    deployed = json.loads((ROOT / "deploy" / "compute88" / "embed_hosts.json").read_text(encoding="utf-8"))
    args = deployed["arc99"]["marker"]["args"]  # what the registry really hands the launcher
    assert args[8] == "--wsl-stop-on-eof"  # a cancelled extract must reach the worker (plan 2.3b)
    built = launcher.build_command([*args[:9], "--input", "C:/w/in", "--output", "C:/w/out", "--gpu", "0", *args[9:]])
    assert built[-len(args[9:]):] == args[9:]  # the worker's own arguments arrive unchanged and in order


def test_stop_on_eof_runs_the_worker_under_the_guard_and_still_passes_the_arguments_through(launcher):
    guard = launcher.windows_to_wsl(str(GUARD))
    command = launcher.build_command([
        "--wsl-distro", "Ubuntu-24.04", "--wsl-cwd", "/home/arc/repo", "--wsl-python", "/home/arc/venv/bin/python",
        "--wsl-worker", "scripts/remote_pvl/pvl_worker.py", "--wsl-stop-on-eof",
        "--input", "C:/work/b1/in", "--output", "C:/work/b1/out", "--gpu", "1", "--max-concurrency", "64"])
    assert command == ["wsl.exe", "-d", "Ubuntu-24.04", "--cd", "/home/arc/repo", "-e", "/home/arc/venv/bin/python",
                       guard, "/home/arc/venv/bin/python", "scripts/remote_pvl/pvl_worker.py", "--input",
                       "/mnt/c/work/b1/in", "--output", "/mnt/c/work/b1/out", "--gpu", "1",
                       "--max-concurrency", "64"]


def _wsl_has_distro() -> bool:
    if shutil.which("wsl.exe") is None:
        return False
    listed = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True, timeout=60).stdout
    return DISTRO in listed.decode("utf-16-le", errors="replace").replace("\x00", "").split()


# A Linux program with the worker's command line: it reports what it was started with.
LINUX_WORKER = """\
import json, os, sys
argv = sys.argv[1:]
code = int(argv[argv.index("--exit-code") + 1])
print(json.dumps({"argv": argv, "cwd": os.getcwd(), "probe": os.environ.get("WSL_LAUNCHER_PROBE")}))
sys.exit(code)
"""


@pytest.mark.skipif(not _wsl_has_distro(), reason=f"no wsl.exe with distro {DISTRO}")
@pytest.mark.parametrize("exit_code", [0, 3])
def test_the_linux_worker_gets_translated_paths_verbatim_arguments_environment_and_its_exit_code(
        launcher, tmp_path, exit_code):
    worker = tmp_path / "linux_worker.py"
    worker.write_text(LINUX_WORKER, encoding="utf-8", newline="\n")
    (tmp_path / "in").mkdir()
    rest = ["--server-command", "/opt/vllm serve m --config {config} --port {port}", "--server-prefix",
            "systemd-run --user --scope -p MemoryHigh=4G --", "--no-enable-prefix-caching",
            "--exit-code", str(exit_code)]
    env = {**os.environ, "WSL_LAUNCHER_PROBE": "from-the-registry env", "WSLENV": "WSL_LAUNCHER_PROBE/u"}

    done = subprocess.run(
        [sys.executable, str(LAUNCHER), "--wsl-distro", DISTRO, "--wsl-cwd", "/tmp",
         "--wsl-python", "/usr/bin/python3", "--wsl-worker", launcher.windows_to_wsl(str(worker)),
         "--input", str(tmp_path / "in"), "--output", str(tmp_path / "out"), "--gpu", "0", *rest],
        capture_output=True, env=env, timeout=180)
    # wsl.exe's own notices are UTF-16 on stderr: read bytes, decode what is readable.
    stdout, stderr = (stream.decode("utf-8", errors="replace") for stream in (done.stdout, done.stderr))

    assert done.returncode == exit_code, stderr[-2000:]
    report = json.loads([line for line in stdout.splitlines() if line.startswith("{")][-1])
    assert report["argv"] == ["--input", launcher.windows_to_wsl(str(tmp_path / "in")),
                              "--output", launcher.windows_to_wsl(str(tmp_path / "out")), "--gpu", "0", *rest]
    assert report["cwd"] == "/tmp" and report["probe"] == "from-the-registry env"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console control events")
def test_a_stop_request_terminates_the_child_the_launcher_started(tmp_path):
    import psutil

    # The child ignores the break event the launcher gets too, so only the launcher can end it.
    child = ("import os, signal, time; signal.signal(signal.SIGBREAK, signal.SIG_IGN); "
             "print(os.getpid(), flush=True); time.sleep(120)")
    driver = (f"import sys; sys.path.insert(0, {str(LAUNCHER.parent)!r}); import wsl_worker; "
              f"sys.exit(wsl_worker.run_child([sys.executable, '-c', {child!r}]))")
    launcher_proc = subprocess.Popen([sys.executable, "-c", driver], stdout=subprocess.PIPE, text=True,
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        child_pid = int(launcher_proc.stdout.readline())
        assert psutil.pid_exists(child_pid)
        launcher_proc.send_signal(signal.CTRL_BREAK_EVENT)
        launcher_proc.wait(timeout=30)
        deadline = time.monotonic() + 10
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not psutil.pid_exists(child_pid)
    finally:
        if launcher_proc.poll() is None:
            launcher_proc.kill()


# ---------------------------------------------------- --wsl-stop-on-eof (plan 2.3b)
# The ssh session that started the launcher ending shows up as EOF on the launcher's stdin (the stage holds an open pipe
# for the whole run, run_local(hold_stdin=True)); the child must then be asked to stop first and killed only after a grace.
def _launcher_driver(child: str, grace_s: float) -> str:
    return (f"import sys; sys.path.insert(0, {str(LAUNCHER.parent)!r}); import wsl_worker; "
            f"sys.exit(wsl_worker.run_child([sys.executable, '-c', {child!r}], stop_on_eof=True, grace_s={grace_s}))")


def _start_driver(child: str, grace_s: float = 60.0) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", _launcher_driver(child, grace_s)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)


# The child ignores the break event the launcher gets too; it ends only by stdin EOF (a clean exit) or by being killed.
EOF_CHILD = ("import os, signal, sys; signal.signal(signal.SIGBREAK, signal.SIG_IGN); "
             "print(os.getpid(), flush=True); sys.stdin.buffer.read(); print('eof', flush=True)")
DEAF_CHILD = ("import os, signal, time; signal.signal(signal.SIGBREAK, signal.SIG_IGN); "
              "print(os.getpid(), flush=True); time.sleep(120)")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console control events")
@pytest.mark.parametrize("how", ["stdin-eof", "ctrl-break"])
def test_a_stop_with_stop_on_eof_closes_the_childs_stdin_and_lets_it_leave_by_itself(how):
    proc = _start_driver(EOF_CHILD)
    try:
        int(proc.stdout.readline())
        if how == "stdin-eof":
            proc.stdin.close()
        else:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        rest = proc.stdout.read()
        assert proc.wait(timeout=30) == 0 and "eof" in rest  # asked, not killed
    finally:
        _kill_tree(proc)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console control events")
def test_a_child_that_ignores_the_closed_stdin_is_terminated_after_the_grace():
    import psutil

    proc = _start_driver(DEAF_CHILD, grace_s=2.0)
    try:
        child_pid = int(proc.stdout.readline())
        started = time.monotonic()
        proc.stdin.close()
        time.sleep(1.0)
        assert psutil.pid_exists(child_pid) and proc.poll() is None  # inside the grace nothing is forced
        assert proc.wait(timeout=30) != 0
        assert time.monotonic() - started >= 2.0
        deadline = time.monotonic() + 10
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not psutil.pid_exists(child_pid)
    finally:
        _kill_tree(proc)


# A Linux "worker": it starts a grandchild, reports its pids and waits. On SIGTERM it does what pvl_worker's handler does
# (stops what it started, here a grandchild in a session of its own that nothing else would reach).
LINUX_STOPPABLE_WORKER = """\
import json, os, signal, subprocess, sys, time
argv = sys.argv[1:]
state, mode = (argv[argv.index(flag) + 1] for flag in ("--state", "--mode"))
if mode == "exit5":
    sys.exit(5)
grandchild = subprocess.Popen(["sleep", "300"], start_new_session=(mode == "term"))
def on_term(*_):
    grandchild.kill()
    open(state + ".term", "w").write("handled")
    os._exit(143)
signal.signal(signal.SIGTERM, on_term if mode == "term" else signal.SIG_IGN)
open(state + ".pids", "w").write(json.dumps([os.getpid(), grandchild.pid]))
print("ready", flush=True)
time.sleep(300)
"""

PROCESS_STATE = """\
import sys
try:
    print(open('/proc/%s/stat' % sys.argv[1]).read().rsplit(')', 1)[1].split()[0])
except OSError:
    print('gone')
"""


def _wsl(*command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["wsl.exe", "-d", DISTRO, "-e", *command], capture_output=True, timeout=60)


def _gone(pid: int) -> bool:
    state = _wsl("/usr/bin/python3", "-c", PROCESS_STATE, str(pid)).stdout.decode().split()
    return bool(state) and state[-1] in ("gone", "Z", "X")


def _read_pids(state: Path) -> list[int]:
    path = Path(str(state) + ".pids")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@pytest.fixture
def linux_worker_run(launcher, tmp_path):
    """Starts the real launcher with --wsl-stop-on-eof around LINUX_STOPPABLE_WORKER and, whatever the test does,
    kills the launcher tree and the pids the worker reported (by pid) when it is over."""
    state = tmp_path / "state"
    worker = tmp_path / "linux_worker.py"
    worker.write_text(LINUX_STOPPABLE_WORKER, encoding="utf-8", newline="\n")
    (tmp_path / "in").mkdir()
    started: list[subprocess.Popen] = []
    errors = (tmp_path / "launcher.stderr").open("wb")

    def start(mode: str, **env) -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, str(LAUNCHER), "--wsl-distro", DISTRO, "--wsl-cwd", "/tmp", "--wsl-python",
             "/usr/bin/python3", "--wsl-worker", launcher.windows_to_wsl(str(worker)), "--wsl-stop-on-eof",
             "--input", str(tmp_path / "in"), "--output", str(tmp_path / "out"), "--gpu", "0",
             "--state", launcher.windows_to_wsl(str(state)), "--mode", mode],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
            env={**os.environ, **env, "WSLENV": ":".join(f"{k}/u" for k in env)})
        started.append(proc)
        return proc

    yield start, state
    for proc in started:
        _kill_tree(proc)
    for pid in _read_pids(state):
        if not _gone(pid):
            _wsl("kill", "-9", str(pid))
    errors.close()


def _wait_for_line(proc: subprocess.Popen, text: bytes, seconds: float = 90.0) -> None:
    import threading

    seen = threading.Event()

    def reader() -> None:
        for line in proc.stdout:
            if text in line:
                seen.set()

    threading.Thread(target=reader, daemon=True).start()
    assert seen.wait(seconds), f"the Linux worker never printed {text!r}"


@pytest.mark.skipif(not _wsl_has_distro(), reason=f"no wsl.exe with distro {DISTRO}")
def test_ending_the_ssh_session_makes_the_linux_worker_stop_itself_and_leaves_nothing_behind(linux_worker_run):
    start, state = linux_worker_run
    launcher_proc = start("term")
    _wait_for_line(launcher_proc, b"ready")
    worker_pid, grandchild_pid = _read_pids(state)
    assert not _gone(worker_pid) and not _gone(grandchild_pid)

    launcher_proc.stdin.close()  # what the session's end looks like to the launcher

    assert launcher_proc.wait(timeout=90) == 143
    assert Path(str(state) + ".term").read_text() == "handled"  # SIGTERM reached the worker's own handler
    assert _gone(worker_pid) and _gone(grandchild_pid)


@pytest.mark.skipif(not _wsl_has_distro(), reason=f"no wsl.exe with distro {DISTRO}")
def test_a_linux_worker_that_ignores_sigterm_is_killed_with_its_process_group_after_the_grace(linux_worker_run):
    start, state = linux_worker_run
    launcher_proc = start("ignore-term", STDIN_GUARD_GRACE_S="2")
    _wait_for_line(launcher_proc, b"ready")
    worker_pid, grandchild_pid = _read_pids(state)

    launcher_proc.stdin.close()

    assert launcher_proc.wait(timeout=90) != 0
    assert not Path(str(state) + ".term").exists()
    assert _gone(worker_pid) and _gone(grandchild_pid)


@pytest.mark.skipif(not _wsl_has_distro(), reason=f"no wsl.exe with distro {DISTRO}")
def test_the_guard_exits_with_the_workers_own_exit_code_while_stdin_stays_open(linux_worker_run):
    start, state = linux_worker_run
    launcher_proc = start("exit5")

    assert launcher_proc.wait(timeout=90) == 5  # stdin was never closed: the exit is the worker's
