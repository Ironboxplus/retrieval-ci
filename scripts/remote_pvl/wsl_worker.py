"""Windows -> WSL launcher: run a worker with marker_worker.py's contract inside a WSL distro.

The host extract stage drives a Windows host through its own Python (makedirs, listdir and rmtree on Windows
paths), so the registered worker has to be a Windows program. This one is that program for a host whose GPU
is used from WSL2: it translates ``--input`` / ``--output`` to ``/mnt/<drive>/...`` and runs the real worker
there with ``wsl.exe``, then exits with the worker's exit status.

    wsl_worker.py --wsl-distro D --wsl-cwd LINUX_DIR --wsl-python LINUX_PYTHON --wsl-worker LINUX_SCRIPT
                  [--wsl-stop-on-eof] --input DIR --output DIR [--gpu N] [worker arguments...]

Arguments it does not know go to the worker unchanged, after the translated contract arguments. The environment
reaches Linux only through ``WSLENV`` (the registry's ``env`` lists the names there); the launcher adds nothing.

A stop request (SIGTERM, SIGINT, Ctrl+Break, console close) terminates the ``wsl.exe`` child. With
``--wsl-stop-on-eof`` it asks first: Windows OpenSSH leaves what a closed session started running, so the stage keeps the
session's stdin open and the end of the session is EOF on this process's stdin. EOF and a stop request then close
``wsl.exe``'s stdin instead, which ``stdin_guard.py`` (run in WSL around the worker) turns into SIGTERM for the worker,
and only a child still running after ``STOP_GRACE_S`` is terminated.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

WSL_EXE = "wsl.exe"
GUARD = Path(__file__).with_name("stdin_guard.py")
STOP_GRACE_S = 90.0


def windows_to_wsl(path: str) -> str:
    """``C:/a/b`` or ``C:\\a\\b`` -> ``/mnt/c/a/b``."""
    text = path.replace("\\", "/")
    if len(text) < 3 or not text[0].isalpha() or text[1] != ":" or text[2] != "/":
        raise ValueError(f"not an absolute Windows drive path: {path!r}")
    return f"/mnt/{text[0].lower()}{text[2:]}".rstrip("/") or "/mnt"


def _parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--wsl-distro", required=True)
    parser.add_argument("--wsl-cwd", required=True, help="Linux directory the worker runs in")
    parser.add_argument("--wsl-python", required=True)
    parser.add_argument("--wsl-worker", required=True)
    parser.add_argument("--wsl-stop-on-eof", action="store_true")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default=None)
    return parser.parse_known_args(argv)


def _command(args: argparse.Namespace, rest: list[str]) -> list[str]:
    contract = ["--input", windows_to_wsl(os.path.abspath(args.input)),
                "--output", windows_to_wsl(os.path.abspath(args.output))]
    if args.gpu is not None:
        contract += ["--gpu", args.gpu]
    worker = [args.wsl_python, args.wsl_worker, *contract, *rest]
    if args.wsl_stop_on_eof:  # the guard runs the worker, with the same interpreter, and watches stdin
        worker = [args.wsl_python, windows_to_wsl(os.path.abspath(GUARD)), *worker]
    return [WSL_EXE, "-d", args.wsl_distro, "--cd", args.wsl_cwd, "-e", *worker]


def build_command(argv: list[str]) -> list[str]:
    return _command(*_parse(argv))


def _stop_on_console_close(stop) -> None:
    """Windows delivers a closing console (and logoff / shutdown) as a control event, not a signal."""
    import ctypes

    handler_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

    def handler(event: int) -> int:
        if event in (2, 5, 6):  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
            stop()
            return 1
        return 0

    _stop_on_console_close.handler = handler_type(handler)  # the OS keeps no reference
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_stop_on_console_close.handler, 1)


def run_child(command: list[str], *, stop_on_eof: bool = False, grace_s: float = STOP_GRACE_S) -> int:
    """Run ``command`` with this process's console and environment; end it when asked to stop; its exit code.

    ``stop_on_eof``: the child gets a stdin pipe that nothing is written to. Its closing (this process's own stdin
    reaching EOF, or a stop request) is the stop signal for the child; the child is terminated only when it is still
    running ``grace_s`` seconds later. Without it a stop request terminates the child at once."""
    try:
        child = subprocess.Popen(command, stdin=subprocess.PIPE if stop_on_eof else None)
    except OSError as exc:
        print(f"wsl_worker: cannot start {command[0]}: {exc}", file=sys.stderr)
        return 127
    lock = threading.RLock()  # a signal handler can interrupt another one
    stopped_at: list[float] = []

    def stop(*_) -> None:
        with lock:
            if child.poll() is not None:
                return
            if not stop_on_eof:
                child.terminate()
            elif not stopped_at:
                stopped_at.append(time.monotonic())
                try:
                    child.stdin.close()
                except OSError:
                    pass

    def watch_stdin() -> None:
        try:
            while os.read(sys.stdin.fileno(), 65536):
                pass
        except (OSError, ValueError):
            pass
        stop()

    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)
    if os.name == "nt":
        _stop_on_console_close(stop)
    if stop_on_eof:
        threading.Thread(target=watch_stdin, daemon=True, name="stdin-eof-watch").start()
    terminated = False
    while True:  # not a bare wait(): Python only runs signal handlers between its own bytecodes
        try:
            return child.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            if stopped_at and not terminated and time.monotonic() - stopped_at[0] >= grace_s:
                terminated = True
                child.terminate()


def main(argv: list[str] | None = None) -> int:
    args, rest = _parse(sys.argv[1:] if argv is None else argv)
    command = _command(args, rest)
    print("wsl_worker: " + " ".join(command), file=sys.stderr, flush=True)
    return run_child(command, stop_on_eof=args.wsl_stop_on_eof)


if __name__ == "__main__":
    sys.exit(main())
