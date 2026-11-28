"""Linux side of ``wsl_worker.py --wsl-stop-on-eof``: run a worker and stop it when this process's stdin reaches EOF.

    python stdin_guard.py WORKER_ARGV...        (WORKER_ARGV is a complete command: the interpreter, the script, its arguments)

The host extract stage keeps an ssh session open for as long as the worker runs, and the session's stdin with it. When the
session ends (a cancelled stage, a restarted backend, a dead ssh client) Windows OpenSSH signals nothing it started, so EOF
on stdin, relayed by the launcher and ``wsl.exe``, is the only notice the Linux side gets. The worker then gets SIGTERM
(pvl_worker.py's handler stops its model servers), and SIGKILL for its whole process group if it is still there after
``STDIN_GUARD_GRACE_S`` seconds (default 90). The exit status is the worker's (128 + N after signal N). Standard library
only: it runs in whatever interpreter the registry names.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

GRACE_S = float(os.environ.get("STDIN_GUARD_GRACE_S", "90"))


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)  # the worker leads its own session, so its pid is its group
    except ProcessLookupError:
        pass


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: stdin_guard.py WORKER_ARGV...", file=sys.stderr)
        return 2
    worker = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True)
    stop = threading.Event()

    def watch_stdin() -> None:
        try:
            while os.read(0, 65536):
                pass
        except OSError:
            pass
        stop.set()

    threading.Thread(target=watch_stdin, daemon=True).start()
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):  # the launcher or wsl.exe going away
        signal.signal(sig, lambda *_: stop.set())
    stopped_at = None
    while True:
        code = worker.poll()
        if code is not None:
            if stopped_at is not None:
                _signal_group(worker.pid, signal.SIGKILL)  # nothing of a stopped worker stays behind
            return code if code >= 0 else 128 - code
        if stop.is_set():
            if stopped_at is None:
                stopped_at = time.monotonic()
                worker.send_signal(signal.SIGTERM)
            elif time.monotonic() - stopped_at >= GRACE_S:
                _signal_group(worker.pid, signal.SIGKILL)
        time.sleep(0.2)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
