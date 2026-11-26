"""run_local(hold_stdin=True): the child's stdin is an open pipe for as long as run_local runs.

The remote end of a host extract learns that its ssh session ended from stdin reaching EOF (plan 2026-11-20, 2.3b);
the backend's own stdin is /dev/null, so without a pipe held open that EOF is there from the start. Real child process.
"""
from __future__ import annotations

import sys

from app.services.pipeline import runners

CHILD = """\
import sys, threading, time
eof = threading.Event()
threading.Thread(target=lambda: (sys.stdin.read(), eof.set()), daemon=True).start()
print("started", flush=True)
time.sleep(1.5)
print("stdin_open=%s" % (not eof.is_set()), flush=True)
"""


def test_hold_stdin_keeps_the_childs_stdin_open_and_closes_it_when_run_local_returns(tmp_path):
    started = []

    def factory(args, cwd, env, **kw):  # the default launcher, recording the process
        started.append(runners._default_popen(args, cwd, env, **kw))
        return started[-1]

    lines: list[str] = []
    runners.run_local([sys.executable, "-c", CHILD], tmp_path, lines.append, popen_factory=factory, hold_stdin=True)

    assert lines == ["started", "stdin_open=True"]
    assert started[0].stdin.closed


def test_without_hold_stdin_a_factory_with_the_old_three_argument_signature_still_works(tmp_path):
    seen = []

    def old_factory(args, cwd, env):
        seen.append((list(args), cwd, env))
        return runners._default_popen(args, cwd, env)

    lines: list[str] = []
    runners.run_local([sys.executable, "-c", "print('x')"], tmp_path, lines.append, popen_factory=old_factory)

    assert lines == ["x"] and seen[0][1] == str(tmp_path)
