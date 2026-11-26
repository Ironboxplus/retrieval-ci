"""A local stand-in for the ssh and scp executables, used where no sshd is reachable.

``python tests/local_ssh_transport.py ssh <ssh options> <host> <command>`` runs the
command in a real shell on this machine; ``... scp <options> <src> <host>:<dst>``
copies the file. The InboxTransfer under test spawns it exactly as it would spawn
the OpenSSH client (same argument vector, same exit-status contract), so the
transfer sequence (mkdir, copy as .part, rename) really executes; only the network
hop between the two "hosts" is absent.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


def _positional(args: list[str]) -> list[str]:
    out, skip = [], False
    for arg in args:
        if skip:
            skip = False
        elif arg == "-o":
            skip = True
        elif not arg.startswith("-"):
            out.append(arg)
    return out


_REMOTE = re.compile(r"^[^:/\\]{2,}:(.*)$")  # host:path (a single letter before ':' is a drive)


def _local_path(spec: str) -> Path:
    """The path of an scp operand: ``host:path`` names a path on the 'remote' host."""
    match = _REMOTE.match(spec)
    return Path(match.group(1) if match else spec)


def main() -> int:
    role, args = sys.argv[1], sys.argv[2:]
    rest = _positional(args)
    if role == "ssh":
        _host, command = rest
        bash = shutil.which("bash")
        if bash is None:
            print("no bash on PATH", file=sys.stderr)
            return 127
        return subprocess.run([bash, "-c", command]).returncode
    if role == "scp":
        *sources, target = rest
        dest = _local_path(target)
        for source in sources:
            path = _local_path(source)
            into = dest / path.name if dest.is_dir() or target.endswith("/") else dest
            into.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, into)
        return 0
    print(f"unknown role {role!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
