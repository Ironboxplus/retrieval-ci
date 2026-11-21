"""Copy a file into another host's inbox over ssh/scp, atomically.

Shared by every downloading-host push (arXiv HTML batches, conference PDF and
Markdown slices): the file is written beside its final name as ``.part`` and
renamed, so a reader polling the inbox never sees half a file.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence, Union

Executable = Union[str, Sequence[str]]

SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
TRANSFER_TIMEOUT_S = 900  # one ~120 MB bundle takes ~10 s on the LAN


def _default_exe(name: str) -> str:
    windows = Path(r"C:\Windows\System32\OpenSSH") / f"{name}.exe"
    return str(windows) if windows.exists() else name


class InboxTransfer:
    """ssh/scp access to one host's inbox directory.

    ``ssh`` / ``scp`` are the executables to run; a sequence is a command prefix
    (an interpreter plus a script), for a transport that is not the OpenSSH client."""

    def __init__(self, *, ssh_target: str, remote_inbox: str,
                 ssh: Executable | None = None, scp: Executable | None = None) -> None:
        self.ssh_target = ssh_target
        self.remote_inbox = remote_inbox.rstrip("/")
        self.ssh = ssh or _default_exe("ssh")
        self.scp = scp or _default_exe("scp")

    def _run(self, exe: Executable, *args: str) -> None:
        # A dead TCP session leaves ssh/scp waiting forever (one mkdir hung for
        # 8 h and stalled every later push): keepalives end it on the ssh side,
        # the timeout is the backstop. stdin is closed so nothing waits on it.
        prefix = [exe] if isinstance(exe, str) else list(exe)
        cmd = [*prefix, *SSH_OPTIONS, *args]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              stdin=subprocess.DEVNULL, timeout=TRANSFER_TIMEOUT_S)
        if proc.returncode != 0:
            raise RuntimeError(f"{Path(prefix[-1]).name} exited {proc.returncode}: {proc.stderr.strip()[-500:]}")

    def send(self, local: Path, name: str) -> None:
        """Put ``local`` into the inbox as ``name`` (visible only once complete)."""
        remote = f"{self.remote_inbox}/{name}"
        self._run(self.ssh, self.ssh_target, f"mkdir -p {self.remote_inbox}")
        self._run(self.scp, "-q", str(local), f"{self.ssh_target}:{remote}.part")
        self._run(self.ssh, self.ssh_target, f"mv -f {remote}.part {remote}")
