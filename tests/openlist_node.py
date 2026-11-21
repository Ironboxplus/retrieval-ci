"""A real OpenList process for tests of what reads a 115 volume.

The volume tests need the node's own HTTP behaviour (login, listing, WebDAV reads,
the refusal of a read-only user's PUT), so they start the same binary the cluster
runs, with a Local storage standing in for a Cfs volume. The binary comes from
OPENLIST_TEST_BINARY, the PATH, or a copy of the Windows service build kept in the
temp directory (the service itself and its data directory are never touched).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ADMIN_PASSWORD = "node-admin-for-tests"
READER_USER = "retrival-reader"
READER_PASSWORD = "reader-for-tests"
# see hidden (1) | access without password (2) | WebDAV read (256): the read-only user
# of the .88 node (env.md), which has no write, rename, move, copy or delete bit.
READER_PERMISSION = 1 | 2 | 256
SERVICE_BINARY = Path(r"C:\Users\Arc\Service\openlist99\openlist.exe")


def find_binary() -> Path | None:
    override = os.environ.get("OPENLIST_TEST_BINARY")
    if override:
        return Path(override) if Path(override).is_file() else None
    found = shutil.which("openlist")
    if found:
        return Path(found)
    if SERVICE_BINARY.is_file():
        cached = Path(tempfile.gettempdir()) / "rp-openlist-bin" / SERVICE_BINARY.name
        if not cached.is_file():
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SERVICE_BINARY, cached)
        return cached
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class OpenListNode:
    def __init__(self, binary: Path, root: Path) -> None:
        self.root = root
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        data = root / "data"
        data.mkdir(parents=True)
        (data / "config.json").write_text(json.dumps(
            {"scheme": {"address": "127.0.0.1", "http_port": self.port}}), encoding="utf-8")
        self.log_path = root / "server.log"
        self.proc = subprocess.Popen(
            [str(binary), "server", "--data", str(data), "--log-std"],
            env=dict(os.environ, OPENLIST_ADMIN_PASSWORD=ADMIN_PASSWORD),
            stdout=open(self.log_path, "wb"), stderr=subprocess.STDOUT)
        try:
            self._wait_ready()
            self.token = self.api("POST", "/api/auth/login",
                                  {"username": "admin", "password": ADMIN_PASSWORD})["token"]
        except BaseException:
            self.stop()
            raise

    def _wait_ready(self, timeout: float = 90) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"openlist exited with {self.proc.returncode}: "
                                   f"{self.log_path.read_text(errors='replace')[-600:]}")
            try:
                with urllib.request.urlopen(self.url + "/ping", timeout=2) as resp:
                    if resp.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError("openlist did not answer /ping")

    def api(self, method: str, path: str, body=None, *, token: str | None = None):
        headers = {"Content-Type": "application/json"}
        auth = token if token is not None else getattr(self, "token", "")
        if auth:
            headers["Authorization"] = auth
        request = urllib.request.Request(
            self.url + path, method=method, headers=headers,
            data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                doc = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{method} {path}: HTTP {exc.code}") from None
        if doc.get("code") != 200:
            raise RuntimeError(f"{method} {path}: code {doc.get('code')} {doc.get('message')}")
        return doc.get("data")

    def mount_local(self, mount: str, directory: Path) -> Path:
        """Expose ``directory`` at ``mount`` (hidden files listed, as a Cfs volume's are)."""
        directory.mkdir(parents=True, exist_ok=True)
        self.api("POST", "/api/admin/storage/create", {
            "mount_path": mount, "order": 0, "driver": "Local", "cache_expiration": 30, "remark": "",
            "addition": json.dumps({"root_folder_path": str(directory), "show_hidden": True})})
        return directory

    def add_reader(self) -> Path:
        """The read-only user; returns the file holding its password (how the backend gets it)."""
        self.api("POST", "/api/admin/user/create", {
            "username": READER_USER, "password": READER_PASSWORD, "role": 0,
            "permission": READER_PERMISSION, "base_path": "/", "disabled": False})
        password_file = self.root / "reader.pass"
        password_file.write_text(READER_PASSWORD + "\n", encoding="utf-8")
        return password_file

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def start_node(tmp_path_factory) -> OpenListNode:
    binary = find_binary()
    if binary is None:
        pytest.skip("no openlist binary (set OPENLIST_TEST_BINARY)")
    return OpenListNode(binary, tmp_path_factory.mktemp("openlist-node"))
