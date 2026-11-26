"""Regression: a relative scope belongs to the CALLING client, not the singleton daemon.

2026-09-06 现场：另一个会话里 `journal_search`/`journal_recent` 返回的是"家庭能源看板"
（`E:/jetb/pythons/playground/homeassitant`）的条目，而不是当前仓库的。

根因不是索引数据，是路径解析的位置：

* daemon 是**每用户唯一的单例**，所有项目的 shim 共用它；
* `_launch_daemon` 用 `cwd=os.getcwd()` 启动它，于是 daemon 的工作目录 =
  **第一个拉起它的那个 shim 的目录**；
* shim 是逐字节转发 JSONL 的桥，从不解析 MCP 消息，因此从来没把自己的目录告诉过 daemon；
* 工具里的 `_scope_path(".")` 走 `os.path.abspath`，在 **daemon 进程**里解析。

三者相乘 = 谁先启动单例，谁的目录就成了所有项目的 `"."`。受影响的不止 journal 读：
`journal_record` 会把条目**写进另一个项目的 namespace**，`index_code(repo_root=".")` /
`journal_index_dir(dir=".")` 会走错目录并注册到错的 scope。

下面用真实子进程验证：两个 shim 从两个不同目录连**同一个**单例，各自的 `"."` 必须
解析到自己的目录。第二个 client 是关键——它复用的是第一个 client 启动的 daemon，
修复前它拿到的是第一个 client 的目录。
"""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from test_mcp_daemon_cli import (
    CLI,
    STDIO_CLIENT,
    _daemon_env,
    _free_loopback_port,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
READY_TIMEOUT_SECONDS = 180.0


class _ScopeEchoHandler(http.server.BaseHTTPRequestHandler):
    """Echoes the requested journal scope back as the entry title.

    Real HTTP, so a passing assertion proves the whole client -> shim -> daemon ->
    tool -> backend path agreed on which directory ``scope_dir="."`` meant.
    """

    def log_message(self, *args: object) -> None:
        return

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        parsed = urlsplit(self.path)
        if parsed.path == "/api/journal/entries":
            scope = (parse_qs(parsed.query).get("project_slug") or [""])[0]
            self._json({
                "entries": [{
                    "title": scope,
                    "entry_id": "scope-echo",
                    "source_kind": "doc",
                }],
                "total": 1,
            })
            return
        if parsed.path == "/api/stats":
            self._json({
                "total_papers": 7,
                "total_chunks": 70,
                "vector_index_size": 70,
                "embedding_model": "scope-echo-stub",
            })
            return
        self.send_error(404)


@pytest.fixture
def scope_echo_backend() -> str:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ScopeEchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _wait_for_ready(ready_file: Path, process: subprocess.Popen[str]) -> dict:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    error_file = ready_file.with_suffix(".error")
    while time.monotonic() < deadline:
        if ready_file.exists():
            return json.loads(ready_file.read_text(encoding="utf-8"))
        if error_file.exists():
            pytest.fail(f"stdio client failed: {error_file.read_text(encoding='utf-8')}")
        if process.poll() is not None:
            out, err = process.communicate()
            pytest.fail(f"stdio client exited early rc={process.returncode}\n{out}\n{err}")
        time.sleep(0.1)
    pytest.fail(f"stdio client never became ready: {ready_file}")


def test_relative_scope_follows_each_client_not_the_shared_daemon(
    scope_echo_backend: str,
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    env = _daemon_env(scope_echo_backend, runtime_dir, test_port=_free_loopback_port())

    # Two directories standing in for two projects that share one account daemon.
    first_project = tmp_path / "project-alpha"
    second_project = tmp_path / "project-beta"
    first_project.mkdir()
    second_project.mkdir()

    release_file = tmp_path / "release.txt"
    ready_files = {
        "first": tmp_path / "client-first.json",
        "second": tmp_path / "client-second.json",
    }

    def start_client(project: Path, ready_file: Path) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [
                sys.executable, str(STDIO_CLIENT),
                "--cli", str(CLI),
                "--cwd", str(project),
                "--ready-file", str(ready_file),
                "--release-file", str(release_file),
                "--tool", "journal_recent",
                "--tool-args", json.dumps({"limit": 1}),
            ],
            cwd=str(project),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    first = start_client(first_project, ready_files["first"])
    processes = [first]
    try:
        # The first client wins the startup race, so the singleton inherits
        # project-alpha as its own working directory.  That is the state in which
        # every later project used to be answered with alpha's scope.
        first_ready = _wait_for_ready(ready_files["first"], first)

        second = start_client(second_project, ready_files["second"])
        processes.append(second)
        second_ready = _wait_for_ready(ready_files["second"], second)

        state = json.loads((runtime_dir / "daemon.json").read_text(encoding="utf-8"))
        assert int(state["pid"]) not in {first.pid, second.pid}

        first_scope = first_ready["journal_recent"]
        second_scope = second_ready["journal_recent"]

        # Scopes are canonicalised (Windows casing is case-folded), so compare on a
        # case-insensitive basis while still requiring the whole path to match.
        assert first_project.name in first_scope.casefold()
        assert str(first_project).replace("\\", "/").casefold() in first_scope.casefold()

        assert second_project.name in second_scope.casefold()
        assert str(second_project).replace("\\", "/").casefold() in second_scope.casefold()

        # The regression itself: the second client must NOT inherit the directory
        # that happened to start the daemon.
        assert second_project.name not in first_scope.casefold()
        assert first_project.name not in second_scope.casefold()
    finally:
        release_file.write_text("release", encoding="utf-8")
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        subprocess.run(
            [sys.executable, str(CLI), "stop", "--state-dir", str(runtime_dir)],
            env=env, capture_output=True, text=True, timeout=60, check=False,
        )


def test_scope_path_resolves_against_the_bound_session_directory(tmp_path: Path) -> None:
    """Unit-level counterpart: the tool module reads the daemon's per-session binding.

    Loaded the way an installed package lays the files out (flat on sys.path), so the
    ``from retrieval_mcp_daemon import client_working_directory`` lookup is the real one.
    """
    import importlib.util

    plugin_dir = REPO_ROOT / "mcp_plugin"
    sys.path.insert(0, str(plugin_dir))
    try:
        import retrieval_mcp_daemon as daemon

        spec = importlib.util.spec_from_file_location(
            "retrieval_mcp_scope_probe", plugin_dir / "retrieval_mcp.py")
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = plugin
        spec.loader.exec_module(plugin)
    finally:
        sys.path.remove(str(plugin_dir))

    client_dir = tmp_path / "caller"
    client_dir.mkdir()
    expected = plugin._scope_path(str(client_dir))

    bound = daemon.CLIENT_CWD.set(str(client_dir))
    try:
        assert plugin._scope_path(".") == expected
        assert plugin._session_abspath(".") == os.path.normpath(str(client_dir))
        # An absolute scope is unaffected by the binding.
        assert plugin._scope_path(str(REPO_ROOT)) == plugin._scope_path(str(REPO_ROOT))
    finally:
        daemon.CLIENT_CWD.reset(bound)

    # With nothing bound the daemon's own directory remains the base, which is what
    # a direct ``serve`` (no shim) should do.
    assert plugin._scope_path(".") == plugin._scope_path(os.getcwd())
