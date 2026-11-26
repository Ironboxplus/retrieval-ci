"""Black-box TDD coverage for the local singleton Retrieval MCP daemon.

The daemon is deliberately tested through the public CLI, real subprocesses and
the official MCP streamable-HTTP client.  The tiny backend below is an actual
loopback HTTP server, not a mocked MCP transport, so a passing test proves the
full client -> daemon -> backend request path without requiring the 4090.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import http.server
import json
import os
import socket
import stat
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from mcp_plugin import retrieval_mcp_daemon as daemon


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "mcp_plugin" / "retrieval_mcp_cli.py"
PLUGIN_SOURCE = REPO_ROOT / "mcp_plugin" / "retrieval_mcp.py"
STDIO_CLIENT = REPO_ROOT / "tests" / "mcp_daemon_stdio_client.py"
DEFAULT_DAEMON_PORT = 17871
# Measured on the real Windows SDK-client burst after the stdlib raw bridge:
# 27.84 MiB maximum shim RSS.  This leaves cross-machine headroom, yet catches
# the previous client-side MCP/HTTP bridge (about 73 MiB) and any full server.
MAX_THIN_SHIM_RSS_BYTES = 48 * 1024 * 1024


def test_full_tool_schema_digest_normalizes_docstring_margin_across_pydantic_versions() -> None:
    """Pydantic 2.13/Python 3.10 preserves a docstring's common margin.

    The 0.4.7 Linux candidate exposed the second spelling while the Windows
    candidate exposed the first.  They represent the same MCP description;
    the singleton fence must retain the description's meaningful content but
    not reject a daemon for this runtime renderer detail.
    """
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    windows_tool = SimpleNamespace(
        name="search_code",
        description=(
            "Search indexed code.\n"
            "Use one precise query.\n\n"
            "Args:\n"
            "    query: Required exact or semantic phrase.\n"
        ),
        inputSchema=schema,
    )
    linux_tool = SimpleNamespace(
        name="search_code",
        description=(
            "Search indexed code.\n"
            "    Use one precise query.\n\n"
            "    Args:\n"
            "        query: Required exact or semantic phrase.\n"
            "    "
        ),
        inputSchema=schema,
    )

    windows_surface = daemon.canonical_tool_surface([windows_tool])
    linux_surface = daemon.canonical_tool_surface([linux_tool])

    assert windows_surface == linux_surface
    assert daemon.canonical_tool_surface_digest(windows_surface) == (
        daemon.canonical_tool_surface_digest(linux_surface)
    )
    # A decoded remote tools/list payload must take the exact same canonical
    # path; otherwise merely deserializing the Linux response would reintroduce
    # the false version fence.
    assert daemon.canonical_tool_surface_digest(windows_surface) == (
        daemon.canonical_tool_surface_digest([
            {
                "name": linux_tool.name,
                "description": linux_tool.description,
                "inputSchema": linux_tool.inputSchema,
            }
        ])
    )


def test_schema_drift_error_reports_digests_and_safe_first_diff_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real drift must be diagnosable without echoing a tool description."""
    expected_description = "expected description must never be logged"
    actual_description = "actual description must never be logged"
    input_schema = {"type": "object", "properties": {}}
    expected_description_digest = hashlib.sha256(
        expected_description.encode("utf-8")
    ).hexdigest()
    input_schema_digest = hashlib.sha256(
        json.dumps(input_schema, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    tool = SimpleNamespace(
        name="corpus_stats",
        description=actual_description,
        inputSchema=input_schema,
    )

    class _McpWithDrift:
        async def list_tools(self) -> list[SimpleNamespace]:
            return [tool]

    monkeypatch.setattr(daemon, "TOOL_SURFACE_NAMES", frozenset({"corpus_stats"}))
    monkeypatch.setattr(
        daemon,
        "TOOL_SURFACE_NAMES_DIGEST",
        daemon._tool_name_surface_digest(frozenset({"corpus_stats"})),
    )
    monkeypatch.setattr(daemon, "TOOL_SURFACE_DIGEST", "expected-full-schema-digest")
    monkeypatch.setattr(
        daemon,
        "TOOL_SURFACE_FIELD_DIGESTS",
        {
            "corpus_stats": {
                "description": expected_description_digest,
                "inputSchema": input_schema_digest,
            }
        },
        raising=False,
    )

    with pytest.raises(daemon.DaemonControlError) as caught:
        daemon.serve_daemon(
            _McpWithDrift(),
            runtime_dir=tmp_path / "runtime",
            instance_token="t" * 32,
        )

    message = str(caught.value)
    assert "expected_digest=expected-full-schema-digest" in message
    assert "actual_digest=" in message
    assert "first_diff_path=tools[corpus_stats].description" in message
    assert expected_description not in message
    assert actual_description not in message


class _BackendHandler(http.server.BaseHTTPRequestHandler):
    """Small real HTTP backend sufficient for the protocol smoke tool call."""

    def log_message(self, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path == "/api/stats":
            body = json.dumps({
                "total_papers": 7,
                "total_chunks": 70,
                "vector_index_size": 70,
                "embedding_model": "daemon-cli-stub",
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


@pytest.fixture
def backend_url() -> str:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _daemon_env(backend_url: str, runtime_dir: Path, *, test_port: int | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "RETRIEVAL_API_URL": backend_url,
        "ACE_PLAYBOOK_DB": str(runtime_dir / "playbook.db"),
        "RETRIEVAL_MCP_STATE_DIR": str(runtime_dir),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    })
    # The public CLI intentionally has no port knob: production configuration is
    # the stable 127.0.0.1:17871/mcp URL.  Tests opt into an isolated loopback port
    # solely so they can never collide with a developer's real daemon.  The raw
    # bridge must receive its *own* independently probed port: ``http + 1`` may
    # already belong to a preceding isolated daemon on Windows.
    if test_port is not None:
        test_raw_port = _free_loopback_port()
        while test_raw_port == test_port:
            test_raw_port = _free_loopback_port()
        env["RETRIEVAL_MCP_TEST_PORT"] = str(test_port)
        env["RETRIEVAL_MCP_TEST_RAW_PORT"] = str(test_raw_port)
    return env


def test_isolated_daemon_environment_probes_a_distinct_raw_bridge_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A test HTTP port must never imply an unprobed adjacent raw port."""
    selected = iter([41000, 41001])
    monkeypatch.setattr(sys.modules[__name__], "_free_loopback_port", lambda: next(selected))

    env = _daemon_env("http://127.0.0.1:8000", tmp_path / "runtime", test_port=41000)

    assert env["RETRIEVAL_MCP_TEST_PORT"] == "41000"
    assert env["RETRIEVAL_MCP_TEST_RAW_PORT"] == "41001"


def _cli(
    *args: str,
    env: dict[str, str],
    timeout: float = 25,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=str(REPO_ROOT),
        env=env,
        input="",
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _server_cli(
    *args: str,
    env: dict[str, str],
    timeout: float = 25,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mcp_server.server", *args],
        cwd=str(REPO_ROOT),
        env=env,
        input="",
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _plugin_source_cli(
    *args: str,
    env: dict[str, str],
    timeout: float = 25,
) -> subprocess.CompletedProcess[str]:
    """Old direct source invocation must dispatch to the same thin control plane."""
    return subprocess.run(
        [sys.executable, str(PLUGIN_SOURCE), *args],
        cwd=str(REPO_ROOT),
        env=env,
        input="",
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _payload(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, (
        f"CLI failed ({result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"CLI must return one JSON status document, not stdio/log output: {result.stdout!r}\n"
            f"stderr:\n{result.stderr}\n{exc}"
        )
    assert isinstance(value, dict)
    return value


def _wait_for_path_absent(path: Path, *, timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not path.exists():
            return True
        time.sleep(0.05)
    return not path.exists()


def _best_effort_stop(runtime_dir: Path, env: dict[str, str]) -> None:
    """Only ask the new CLI to stop a state file created in this test's temp dir."""
    if not (runtime_dir / "daemon.json").exists():
        return
    try:
        _cli("stop", "--state-dir", str(runtime_dir), env=env, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        # Do not guess a PID or kill anything here: that is exactly the behaviour
        # this test suite is meant to prevent.
        pass


async def _mcp_stats(
    url: str,
    *,
    instance_token: str | None = None,
) -> tuple[list[str], str]:
    """Drive the real HTTP MCP endpoint, optionally as its daemon owner."""
    headers = ({"X-Retrieval-MCP-Instance": instance_token} if instance_token else {})
    async with httpx.AsyncClient(headers=headers, trust_env=False) as http_client:
        async with streamable_http_client(
            url,
            http_client=http_client,
            terminate_on_close=False,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = [tool.name for tool in (await session.list_tools()).tools]
                reply = await session.call_tool("corpus_stats", {})
                text = "\n".join(getattr(part, "text", "") for part in (reply.content or []))
                return names, text


async def _two_mcp_stats(url: str, *, instance_token: str) -> list[tuple[list[str], str]]:
    return list(await asyncio.gather(
        _mcp_stats(url, instance_token=instance_token),
        _mcp_stats(url, instance_token=instance_token),
    ))


def _wait_for_files(
    paths: list[Path],
    *,
    timeout: float = 15,
    processes_by_ready_file: dict[Path, subprocess.Popen[str]] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(path.exists() for path in paths):
            return
        time.sleep(0.05)
    missing = [str(path) for path in paths if not path.exists()]
    errors = {
        str(path.with_suffix(".error")): path.with_suffix(".error").read_text(encoding="utf-8")
        for path in paths
        if path.with_suffix(".error").exists()
    }
    exited_clients: dict[str, dict[str, object]] = {}
    for path, process in (processes_by_ready_file or {}).items():
        if process.poll() is None:
            continue
        stdout, stderr = process.communicate()
        exited_clients[str(path)] = {
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    pytest.fail(
        "stdio MCP clients never reached initialize/list_tools/call: "
        f"{missing}; errors={errors}; exited_clients={exited_clients}"
    )


def _resident_bytes(pid: int) -> int:
    """Read one process's RSS from the OS directly, independent of the daemon's own code."""
    if os.name == "nt":
        probe = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                f"(Get-Process -Id {int(pid)} -ErrorAction Stop).WorkingSet64",
            ],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert probe.returncode == 0, probe.stderr
        return int(probe.stdout.strip())
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    probe = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(int(pid))],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert probe.returncode == 0 and probe.stdout.strip(), probe.stderr
    return int(probe.stdout.strip()) * 1024


def _resident_bytes_many(pids: set[int]) -> dict[int, int]:
    """Collect RSS for a client burst in one OS query, independent of the daemon's own code."""
    if not pids:
        return {}
    if os.name == "nt":
        wanted = ",".join(str(int(pid)) for pid in sorted(pids))
        probe = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "$wanted = @(" + wanted + "); "
                "Get-Process | Where-Object { $wanted -contains $_.Id } | "
                "Select-Object Id,WorkingSet64 | ConvertTo-Json -Compress",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert probe.returncode == 0, probe.stderr
        raw = probe.stdout.strip()
        records = [] if not raw else json.loads(raw)
        if isinstance(records, dict):
            records = [records]
        return {int(record["Id"]): int(record["WorkingSet64"]) for record in records}
    return {pid: _resident_bytes(pid) for pid in pids if Path(f"/proc/{pid}").exists()}


def _live_pids(pids: set[int]) -> set[int]:
    return set(_resident_bytes_many(pids))


def _parent_pids(pids: set[int]) -> dict[int, int]:
    """Capture an OS process tree without adding a runtime process library."""
    if not pids:
        return {}
    if os.name == "nt":
        wanted = ",".join(str(int(pid)) for pid in sorted(pids))
        probe = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "$wanted = @("
                + wanted
                + "); Get-CimInstance Win32_Process | Where-Object { $wanted -contains $_.ProcessId } | "
                "Select-Object ProcessId,ParentProcessId | ConvertTo-Json -Compress",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert probe.returncode == 0, probe.stderr
        raw = probe.stdout.strip()
        records = [] if not raw else json.loads(raw)
        if isinstance(records, dict):
            records = [records]
        return {
            int(record["ProcessId"]): int(record["ParentProcessId"])
            for record in records
        }
    result: dict[int, int] = {}
    for pid in pids:
        stat_path = Path(f"/proc/{pid}/stat")
        if stat_path.exists():
            fields = stat_path.read_text(encoding="utf-8").split()
            if len(fields) > 3:
                result[pid] = int(fields[3])
    return result


def test_default_cli_is_a_thin_stdio_proxy_with_one_daemon(
    backend_url: str,
    tmp_path: Path,
) -> None:
    """Four existing command-style clients share one daemon and never load its tools."""
    runtime_dir = tmp_path / "runtime"
    shim_dir = tmp_path / "shims"
    port = _free_loopback_port()
    env = _daemon_env(backend_url, runtime_dir, test_port=port)
    env["RETRIEVAL_MCP_TEST_SHIM_DIR"] = str(shim_dir)
    ready_files = [tmp_path / f"stdio-client-{index}.json" for index in range(4)]

    def start_stdio_client(ready_file: Path, *, hold_seconds: str) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [
                sys.executable, str(STDIO_CLIENT), "--cli", str(CLI), "--cwd", str(REPO_ROOT),
                "--ready-file", str(ready_file), "--hold-seconds", hold_seconds,
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    processes = [
        start_stdio_client(ready_file, hold_seconds="15")
        for ready_file in ready_files
    ]
    try:
        _wait_for_files(ready_files)
        shim_files = sorted(shim_dir.glob("shim-*.json"))
        assert len(shim_files) == 4, [str(path) for path in shim_files]
        shims = [json.loads(path.read_text(encoding="utf-8")) for path in shim_files]
        shim_pids = {int(record["pid"]) for record in shims}
        assert len(shim_pids) == 4
        assert all(record["full_server_loaded"] is False for record in shims)
        assert all(record["mcp_sdk_loaded"] is False for record in shims)

        live_state = json.loads((runtime_dir / "daemon.json").read_text(encoding="utf-8"))
        assert int(live_state["pid"]) not in shim_pids
        assert live_state["url"] == f"http://127.0.0.1:{port}/mcp"

        # A full FastMCP/chonkie tool process is materially larger than a stdlib
        # bridge.  This threshold was set only after the 24-client real SDK burst.
        rss = {pid: _resident_bytes(pid) for pid in shim_pids}
        assert max(rss.values()) <= MAX_THIN_SHIM_RSS_BYTES, rss

        for ready_file in ready_files:
            ready = json.loads(ready_file.read_text(encoding="utf-8"))
            assert "corpus_stats" in ready["tools"]
            assert "7 papers" in ready["corpus_stats"]

        # This fifth control client must diagnose a healthy shared daemon while
        # the first four stdio sessions are still held open.  It performs a real
        # HTTP MCP initialize/list_tools after token-authenticated health.
        status = _payload(_cli("status", "--state-dir", str(runtime_dir), env=env))
        assert status["running"] is True, status
        assert status["health"] is True
        assert status["mcp_handshake"] is True
        assert status["pid"] == live_state["pid"]

        sixth_ready = tmp_path / "stdio-client-sixth.json"
        processes.append(start_stdio_client(sixth_ready, hold_seconds="2"))
        _wait_for_files([sixth_ready])
        sixth = json.loads(sixth_ready.read_text(encoding="utf-8"))
        assert "corpus_stats" in sixth["tools"]
        assert "7 papers" in sixth["corpus_stats"]

        # Stop is also exercised with active client sessions.  It must address the
        # exact token-authenticated daemon, rather than any of the four shims.
        stopped = _payload(_cli("stop", "--state-dir", str(runtime_dir), env=env))
        assert stopped["stopped"] is True
        assert stopped["pid"] == live_state["pid"]
        assert _wait_for_path_absent(runtime_dir / "daemon.json")

        for process in processes:
            if process.poll() is None:
                process.terminate()  # exact test-owned MCP client helper
            process.wait(timeout=5)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()  # exact test-owned helper process
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        _best_effort_stop(runtime_dir, env)


@pytest.mark.parametrize("transport", ["sdk", "native-subprocess"])
def test_stdio_proxy_exits_naturally_after_client_eof_and_keeps_daemon_running(
    backend_url: str,
    tmp_path: Path,
    transport: str,
) -> None:
    """A normal MCP client close must not leave the transparent shim stranded.

    This covers the stdio EOF path independently of ``stop``.  Terminating a
    test-owned client is useful cleanup for a failed test, but it is not an
    acceptable substitute for a bridge that lets the client SDK close normally.
    """
    runtime_dir = tmp_path / "runtime"
    shim_dir = tmp_path / "shims"
    port = _free_loopback_port()
    env = _daemon_env(backend_url, runtime_dir, test_port=port)
    env["RETRIEVAL_MCP_TEST_SHIM_DIR"] = str(shim_dir)
    ready_file = tmp_path / "stdio-client.json"
    process = subprocess.Popen(
        [
            sys.executable, str(STDIO_CLIENT), "--cli", str(CLI), "--cwd", str(REPO_ROOT),
            "--ready-file", str(ready_file), "--hold-seconds", "0.1",
            "--transport", transport,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_files([ready_file])
        shim_files = list(shim_dir.glob("shim-*.json"))
        assert len(shim_files) == 1, [str(path) for path in shim_files]
        shim = json.loads(shim_files[0].read_text(encoding="utf-8"))
        # Default command-mode clients must stay a stdlib pipe/socket forwarder.
        # The full MCP SDK belongs exclusively to the singleton daemon process.
        assert shim["full_server_loaded"] is False
        assert shim["mcp_sdk_loaded"] is False
        assert shim["httpx_loaded"] is False
        assert shim["anyio_loaded"] is False
        live_state = json.loads((runtime_dir / "daemon.json").read_text(encoding="utf-8"))
        live_pid = int(live_state["pid"])
        daemon_parent_pid = _parent_pids({live_pid}).get(live_pid)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, f"client failed\nstdout:\n{stdout}\nstderr:\n{stderr}"

        # A shim EOF only closes its session.  The singleton is deliberately
        # independent of a particular agent and stays available for the next one.
        state_before_status = json.loads((runtime_dir / "daemon.json").read_text(encoding="utf-8"))
        daemon_pid = int(state_before_status["pid"])
        if os.name == "nt":
            daemon_exists = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", f"Get-Process -Id {daemon_pid}"],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            ).returncode == 0
        else:
            daemon_exists = True
        status = _payload(_cli("status", "--state-dir", str(runtime_dir), env=env))
        assert status["running"] is True, {
            "status": status,
            "daemon_pid": daemon_pid,
            "daemon_exists": daemon_exists,
            "daemon_parent_pid_while_live": daemon_parent_pid,
        }
        assert status["health"] is True
        assert status["mcp_handshake"] is True
    finally:
        if process.poll() is None:
            process.terminate()  # exact child only if the assertion timed out
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        _best_effort_stop(runtime_dir, env)


def test_twenty_four_default_stdio_clients_share_one_daemon_and_exit_cleanly(
    backend_url: str,
    tmp_path: Path,
) -> None:
    """A 24-agent burst keeps one full server, then releases every thin shim.

    The two batches model a real workstation's first wave of agents followed by
    another wave while the first sessions are still connected.  All clients use
    the official SDK stdio transport; no transport, daemon, or backend call is
    mocked.  The printed metric record is intentionally consumed with ``-s``
    before this test's RSS budget is tightened from measured values.
    """
    runtime_dir = tmp_path / "runtime"
    shim_dir = tmp_path / "shims"
    port = _free_loopback_port()
    env = _daemon_env(backend_url, runtime_dir, test_port=port)
    env["RETRIEVAL_MCP_TEST_SHIM_DIR"] = str(shim_dir)
    ready_files = [tmp_path / f"stdio-client-{index}.json" for index in range(24)]
    release_file = tmp_path / "release-clients"

    def start_client(ready_file: Path) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [
                sys.executable, str(STDIO_CLIENT), "--cli", str(CLI), "--cwd", str(REPO_ROOT),
                "--ready-file", str(ready_file), "--hold-seconds", "0", "--transport", "sdk",
                "--release-file", str(release_file),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    processes: list[subprocess.Popen[str]] = []
    try:
        # Deliberately burst in two groups: all four control paths must keep a
        # single lock/state identity while existing SDK sessions stay open.
        for batch in (ready_files[:8], ready_files[8:16], ready_files[16:]):
            processes.extend(start_client(path) for path in batch)
            # The batches are launched without waiting for prior sessions, so
            # all 24 contend on the first singleton startup rather than merely
            # serializing test setup.
            time.sleep(0.1)
        _wait_for_files(
            ready_files,
            timeout=55,
            processes_by_ready_file={path: process for path, process in zip(ready_files, processes)},
        )

        shim_files = sorted(shim_dir.glob("shim-*.json"))
        assert len(shim_files) == 24, [str(path) for path in shim_files]
        shims = [json.loads(path.read_text(encoding="utf-8")) for path in shim_files]
        shim_pids = {int(record["pid"]) for record in shims}
        assert len(shim_pids) == 24
        assert all(record["full_server_loaded"] is False for record in shims)
        assert all(record["mcp_sdk_loaded"] is False for record in shims)

        state = json.loads((runtime_dir / "daemon.json").read_text(encoding="utf-8"))
        daemon_pid = int(state["pid"])
        assert daemon_pid not in shim_pids
        shim_rss = _resident_bytes_many(shim_pids)
        assert set(shim_rss) == shim_pids, shim_rss
        daemon_rss = _resident_bytes_many({daemon_pid})
        assert daemon_rss[daemon_pid] > 0
        shim_parent_pids = _parent_pids(shim_pids)
        daemon_parent_pids = _parent_pids({daemon_pid})
        assert set(shim_parent_pids) == shim_pids, shim_parent_pids
        assert daemon_pid in daemon_parent_pids, daemon_parent_pids
        assert max(shim_rss.values()) <= MAX_THIN_SHIM_RSS_BYTES, shim_rss

        for ready_file in ready_files:
            ready = json.loads(ready_file.read_text(encoding="utf-8"))
            assert "corpus_stats" in ready["tools"]
            assert "7 papers" in ready["corpus_stats"]

        # This is a 25th concurrent control client: health plus actual MCP
        # initialize/list_tools must work before any held stdio client closes.
        status = _payload(_cli("status", "--state-dir", str(runtime_dir), env=env))
        assert status["running"] is True
        assert status["health"] is True
        assert status["mcp_handshake"] is True
        assert status["pid"] == daemon_pid

        release_file.write_text("release\n", encoding="utf-8")
        for process in processes:
            stdout, stderr = process.communicate(timeout=35)
            assert process.returncode == 0, f"client failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
        assert not _live_pids(shim_pids), "thin shims must disappear after normal client EOF"

        # EOF releases only sessions.  A later agent can still use the one
        # daemon until a user explicitly token-authenticates ``stop``.
        after_clients = _payload(_cli("status", "--state-dir", str(runtime_dir), env=env))
        assert after_clients["running"] is True
        assert after_clients["pid"] == daemon_pid

        rss_values = list(shim_rss.values())
        combined_rss = daemon_rss[daemon_pid] + sum(rss_values)
        baseline_bytes = int(2.3 * 1024**3)
        metrics = {
            "daemon_pid": daemon_pid,
            "daemon_rss_bytes": daemon_rss[daemon_pid],
            "shim_count": len(shim_pids),
            "shim_rss_min_bytes": min(rss_values),
            "shim_rss_median_bytes": int(statistics.median(rss_values)),
            "shim_rss_max_bytes": max(rss_values),
            "shim_rss_total_bytes": sum(rss_values),
            "daemon_plus_shims_bytes": combined_rss,
            "reduction_vs_2_3_gib_bytes": baseline_bytes - combined_rss,
            "daemon_parent_pid": daemon_parent_pids[daemon_pid],
            "shim_parent_pids": {str(pid): shim_parent_pids[pid] for pid in sorted(shim_pids)},
        }
        print("MCP_DAEMON_STRESS_METRICS=" + json.dumps(metrics, sort_keys=True))

        stopped = _payload(_cli("stop", "--state-dir", str(runtime_dir), env=env))
        assert stopped["stopped"] is True
        assert stopped["pid"] == daemon_pid
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()  # exact test-owned helper after a failed assertion
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        _best_effort_stop(runtime_dir, env)


@pytest.mark.parametrize(
    "runner",
    [_cli, _plugin_source_cli, _server_cli],
    ids=["thin-plugin", "plugin-source", "repo-server"],
)
def test_status_is_a_non_starting_json_control_command(
    runner: Any,
    backend_url: str,
    tmp_path: Path,
) -> None:
    """Both distributions expose the same non-starting control-plane boundary."""
    runtime_dir = tmp_path / "runtime"
    payload = _payload(runner("status", "--state-dir", str(runtime_dir), env=_daemon_env(backend_url, runtime_dir)))
    assert payload["running"] is False
    assert payload["url"] == f"http://127.0.0.1:{DEFAULT_DAEMON_PORT}/mcp"
    assert payload["health"] is False
    assert payload["mcp_handshake"] is False


def test_http_mcp_requires_instance_token_while_raw_and_stdio_paths_remain_private(
    backend_url: str,
    tmp_path: Path,
) -> None:
    """The loopback HTTP compatibility endpoint is not an unauthenticated LAN API."""
    runtime_dir = tmp_path / "runtime"
    port = _free_loopback_port()
    env = _daemon_env(backend_url, runtime_dir, test_port=port)
    try:
        started = _payload(_cli("start", "--state-dir", str(runtime_dir), env=env))
        url = str(started["url"])
        token = str(started["instance_token"])
        initialize = {
            "jsonrpc": "2.0",
            "id": "unauthorized-initialize",
            "method": "initialize",
            "params": {},
        }

        missing = httpx.post(url, json=initialize, timeout=5, trust_env=False)
        wrong = httpx.post(
            url,
            json=initialize,
            headers={"X-Retrieval-MCP-Instance": "wrong-instance-token"},
            timeout=5,
            trust_env=False,
        )
        assert missing.status_code == 401
        assert wrong.status_code == 403

        names, text = asyncio.run(_mcp_stats(url, instance_token=token))
        assert "corpus_stats" in names
        assert "7 papers" in text

        # The stdio proxy's private JSONL transport has its own token handshake;
        # tightening HTTP must not degrade its already-authenticated control path.
        state = daemon.DaemonState.from_mapping(
            json.loads((runtime_dir / "daemon.json").read_text(encoding="utf-8")))
        assert state is not None
        raw_health = daemon._raw_control(state, "health", timeout=5)
        assert raw_health["ready"] is True
        assert raw_health["instance_token"] == token
    finally:
        _best_effort_stop(runtime_dir, env)


def test_start_is_idempotent_singleton_and_serves_two_real_mcp_clients(
    backend_url: str,
    tmp_path: Path,
) -> None:
    """Concurrent ``start`` calls must create exactly one daemon, not one per agent."""
    runtime_dir = tmp_path / "runtime"
    port = _free_loopback_port()
    env = _daemon_env(backend_url, runtime_dir, test_port=port)
    args = ("start", "--state-dir", str(runtime_dir))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: _cli(*args, env=env), range(4)))
        payloads = [_payload(result) for result in results]
        pids = {int(payload["pid"]) for payload in payloads}
        tokens = {str(payload["instance_token"]) for payload in payloads}
        assert len(pids) == 1
        assert len(tokens) == 1
        assert {payload["url"] for payload in payloads} == {f"http://127.0.0.1:{port}/mcp"}
        assert all(payload["health"] and payload["mcp_handshake"] for payload in payloads)

        status = _payload(_cli("status", "--state-dir", str(runtime_dir), env=env))
        assert status["running"] is True
        assert status["pid"] in pids
        assert status["instance_token"] in tokens
        assert status["health"] is True
        assert status["mcp_handshake"] is True

        clients = asyncio.run(_two_mcp_stats(
            status["url"],
            instance_token=str(status["instance_token"]),
        ))
        assert all("corpus_stats" in names for names, _ in clients)
        assert all("7 papers" in text for _, text in clients)

        # The clients did not execute ``start`` and the same daemon identity remains.
        after_clients = _payload(_cli("status", "--state-dir", str(runtime_dir), env=env))
        assert after_clients["pid"] == status["pid"]
        assert after_clients["instance_token"] == status["instance_token"]

        stopped = _payload(_cli("stop", "--state-dir", str(runtime_dir), env=env))
        assert stopped["stopped"] is True
        assert stopped["pid"] == status["pid"]
        assert _wait_for_path_absent(runtime_dir / "daemon.json")
    finally:
        _best_effort_stop(runtime_dir, env)


def test_incompatible_daemon_version_is_visible_and_new_shim_refuses_to_reuse_it(
    backend_url: str,
    tmp_path: Path,
) -> None:
    """A newer shim must not silently use or kill an older healthy daemon."""
    runtime_dir = tmp_path / "runtime"
    port = _free_loopback_port()
    env = _daemon_env(backend_url, runtime_dir, test_port=port)
    try:
        started = _payload(_cli("start", "--state-dir", str(runtime_dir), env=env))
        state_path = runtime_dir / "daemon.json"
        old_state = json.loads(state_path.read_text(encoding="utf-8"))

        # This simulates a still-live previous release.  The old token remains
        # correct, so the test proves that compatibility—not a missing identity—
        # makes the new shim refuse reuse.
        old_state.update({
            "package_version": "0.0.0-previous",
            "protocol_schema": 0,
            "tool_surface_digest": "previous-tool-surface",
        })
        state_path.write_text(json.dumps(old_state), encoding="utf-8")

        health = httpx.get(
            f"http://127.0.0.1:{port}/health",
            headers={"X-Retrieval-MCP-Instance": str(started["instance_token"])},
            timeout=5,
            trust_env=False,
        )
        assert health.status_code == 200
        health_body = health.json()
        assert health_body["package_version"] != "0.0.0-previous"
        assert health_body["protocol_schema"] != 0
        assert health_body["tool_surface_digest"] != "previous-tool-surface"

        status = _payload(_cli("status", "--state-dir", str(runtime_dir), env=env))
        assert status["health"] is True
        assert status["mcp_handshake"] is True
        assert status["identity"] is True
        assert status["compatible"] is False
        assert status["running"] is False
        assert status["package_version"] == "0.0.0-previous"
        assert status["expected_package_version"] != "0.0.0-previous"
        assert "retrieval-mcp stop" in status["reason"]

        retry = _cli("start", "--state-dir", str(runtime_dir), env=env)
        assert retry.returncode != 0
        retry_payload = json.loads(retry.stdout)
        assert "retrieval-mcp stop" in retry_payload["reason"]

        # Refusal is non-destructive: the original token-authenticated daemon
        # remains live until the user explicitly stops that exact identity.
        still_live = httpx.get(
            f"http://127.0.0.1:{port}/health",
            headers={"X-Retrieval-MCP-Instance": str(started["instance_token"])},
            timeout=5,
            trust_env=False,
        )
        assert still_live.status_code == 200

        stopped = _payload(_cli("stop", "--state-dir", str(runtime_dir), env=env))
        assert stopped["stopped"] is True
    finally:
        _best_effort_stop(runtime_dir, env)


def test_stale_reused_pid_state_is_recovered_without_terminating_that_process(
    backend_url: str,
    tmp_path: Path,
) -> None:
    """A stale state file may be reclaimed, but an unrelated reused PID is untouchable."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    port = _free_loopback_port()
    env = _daemon_env(backend_url, runtime_dir, test_port=port)
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stale_token = "stale-token-that-is-not-a-daemon"
    stale = {
        "pid": unrelated.pid,
        "instance_token": stale_token,
        "port": port,
        "url": f"http://127.0.0.1:{port}/mcp",
    }
    lock_dir = runtime_dir / "daemon.lock"
    lock_dir.mkdir()
    (lock_dir / "owner.json").write_text(json.dumps(stale), encoding="utf-8")
    (runtime_dir / "daemon.json").write_text(json.dumps(stale), encoding="utf-8")
    try:
        started = _payload(_cli(
            "start", "--state-dir", str(runtime_dir), env=env,
        ))
        assert started["pid"] != unrelated.pid
        assert started["instance_token"] != stale_token
        assert unrelated.poll() is None, "stale-state recovery must never kill a PID-reused process"
    finally:
        _best_effort_stop(runtime_dir, env)
        if unrelated.poll() is None:
            unrelated.terminate()  # exact test-owned child, never an ambient process
            try:
                unrelated.wait(timeout=5)
            except subprocess.TimeoutExpired:
                unrelated.kill()


def test_stop_refuses_a_non_daemon_pid_even_when_its_number_is_in_state(
    backend_url: str,
    tmp_path: Path,
) -> None:
    """``stop`` needs the running command-line token, not merely a PID from JSON."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    env = _daemon_env(backend_url, runtime_dir)
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    state = {
        "pid": unrelated.pid,
        "instance_token": "not-present-in-this-child-command-line",
        "port": _free_loopback_port(),
        "url": "http://127.0.0.1:1/mcp",
    }
    (runtime_dir / "daemon.json").write_text(json.dumps(state), encoding="utf-8")
    try:
        result = _cli("stop", "--state-dir", str(runtime_dir), env=env)
        assert result.returncode != 0
        assert unrelated.poll() is None, "stop must not terminate a non-daemon PID"
        assert "identity" in (result.stdout + result.stderr).lower()
    finally:
        if unrelated.poll() is None:
            unrelated.terminate()  # exact test-owned child
            try:
                unrelated.wait(timeout=5)
            except subprocess.TimeoutExpired:
                unrelated.kill()


def test_stop_fails_closed_if_pid_could_change_after_identity_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-termination token check cannot make a later PID kill safe.

    This models the narrow race in which a command-line identity check succeeds,
    then the OS recycles that numeric PID before a fallback termination call.
    Raw authenticated shutdown is unavailable here, so the only safe outcome is
    a diagnostic refusal—not a best-effort kill of whichever process owns the
    number at the next instruction.
    """
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    state = daemon.DaemonState(
        pid=424242,
        instance_token="identity-race-token",
        port=_free_loopback_port(),
        started_at=time.time(),
    )
    (runtime_dir / "daemon.json").write_text(json.dumps(state.as_dict()), encoding="utf-8")
    termination_attempts: list[int] = []

    monkeypatch.setattr(daemon, "_probe_health", lambda _state: False)
    monkeypatch.setattr(daemon, "_identity_matches", lambda _state: True)
    monkeypatch.setattr(
        daemon,
        "_terminate_exact_process",
        lambda pid: termination_attempts.append(pid) or True,
        raising=False,
    )
    monkeypatch.setattr(daemon, "_wait_for_exit", lambda _pid, _timeout: True)

    code, payload = daemon._stop_daemon(runtime_dir)

    assert code == 3
    assert payload["stopped"] is False
    assert payload["reason"] == (
        "authenticated shutdown unavailable; refusing unsafe PID termination"
    )
    assert termination_attempts == []


def test_startup_timeout_never_falls_back_to_pid_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch timeout also fails closed when no authenticated daemon exists yet."""
    runtime_dir = tmp_path / "runtime"
    termination_attempts: list[int] = []

    class _UnverifiedProcess:
        pid = 515151

        @staticmethod
        def poll() -> None:
            return None

    monkeypatch.setattr(daemon, "STARTUP_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(daemon, "_cleanup_stale_state", lambda _runtime_dir: None)
    monkeypatch.setattr(daemon, "_recover_stale_daemon_lock", lambda _runtime_dir: True)
    monkeypatch.setattr(daemon, "_launch_daemon", lambda *_args, **_kwargs: _UnverifiedProcess())
    monkeypatch.setattr(daemon, "_identity_matches", lambda _state: True)
    monkeypatch.setattr(
        daemon,
        "_terminate_exact_process",
        lambda pid: termination_attempts.append(pid) or True,
        raising=False,
    )

    with pytest.raises(
        daemon.DaemonControlError,
        match="startup timeout left daemon unverified; refusing unsafe PID termination",
    ):
        daemon._ensure_daemon(runtime_dir, ["unused-test-serve-command"])

    assert termination_attempts == []


def test_startup_lock_recovery_passes_the_observed_owner_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contender may reclaim only the exact stale owner it first observed."""
    runtime_dir = tmp_path / "runtime"
    lock_dir = runtime_dir / "startup.lock"
    lock_dir.mkdir(parents=True)
    observed_token = "stale-owner-token"
    (lock_dir / "owner.json").write_text(json.dumps({
        "pid": 999999,
        "lock_token": observed_token,
        "started_at": 0,
    }), encoding="utf-8")
    captured: list[str | None] = []

    class _ObservedToken(Exception):
        pass

    def capture(path: Path, token: str | None = None) -> bool:
        assert path == lock_dir
        captured.append(token)
        raise _ObservedToken

    monkeypatch.setattr(daemon, "_pid_exists", lambda _pid: False)
    monkeypatch.setattr(daemon, "_remove_startup_lock", capture)

    with pytest.raises(_ObservedToken):
        with daemon._StartupLock(runtime_dir):
            pass
    assert captured == [observed_token]


def test_startup_lock_reclaim_does_not_delete_replaced_owner_across_processes(
    tmp_path: Path,
) -> None:
    """A stale-reclaim race must preserve a newer lock claimed by another process."""
    runtime_dir = tmp_path / "runtime"
    lock_dir = runtime_dir / "startup.lock"
    lock_dir.mkdir(parents=True)
    old_token = "observed-stale-owner"
    new_token = "new-owner-must-survive"
    owner_path = lock_dir / "owner.json"
    owner_path.write_text(json.dumps({
        "pid": 999999,
        "lock_token": old_token,
        "started_at": 0,
    }), encoding="utf-8")
    ready_path = tmp_path / "reclaimer-read-owner"
    proceed_path = tmp_path / "replace-owner-now"
    result_path = tmp_path / "reclaimer-result.json"
    child = r"""
import json
import sys
import time
from pathlib import Path
from mcp_plugin import retrieval_mcp_daemon as daemon

lock_path = Path(sys.argv[1])
expected_token = sys.argv[2]
ready_path = Path(sys.argv[3])
proceed_path = Path(sys.argv[4])
result_path = Path(sys.argv[5])
owner_path = lock_path / "owner.json"
original_read = daemon._read_json
seen_owner = False

def paused_read(path):
    global seen_owner
    value = original_read(path)
    if path == owner_path and not seen_owner:
        seen_owner = True
        ready_path.write_text("ready", encoding="utf-8")
        deadline = time.monotonic() + 10
        while not proceed_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
    return value

daemon._read_json = paused_read
try:
    result = {"removed": daemon._remove_startup_lock(lock_path, expected_token)}
except BaseException as exc:
    result = {"error": repr(exc)}
result_path.write_text(json.dumps(result), encoding="utf-8")
"""
    process = subprocess.Popen(
        [
            sys.executable, "-c", child, str(lock_dir), old_token,
            str(ready_path), str(proceed_path), str(result_path),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_files([ready_path], timeout=10)
        owner_path.unlink()
        lock_dir.rmdir()
        lock_dir.mkdir()
        owner_path.write_text(json.dumps({
            "pid": os.getpid(),
            "lock_token": new_token,
            "started_at": time.time(),
        }), encoding="utf-8")
        proceed_path.write_text("replace complete", encoding="utf-8")
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, f"reclaimer failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result == {"removed": False}
        current = json.loads(owner_path.read_text(encoding="utf-8"))
        assert current["lock_token"] == new_token
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def test_backend_profile_normalizes_url_and_fingerprints_secret_without_storing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equivalent backend spellings share a profile; raw API keys never do."""
    monkeypatch.setenv("RETRIEVAL_API_URL", "HTTP://Example.TEST:80/api/")
    monkeypatch.setenv("RETRIEVAL_API_KEY", "profile-secret-a")
    first = daemon.backend_profile_from_environment("http://unused-default:8000")
    monkeypatch.setenv("RETRIEVAL_API_URL", "http://example.test/api")
    second = daemon.backend_profile_from_environment("http://unused-default:8000")

    assert first.normalized_url == "http://example.test/api"
    assert first.url_fingerprint == second.url_fingerprint
    assert first.api_key_fingerprint == second.api_key_fingerprint
    assert first.fingerprint == second.fingerprint
    assert "profile-secret-a" not in repr(first)


def test_live_daemon_refuses_reuse_with_a_different_backend_profile(
    backend_url: str,
    tmp_path: Path,
) -> None:
    """One per-user daemon cannot silently carry another client's backend key."""
    runtime_dir = tmp_path / "runtime"
    port = _free_loopback_port()
    first_env = _daemon_env(backend_url + "/", runtime_dir, test_port=port)
    first_env["RETRIEVAL_API_KEY"] = "profile-secret-a"
    equivalent_env = dict(first_env)
    equivalent_env["RETRIEVAL_API_URL"] = backend_url
    changed_env = dict(first_env)
    changed_env["RETRIEVAL_API_KEY"] = "profile-secret-b"
    try:
        first = _payload(_cli("start", "--state-dir", str(runtime_dir), env=first_env))
        persisted = (runtime_dir / "daemon.json").read_text(encoding="utf-8")
        assert "profile-secret-a" not in persisted
        state = json.loads(persisted)
        assert state["backend_profile_fingerprint"]
        assert state["backend_url_fingerprint"]
        assert state["api_key_fingerprint"]

        equivalent = _payload(_cli(
            "start", "--state-dir", str(runtime_dir), env=equivalent_env,
        ))
        assert equivalent["action"] == "already_running"
        assert equivalent["pid"] == first["pid"]

        refused = _cli("start", "--state-dir", str(runtime_dir), env=changed_env)
        assert refused.returncode != 0
        refused_payload = json.loads(refused.stdout)
        assert "backend_profile" in refused_payload["reason"]

        status = _payload(_cli("status", "--state-dir", str(runtime_dir), env=changed_env))
        assert status["running"] is False
        assert status["health"] is True
        assert status["mcp_handshake"] is True
        assert status["compatible"] is False
        assert "backend_profile" in status["reason"]
    finally:
        _best_effort_stop(runtime_dir, first_env)


def test_runtime_directory_and_control_files_are_private(tmp_path: Path) -> None:
    """State/token files and daemon diagnostics are never group-readable on POSIX."""
    runtime_dir = daemon._ensure_private_runtime_dir(tmp_path / "runtime")
    state_path = runtime_dir / "daemon.json"
    daemon._write_json_atomic(state_path, {"instance_token": "private-token"})
    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "instance_token": "private-token"
    }, "Windows ACL hardening must not lock the creating/current daemon process out"
    log_handle = daemon._open_private_log(runtime_dir / "daemon.log")
    try:
        log_handle.write(b"private daemon diagnostic\n")
        log_handle.flush()
    finally:
        log_handle.close()

    if os.name != "nt":
        assert stat.S_IMODE(runtime_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
        assert stat.S_IMODE((runtime_dir / "daemon.log").stat().st_mode) == 0o600


def test_runtime_security_rejects_symlinked_runtime_and_state_paths(tmp_path: Path) -> None:
    """The state namespace must not follow a user-controlled link or junction."""
    target = tmp_path / "outside"
    target.mkdir()
    runtime_link = tmp_path / "runtime-link"
    try:
        os.symlink(target, runtime_link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"this Windows test account cannot create a test symlink: {exc}")

    with pytest.raises(daemon.DaemonControlError, match="symlink"):
        daemon._ensure_private_runtime_dir(runtime_link)

    runtime_dir = daemon._ensure_private_runtime_dir(tmp_path / "runtime")
    state_target = tmp_path / "outside-state.json"
    state_target.write_text("{}", encoding="utf-8")
    os.symlink(state_target, runtime_dir / "daemon.json")
    with pytest.raises(daemon.DaemonControlError, match="symlink"):
        daemon._write_json_atomic(runtime_dir / "daemon.json", {"pid": 1})


def test_runtime_security_requests_windows_acl_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Windows branch must attempt ACL hardening without making it a service."""
    calls: list[Path] = []
    monkeypatch.setattr(daemon, "_is_windows", lambda: True)
    monkeypatch.setattr(daemon, "_apply_windows_private_acl", lambda path: calls.append(path))

    runtime_dir = daemon._ensure_private_runtime_dir(tmp_path / "runtime")

    assert calls == [runtime_dir]
