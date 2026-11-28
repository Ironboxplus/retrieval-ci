"""Black-box wheel/sdist gate for the singleton Retrieval MCP console entry."""
from __future__ import annotations

import asyncio
import http.server
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAEMON_PORT = 17871
DEFAULT_RAW_MCP_PORT = 17872
MAX_THIN_SHIM_RSS_BYTES = 48 * 1024 * 1024


# This gate skips whenever a local daemon owns 17871/17872, which is the normal
# state on a developer box — so anything hardcoded in it goes stale unnoticed and
# only fails during release verification on a clean machine. It did: the tool
# count sat at 32 after `find_duplicate_code` made the surface 33. Both figures
# are therefore derived from the same sources the package itself ships, never
# retyped here.
def _package_version() -> str:
    import tomllib

    return tomllib.loads(
        (REPO_ROOT / "mcp_plugin" / "pyproject.toml").read_text(encoding="utf-8"),
    )["project"]["version"]


def _expected_tool_count() -> int:
    plugin_dir = str(REPO_ROOT / "mcp_plugin")
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)
    import retrieval_mcp_daemon

    return len(retrieval_mcp_daemon.TOOL_SURFACE_NAMES)


class _StatsHandler(http.server.BaseHTTPRequestHandler):
    """A real loopback backend for the one tool needed by this artifact smoke."""

    def log_message(self, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        if self.path != "/api/stats":
            self.send_error(404)
            return
        body = json.dumps({
            "total_papers": 7,
            "total_chunks": 70,
            "vector_index_size": 70,
            "embedding_model": "package-artifact-stub",
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _free_loopback_port_pair() -> tuple[int, int]:
    """Choose independent test-only HTTP and raw bridge ports.

    The daemon production defaults are fixed, but the artifact gate deliberately
    overrides them.  ``port + 1`` is not a valid availability check on Windows:
    an unrelated local listener can own the adjacent port after the first probe
    closes.  Two independent ephemeral selections avoid making the package
    smoke depend on that unrelated process.
    """
    first = _free_loopback_port()
    second = _free_loopback_port()
    while second == first:
        second = _free_loopback_port()
    return first, second


def _fixed_daemon_ports_are_available() -> bool:
    """Preflight the owned fixed loopback pair without disturbing another daemon."""
    listeners: list[socket.socket] = []
    try:
        for port in (DEFAULT_DAEMON_PORT, DEFAULT_RAW_MCP_PORT):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", port))
            listeners.append(listener)
        return True
    except OSError:
        return False
    finally:
        for listener in listeners:
            listener.close()


def _resident_bytes_many(pids: set[int]) -> dict[int, int]:
    """Read a real process working set without adding a package dependency."""
    if not pids:
        return {}
    if os.name == "nt":
        wanted = ",".join(str(int(pid)) for pid in sorted(pids))
        probe = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "$wanted = @("
                + wanted
                + "); Get-Process | Where-Object { $wanted -contains $_.Id } | "
                "Select-Object Id,WorkingSet64 | ConvertTo-Json -Compress",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert probe.returncode == 0, probe.stderr
        payload = probe.stdout.strip()
        records = [] if not payload else json.loads(payload)
        if isinstance(records, dict):
            records = [records]
        return {int(record["Id"]): int(record["WorkingSet64"]) for record in records}
    result: dict[int, int] = {}
    for pid in pids:
        status = Path(f"/proc/{pid}/status")
        if not status.exists():
            continue
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                result[pid] = int(line.split()[1]) * 1024
                break
    return result


def _live_pids(pids: set[int]) -> set[int]:
    return set(_resident_bytes_many(pids))


def _console_path(venv_dir: Path) -> Path:
    directory = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    return directory / ("retrieval-mcp.exe" if os.name == "nt" else "retrieval-mcp")


def _run_console(console: Path, *args: str, env: dict[str, str]) -> dict:
    result = subprocess.run(
        [str(console), *args],
        cwd=str(console.parent.parent),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"console failed ({result.returncode}): {result.stderr}"
    value = json.loads(result.stdout)
    assert isinstance(value, dict)
    return value


async def _hold_real_default_client(
    console: Path,
    env: dict[str, str],
    ready: asyncio.Event,
    release: asyncio.Event,
) -> tuple[set[str], str]:
    """Use the official SDK over the installed no-argument console entry."""
    params = StdioServerParameters(
        command=str(console),
        args=[],
        env=env,
        cwd=str(console.parent.parent),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {tool.name for tool in (await session.list_tools()).tools}
            reply = await session.call_tool("corpus_stats", {})
            text = "\n".join(getattr(part, "text", "") for part in (reply.content or []))
            ready.set()
            await release.wait()
            return tools, text


def test_built_wheel_console_is_a_real_singleton_for_isolated_and_fixed_port_clients(
    tmp_path: Path,
) -> None:
    """Build/install the artifact and drive its default console through real clients.

    The package and its declared dependencies are installed into a fresh venv
    from the local uv cache with ``--offline``.  This verifies the built artifact
    rather than the repository source without downloading or publishing anything.
    The second half is deliberately the production fixed loopback pair.  It is
    skipped rather than displacing an already-running user daemon.
    """
    if not _fixed_daemon_ports_are_available():
        pytest.skip("127.0.0.1:17871/17872 belongs to an existing local daemon")
    supplied_wheel = os.environ.get("RETRIEVAL_MCP_ARTIFACT_WHEEL")
    if supplied_wheel:
        # Release verification may point this same black-box gate at a freshly
        # created candidate.  It still creates a new venv and never imports the
        # repository module as the installed console.
        wheel = Path(supplied_wheel).resolve()
        version = _package_version()
        assert wheel.is_file(), f"supplied Retrieval MCP wheel is absent: {wheel}"
        assert wheel.name == f"retrieval_mcp-{version}-py3-none-any.whl"
        assert (wheel.parent / f"retrieval_mcp-{version}.tar.gz").is_file()
    else:
        artifact_dir = tmp_path / "dist"
        artifact_dir.mkdir()
        build = subprocess.run(
            [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--sdist", "--outdir", str(artifact_dir), "mcp_plugin"],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert build.returncode == 0, build.stderr
        wheel = next(artifact_dir.glob("retrieval_mcp-*.whl"))
        assert list(artifact_dir.glob("retrieval_mcp-*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert {"retrieval_mcp.py", "retrieval_mcp_cli.py", "retrieval_mcp_daemon.py",
                "retrieval_mcp_conversations.py"} <= names
        entrypoints = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        assert "retrieval-mcp = retrieval_mcp_cli:main" in archive.read(entrypoints).decode("utf-8")

    artifact_venv = tmp_path / "artifact-venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", str(artifact_venv)],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    artifact_python = artifact_venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    installed = subprocess.run(
        ["uv", "pip", "install", "--offline", "--python", str(artifact_python), str(wheel)],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    console = _console_path(artifact_venv)
    assert console.exists()

    env = dict(os.environ)
    port, raw_port = _free_loopback_port_pair()
    env.update({
        "PYTHONNOUSERSITE": "1",
        "RETRIEVAL_API_URL": "",
        "RETRIEVAL_MCP_STATE_DIR": str(tmp_path / "runtime"),
        "RETRIEVAL_MCP_TEST_PORT": str(port),
        "RETRIEVAL_MCP_TEST_RAW_PORT": str(raw_port),
    })
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StatsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env["RETRIEVAL_API_URL"] = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        provenance = subprocess.run(
            [str(artifact_python), "-c", "import retrieval_mcp_cli; print(retrieval_mcp_cli.__file__)"],
            cwd=str(artifact_venv),
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        assert provenance.returncode == 0, provenance.stderr
        assert str(artifact_venv) in provenance.stdout

        async def drive() -> tuple[list[tuple[set[str], str]], dict]:
            ready = [asyncio.Event(), asyncio.Event()]
            release = asyncio.Event()
            clients = [
                asyncio.create_task(_hold_real_default_client(console, env, item, release))
                for item in ready
            ]
            try:
                await asyncio.wait_for(asyncio.gather(*(item.wait() for item in ready)), timeout=45)
                # A third process proves status performs real health/handshake
                # while both official SDK stdio sessions remain connected.
                status = await asyncio.to_thread(
                    _run_console,
                    console,
                    "status",
                    "--state-dir",
                    env["RETRIEVAL_MCP_STATE_DIR"],
                    env=env,
                )
                release.set()
                return await asyncio.gather(*clients), status
            finally:
                release.set()
                await asyncio.gather(*clients, return_exceptions=True)

        results, status = asyncio.run(drive())
        assert status["running"] is True
        assert status["health"] is True
        assert status["mcp_handshake"] is True
        live_state = json.loads(
            (Path(env["RETRIEVAL_MCP_STATE_DIR"]) / "daemon.json").read_text(encoding="utf-8"))
        assert status["pid"] == live_state["pid"]
        assert status["instance_token"] == live_state["instance_token"]
        for tools, text in results:
            assert len(tools) == _expected_tool_count()
            assert "corpus_stats" in tools
            assert "7 papers" in text

        stopped = _run_console(
            console,
            "stop",
            "--state-dir",
            env["RETRIEVAL_MCP_STATE_DIR"],
            env=env,
        )
        assert stopped["stopped"] is True
        assert stopped["pid"] == status["pid"]

        # A fresh installed wheel must also work through the fixed production
        # loopback pair.  It has its own temp state namespace but intentionally
        # no test port override, so this catches an accidental artifact-only
        # default-port or entrypoint drift without touching user state.
        fixed_env = dict(env)
        fixed_env.pop("RETRIEVAL_MCP_TEST_PORT", None)
        fixed_env.pop("RETRIEVAL_MCP_TEST_RAW_PORT", None)
        fixed_runtime = tmp_path / "fixed-runtime"
        fixed_shims = tmp_path / "fixed-shims"
        fixed_env["RETRIEVAL_MCP_STATE_DIR"] = str(fixed_runtime)
        fixed_env["RETRIEVAL_MCP_TEST_SHIM_DIR"] = str(fixed_shims)

        async def drive_twenty_four_fixed_port_clients() -> tuple[
            list[tuple[set[str], str]], dict, dict, dict[str, int], int,
        ]:
            ready = [asyncio.Event() for _ in range(24)]
            release = asyncio.Event()
            clients = [
                asyncio.create_task(_hold_real_default_client(console, fixed_env, item, release))
                for item in ready
            ]
            try:
                await asyncio.wait_for(asyncio.gather(*(item.wait() for item in ready)), timeout=60)
                state = json.loads((fixed_runtime / "daemon.json").read_text(encoding="utf-8"))
                shim_records = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted(fixed_shims.glob("shim-*.json"))
                ]
                assert len(shim_records) == 24
                shim_pids = {int(record["pid"]) for record in shim_records}
                assert len(shim_pids) == 24
                assert all(record["full_server_loaded"] is False for record in shim_records)
                assert all(record["mcp_sdk_loaded"] is False for record in shim_records)
                assert all(record["httpx_loaded"] is False for record in shim_records)
                assert all(record["anyio_loaded"] is False for record in shim_records)

                daemon_pid = int(state["pid"])
                assert daemon_pid not in shim_pids
                shim_rss = await asyncio.to_thread(_resident_bytes_many, shim_pids)
                assert set(shim_rss) == shim_pids, shim_rss
                daemon_rss = await asyncio.to_thread(_resident_bytes_many, {daemon_pid})
                assert daemon_rss[daemon_pid] > 0
                assert max(shim_rss.values()) <= MAX_THIN_SHIM_RSS_BYTES, shim_rss

                # Status is a 25th command-mode client while all 24 official SDK
                # sessions remain live, so it proves one healthy shared daemon.
                status_while_live = await asyncio.to_thread(
                    _run_console,
                    console,
                    "status",
                    "--state-dir",
                    str(fixed_runtime),
                    env=fixed_env,
                )
                release.set()
                results = await asyncio.gather(*clients)
                assert not await asyncio.to_thread(_live_pids, shim_pids)
                status_after_eof = await asyncio.to_thread(
                    _run_console,
                    console,
                    "status",
                    "--state-dir",
                    str(fixed_runtime),
                    env=fixed_env,
                )
                return results, status_while_live, status_after_eof, shim_rss, daemon_rss[daemon_pid]
            finally:
                release.set()
                await asyncio.gather(*clients, return_exceptions=True)

        fixed_results, fixed_status, fixed_after_eof, fixed_shim_rss, fixed_daemon_rss = asyncio.run(
            drive_twenty_four_fixed_port_clients())
        fixed_state = json.loads((fixed_runtime / "daemon.json").read_text(encoding="utf-8"))
        assert fixed_state["url"] == "http://127.0.0.1:17871/mcp"
        assert fixed_state["port"] == DEFAULT_DAEMON_PORT
        assert fixed_state["raw_mcp_port"] == DEFAULT_RAW_MCP_PORT
        assert fixed_state["package_version"] == _package_version()
        assert fixed_status["running"] is True
        assert fixed_status["health"] is True
        assert fixed_status["mcp_handshake"] is True
        assert fixed_status["pid"] == fixed_state["pid"]
        assert fixed_after_eof["running"] is True
        assert fixed_after_eof["pid"] == fixed_state["pid"]
        assert all(len(tools) == _expected_tool_count() and "corpus_stats" in tools
                   for tools, _ in fixed_results)
        assert all("7 papers" in text for _, text in fixed_results)
        daemon_log = (fixed_runtime / "daemon.log").read_text(encoding="utf-8")
        assert daemon_log.count("StreamableHTTP session manager started") == 1

        rss_values = list(fixed_shim_rss.values())
        metrics = {
            "daemon_pid": fixed_state["pid"],
            "daemon_rss_bytes": fixed_daemon_rss,
            "shim_count": len(rss_values),
            "shim_rss_min_bytes": min(rss_values),
            "shim_rss_median_bytes": int(statistics.median(rss_values)),
            "shim_rss_max_bytes": max(rss_values),
            "shim_rss_total_bytes": sum(rss_values),
            "daemon_plus_shims_bytes": fixed_daemon_rss + sum(rss_values),
        }
        (tmp_path / "artifact-fixed-port-stress-metrics.json").write_text(
            json.dumps(metrics, sort_keys=True),
            encoding="utf-8",
        )
        print("MCP_ARTIFACT_FIXED_PORT_STRESS_METRICS=" + json.dumps(metrics, sort_keys=True))
        fixed_stopped = _run_console(
            console,
            "stop",
            "--state-dir",
            str(fixed_runtime),
            env=fixed_env,
        )
        assert fixed_stopped["stopped"] is True
        assert fixed_stopped["pid"] == fixed_state["pid"]
    finally:
        runtime_state = Path(env["RETRIEVAL_MCP_STATE_DIR"]) / "daemon.json"
        if runtime_state.exists():
            try:
                _run_console(
                    console,
                    "stop",
                    "--state-dir",
                    env["RETRIEVAL_MCP_STATE_DIR"],
                    env=env,
                )
            except (OSError, AssertionError, subprocess.SubprocessError):
                pass
        fixed_runtime = locals().get("fixed_runtime")
        fixed_env = locals().get("fixed_env")
        if isinstance(fixed_runtime, Path) and isinstance(fixed_env, dict):
            fixed_state_path = fixed_runtime / "daemon.json"
            if fixed_state_path.exists():
                try:
                    _run_console(
                        console,
                        "stop",
                        "--state-dir",
                        str(fixed_runtime),
                        env=fixed_env,
                    )
                except (OSError, AssertionError, subprocess.SubprocessError):
                    pass
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
