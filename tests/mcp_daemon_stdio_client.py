"""One real stdio MCP client used by ``test_mcp_daemon_cli``.

It intentionally uses the official SDK rather than speaking JSON-RPC itself.
Keeping the session open gives the parent test time to inspect the thin shim
process that the SDK launched.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage


@asynccontextmanager
async def _native_subprocess_stdio_client(
    params: StdioServerParameters,
) -> AsyncIterator[tuple[object, object]]:
    """Real pipe transport without the SDK's Windows test Job Object wrapper.

    The wire framing and ``ClientSession`` are still the official MCP SDK.  The
    process itself is deliberately opened with the stdlib/AnyIO primitive so this
    helper can separately prove normal EOF behavior; the SDK's Windows wrapper
    deliberately kills every child in its Job after a stuck server deadline.
    """
    read_writer, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](0)
    write_stream, write_reader = anyio.create_memory_object_stream[SessionMessage](0)
    process = await anyio.open_process(
        [params.command, *params.args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        env=params.env,
        cwd=params.cwd,
    )
    assert process.stdin is not None
    assert process.stdout is not None

    async def stdout_reader() -> None:
        try:
            buffer = b""
            async with read_writer:
                async for chunk in process.stdout:
                    lines = (buffer + chunk).split(b"\n")
                    buffer = lines.pop()
                    for line in lines:
                        if not line:
                            continue
                        try:
                            message = JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:
                            await read_writer.send(exc)
                        else:
                            await read_writer.send(SessionMessage(message))
        finally:
            await read_writer.aclose()

    async def stdin_writer() -> None:
        try:
            async with write_reader:
                async for session_message in write_reader:
                    raw = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    await process.stdin.send((raw + "\n").encode("utf-8"))
        finally:
            await process.stdin.aclose()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(stdout_reader)
        task_group.start_soon(stdin_writer)
        try:
            yield read_stream, write_stream
        finally:
            await write_stream.aclose()
            with anyio.move_on_after(5):
                await process.wait()
            if process.returncode is None:
                process.terminate()  # exact helper-owned shim after bounded EOF wait
                with anyio.move_on_after(2):
                    await process.wait()
            task_group.cancel_scope.cancel()
            await read_stream.aclose()
            await read_writer.aclose()


async def _drive(args: argparse.Namespace) -> dict[str, object]:
    params = StdioServerParameters(
        command=sys.executable,
        args=[args.cli],
        env=dict(os.environ),
        cwd=args.cwd,
    )
    transport = stdio_client if args.transport == "sdk" else _native_subprocess_stdio_client
    async with transport(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [tool.name for tool in (await session.list_tools()).tools]
            reply = await session.call_tool(args.tool, json.loads(args.tool_args))
            text = "\n".join(getattr(part, "text", "") for part in (reply.content or []))
            # Keyed by tool name as well so a caller can read the result without
            # knowing which tool this client was pointed at.
            ready = {"tools": names, "tool": args.tool, "tool_result": text, args.tool: text}
            Path(args.ready_file).write_text(json.dumps(ready), encoding="utf-8")
            if args.release_file:
                release_path = Path(args.release_file)
                while not release_path.exists():
                    await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(args.hold_seconds)
            return ready


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--transport", choices=["sdk", "native-subprocess"], default="sdk")
    parser.add_argument("--tool", default="corpus_stats", help="tool this session calls once")
    parser.add_argument("--tool-args", default="{}", help="JSON arguments for --tool")
    parser.add_argument("--release-file", help="test-only gate that holds a live client session")
    args = parser.parse_args()
    try:
        print(json.dumps(asyncio.run(_drive(args))))
    except Exception as exc:  # parent reports stderr/output as test evidence
        Path(args.ready_file).with_suffix(".error").write_text(repr(exc), encoding="utf-8")
        print(f"stdio MCP client failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
