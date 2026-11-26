"""Invoke the real MCP ``journal_index_dir`` tool against a test Uvicorn process.

This is intentionally a subprocess rather than a test-side direct function call:
the subprocess imports the actual MCP HTTP client, inherits only explicit endpoint
configuration, and emits its HTTP request log for the no-upload assertions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time


def _job_id(text: str) -> str | None:
    match = re.search(r"job '([^']+)'", text)
    return match.group(1) if match else None


async def _run(args: argparse.Namespace) -> dict[str, object]:
    from mcp_server import server

    result = await server.journal_index_dir(
        dir=args.directory,
        recursive=args.recursive,
        pattern=args.pattern,
        max_depth=args.max_depth,
        respect_gitignore=False,
    )
    payload: dict[str, object] = {"result": result, "job": None}
    job_id = _job_id(result)
    if job_id:
        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            job = await server._client.journal_bulk_index_job(job_id)
            payload["job"] = job
            if job.get("status") in {"completed", "failed"}:
                break
            await asyncio.sleep(0.25)
        else:
            raise TimeoutError(f"Journal bulk job {job_id} did not reach a terminal state")
    await server._client.aclose()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--pattern", default="*.md")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    try:
        print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False))
    except Exception as exc:  # CLI contract: retain exception class and nonzero exit.
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
