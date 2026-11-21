"""Thin console entry point for the singleton Retrieval MCP daemon.

The installed ``retrieval-mcp`` command imports this file first.  Ordinary
stdio clients therefore never import the large tool-registration module; only
the daemon's internal ``serve`` and explicit ``stdio-server`` commands do.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from retrieval_mcp_daemon import DEFAULT_PLUGIN_BACKEND_URL, run_thin_cli


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"serve", "stdio-server"}:
        # Delayed on purpose: this is the only path that imports FastMCP, chonkie,
        # and the 32 registered Retrieval tools in this process.
        import retrieval_mcp

        return retrieval_mcp.main(args)
    return run_thin_cli(
        args,
        serve_command=[str(Path(__file__).with_name("retrieval_mcp.py"))],
        backend_default_url=DEFAULT_PLUGIN_BACKEND_URL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
