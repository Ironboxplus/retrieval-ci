"""Drive the real plugin tools against a test Uvicorn process, as a separate process.

``sync``       -> start the background conversation sync exactly as the tools do, wait for it,
                  and make NO other request (so the server log shows the sync's own traffic).
``search``     -> ``journal_search(path_glob=".conversations/")``, then wait for the sync.
``index_dir``  -> recursive ``journal_index_dir(pattern="*.md")``, then wait for the sync.

The result is one JSON line on stdout.  Endpoint, deja roots, state dir and the feature
switch all arrive through the environment, like a real MCP client configuration.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "mcp_plugin"


async def _run(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(PLUGIN_DIR))
    import retrieval_mcp as plugin

    conversations = plugin._conversations_module()
    plugin._CONVERSATION_SYNCER = conversations.ConversationSyncer(
        post=plugin._post, get=plugin._get, state_dir=plugin._conversation_state_dir,
        min_interval_s=0)

    payload: dict = {"scope": plugin._scope_path(args.scope)}
    if args.op == "sync":
        payload["kick_line"] = plugin._conversation_sync_line(args.scope)
    elif args.op == "search":
        payload["result"] = await plugin.journal_search(
            query=args.query, scope_dir=args.scope, mode="keyword", top_k=20,
            chunk_limit=1, path_glob=".conversations/")
    elif args.op == "index_dir":
        payload["result"] = await plugin.journal_index_dir(
            dir=args.scope, recursive=True, pattern="*.md", respect_gitignore=False)
    else:
        raise SystemExit(f"unknown op {args.op}")

    tasks = list(plugin._CONVERSATION_SYNCER._tasks.values())
    if tasks:
        await asyncio.gather(*tasks)
    key = next(iter(plugin._CONVERSATION_SYNCER._tasks), None)
    payload["final_line"] = plugin._CONVERSATION_SYNCER._line(key, False) if key else None
    last = plugin._CONVERSATION_SYNCER._last.get(key) if key else None
    result = (last or {}).get("result")
    if result is not None:
        payload["summary"] = {
            "sessions_seen": result.sessions_seen, "sessions_changed": result.sessions_changed,
            "segments_uploaded": result.segments_uploaded, "requests": result.requests,
            "errors": result.errors}
    elif last:
        payload["summary"] = {k: v for k, v in last.items() if k != "at"}
    await plugin._client().aclose()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=["sync", "search", "index_dir"])
    parser.add_argument("scope")
    parser.add_argument("--query", default="alpha0_0")
    args = parser.parse_args()
    try:
        print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False))
    except Exception as exc:  # CLI contract: keep the class name and exit nonzero.
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
