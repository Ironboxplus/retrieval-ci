"""Guard: an `async def` route/generator must not run synchronous DB work on the event loop.

2026-11-09 production measurement: `/api/papers/tags` is an `async def` that ran its
GROUP BY queries inline; for the 11 s it executed even a 30 ms `/api/dashboard/system`
waited, because every request shares the one event loop. The route modules are scanned
statically; each offending call must be a plain `def` route (FastAPI runs those in the
threadpool) or go through `asyncio.to_thread` / `run_in_threadpool`.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Every module whose router the compute app registers. Completeness is asserted against
# the real app's route table below, so a new router file cannot silently escape the scan.
SCANNED_FILES = sorted(
    [p for p in (ROOT / "app" / "api" / "routes").glob("*.py")]
    + [
        ROOT / "main.py",
        ROOT / "pipeline" / "paper_crawler" / "integration" / "api_routes.py",
        ROOT / "core" / "observability.py",
    ]
)

DB_FACTORIES = {"get_db_session", "SessionLocal", "sqlite3.connect"}
DB_RECEIVERS = {"db", "session", "sess", "conn", "connection", "cursor"}
OFFLOADERS = {
    "asyncio.to_thread",
    "to_thread",
    "run_in_threadpool",
    "anyio.to_thread.run_sync",
    "loop.run_in_executor",
}


def _root_name(node: ast.AST) -> str | None:
    while True:
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            return node.id if isinstance(node, ast.Name) else None


def _blocking_db_calls(fn: ast.AsyncFunctionDef) -> list[tuple[int, str]]:
    """Calls in `fn`'s own body (not nested defs/lambdas) that touch a DB on the loop."""
    found: dict[int, str] = {}

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return  # nested sync callables are the offload payload; nested async defs are scanned on their own
        if isinstance(node, ast.Call):
            callee = ast.unparse(node.func)
            if callee in OFFLOADERS or callee.endswith(".run_in_executor"):
                return
            db_arg = any(
                isinstance(a, ast.Name) and a.id in DB_RECEIVERS
                for a in [*node.args, *(k.value for k in node.keywords)]
            )
            if callee in DB_FACTORIES or _root_name(node.func) in DB_RECEIVERS or db_arg:
                found.setdefault(node.lineno, callee)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in fn.body:
        visit(stmt)
    return sorted(found.items())


def _scan(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            for lineno, callee in _blocking_db_calls(node):
                problems.append(f"{path.relative_to(ROOT)}:{lineno} async def {node.name} calls {callee}")
    return problems


def test_scanner_flags_inline_db_and_accepts_offload():
    bad = ast.parse(
        "async def route(db=None):\n"
        "    return db.query(1).all()\n"
        "async def route2():\n"
        "    with get_db_session() as s:\n"
        "        return helper(s)\n"
        "async def route3(db=None):\n"
        "    return build(db)\n"
    )
    good = ast.parse(
        "async def route(db=None):\n"
        "    def work():\n"
        "        return db.query(1).all()\n"
        "    return await asyncio.to_thread(work)\n"
        "async def route2(db=None):\n"
        "    return await run_in_threadpool(helper, db)\n"
    )
    flagged = [n.name for n in ast.walk(bad) if isinstance(n, ast.AsyncFunctionDef) and _blocking_db_calls(n)]
    assert flagged == ["route", "route2", "route3"]
    assert not [n for n in ast.walk(good) if isinstance(n, ast.AsyncFunctionDef) and _blocking_db_calls(n)]


@pytest.mark.parametrize("path", SCANNED_FILES, ids=lambda p: p.name)
def test_async_routes_do_not_run_sync_db_on_event_loop(path):
    assert _scan(path) == []


# The DB-call scan cannot see through a service layer: after the first fix,
# `/api/admin/console/overview` (`async def` calling `get_journal_service().stats()`)
# still took 7.7 s on the loop. An `async def` endpoint that never awaits gains nothing
# from the loop and blocks it on any synchronous call, so it must be a plain `def`.
# Kept async on purpose:
AWAITLESS_ASYNC_ALLOWED = {
    "search_stream",  # builds a request, returns an async streamer that offloads the pipeline
    "serve_frontend_asset",  # static files: must not queue behind slow threadpool routes
    "reject_frontend_asset_directory",
    "serve_spa",
}
ROUTE_DECORATOR_METHODS = ("get", "post", "put", "patch", "delete", "api_route", "websocket")


def _awaits(fn: ast.AsyncFunctionDef) -> bool:
    def visit(node: ast.AST) -> bool:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(child, (ast.Await, ast.AsyncFor, ast.AsyncWith)) or visit(child):
                return True
        return False

    return visit(fn)


def _is_route(fn: ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr in ROUTE_DECORATOR_METHODS
        for d in fn.decorator_list
    )


def _awaitless_async_routes(tree: ast.AST) -> list[ast.AsyncFunctionDef]:
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and _is_route(n) and not _awaits(n)
    ]


def test_awaitless_scanner():
    tree = ast.parse(
        "@router.get('/a')\n"
        "async def bad():\n"
        "    return service().stats()\n"
        "@router.get('/b')\n"
        "async def good():\n"
        "    return await asyncio.to_thread(service().stats)\n"
        "@router.get('/c')\n"
        "async def nested_only():\n"
        "    async def inner():\n"
        "        await x()\n"
        "    return inner\n"
        "async def not_a_route():\n"
        "    return 1\n"
    )
    assert [n.name for n in _awaitless_async_routes(tree)] == ["bad", "nested_only"]


@pytest.mark.parametrize("path", SCANNED_FILES, ids=lambda p: p.name)
def test_async_routes_await_something(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    offenders = [
        f"{path.relative_to(ROOT)}:{n.lineno} async def {n.name}"
        for n in _awaitless_async_routes(tree)
        if n.name not in AWAITLESS_ASYNC_ALLOWED
    ]
    assert offenders == []


def test_every_registered_route_lives_in_a_scanned_module():
    import main

    scanned = {p.resolve() for p in SCANNED_FILES}
    strays = set()
    for route in main.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        source = inspect.getsourcefile(inspect.unwrap(endpoint))
        if source and Path(source).resolve() not in scanned and ".venv" not in source:
            if "site-packages" not in source:
                strays.add(source)
    assert strays == set(), f"route endpoints outside the scanned modules: {sorted(strays)}"
