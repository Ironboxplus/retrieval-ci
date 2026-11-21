"""Real-HTTP proof of the inventory-baseline traffic of the MCP plugin.

A real uvicorn process (tests/index_baseline_http_app.py: production Index Catalog and Code
KB routes, temp SQLite, scratch Qdrant collection, request log) is driven by the real
plugin module. The same scenarios also run through the plugin as of ``HEAD`` -- loaded
from ``git show`` -- so each "before -> after" is two measurements on one server, and the
old client doubles as the old-client/new-server compatibility check.

Skipped unless a Qdrant answers on 127.0.0.1:6333 (the code jobs write real vectors to a
test-private collection that is dropped afterwards).
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402

from tests.qdrant_scratch import drop  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "mcp_plugin" / "retrieval_mcp.py"
# The commit the "old" client comes from.  A CI snapshot commit already contains the
# working-tree change under test, so CI sets this to the snapshot's parent (HEAD^).
OLD_PLUGIN_REF = os.environ.get("INDEX_BASELINE_OLD_REF", "HEAD")
QDRANT_URL = "http://127.0.0.1:6333"
OWNER = "e2e-user"
HOST = "e2e-host"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_plugin(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    try:
        qdrant = QdrantClient(url=QDRANT_URL, timeout=10)
        qdrant.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {QDRANT_URL}: {exc}")
    root = tmp_path_factory.mktemp("baseline_http")
    log = root / "requests.jsonl"
    log.write_text("", encoding="utf-8")
    collection = f"e2e_baseline_code_{os.getpid()}_test"
    drop(qdrant, [collection])
    port = _free_port()
    env = os.environ.copy()
    env.update({"DATA_DIR": str(root / "data"), "REQUEST_LOG": str(log),
                "E2E_CODE_COLLECTION": collection, "PYTHONPATH": str(REPO),
                "PYTHONUNBUFFERED": "1"})
    out = (root / "uvicorn.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tests.index_baseline_http_app:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"uvicorn exited early: {(root / 'uvicorn.log').read_text()[-2000:]}")
        try:
            if httpx.get(f"{url}/__ready", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.3)
    else:
        proc.kill()
        pytest.fail("uvicorn did not become ready")
    yield type("Server", (), {"url": url, "log": log})
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    out.close()
    drop(qdrant, [collection])


@pytest.fixture(scope="module")
def plugins(tmp_path_factory):
    old_dir = tmp_path_factory.mktemp("head_plugin")
    old_path = old_dir / "retrieval_mcp.py"
    old_path.write_bytes(subprocess.check_output(
        ["git", "show", f"{OLD_PLUGIN_REF}:mcp_plugin/retrieval_mcp.py"], cwd=REPO))
    return _load_plugin(PLUGIN, "e2e_new_plugin"), _load_plugin(old_path, "e2e_head_plugin")


class Traffic:
    def __init__(self, server):
        self.server = server

    def mark(self) -> int:
        return len(self._entries())

    def _entries(self) -> list[dict]:
        text = self.server.log.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def since(self, mark: int) -> list[dict]:
        return self._entries()[mark:]

    @staticmethod
    def requests(entries: list[dict]) -> list[tuple[str, str]]:
        return [(e["method"], e["path"]) for e in entries if e["event"] == "request"]


@pytest.fixture()
def traffic(server, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_HOST", HOST)
    return Traffic(server)


def _connect(mod, server):
    mod._CLIENT = httpx.AsyncClient(
        base_url=server.url, timeout=60,
        headers={**getattr(mod, "_HEADERS", {}), "X-Retrival-User-Id": OWNER})


def _run(plugins, server, scenario):
    """Run ``scenario(new, old)`` on one loop with both plugin clients pointed at the server."""
    new, old = plugins

    async def main():
        for mod in (new, old):
            _connect(mod, server)
        try:
            return await scenario(new, old)
        finally:
            for mod in (new, old):
                await mod._CLIENT.aclose()
                mod._CLIENT = None

    return asyncio.run(main())


async def _post(mod, path, payload):
    return await mod._post(path, payload)


def _doc_record(root: Path, rel: str) -> dict:
    data = (root / rel).read_bytes()
    return {"rel_path": rel, "kind": "doc", "collection": "journal_chunks", "chunk_count": 2,
            "content_hash": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


# ---------------------------------------------------------------- journal no-op ----


def test_journal_rerun_with_nothing_changed_skips_report_and_reconcile(
        server, plugins, traffic, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (docs / name).write_text(f"# {name}\nbody of {name}\n", encoding="utf-8")

    async def scenario(new, old):
        scope = new._scope_path(str(docs))
        await _post(new, "/api/index/report", {
            "host": HOST, "project": scope, "root_dir": str(docs), "replace": True,
            "files": [_doc_record(docs, n) for n in ("a.md", "b.md", "c.md")]})

        mark = traffic.mark()
        old_text = await old.journal_index_dir(dir=str(docs), pattern="*.md")
        old_traffic = traffic.requests(traffic.since(mark))

        mark = traffic.mark()
        new_text = await new.journal_index_dir(dir=str(docs), pattern="*.md")
        new_traffic = traffic.requests(traffic.since(mark))
        return old_text, old_traffic, new_text, new_traffic

    old_text, old_traffic, new_text, new_traffic = _run(plugins, server, scenario)

    assert new_text == old_text  # the same words for the same situation
    assert ("POST", "/api/index/report") in old_traffic
    assert ("POST", "/api/journal/docs/reconcile") in old_traffic
    assert new_traffic == [("GET", "/api/index/scope-overlap"), ("GET", "/api/index/inventory")]
    assert len(old_traffic) == 4 and len(new_traffic) == 2


def test_journal_rerun_still_sweeps_a_file_deleted_locally(server, plugins, traffic, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a.md", "b.md"):
        (docs / name).write_text(f"# {name}\n", encoding="utf-8")

    async def scenario(new, old):
        scope = new._scope_path(str(docs))
        await _post(new, "/api/index/report", {
            "host": HOST, "project": scope, "root_dir": str(docs), "replace": True,
            "files": [_doc_record(docs, n) for n in ("a.md", "b.md")]})
        (docs / "b.md").unlink()
        mark = traffic.mark()
        text = await new.journal_index_dir(dir=str(docs), pattern="*.md")
        sent = traffic.requests(traffic.since(mark))
        inventory = await new._get("/api/index/inventory", {
            "host": HOST, "project": scope, "kind": "doc", "detail": True})
        live = sorted(f["rel_path"] for u in inventory["users"] for h in u["hosts"]
                      for p in h["projects"] for d in p["dirs"] for f in d["files"])
        return text, sent, live

    text, sent, live = _run(plugins, server, scenario)

    assert ("POST", "/api/index/report") in sent
    assert ("POST", "/api/journal/docs/reconcile") in sent
    assert live == ["a.md"]  # b.md swept from the catalog
    assert "reported 1 to inventory" in text


def test_journal_rerun_with_an_edited_file_is_not_a_noop(server, plugins, traffic, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# a\none\n", encoding="utf-8")

    async def scenario(new, old):
        scope = new._scope_path(str(docs))
        await _post(new, "/api/index/report", {
            "host": HOST, "project": scope, "root_dir": str(docs), "replace": True,
            "files": [_doc_record(docs, "a.md")]})
        (docs / "a.md").write_text("# a\ntwo\n", encoding="utf-8")
        mark = traffic.mark()
        await new.journal_index_dir(dir=str(docs), pattern="*.md")
        return traffic.requests(traffic.since(mark))

    sent = _run(plugins, server, scenario)
    # The changed body is offered to the Journal (bulk, then per-entry compatibility path;
    # this server has no Journal store, so those calls fail -- what matters is that they
    # are made at all instead of the file being declared unchanged).
    assert ("POST", "/api/journal/bulk-index") in sent or ("POST", "/api/journal/entries") in sent


# ------------------------------------------------------------------ lean baseline ----


def test_lean_baseline_has_the_same_diff_data_in_a_fraction_of_the_bytes(
        server, plugins, traffic, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    records = [{
        "rel_path": f"pkg{i % 9}/module_{i}.py", "kind": "code", "collection": "code_chunks",
        "language": "python", "chunk_count": i % 5 + 1,
        "content_hash": hashlib.sha256(str(i).encode()).hexdigest(), "size_bytes": 900 + i,
        "metadata": {"git_commit": "abc1234", "note": "x" * 80}} for i in range(600)]

    async def scenario(new, old):
        scope = new._scope_path(str(repo))
        await _post(new, "/api/index/report", {
            "host": HOST, "project": scope, "root_dir": str(repo), "files": records,
            "replace": True})
        mark = traffic.mark()
        old_known = await old._known_code_hashes(scope)
        old_entries = traffic.since(mark)
        mark = traffic.mark()
        new_known = await new._known_code_hashes(scope)
        new_entries = traffic.since(mark)
        return old_known, old_entries, new_known, new_entries

    old_known, old_entries, new_known, new_entries = _run(plugins, server, scenario)

    fields = ("rel_path", "content_hash", "chunk_count", "language", "kind", "size_bytes")
    assert old_known.keys() == new_known.keys() and len(new_known) == 600
    for rel, node in new_known.items():
        assert {f: node.get(f) for f in fields} == {f: old_known[rel].get(f) for f in fields}
    old_bytes, new_bytes = old_entries[0]["length"], new_entries[0]["length"]
    assert len(old_entries) == len(new_entries) == 1
    assert parse_qs(new_entries[0]["query"]).get("lean") == ["true"]
    assert "lean" not in parse_qs(old_entries[0]["query"])
    print(f"BASELINE bytes full={old_bytes} lean={new_bytes} "
          f"ms full={old_entries[0]['ms']} lean={new_entries[0]['ms']}")
    assert new_bytes < old_bytes * 0.7, (old_bytes, new_bytes)


# ------------------------------------------------------------- inventory filter ----


def test_index_inventory_project_filter_is_answered_by_the_server(
        server, plugins, traffic, tmp_path):
    async def scenario(new, old):
        for i in range(40):
            await _post(new, "/api/index/report", {
                "host": HOST, "project": f"/e2e/filter/proj{i:02d}", "root_dir": f"/e2e/filter/proj{i:02d}",
                "replace": True, "files": [{
                    "rel_path": "main.py", "kind": "code", "chunk_count": 1,
                    "content_hash": "h", "metadata": {"git_commit": f"c{i}"}}]})
        await _post(new, "/api/index/report", {
            "host": HOST, "project": "C:/Users/Me/needle-project", "root_dir": "C:/Users/Me/needle-project",
            "replace": True, "files": [{"rel_path": "x.py", "kind": "code", "chunk_count": 3,
                                        "content_hash": "n"}]})
        outputs = {}
        for label, mod in (("old", old), ("new", new)):
            mark = traffic.mark()
            text = await mod.index_inventory(project="Needle")
            outputs[label] = (text, traffic.since(mark))
        return outputs

    outputs = _run(plugins, server, scenario)

    assert outputs["new"][0] == outputs["old"][0]
    assert "needle-project" in outputs["new"][0] and "proj00" not in outputs["new"][0]
    (old_entry,), (new_entry,) = outputs["old"][1], outputs["new"][1]
    assert parse_qs(new_entry["query"]).get("project_contains") == ["Needle"]
    assert "project_contains" not in parse_qs(old_entry["query"])
    print(f"INVENTORY bytes full={old_entry['length']} filtered={new_entry['length']} "
          f"ms full={old_entry['ms']} filtered={new_entry['ms']}")
    assert new_entry["length"] < old_entry["length"] * 0.5, (old_entry["length"], new_entry["length"])


# ------------------------------------------------------ code: removals + watcher ----


def _code_repo(root: Path) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in ("alpha", "beta", "gamma", "delta"):
        (root / f"{name}.py").write_text(
            f"def {name}_fn():\n    return '{name}-v1'\n", encoding="utf-8")
    return root


async def _index_and_wait(mod, repo: Path, **kwargs) -> dict:
    reply = await mod.index_code(str(repo), **kwargs)
    match = re.search(r"job ([0-9a-f]{32})", reply)
    assert match, reply
    local = match.group(1)
    task = mod._CODE_PREPARATION_TASKS.get(local)
    if task is not None:
        await task
    backend = mod._CODE_PREPARATION_STATUS[local].get("backend_job_id")
    assert backend, mod._CODE_PREPARATION_STATUS[local]
    job = await mod._wait_code_job(backend)
    assert job["status"] == "completed", job
    return job


def _qdrant_paths(collection_prefix: str, scope: str) -> set[str]:
    client = QdrantClient(url=QDRANT_URL, timeout=30)
    name = next(c.name for c in client.get_collections().collections
                if c.name.startswith(collection_prefix))
    paths: set[str] = set()
    offset = None
    while True:
        rows, offset = client.scroll(name, limit=500, offset=offset, with_payload=True,
                                     with_vectors=False)
        paths.update(r.payload["path"] for r in rows if r.payload.get("project") == scope)
        if offset is None:
            return paths


def _stop_watcher(mod, scope):
    for key, task in list(mod._CODE_WATCH_TASKS.items()):
        if key[1] == scope and not task.done():
            task.cancel()


@pytest.mark.parametrize("which", ["old", "new"])
def test_incremental_index_code_removes_deleted_files_and_only_the_new_client_skips_the_path_facet(
        server, plugins, traffic, tmp_path, which):
    repo = _code_repo(tmp_path / f"repo_{which}")

    async def scenario(new, old):
        mod = new if which == "new" else old
        scope = new._scope_path(str(repo))
        await _index_and_wait(mod, repo, auto_refresh=False)
        assert _qdrant_paths("e2e_baseline_code_", scope) == {
            "alpha.py", "beta.py", "gamma.py", "delta.py"}
        (repo / "beta.py").unlink()
        (repo / "gamma.py").write_text("def gamma_fn():\n    return 'gamma-v2'\n", encoding="utf-8")
        (repo / "epsilon.py").write_text("def eps():\n    return 1\n", encoding="utf-8")
        mark = traffic.mark()
        await _index_and_wait(mod, repo, incremental=True, auto_refresh=False)
        entries = traffic.since(mark)
        return scope, entries

    scope, entries = _run(plugins, server, scenario)

    # the same end state for both clients: the deleted file is gone, edits/new files in
    assert _qdrant_paths("e2e_baseline_code_", scope) == {
        "alpha.py", "gamma.py", "delta.py", "epsilon.py"}
    facets = [e for e in entries if e["event"] == "qdrant_list_paths" and e["project"] == scope]
    inventory_reads = [e for e in entries if e["event"] == "request"
                       and e["path"] == "/api/index/inventory"]
    assert len(inventory_reads) == 1
    assert len(facets) == (0 if which == "new" else 1)


@pytest.mark.parametrize("which", ["old", "new"])
def test_gitignore_edit_reconciles_with_one_inventory_read_instead_of_two(
        server, plugins, traffic, tmp_path, which):
    repo = _code_repo(tmp_path / f"watch_{which}")

    async def scenario(new, old):
        mod = new if which == "new" else old
        scope = new._scope_path(str(repo))
        await _index_and_wait(mod, repo, auto_refresh=True)  # also starts the watcher
        key = (HOST, scope)
        deadline = time.monotonic() + 60
        while mod._CODE_WATCH_STATUS.get(key, {}).get("state") != "watching":
            assert time.monotonic() < deadline, mod._CODE_WATCH_STATUS.get(key)
            await asyncio.sleep(0.2)
        await asyncio.sleep(1.5)  # let the file watcher settle on this directory
        mark = traffic.mark()
        # one ignore-file edit that changes the tree: beta.py becomes ignored, gamma edited
        (repo / ".gitignore").write_text("beta.py\n", encoding="utf-8")
        (repo / "gamma.py").write_text("def gamma_fn():\n    return 'gamma-v2'\n", encoding="utf-8")
        deadline = time.monotonic() + 90
        while True:
            entries = traffic.since(mark)
            submitted = [e for e in entries if e["event"] == "request"
                         and e["method"] == "POST" and e["path"] == "/api/code/index"]
            if submitted:
                break
            assert time.monotonic() < deadline, "watcher never reacted to the .gitignore edit"
            await asyncio.sleep(0.3)
        # wait for the watcher to finish the cycle (job done -> baseline refreshed -> idle)
        await asyncio.sleep(1)
        deadline = time.monotonic() + 60
        while mod._CODE_WATCH_STATUS.get(key, {}).get("state") != "watching":
            assert time.monotonic() < deadline, mod._CODE_WATCH_STATUS.get(key)
            await asyncio.sleep(0.2)
        await asyncio.sleep(0.5)
        entries = traffic.since(mark)
        _stop_watcher(mod, scope)
        return scope, entries

    scope, entries = _run(plugins, server, scenario)

    # The watcher also loads the scope's document baseline once, lazily, on its first
    # batch (_resolve_watch_docs); that kind="doc" read is not part of the code cycle
    # this test measures.
    inventory_reads = [e for e in entries if e["event"] == "request"
                       and e["path"] == "/api/index/inventory" and "kind=code" in e["query"]]
    # The refresh job's own report is what the watcher keeps as its baseline.
    assert len(inventory_reads) == (1 if which == "new" else 2), [
        (e["path"], e["query"]) for e in entries if e["event"] == "request"]
    assert _qdrant_paths("e2e_baseline_code_", scope) == {"alpha.py", "gamma.py", "delta.py"}
