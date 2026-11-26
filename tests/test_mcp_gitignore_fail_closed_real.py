"""Every gitignore check of the MCP client fails closed (real git, real files).

Incident 2026-11-21: the local daemon's filesystem watcher sent every file under a changed
directory to one ``git check-ignore`` call with a 10 s timeout and read a timeout / error as
"nothing is ignored". 128k gitignored JSON files were chunked and the daemon grew to 77 GB.
These tests drive git itself (a corrupt ``.git/index``, a timeout shorter than any process
start) and assert that no path becomes a candidate on an ignore answer git did not give.

Plugin (``mcp_plugin/retrieval_mcp.py``) and server (``mcp_server/server.py`` +
``core.code_kb.chunker``) carry parallel copies; the cases run against both.
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import core.code_kb.chunker as chunker
import mcp_server.server as server
from tests.test_mcp_plugin import (
    _Backend, _load, _sha, _start_watcher, _stop, _wait_until, _wire, _write)

GITIGNORE = b"tmp/**\n!tmp/**/\n!tmp/**/*.py\n"


class _NeverRaised(Exception):
    """Stands in for GitIgnoreCheckFailed on code that predates it, so a test fails with
    "DID NOT RAISE" rather than an AttributeError."""


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True,
                   stdin=subprocess.DEVNULL, capture_output=True)


def _make_repo(root: Path, json_count: int = 50, py_count: int = 1) -> Path:
    """A committed repo whose .gitignore ignores everything under tmp/ except .py files,
    with tmp/run/ holding json_count ignored .json files and py_count allowed .py files."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    (root / ".gitignore").write_bytes(GITIGNORE)
    run = root / "tmp" / "run"
    run.mkdir(parents=True)
    for i in range(json_count):
        (run / f"f{i:05d}.json").write_bytes(b'{"i": %d}\n' % i)
    for i in range(py_count):
        (run / f"m{i}.py").write_bytes(b"value = %d\n" % i)
    _git(root, "add", ".gitignore")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", "init")
    return root


def _corrupt_index(root: Path) -> bytes:
    index = root / ".git" / "index"
    original = index.read_bytes()
    index.write_bytes(b"garbage")
    return original


def _impl(name: str) -> SimpleNamespace:
    if name == "plugin":
        mod = _load()
        return SimpleNamespace(
            name=name, mod=mod, collect=mod._collect_watched_changes,
            allowed=mod._gitignore_allowed, ignored=mod._git_ignored_watch_paths,
            walk=lambda root: list(mod._walk_repo_files(root)),
            exc=getattr(mod, "GitIgnoreCheckFailed", _NeverRaised),
            list_timeout_owner=mod, check_timeout_owner=mod, batch_owner=mod)
    return SimpleNamespace(
        name=name, mod=server, collect=server._collect_watched_changes,
        allowed=chunker.gitignore_allowed, ignored=server._git_ignored_watch_paths,
        walk=lambda root: list(server._walk_repo_files(root)),
        exc=getattr(chunker, "GitIgnoreCheckFailed", _NeverRaised),
        list_timeout_owner=chunker, check_timeout_owner=server, batch_owner=server)


@pytest.fixture(params=["plugin", "server"])
def impl(request) -> SimpleNamespace:
    return _impl(request.param)


def _dir_event(impl, root: Path, directory: Path):
    return impl.collect(str(root), {str(directory)}, {}, None, True, [])


# ---- 1. corrupt index, directory event ------------------------------------------------

def test_corrupt_index_directory_event_chunks_nothing_and_requires_reconcile(impl, tmp_path):
    root = _make_repo(tmp_path / "repo")
    _corrupt_index(root)

    result = _dir_event(impl, root, root / "tmp" / "run")

    assert result.payloads == [], (
        f"{len(result.payloads)} ignored file(s) were chunked on a failed git answer")
    assert result.requires_reconcile is True


# ---- 2. corrupt index, repository walk ------------------------------------------------

def test_corrupt_index_repository_walk_raises(impl, tmp_path):
    root = _make_repo(tmp_path / "repo")
    _corrupt_index(root)

    with pytest.raises(impl.exc):
        impl.walk(str(root))


# ---- 3. timeouts ----------------------------------------------------------------------

def test_list_timeout_fails_closed_on_a_directory_event(impl, tmp_path, monkeypatch):
    root = _make_repo(tmp_path / "repo")
    monkeypatch.setattr(impl.list_timeout_owner, "_GIT_LIST_TIMEOUT_S", 1e-6, raising=False)

    result = _dir_event(impl, root, root / "tmp" / "run")

    assert result.payloads == [], (
        f"{len(result.payloads)} file(s) were chunked although git ls-files timed out")
    assert result.requires_reconcile is True


def test_check_ignore_timeout_fails_closed_on_file_events(impl, tmp_path, monkeypatch):
    root = _make_repo(tmp_path / "repo")
    monkeypatch.setattr(impl.check_timeout_owner, "_GIT_CHECK_IGNORE_TIMEOUT_S", 1e-6,
                        raising=False)
    changed = {str(root / "tmp" / "run" / f"f{i:05d}.json") for i in range(5)}

    result = impl.collect(str(root), changed, {}, None, True, [])

    assert result.payloads == [], (
        f"{len(result.payloads)} ignored file(s) were chunked although check-ignore timed out")
    assert result.requires_reconcile is True


def test_list_timeout_raises_from_gitignore_allowed(impl, tmp_path, monkeypatch):
    root = _make_repo(tmp_path / "repo")
    monkeypatch.setattr(impl.list_timeout_owner, "_GIT_LIST_TIMEOUT_S", 1e-6, raising=False)

    with pytest.raises(impl.exc):
        impl.allowed(str(root))


# ---- 4. healthy repository: only the allowed files -------------------------------------

def test_healthy_directory_event_with_many_ignored_files_yields_only_the_python(
        impl, tmp_path):
    root = _make_repo(tmp_path / "repo", json_count=3000, py_count=2)

    result = _dir_event(impl, root, root / "tmp" / "run")

    assert sorted(p["rel_path"] for p in result.payloads) == [
        "tmp/run/m0.py", "tmp/run/m1.py"]
    assert result.requires_reconcile is False


def test_healthy_directory_event_skips_tracked_but_deleted_files(impl, tmp_path):
    root = _make_repo(tmp_path / "repo", json_count=0, py_count=0)
    (root / "src").mkdir()
    (root / "src" / "gone.py").write_bytes(b"x = 1\n")
    (root / "src" / "kept.py").write_bytes(b"y = 2\n")
    _git(root, "add", "src")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", "src")
    (root / "src" / "gone.py").unlink()

    result = _dir_event(impl, root, root / "src")

    assert [p["rel_path"] for p in result.payloads] == ["src/kept.py"]


# ---- 5. not a repository: unchanged behaviour ------------------------------------------

def test_non_repository_means_no_filtering(impl, tmp_path):
    root = tmp_path / "plain"
    (root / "d").mkdir(parents=True)
    names = ["a.py", "b.json", "c.json", "e.py", "f.json"]
    for name in names:
        (root / "d" / name).write_bytes(b"1\n")

    assert impl.allowed(str(root)) is None
    result = _dir_event(impl, root, root / "d")

    assert sorted(p["rel_path"] for p in result.payloads) == sorted(f"d/{n}" for n in names)
    assert result.requires_reconcile is False


def test_pathspec_listing_stays_relative_to_the_root_and_is_literal(impl, tmp_path):
    root = _make_repo(tmp_path / "repo", json_count=3, py_count=2)
    (root / "tmp" / "[x]").mkdir()
    (root / "tmp" / "[x]" / "lit.py").write_bytes(b"z = 1\n")
    (root / "tmp" / "x").mkdir()
    (root / "tmp" / "x" / "glob.py").write_bytes(b"z = 2\n")

    assert impl.allowed(str(root), ["tmp/run"]) == {"tmp/run/m0.py", "tmp/run/m1.py"}
    assert impl.allowed(str(root), ["tmp/[x]"]) == {"tmp/[x]/lit.py"}


# ---- 6. batching ------------------------------------------------------------------------

def _expected_ignored(rels: list[str]) -> set[str]:
    return {rel for rel in rels if rel.endswith(".json")}


def test_check_ignore_batches_match_git_for_4500_paths(impl, tmp_path):
    root = _make_repo(tmp_path / "repo", json_count=3000, py_count=1500)
    rels = sorted(
        p.relative_to(root).as_posix() for p in (root / "tmp" / "run").iterdir())
    assert len(rels) == 4500

    ignored = impl.ignored(str(root), set(rels))

    direct = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--stdin", "-z"],
        input=("\0".join(rels) + "\0").encode(), capture_output=True, check=False)
    assert direct.returncode == 0
    git_says = {p.decode() for p in direct.stdout.split(b"\0") if p}
    assert ignored == git_says == _expected_ignored(rels)


def test_check_ignore_batch_boundaries(impl, tmp_path, monkeypatch):
    root = _make_repo(tmp_path / "repo", json_count=60, py_count=40)
    rels = sorted(
        p.relative_to(root).as_posix() for p in (root / "tmp" / "run").iterdir())
    monkeypatch.setattr(impl.batch_owner, "_CHECK_IGNORE_BATCH", 7, raising=False)

    assert impl.ignored(str(root), set(rels)) == _expected_ignored(rels)


# ---- 7. the watcher survives a failed reconcile ----------------------------------------

def _plugin_scenario(root: Path):
    """Real plugin watcher over a repo whose index gets corrupted, then restored."""
    mod = _load()
    _write(root / "a.py", "value = 1\n")
    backend = _Backend()
    backend.seed_policies(str(root), docs=False, interval=1)
    backend.policies["code"]["respect_gitignore"] = True
    backend.inventory["code"] = [{
        "rel_path": "a.py", "kind": "code", "content_hash": _sha("value = 1\n"),
        "chunk_count": 1, "language": "python"}]
    _wire(mod, backend)

    def code_post_files() -> list[set[str]]:
        return [{item["rel_path"] for item in body["files"]} for body in backend.code_posts()]

    def reconcile_attempts() -> int:
        return len([1 for path, _ in backend.gets if path == "/api/code/jobs"])

    async def start():
        project, task = _start_watcher(mod, root)
        key = ("desktop", project)
        await _wait_until(
            lambda: mod._CODE_WATCH_STATUS.get(key, {}).get("state") == "watching",
            what="watcher start")
        return task, (lambda: mod._CODE_WATCH_STATUS.get(key, {}))

    return SimpleNamespace(start=start, code_post_files=code_post_files,
                           reconcile_attempts=reconcile_attempts, stop=_stop,
                           fmt=mod._format_refresh_status)


class _ServerBackend:
    """The server-side `_client` stand-in for the routes the watcher touches."""

    def __init__(self, inventory_rows: list[dict], policy: dict):
        self.rows = inventory_rows
        self.policy = policy
        self.code_requests: list[dict] = []
        self.job_queries = 0

    async def get_index_policy(self, *, host, project, kind="code"):
        if kind != "code":
            request = httpx.Request("GET", "http://backend/api/index/policy")
            raise httpx.HTTPStatusError(
                "HTTP 404", request=request, response=httpx.Response(404, request=request))
        return self.policy

    async def index_inventory(self, *, host=None, project=None, kind=None, detail=False,
                              lean=False, project_contains=None):
        rows = self.rows if kind == "code" else []
        return {"users": [{"hosts": [{"projects": [{"dirs": [{"files": rows}]}]}]}]}

    async def code_index_jobs(self, **kwargs):
        self.job_queries += 1
        return {"jobs": []}

    async def code_index(self, **request):
        self.code_requests.append(request)
        return {"job_id": f"code-{len(self.code_requests)}", "status": "queued"}

    async def code_index_job(self, job_id):
        return {"job_id": job_id, "status": "completed", "progress": 1.0}

    async def upsert_index_policy(self, **policy):
        return policy


def _server_scenario(root: Path, monkeypatch):
    _write(root / "a.py", "value = 1\n")
    backend = _ServerBackend(
        [{"rel_path": "a.py", "kind": "code", "content_hash": _sha("value = 1\n"),
          "chunk_count": 1, "language": "python"}],
        {"host": "desktop", "kind": "code", "root_dir": str(root), "auto_refresh": True,
         "respect_gitignore": True, "exclude_globs": [], "min_check_interval_s": 1})
    monkeypatch.setattr(server, "_client", backend)
    monkeypatch.setattr(server, "_AUTO_INDEX_MIN_INTERVAL_S", 0)

    def code_post_files() -> list[set[str]]:
        return [{item["rel_path"] for item in body["files"]} for body in backend.code_requests]

    async def start():
        project = server._scope_path(str(root))
        task = server._ensure_code_watcher(
            project=project, host="desktop", repo_root=str(root))
        assert task is not None
        key = ("desktop", project)
        await _wait_until(
            lambda: server._CODE_WATCH_STATUS.get(key, {}).get("state") == "watching",
            what="watcher start")
        return task, (lambda: server._CODE_WATCH_STATUS.get(key, {}))

    return SimpleNamespace(start=start, code_post_files=code_post_files,
                           reconcile_attempts=lambda: backend.job_queries, stop=_stop,
                           fmt=server._format_refresh_status)


@pytest.mark.parametrize("which", ["plugin", "server"])
def test_watcher_survives_a_failed_reconcile_and_processes_a_later_healthy_event(
        which, tmp_path, monkeypatch):
    root = _make_repo(tmp_path / "repo", json_count=5, py_count=0)
    scenario = (_plugin_scenario(root) if which == "plugin"
                else _server_scenario(root, monkeypatch))
    original_index = (root / ".git" / "index").read_bytes()

    async def run():
        task, status = await scenario.start()
        assert (root / ".git" / "index").read_bytes() == original_index

        (root / ".git" / "index").write_bytes(b"garbage")
        (root / "tmp" / "run" / "burst.json").write_bytes(b'{"burst": 1}\n')
        await _wait_until(
            lambda: status().get("state") == "failed" or status().get("last_error"),
            what="the failed reconcile to show in the watcher status")
        assert "git" in str(status().get("message") or status().get("last_error")).lower()
        # The user sees it in the freshness line search_code shows, while the retry waits.
        await _wait_until(
            lambda: "last refresh failed" in scenario.fmt(dict(status())),
            what="the failure reason in the formatted freshness line")
        line = scenario.fmt(dict(status()))
        assert "filesystem watcher active" in line and "git" in line.lower(), line

        # Alive, nothing submitted, and not spinning: policy interval is 1 s.
        began = time.monotonic()
        attempts_before = scenario.reconcile_attempts()
        await asyncio.sleep(2.5)
        assert not task.done(), "the watcher exited on a failed reconcile"
        assert scenario.code_post_files() == [], (
            "a job was submitted although git could not answer: "
            f"{scenario.code_post_files()}")
        spent = time.monotonic() - began
        assert scenario.reconcile_attempts() - attempts_before <= int(spent) + 2, (
            "the watcher retried faster than its interval")

        (root / ".git" / "index").write_bytes(original_index)
        (root / "tmp" / "run" / "healthy.py").write_bytes(b"value = 7\n")
        await _wait_until(lambda: len(scenario.code_post_files()) >= 1,
                          timeout_seconds=20.0, what="the healthy event to be submitted")
        assert not task.done()
        await asyncio.sleep(0.5)
        submitted = set().union(*scenario.code_post_files())
        assert "tmp/run/healthy.py" in submitted
        assert not any(rel.endswith(".json") for rel in submitted), submitted
        await scenario.stop(task)

    asyncio.run(run())


# ---- section 7: a line-window flush never emits an all-blank chunk (HTTP 422) ----------

def _line_chunkers():
    mod = _load()
    return {
        "plugin": (lambda src: mod._chunk_lines("f.js", src), mod._MAX_CHUNK_SIZE),
        "core": (lambda src: chunker.chunk_lines("f.js", src), chunker.DEFAULT_MAX_CHUNK_SIZE),
    }


@pytest.mark.parametrize("which", ["plugin", "core"])
def test_line_windows_never_emit_a_blank_chunk_after_a_hard_split_line(which):
    chunk, size = _line_chunkers()[which]
    hard = "a" * (4 * size + 10)
    big = "b" * (size + 100)

    chunks = chunk("\n".join([hard, "", big]) + "\n")

    assert [c for c in chunks if not c.code.strip()] == [], "an all-blank chunk was emitted"
    text = "".join(c.code for c in chunks)
    assert text.count("a") == len(hard) and text.count("b") == len(big)


@pytest.mark.parametrize("which", ["plugin", "core"])
def test_line_windows_keep_their_boundaries_for_ordinary_input(which):
    chunk, size = _line_chunkers()[which]
    width = (size * 2) // 5 + 1
    lines = ["a" * width, "b" * width, "", "c" * width, "d" * width, "e" * width,
             "f" * width]

    chunks = chunk("\n".join(lines) + "\n")

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 3), (4, 5), (6, 7)]
    assert [c.code for c in chunks] == [
        "\n".join(lines[0:3]), "\n".join(lines[3:5]), "\n".join(lines[5:7])]


# ---- the watcher's last failure is part of the freshness line ---------------------------

def test_watching_freshness_line_shows_the_last_refresh_failure(impl):
    fmt = impl.mod._format_refresh_status
    status = {"state": "watching", "files": 3, "pending_changes": 2, "next_refresh_in_s": 120,
              "last_error": "git ls-files failed in /r (exit 128): fatal: index file corrupt"}

    assert fmt(status) == (
        "Index freshness: filesystem watcher active (3 files); 2 pending change(s), next "
        "refresh in 120s; last refresh failed: git ls-files failed in /r (exit 128): "
        "fatal: index file corrupt.")
    assert fmt({"state": "watching", "files": 3}) == (
        "Index freshness: filesystem watcher active (3 files).")
