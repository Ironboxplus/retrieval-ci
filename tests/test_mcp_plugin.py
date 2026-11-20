"""The standalone installable MCP plugin registers the FULL tool surface.

After W144 (ACE playbook → server-side /api/playbook/*) and W146 (code KB →
server-side /api/code/*), the plugin is feature-complete: paper search + journal +
index inventory + code KB (index_code/search_code) + the 5 drop-in ace-mcp tools.
Only ``index_code`` chunks locally (chonkie CodeChunker); everything else is pure REST.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import re
import sys
import threading
import tomllib
from pathlib import Path

import httpx
import pytest

from conftest import git_init_repo

PLUGIN = Path(__file__).resolve().parents[1] / "mcp_plugin" / "retrieval_mcp.py"
REPO_ROOT = PLUGIN.parents[1]

EXPECTED = {
    # paper retrieval
    "search_papers", "search_arxiv", "search_within_paper", "match_document", "match_paper",
    "match_paper_status",
    "get_paper_markdown", "download_paper_pdf", "download_papers",
    "list_conferences", "corpus_stats",
    # journal memory + inventory
    "journal_record", "journal_search", "journal_recent", "journal_index_dir",
    "journal_index_status", "delete_journal_docs", "index_inventory",
    # code KB (server-side; client chunks locally; async job queue since W156)
    "index_code", "index_code_status", "search_code",
    "find_duplicate_code", "delete_code",
    # curated public code control plane (admin/internal)
    "list_public_code_sources", "add_public_code_source", "check_public_code_updates",
    "index_public_code_sources", "set_public_code_update_policy",
    "public_code_job_status", "import_public_code_repo",
    # ACE playbook (drop-in ace-mcp signatures)
    "ace_context_aware", "ace_enhance_prompt", "ace_smart_generate",
    "ace_smart_reflect", "ace_playbook",
}


def _load():
    spec = importlib.util.spec_from_file_location("retrieval_mcp_plugin", PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass with `from __future__ import annotations`
    # resolves field-type strings via sys.modules[cls.__module__] (as a normal
    # `uvx retrieval-mcp` import would). Without this the synthetic load can't
    # resolve _CodeChunk's annotations.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Capture:
    """Records _post/_get calls and returns canned responses keyed by path."""

    def __init__(self, responses: dict[str, dict] | None = None):
        self.responses = responses or {}
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[tuple[str, dict | None]] = []
        self.deletes: list[tuple[str, dict | None]] = []
        self.puts: list[tuple[str, dict]] = []

    async def post(self, path: str, payload: dict) -> dict:
        self.posts.append((path, payload))
        return self.responses.get(path, {})

    async def get(
        self,
        path: str,
        params: dict | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict:
        self.gets.append((path, params))
        if path.startswith("/api/code/jobs/") and path not in self.responses:
            job_id = path.rsplit("/", 1)[-1]
            return {"job_id": job_id, "status": "completed", "progress": 1.0,
                    "indexed": 0}
        return self.responses.get(path, {})

    async def delete(self, path: str, params: dict | None = None) -> dict:
        self.deletes.append((path, params))
        return self.responses.get(path, {})

    async def put(self, path: str, payload: dict) -> dict:
        self.puts.append((path, payload))
        return self.responses.get(path, {})


def _wire(mod, cap: _Capture):
    mod._post = cap.post      # type: ignore[assignment]
    mod._get = cap.get        # type: ignore[assignment]
    mod._delete = cap.delete  # type: ignore[assignment]
    mod._put = cap.put        # type: ignore[assignment]


def _index_code_to_completion(mod, **kwargs) -> str:
    async def scenario() -> str:
        reply = await mod.index_code(**kwargs)
        match = re.search(r"job ([0-9a-f]+)", reply)
        if not match:
            return reply
        local_job_id = match.group(1)
        task = mod._CODE_PREPARATION_TASKS.get(local_job_id)
        if task is not None:
            await task
        return reply + "\n" + await mod.index_code_status(local_job_id)

    return asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Tool surface
# --------------------------------------------------------------------------- #

def test_plugin_registers_full_tool_surface_and_defaults_to_compute_box():
    mod = _load()
    assert mod.BASE_URL == "http://10.100.100.111:8000"
    names = {t.name for t in asyncio.run(mod.mcp.list_tools())}
    assert EXPECTED <= names, f"missing: {EXPECTED - names}"
    assert len(EXPECTED) == 35


def test_public_code_tools_forward_profiles_and_render_dense_results():
    mod = _load()
    source = {
        "source_id": "pion", "domain": "networking", "relative_path": "networking/pion",
        "active_revision": "abcdef123456", "active_manifest_file_count": 80,
        "active_chunk_count": 713, "active_vector_count": 713,
        "update_policy": "scheduled", "update_interval_minutes": 1440,
        "last_check_error": None,
    }
    job = {
        "job_id": "job-1", "operation": "index", "status": "running",
        "completed_sources": 0, "total_sources": 1, "progress": 0.25,
        "phase": "chunking", "files_done": 20, "files_total": 80,
        "chunks_done": 200, "current_source_id": "pion", "current_path": "rtp/packet.go",
        "errors": {},
    }
    cap = _Capture({
        "/api/public-code/sources": [source],
        "/api/public-code/sources/pion/update-policy": source,
        "/api/public-code/jobs/job-1": job,
        "/api/public-code/search": {
            "query": "rtp packetizer", "selected_source_ids": ["pion"],
            "requested_mode": "hybrid", "effective_mode": "hybrid",
            "rerank_applied": True, "candidate_count": 20, "rerank_candidate_count": 12,
            "hits": [{
                "score": 0.91, "source_id": "pion", "domain": "networking",
                "retrieval_score": 0.42, "reranker_score": 0.91,
                "revision": "abcdef123456", "path": "rtp/packet.go",
                "start_line": 10, "end_line": 30, "language": "go",
                "label": "Packetize", "code": "func Packetize() {}",
            }],
        },
    })
    _wire(mod, cap)

    listed = asyncio.run(mod.list_public_code_sources())
    cap.responses["/api/public-code/sources"] = {
        "source": {**source, "source_id": "pion-rtp", "active_revision": None},
        "index_job": None,
    }
    added = asyncio.run(mod.add_public_code_source(
        source_id="pion-rtp", domain="networking",
        upstream_url="https://github.com/pion/rtp.git",
        relative_path="networking/pion-rtp", proxy_profile="local-10808",
        credential_profile="github-readonly", index_now=False))
    policy = asyncio.run(mod.set_public_code_update_policy(
        source_id="pion", update_policy="scheduled", update_interval_minutes=1440))
    status = asyncio.run(mod.public_code_job_status("job-1"))
    searched = asyncio.run(mod.search_code(
        query="rtp packetizer", sources="public", source_ids=["pion"],
        domains=["networking"], top_k=7, language="go", rerank=True))

    assert "1 configured, 1 indexed" in listed
    assert "pion-rtp" in added
    assert cap.posts[0][0] == "/api/public-code/sources"
    assert cap.posts[0][1]["proxy_profile"] == "local-10808"
    assert cap.posts[0][1]["credential_profile"] == "github-readonly"
    assert "token" not in cap.posts[0][1]
    assert cap.puts == [("/api/public-code/sources/pion/update-policy", {
        "update_policy": "scheduled", "update_interval_minutes": 1440,
    })]
    assert "scheduled/1440m" in policy
    assert "job-1 [running]" in status
    assert "pion@abcdef1234" in searched
    assert "func Packetize()" in searched
    assert "retrieval_score=0.420" in searched
    assert "reranker_score=0.910" in searched
    assert "requested='hybrid', effective='hybrid'" in searched
    assert "applied to 12 of 20" in searched


def test_public_code_tool_signatures_and_formatters_match_repository_server():
    import mcp_server.server as repository_server
    from mcp_server.backend import (
        format_public_code_job,
        format_public_code_search,
        format_public_code_sources,
    )

    mod = _load()
    tool_names = (
        "list_public_code_sources",
        "add_public_code_source",
        "check_public_code_updates",
        "index_public_code_sources",
        "set_public_code_update_policy",
        "public_code_job_status",
        "import_public_code_repo",
        "search_code",
    )
    for tool_name in tool_names:
        assert inspect.signature(getattr(mod, tool_name)) == inspect.signature(
            getattr(repository_server, tool_name)
        )

    source = {
        "source_id": "pion", "domain": "networking", "relative_path": "networking/pion",
        "active_revision": "abcdef123456", "active_manifest_file_count": 80,
        "active_chunk_count": 713, "active_vector_count": 713,
        "update_policy": "manual", "update_interval_minutes": None,
        "last_check_error": None,
    }
    job = {
        "job_id": "job-1", "operation": "index", "status": "completed",
        "completed_sources": 1, "total_sources": 1, "progress": 1.0,
        "phase": "completed", "files_done": 80, "files_total": 80,
        "chunks_done": 713, "errors": {},
    }
    search = {
        "query": "packetizer", "selected_source_ids": ["pion"],
        "hits": [{
            "score": 0.42, "retrieval_score": 0.42,
            "source_id": "pion", "revision": "abcdef123456",
            "path": "rtp/packet.go", "start_line": 1, "end_line": 3,
            "language": "go", "label": "Packetize", "code": "func Packetize() {}",
        }],
    }
    assert mod._format_public_code_sources([source]) == format_public_code_sources([source])
    assert mod._format_public_code_job(job) == format_public_code_job(job)
    assert mod._format_public_code_search(search) == format_public_code_search(search)
    assert "score=0.420 retrieval_score=0.420" in format_public_code_search(search)


def test_release_metadata_and_owned_mcp_sample_are_command_only():
    # pyproject is the single source for the release version. Retyping the
    # literal here and in every doc turned each bump into a scavenger hunt
    # across nine files, and a copy in the usually-skipped artifact gate had
    # already gone stale. Everything below is checked AGAINST this value.
    package = tomllib.loads((REPO_ROOT / "mcp_plugin" / "pyproject.toml").read_text(
        encoding="utf-8"))
    version = package["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), (
        f"release version must be a plain three-part version, got {version!r}")

    # PEP 723 has no project-version field; this source marker keeps direct
    # script users on the same release documentation without inventing invalid
    # script metadata.  Its dependency block is covered separately below.
    plugin_source = (REPO_ROOT / "mcp_plugin" / "retrieval_mcp.py").read_text(
        encoding="utf-8")
    assert f"Package release: {version}" in plugin_source

    # The checked-in root config is user-selected legacy state.  A new package
    # candidate must not rewrite it merely to advance its own release version.
    root_config = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    root_entry = root_config["mcpServers"]["retrieval"]
    assert root_entry["command"]
    assert "url" not in root_entry
    assert "transport" not in root_entry

    # The package-owned sample recommends the already-installed command that
    # reaches the shared singleton without a per-client package runner or a
    # client-side backend/transport setting.
    plugin_config = json.loads((REPO_ROOT / "mcp_plugin" / ".mcp.json").read_text(encoding="utf-8"))
    plugin_entry = plugin_config["mcpServers"]["retrieval"]
    assert plugin_entry["command"] == "retrieval-mcp"
    assert plugin_entry["args"] == []
    assert "env" not in plugin_entry

    for document in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "mcp_plugin" / "README.md",
        REPO_ROOT / "mcp_server" / "README.md",
        REPO_ROOT / "overview.md",
        # env.md carries the uvx install command operators copy-paste. It was
        # outside this set, so it could go stale silently — the one document
        # where a wrong version is executed rather than read.
        REPO_ROOT / "env.md",
    ):
        text = document.read_text(encoding="utf-8")
        assert version in text, f"{document.name} does not mention release {version}"
        assert '"url":' not in text
        assert "--transport" not in text
        assert "retrieval-mcp start" not in text


def test_singleton_cli_package_contract_and_command_only_examples():
    """The published entry stays light and documented client configs stay stdio-only."""
    from mcp_plugin import retrieval_mcp_daemon as daemon

    package_path = REPO_ROOT / "mcp_plugin" / "pyproject.toml"
    package = tomllib.loads(package_path.read_text(encoding="utf-8"))
    assert package["project"]["scripts"] == {"retrieval-mcp": "retrieval_mcp_cli:main"}
    wheel = package["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert set(wheel["only-include"]) == {
        "retrieval_mcp.py", "retrieval_mcp_cli.py", "retrieval_mcp_daemon.py",
    }
    assert set(wheel["sources"]) == {
        "retrieval_mcp.py", "retrieval_mcp_cli.py", "retrieval_mcp_daemon.py",
    }
    assert daemon.TOOL_SURFACE_NAMES == EXPECTED
    assert len(daemon.TOOL_SURFACE_NAMES) == 35

    # The repository-level file may remain a user-selected legacy `uvx` command,
    # but no client configuration is allowed to become a visible URL transport.
    for config_path in (REPO_ROOT / ".mcp.json", REPO_ROOT / "mcp_plugin" / ".mcp.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        retrieval = config["mcpServers"]["retrieval"]
        assert "url" not in retrieval
        assert "transport" not in retrieval

    packaged_example = json.loads((REPO_ROOT / "mcp_plugin" / ".mcp.json").read_text(encoding="utf-8"))
    packaged_retrieval = packaged_example["mcpServers"]["retrieval"]
    assert packaged_retrieval["command"] == "retrieval-mcp"
    assert packaged_retrieval.get("args", []) == []

    for document in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "mcp_plugin" / "README.md",
        REPO_ROOT / "mcp_server" / "README.md",
    ):
        text = document.read_text(encoding="utf-8")
        assert '"command": "retrieval-mcp"' in text
        assert "--transport" not in text


def test_plugin_code_chunker_dependencies_stay_on_verified_parser_line():
    """Do not let a fresh install silently cross an unvalidated parser boundary.

    ``chonkie[code]>=1.6`` accepts 1.7.0, which in turn may resolve a newer
    tree-sitter stack.  The standalone ``index_code`` path is validated on the
    1.6.x CodeChunker line.  Chonkie's 1.6.x code extra itself leaves
    ``tree-sitter-language-pack`` open-ended, so this package also records the
    verified 1.12.x language-pack boundary in both metadata forms.  Those
    language-pack releases still advertise native ``tree-sitter>=0.23`` without
    an upper bound, so this package records the verified native ABI line as an
    explicit compatibility dependency rather than letting a native parser
    upgrade alter the crash boundary of local code indexing.
    """
    from packaging.requirements import Requirement

    package = tomllib.loads((REPO_ROOT / "mcp_plugin" / "pyproject.toml").read_text(
        encoding="utf-8"))
    source = (REPO_ROOT / "mcp_plugin" / "retrieval_mcp.py").read_text(encoding="utf-8")
    chonkie_requirements = [
        requirement for requirement in package["project"]["dependencies"]
        if requirement.lower().startswith("chonkie")
    ]
    language_pack_requirements = [
        requirement for requirement in package["project"]["dependencies"]
        if requirement.lower().startswith("tree-sitter-language-pack")
    ]
    native_parser_requirements = [
        requirement for requirement in package["project"]["dependencies"]
        if Requirement(requirement).name == "tree-sitter"
    ]

    assert chonkie_requirements == ["chonkie[code]>=1.6,<1.7"]
    parsed = Requirement(chonkie_requirements[0])
    assert parsed.specifier.contains("1.6.8")
    assert not parsed.specifier.contains("1.7.0"), (
        "an unbounded Chonkie requirement can pull the unvalidated 1.7/parser line"
    )
    assert '"chonkie[code]>=1.6,<1.7"' in source.splitlines()[3]
    assert language_pack_requirements == ["tree-sitter-language-pack>=1.8,<1.13"]
    parser_line = Requirement(language_pack_requirements[0])
    assert parser_line.specifier.contains("1.12.2")
    assert not parser_line.specifier.contains("1.13.0")
    assert '"tree-sitter-language-pack>=1.8,<1.13"' in source.splitlines()[3]
    assert native_parser_requirements == ["tree-sitter>=0.23,<0.26"]
    native_line = Requirement(native_parser_requirements[0])
    assert native_line.specifier.contains("0.25.2")
    assert not native_line.specifier.contains("0.26.0")
    assert '"tree-sitter>=0.23,<0.26"' in source.splitlines()[3]


def test_plugin_dependency_prevents_unmigrated_mcp_v2_install():
    """The raw bridge uses v1.29 internals, so a broad v1 range is unsafe too."""
    from packaging.requirements import Requirement

    package = tomllib.loads((REPO_ROOT / "mcp_plugin" / "pyproject.toml").read_text(
        encoding="utf-8"))
    mcp_requirements = [
        requirement for requirement in package["project"]["dependencies"]
        if requirement.startswith("mcp")
    ]
    source = (REPO_ROOT / "mcp_plugin" / "retrieval_mcp.py").read_text(encoding="utf-8")

    assert mcp_requirements == ["mcp>=1.29,<1.30"]
    parsed = Requirement(mcp_requirements[0])
    assert parsed.specifier.contains("1.29.0")
    assert not parsed.specifier.contains("1.30.0")
    assert not parsed.specifier.contains("2.0.0")
    assert '"mcp>=1.29,<1.30"' in source.splitlines()[3]
    daemon_source = (REPO_ROOT / "mcp_plugin" / "retrieval_mcp_daemon.py").read_text(
        encoding="utf-8")
    assert "from mcp.shared.message import SessionMessage" in daemon_source
    assert "mcp._mcp_server.run" in daemon_source
    package_readme = (REPO_ROOT / "mcp_plugin" / "README.md").read_text(encoding="utf-8")
    assert "mcp>=1.29,<1.30" in package_readme
    assert "private" in package_readme.lower()


def test_daemon_keeps_a_real_python_3_10_tomli_fallback():
    """The advertised Python 3.10 floor cannot import the 3.11-only ``tomllib``."""
    import builtins
    import types
    from packaging.requirements import Requirement

    package = tomllib.loads((REPO_ROOT / "mcp_plugin" / "pyproject.toml").read_text(
        encoding="utf-8"))
    source_path = REPO_ROOT / "mcp_plugin" / "retrieval_mcp_daemon.py"
    source = source_path.read_text(encoding="utf-8")
    tomli_requirements = [
        requirement for requirement in package["project"]["dependencies"]
        if Requirement(requirement).name == "tomli"
    ]

    assert package["project"]["requires-python"] == ">=3.10"
    assert tomli_requirements == ["tomli>=2.0; python_version < '3.11'"]
    assert '"tomli>=2.0; python_version < \'3.11\'"' in (
        (REPO_ROOT / "mcp_plugin" / "retrieval_mcp.py").read_text(encoding="utf-8").splitlines()[3]
    )

    # Execute the import prefix with the builtin module deliberately unavailable.
    # A fake tomli is enough to prove that the daemon actually takes its fallback,
    # rather than merely mentioning it in package metadata or documentation.
    fake_tomli = types.ModuleType("tomli")
    fake_tomli.TOMLDecodeError = ValueError
    fake_tomli.load = lambda _stream: {"project": {"version": "fallback-probe"}}
    original_import = builtins.__import__

    def import_without_tomllib(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tomllib":
            raise ModuleNotFoundError("No module named 'tomllib'")
        if name == "tomli":
            return fake_tomli
        return original_import(name, globals, locals, fromlist, level)

    prefix = source[:source.index("\n\nLOG =")]
    namespace = {
        "__name__": "daemon_tomli_fallback_probe",
        "__file__": str(source_path),
        "__builtins__": {**vars(builtins), "__import__": import_without_tomllib},
    }
    exec(compile(prefix, str(source_path), "exec"), namespace)  # noqa: S102 - isolated import probe
    assert namespace["tomllib"] is fake_tomli


def test_get_paper_markdown_returns_the_untruncated_full_text():
    mod = _load()
    full_text = "# Full paper\n\n" + ("body paragraph\n" * 400) + "FINAL-MARKER"
    cap = _Capture({
        "/api/papers/neurips-2024-full/fulltext": {
            "paper_id": "neurips-2024-full",
            "title": "Full Paper",
            "content": full_text,
            "size": len(full_text),
        }
    })
    _wire(mod, cap)

    result = asyncio.run(mod.get_paper_markdown("neurips-2024-full"))

    assert result == full_text
    assert cap.gets == [("/api/papers/neurips-2024-full/fulltext", None)]


def test_get_paper_markdown_can_save_to_a_local_directory(tmp_path):
    mod = _load()
    content = "# Saved paper\n\nExact Markdown bytes.\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/papers/acl:2025-saved/markdown"
        assert request.headers["range"].startswith("bytes=0-")
        body = content.encode("utf-8")
        return httpx.Response(206, headers={
            "content-range": f"bytes 0-{len(body) - 1}/{len(body)}",
            "content-length": str(len(body)),
        }, content=body)

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")

    async def scenario() -> str:
        try:
            return await mod.get_paper_markdown(
                "acl:2025-saved", save_path=str(tmp_path))
        finally:
            await mod._CLIENT.aclose()

    result = asyncio.run(scenario())

    destination = tmp_path / "acl_2025-saved.md"
    assert destination.read_text(encoding="utf-8") == content
    assert str(destination.resolve()) in result
    assert "Saved Markdown" in result


def test_saved_markdown_reuses_one_idempotency_key_for_all_ranges(tmp_path):
    mod = _load()
    mod._PAPER_ARTIFACT_RANGE_BYTES = 16
    content = ("# Ranged paper\n\n" + "range body\n" * 20).encode("utf-8")
    idempotency_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        idempotency_keys.append(request.headers["x-idempotency-key"])
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["range"])
        assert match is not None
        start, requested_end = map(int, match.groups())
        end = min(requested_end, len(content) - 1)
        return httpx.Response(206, headers={
            "content-range": f"bytes {start}-{end}/{len(content)}",
            "content-length": str(end - start + 1),
        }, content=content[start:end + 1])

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")

    async def scenario() -> str:
        try:
            return await mod.get_paper_markdown(
                "ranged-markdown", save_path=str(tmp_path))
        finally:
            await mod._CLIENT.aclose()

    asyncio.run(scenario())

    assert len(idempotency_keys) > 2
    assert len(set(idempotency_keys)) == 1
    assert idempotency_keys[0].startswith("paper-markdown-")
    assert (tmp_path / "ranged-markdown.md").read_bytes() == content


def test_saved_markdown_retry_keeps_the_same_idempotency_key(tmp_path):
    mod = _load()
    mod._PAPER_ARTIFACT_RETRY_BASE_SECONDS = 0
    content = b"# Retry\n\nComplete body.\n"
    idempotency_keys: list[str] = []
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        idempotency_keys.append(request.headers["x-idempotency-key"])
        if requests == 1:
            return httpx.Response(503, text="retry")
        return httpx.Response(200, content=content)

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")

    async def scenario() -> str:
        try:
            return await mod.get_paper_markdown(
                "retry-markdown", save_path=str(tmp_path))
        finally:
            await mod._CLIENT.aclose()

    asyncio.run(scenario())

    assert requests == 2
    assert len(set(idempotency_keys)) == 1
    assert (tmp_path / "retry-markdown.md").read_bytes() == content


def test_ranged_download_rejects_a_file_version_change(tmp_path):
    mod = _load()
    mod._PAPER_ARTIFACT_RANGE_BYTES = 8
    version_a = b"%PDF-A1" + b"a" * 24
    version_b = b"%PDF-B2" + b"b" * 24
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["range"])
        assert match is not None
        start, requested_end = map(int, match.groups())
        content = version_a if requests == 1 else version_b
        etag = '"version-a"' if requests == 1 else '"version-b"'
        if requests > 1:
            assert request.headers["if-range"] == '"version-a"'
        end = min(requested_end, len(content) - 1)
        return httpx.Response(206, headers={
            "content-range": f"bytes {start}-{end}/{len(content)}",
            "content-length": str(end - start + 1),
            "etag": etag,
        }, content=content[start:end + 1])

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")
    destination = tmp_path / "mixed-version.pdf"

    async def scenario() -> None:
        try:
            with pytest.raises(ValueError, match="changed during ranged download"):
                await mod.download_paper_pdf(
                    "versioned-paper", save_path=str(destination))
        finally:
            await mod._CLIENT.aclose()

    asyncio.run(scenario())

    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_get_paper_markdown_refuses_to_overwrite_by_default(tmp_path):
    mod = _load()
    destination = tmp_path / "paper.md"
    destination.write_text("keep me", encoding="utf-8")
    cap = _Capture({
        "/api/papers/paper/fulltext": {
            "paper_id": "paper",
            "title": "Paper",
            "content": "replacement",
        }
    })
    _wire(mod, cap)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        asyncio.run(mod.get_paper_markdown("paper", save_path=str(destination)))

    assert destination.read_text(encoding="utf-8") == "keep me"
    assert cap.gets == []


def test_get_paper_markdown_retries_an_incomplete_response():
    mod = _load()
    mod._PAPER_ARTIFACT_RETRY_BASE_SECONDS = 0
    full_text = "# Retried paper\n\nComplete body after reconnect.\n"
    requests = 0
    idempotency_keys: list[str] = []

    class InterruptedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"paper_id":"retry","content":"partial'
            raise httpx.RemoteProtocolError("peer closed incomplete response")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        idempotency_keys.append(request.headers["x-idempotency-key"])
        if requests == 1:
            return httpx.Response(200, stream=InterruptedStream())
        return httpx.Response(200, json={
            "paper_id": "retry",
            "title": "Retried Paper",
            "content": full_text,
        })

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")

    async def scenario() -> str:
        try:
            return await mod.get_paper_markdown("retry")
        finally:
            await mod._CLIENT.aclose()

    result = asyncio.run(scenario())

    assert requests == 2
    assert len(set(idempotency_keys)) == 1
    assert idempotency_keys[0].startswith("paper-fulltext-")
    assert result == full_text


def test_download_paper_pdf_streams_to_a_local_path(tmp_path):
    mod = _load()
    pdf_bytes = b"%PDF-1.7\n" + (b"binary-payload\n" * 100)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/papers/cvpr-2026-pdf/pdf"
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=pdf_bytes,
        )

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")
    destination = tmp_path / "downloaded.pdf"

    async def scenario() -> str:
        try:
            return await mod.download_paper_pdf(
                "cvpr-2026-pdf", save_path=str(destination))
        finally:
            await mod._CLIENT.aclose()

    result = asyncio.run(scenario())

    assert destination.read_bytes() == pdf_bytes
    assert str(destination.resolve()) in result
    assert "Downloaded PDF" in result


def test_download_paper_pdf_uses_validated_http_ranges(tmp_path):
    mod = _load()
    mod._PAPER_ARTIFACT_RANGE_BYTES = 64
    pdf_bytes = b"%PDF-1.7\n" + bytes(range(256)) * 2
    ranges: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        range_header = request.headers.get("range")
        assert range_header is not None
        ranges.append(range_header)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
        assert match is not None
        start, requested_end = map(int, match.groups())
        end = min(requested_end, len(pdf_bytes) - 1)
        payload = pdf_bytes[start:end + 1]
        return httpx.Response(
            206,
            headers={
                "content-type": "application/pdf",
                "content-range": f"bytes {start}-{end}/{len(pdf_bytes)}",
                "content-length": str(len(payload)),
                "accept-ranges": "bytes",
            },
            content=payload,
        )

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")
    destination = tmp_path / "ranged.pdf"

    async def scenario() -> str:
        try:
            return await mod.download_paper_pdf(
                "range-paper", save_path=str(destination))
        finally:
            await mod._CLIENT.aclose()

    result = asyncio.run(scenario())

    assert destination.read_bytes() == pdf_bytes
    assert ranges[0] == "bytes=0-63"
    assert ranges[-1].endswith(f"-{len(pdf_bytes) - 1}")
    assert len(ranges) > 2
    assert "Downloaded PDF" in result


def test_download_paper_pdf_discards_bytes_beyond_declared_range(tmp_path):
    mod = _load()
    mod._PAPER_ARTIFACT_RANGE_BYTES = 4
    mod._PAPER_ARTIFACT_RETRY_BASE_SECONDS = 0
    pdf_bytes = b"%PDF-OK!"
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["range"])
        assert match is not None
        start, requested_end = map(int, match.groups())
        end = min(requested_end, len(pdf_bytes) - 1)
        headers = {
            "content-range": f"bytes {start}-{end}/{len(pdf_bytes)}",
            "content-length": str(end - start + 1),
        }
        if requests == 1:
            return httpx.Response(206, headers=headers, content=b"%PDF-EXTRA")
        return httpx.Response(206, headers=headers, content=pdf_bytes[start:end + 1])

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")
    destination = tmp_path / "validated.pdf"

    async def scenario() -> str:
        try:
            return await mod.download_paper_pdf(
                "validated", save_path=str(destination))
        finally:
            await mod._CLIENT.aclose()

    asyncio.run(scenario())

    assert requests == 3
    assert destination.read_bytes() == pdf_bytes


def test_download_papers_reuses_single_artifact_downloads(tmp_path):
    mod = _load()
    pdfs = {
        "p1": b"%PDF-1.7\np1",
        "p2": b"%PDF-1.7\np2",
    }
    markdown = {"p1": "# P1\n", "p2": "# P2\n"}
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        paper_id = request.url.path.split("/")[-2]
        if request.url.path.endswith("/markdown"):
            body = markdown[paper_id].encode("utf-8")
            return httpx.Response(200, content=body)
        range_header = request.headers["range"]
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
        assert match is not None
        start, requested_end = map(int, match.groups())
        content = pdfs[paper_id]
        end = min(requested_end, len(content) - 1)
        return httpx.Response(206, headers={
            "content-range": f"bytes {start}-{end}/{len(content)}",
            "content-length": str(end - start + 1),
        }, content=content[start:end + 1])

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")

    async def scenario() -> str:
        try:
            return await mod.download_papers(
                ["p1", "p2"], save_dir=str(tmp_path), artifacts="both")
        finally:
            await mod._CLIENT.aclose()

    result = asyncio.run(scenario())

    assert (tmp_path / "p1.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path / "p1.md").read_text(encoding="utf-8") == "# P1\n"
    assert (tmp_path / "p2.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path / "p2.md").read_text(encoding="utf-8") == "# P2\n"
    assert "4 artifact(s)" in result
    assert all(method == "GET" for method, _ in seen)
    assert not any(path == "/api/papers/download-batch" for _, path in seen)


def test_download_papers_resumes_from_the_partial_range_offset(tmp_path):
    mod = _load()
    mod._PAPER_ARTIFACT_RANGE_BYTES = 64
    mod._PAPER_STREAM_CHUNK_BYTES = 4
    mod._PAPER_ARTIFACT_RETRY_BASE_SECONDS = 0
    pdf_bytes = b"%PDF-1.7\n" + b"a" * 200
    ranges: list[str] = []
    attempts: dict[str, int] = {}

    class InterruptedStream(httpx.AsyncByteStream):
        def __init__(self, content: bytes):
            self.content = content

        async def __aiter__(self):
            yield self.content[:8]
            raise httpx.RemoteProtocolError("batch range interrupted")

    def handler(request: httpx.Request) -> httpx.Response:
        range_header = request.headers["range"]
        ranges.append(range_header)
        attempts[range_header] = attempts.get(range_header, 0) + 1
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
        assert match is not None
        start, requested_end = map(int, match.groups())
        end = min(requested_end, len(pdf_bytes) - 1)
        content = pdf_bytes[start:end + 1]
        headers = {
            "content-range": f"bytes {start}-{end}/{len(pdf_bytes)}",
            "content-length": str(len(content)),
        }
        if start == 64 and attempts[range_header] == 1:
            return httpx.Response(206, headers=headers, stream=InterruptedStream(content))
        return httpx.Response(206, headers=headers, content=content)

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")

    async def scenario() -> str:
        try:
            return await mod.download_papers(["p1"], save_dir=str(tmp_path))
        finally:
            await mod._CLIENT.aclose()

    result = asyncio.run(scenario())

    assert ranges[0] == "bytes=0-63"
    assert attempts["bytes=64-127"] == 1
    assert "bytes=72-135" in ranges
    assert (tmp_path / "p1.pdf").read_bytes().startswith(b"%PDF-")
    assert "1 artifact(s)" in result


def test_download_papers_rejects_more_than_ten_before_http(tmp_path):
    mod = _load()

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid batch must not make an HTTP request")

    mod._client = fail_if_called

    with pytest.raises(ValueError, match="at most 10"):
        asyncio.run(mod.download_papers(
            [f"paper-{index}" for index in range(11)],
            save_dir=str(tmp_path),
        ))


def test_download_papers_tool_schema_caps_batch_at_ten():
    mod = _load()
    tool = {
        item.name: item for item in asyncio.run(mod.mcp.list_tools())
    }["download_papers"]
    ids_schema = tool.inputSchema["properties"]["paper_ids"]

    assert ids_schema["minItems"] == 1
    assert ids_schema["maxItems"] == 10


def test_download_papers_disambiguates_sanitized_filename_collisions(tmp_path):
    mod = _load()
    ids = ["foo:1", "foo_1"]

    def handler(request: httpx.Request) -> httpx.Response:
        paper_id = request.url.path.split("/")[-2]
        body = f"%PDF-1.7\n{paper_id}".encode()
        return httpx.Response(200, content=body)

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")

    async def scenario() -> str:
        try:
            return await mod.download_papers(ids, save_dir=str(tmp_path))
        finally:
            await mod._CLIENT.aclose()

    result = asyncio.run(scenario())
    files = sorted(path.name for path in tmp_path.glob("*.pdf"))

    assert len(files) == 2
    assert files[0] != files[1]
    assert "2 artifact(s)" in result


def test_download_papers_preflights_existing_files_before_http(tmp_path):
    mod = _load()
    (tmp_path / "p2.pdf").write_bytes(b"keep")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("preflight failure must happen before HTTP")

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")

    async def scenario() -> None:
        try:
            await mod.download_papers(
                ["p1", "p2"], save_dir=str(tmp_path), artifacts="pdf")
        finally:
            await mod._CLIENT.aclose()

    with pytest.raises(FileExistsError, match="overwrite=True"):
        asyncio.run(scenario())

    assert not (tmp_path / "p1.pdf").exists()
    assert (tmp_path / "p2.pdf").read_bytes() == b"keep"


def test_download_paper_pdf_rejects_non_pdf_without_leaving_a_partial_file(tmp_path):
    mod = _load()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html>gateway error</html>",
        )

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")
    destination = tmp_path / "bad.pdf"

    async def scenario() -> None:
        try:
            with pytest.raises(ValueError, match="not a PDF"):
                await mod.download_paper_pdf("broken-paper", save_path=str(destination))
        finally:
            await mod._CLIENT.aclose()

    asyncio.run(scenario())

    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_paper_pdf_retries_an_incomplete_response(tmp_path):
    mod = _load()
    mod._PAPER_ARTIFACT_RETRY_BASE_SECONDS = 0
    pdf_bytes = b"%PDF-1.7\n" + (b"complete-pdf-body\n" * 100)
    requests = 0

    class InterruptedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield pdf_bytes[:128]
            raise httpx.RemoteProtocolError("peer closed incomplete response")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(200, stream=InterruptedStream())
        return httpx.Response(200, content=pdf_bytes)

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")
    destination = tmp_path / "retried.pdf"

    async def scenario() -> str:
        try:
            return await mod.download_paper_pdf(
                "retry-paper", save_path=str(destination))
        finally:
            await mod._CLIENT.aclose()

    result = asyncio.run(scenario())

    assert requests == 2
    assert destination.read_bytes() == pdf_bytes
    assert "Downloaded PDF" in result
    assert not list(tmp_path.glob("*.part"))


def test_atomic_no_overwrite_commit_never_uses_replacing_rename(tmp_path, monkeypatch):
    mod = _load()
    destination = tmp_path / "paper.md"
    temporary = tmp_path / ".paper.md.test.part"
    temporary.write_bytes(b"safe content")

    def replacing_commit_would_be_a_bug(*args, **kwargs):
        raise AssertionError("overwrite=False must not call os.replace")

    monkeypatch.setattr(mod.os, "replace", replacing_commit_would_be_a_bug)

    try:
        mod._commit_temporary_file(temporary, destination, overwrite=False)
    finally:
        temporary.unlink(missing_ok=True)

    assert destination.read_bytes() == b"safe content"


def test_atomic_no_overwrite_reports_unsupported_hard_links_without_side_effects(
    tmp_path, monkeypatch,
):
    mod = _load()
    destination = tmp_path / "portable.md"
    temporary = tmp_path / ".portable.md.test.part"
    temporary.write_bytes(b"portable content")

    def unsupported_hard_link(*args, **kwargs):
        raise OSError("hard links are not supported on this filesystem")

    monkeypatch.setattr(mod.os, "link", unsupported_hard_link)

    try:
        with pytest.raises(OSError, match="cannot guarantee overwrite=False"):
            mod._commit_temporary_file(temporary, destination, overwrite=False)
    finally:
        temporary.unlink(missing_ok=True)

    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_paper_pdf_retries_a_transient_server_error(tmp_path):
    mod = _load()
    mod._PAPER_ARTIFACT_RETRY_BASE_SECONDS = 0
    pdf_bytes = b"%PDF-1.7\n" + (b"healthy-after-retry\n" * 100)
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(200, content=pdf_bytes)

    mod._CLIENT = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test")
    destination = tmp_path / "server-retry.pdf"

    async def scenario() -> str:
        try:
            return await mod.download_paper_pdf(
                "server-retry", save_path=str(destination))
        finally:
            await mod._CLIENT.aclose()

    result = asyncio.run(scenario())

    assert requests == 2
    assert destination.read_bytes() == pdf_bytes
    assert "Downloaded PDF" in result


def test_code_preparation_lifecycle_helpers_match_repo_server():
    """The standalone package intentionally vendors these dependency-free helpers."""
    import mcp_server.server as server

    mod = _load()
    helpers = (
        "_wait_code_job_shared",
        "_ensure_code_watcher",
        "_trim_code_preparation_history",
        "_active_code_preparation",
        "_code_scope_operation_lock",
        "_submit_code_index_with_receipt",
        "_backend_job_snapshot_is_stale",
        "_update_preparation_from_backend",
        "_refresh_preparation_from_backend",
        "_run_code_index_preparation",
        "_start_code_index_preparation",
        "_fmt_job_line",
        "_is_windows_absolute_path",
        "_is_windows_device_path",
        "_canonical_windows_scope",
        "_session_cwd",
        "_session_abspath",
        "_scope_path",
        "_normalize_path_glob",
        "_local_repo_root",
        "_format_scope_miss",
    )
    for helper in helpers:
        assert inspect.getsource(getattr(mod, helper)) == inspect.getsource(
            getattr(server, helper)), f"standalone plugin drifted in {helper}"


def test_paper_artifact_helpers_match_repo_client():
    import mcp_server.backend as backend

    mod = _load()
    helpers = (
        "_paper_artifact_filename",
        "_paper_artifact_destination",
        "_commit_temporary_file",
        "_retryable_paper_artifact_error",
        "_paper_artifact_validator",
        "_parse_content_range",
        "_truncate_temporary_file",
        "_append_stream",
        "_stream_ranged_artifact",
        "_stream_pdf_ranges",
        "_stream_markdown_ranges",
        "_validate_paper_batch",
        "_paper_batch_destination",
        "_unique_paper_artifact_filename",
        "_paper_batch_artifact_plan",
        "format_paper_batch_download",
    )
    for helper in helpers:
        assert inspect.getsource(getattr(mod, helper)) == inspect.getsource(
            getattr(backend, helper)), f"standalone plugin drifted in {helper}"


def test_plugin_tells_clients_to_search_with_retrieval_before_shell_tools():
    mod = _load()
    instructions = mod.mcp.instructions
    assert "Retrieval-first" in instructions
    assert "use search_code for code (your indexed repositories, the curated public corpus, or both)" in instructions
    assert "the grep replacement" in instructions
    assert "search_papers, search_arxiv (arXiv preprints) and search_within_paper for papers" in instructions
    assert "journal_search for Journal/project history" in instructions
    assert "ordered list" in instructions
    assert "independent searches would otherwise be serial" in instructions
    assert "This is not Retrieval-only" in instructions
    for permitted_evidence in (
        "rg", "Git history", "direct working-tree reads", "official paper sources",
        "unindexed/current files", "exhaustive every-occurrence listing", "cross-checking",
        "supplementation", "MCP/tool failure",
    ):
        assert permitted_evidence in instructions

    tools = {tool.name: tool for tool in asyncio.run(mod.mcp.list_tools())}
    # Recommendations state the advantage over rg/grep, not just "prefer this tool".
    code_description = " ".join((tools["search_code"].description or "").split())
    assert "What it does better than rg/grep" in code_description
    assert 'mode="keyword" (sparse, BM25-like) is the grep replacement for exact lookups' in code_description
    assert "Use it first whenever you know the token" in code_description
    assert "they recover matches grep cannot see" in code_description
    journal_description = " ".join((tools["journal_search"].description or "").split())
    assert "What it does better than rg/grep on the same files" in journal_description
    for name in (
        "search_code", "search_papers", "search_arxiv",
        "search_within_paper", "journal_search",
    ):
        # The list contract is published once per tool on `query`, not in the prose.
        assert "Batch query contract" not in (tools[name].description or "")
        query = tools[name].inputSchema["properties"]["query"]
        assert "list of independent queries" in query["description"]
        assert "same other arguments" in query["description"]
        assert "input order" in query["description"]
    assert "each list item runs independently" in mod.mcp.instructions


def test_published_schemas_carry_no_generated_titles():
    """Pydantic's auto titles ("Top K", "<fn>Arguments") only repeat parameter names;
    they are stripped from the advertised surface. A parameter NAMED title survives."""
    mod = _load()
    tools = {t.name: t for t in asyncio.run(mod.mcp.list_tools())}

    def titles(node):
        if isinstance(node, dict):
            own = 1 if isinstance(node.get("title"), str) else 0
            return own + sum(titles(v) for k, v in node.items() if k != "properties") + sum(
                titles(v) for v in (node.get("properties") or {}).values())
        if isinstance(node, list):
            return sum(titles(v) for v in node)
        return 0

    assert {name: titles(t.inputSchema) for name, t in tools.items() if titles(t.inputSchema)} == {}
    assert "title" in tools["journal_record"].inputSchema["properties"]


def test_tools_publish_text_only_output():
    """Every tool returns formatted text. FastMCP's automatic structured output added
    an outputSchema of {"result": string} to all of them and sent each result twice
    (text + {"result": <same text>}); tools now publish no outputSchema. The
    client-visible half (no structuredContent over real stdio) is asserted in
    tests/test_mcp_multi_query_real.py against the live backend."""
    mod = _load()
    tools = asyncio.run(mod.mcp.list_tools())
    assert [t.name for t in tools if t.outputSchema is not None] == []
    assert all(t.fn_metadata.output_schema is None for t in mod.mcp._tool_manager.list_tools())


# A relative scope is resolved against the CALLING client's directory, not the
# singleton daemon's, so `_session_cwd` is the base these two tests vary.  It is a
# real seam: the daemon binds it per raw session from the shim's handshake.
def test_scope_path_canonicalises_windows_path_case_and_separators(monkeypatch):
    """The whole Windows path, not merely its drive letter, is one scope identity."""
    mod = _load()
    monkeypatch.setattr(
        mod, "_session_cwd",
        lambda: "e:\\JETB\\Pythons\\Playground\\Octopus\\AxonHub\\LLM\\.\\",
    )
    assert mod._scope_path(".") == "E:/jetb/pythons/playground/octopus/axonhub/llm"
    assert mod._scope_path("Sub\\Dir") == (
        "E:/jetb/pythons/playground/octopus/axonhub/llm/sub/dir")


def test_scope_path_leaves_posix_paths_untouched(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "_session_cwd", lambda: "/home/arc/repo")
    assert mod._scope_path(".") == "/home/arc/repo"
    # POSIX casing identifies different directories, so it must survive the join.
    assert mod._scope_path("Sub/Dir") == "/home/arc/repo/Sub/Dir"


def test_scope_path_canonicalises_unc_and_rejects_drive_relative_glob():
    mod = _load()
    assert mod._scope_path(
        r"\\BuildServer\SharedRepos\AxonHub\LLM"
    ) == "//buildserver/sharedrepos/axonhub/llm"

    with pytest.raises(ValueError, match="scope-relative"):
        mod._normalize_path_glob(r"E:conversion\planner")


def test_scope_path_rejects_windows_device_prefix_with_actionable_guidance():
    mod = _load()
    with pytest.raises(ValueError, match="extended-length/device.*ordinary drive or UNC"):
        mod._scope_path(r"\\?\C:\Repo")


def test_journal_search_surfaces_path_glob_notice_when_empty():
    """An empty path_glob result carries a server 'notice' (0-of-N + sample title) so the
    user learns the glob is scope-relative and their depth is wrong — not a bare miss that
    hides the docs the scope really has."""
    mod = _load()
    cap = _Capture({"/api/journal/search": {
        "hits": [], "notice": "path_glob 'x/*.md' matched 0 of 3 doc(s) in scope 'p'."}})
    _wire(mod, cap)
    out = asyncio.run(mod.journal_search(query="q", path_glob="x/*.md"))
    assert "matched 0 of 3 doc(s)" in out


def test_journal_search_empty_guides_to_the_exact_scope():
    mod = _load()
    cap = _Capture({"/api/journal/search": {"hits": []}})
    _wire(mod, cap)

    out = asyncio.run(mod.journal_search(query="q"))
    scope = mod._scope_path(".")

    assert f"scope {scope!r}" in out
    assert "does not prove that scope_dir is wrong" in out
    assert f"journal_search(query='q', scope_dir={scope!r})" in out
    assert "index_inventory()" in out
    assert f"journal_index_dir(dir={scope!r})" in out
    assert "index_code_status" not in out


def test_journal_search_tool_schema_bounds_chunk_count():
    mod = _load()
    tool = {
        item.name: item for item in asyncio.run(mod.mcp.list_tools())
    }["journal_search"]
    chunk_schema = tool.inputSchema["properties"]["chunk_limit"]

    assert chunk_schema["minimum"] == 1
    assert chunk_schema["maximum"] == 20
    assert "number of chunks" in chunk_schema["description"].lower()


def test_journal_search_rejects_invalid_chunk_count_before_http():
    mod = _load()
    cap = _Capture()
    _wire(mod, cap)

    with pytest.raises(ValueError, match="number of chunks.*1 through 20"):
        asyncio.run(mod.journal_search(query="cscc hook TODO", chunk_limit=16000))

    assert cap.posts == []


def test_journal_search_rejects_absolute_windows_path_glob_before_http():
    mod = _load()
    cap = _Capture()
    _wire(mod, cap)

    with pytest.raises(ValueError, match="scope-relative"):
        asyncio.run(mod.journal_search(
            query="compact", scope_dir=r"E:\JETB\repo",
            path_glob=r"E:\JETB\repo\docs"))

    assert cap.posts == []


def test_index_inventory_uses_server_freshness_summary():
    """index_inventory renders the compact SUMMARY (project counts + server freshness), never
    the per-file tree — so it no longer downloads/echoes the ~10MB content_hash listing."""
    mod = _load()
    cap = _Capture({"/api/index/inventory": {
        "totals": {"hosts": 1, "projects": 1, "files": 2, "chunks": 9},
        "users": [{"owner_user_id": "u1", "file_count": 2, "hosts": [
            {"host": "box", "projects": [
                {"project": "P", "root_dir": "/repo", "file_count": 2,
                 "last_indexed_at": "2026-07-03T12:34:56", "last_commit": "abc1234",
                 "dirs": []}]}]}]}})
    _wire(mod, cap)
    out = asyncio.run(mod.index_inventory())
    assert "last indexed 2026-07-03 12:34:56" in out and "@abc1234" in out
    # summary is requested (no per-file detail) so the wire payload stays KB-scale
    assert cap.gets and cap.gets[0][0] == "/api/index/inventory"


def test_index_inventory_surfaces_preparing_code_scope():
    mod = _load()
    cap = _Capture({"/api/index/inventory": {
        "totals": {"hosts": 1, "projects": 1, "files": 0, "chunks": 0},
        "users": [{"owner_user_id": "u1", "file_count": 0, "hosts": [
            {"host": "box", "projects": [
                {"project": "/repo/sing-box", "root_dir": "/repo/sing-box", "file_count": 0,
                 "code_snapshot_state": "preparing", "code_active_index_job_id": "sing-box-job",
                 "code_previous_completed_snapshot_available": False, "dirs": []}]}]}],
    }})
    _wire(mod, cap)

    out = asyncio.run(mod.index_inventory())

    assert "code snapshot: preparing" in out
    assert "job=sing-box-job" in out
    assert "completed snapshot unavailable" in out



def _inventory(projects_by_host: dict[str, list[dict]]) -> dict:
    count = sum(len(projects) for projects in projects_by_host.values())
    return {
        "totals": {"hosts": len(projects_by_host), "projects": count, "files": count,
                   "chunks": count},
        "users": [{"owner_user_id": "u1", "file_count": count, "hosts": [
            {"host": host, "projects": projects}
            for host, projects in projects_by_host.items()]}],
    }


def _completed_scope(path: str, day: int) -> dict:
    return {"project": path, "root_dir": path, "file_count": 1,
            "last_indexed_at": f"2026-09-{day:02d}T10:00:00",
            "code_snapshot_state": "completed", "code_snapshot_status": "completed",
            "code_previous_completed_snapshot_available": True, "dirs": []}


def test_index_inventory_stays_bounded_for_hundreds_of_scopes():
    """305 scopes rendered as a 565-line, 56,159-character result on 2026-10-09,
    more than the MCP client accepts. The default lists the newest 40 and says so."""
    mod = _load()
    hosts = {
        f"HOST-{h}": [_completed_scope(f"C:/users/arc/prd/very/long/project/path/{h}/{i:03d}",
                                       1 + (i % 28)) for i in range(100)]
        for h in range(3)
    }
    cap = _Capture({"/api/index/inventory": _inventory(hosts)})
    _wire(mod, cap)

    out = asyncio.run(mod.index_inventory())

    assert len(out) < 8000, len(out)
    assert "Listing 40 of 300" in out
    assert "project=<path fragment>" in out
    assert "code snapshot: completed" not in out
    assert "[root " not in out  # root_dir equal to the project key is not repeated
    shown_days = sorted((line.split("last indexed 2026-09-")[1][:2]
                         for line in out.splitlines() if "last indexed" in line), reverse=True)
    newest_days = sorted((p["last_indexed_at"][8:10] for scopes in hosts.values()
                          for p in scopes), reverse=True)[:40]
    assert shown_days == newest_days


def test_index_inventory_lists_scopes_needing_attention_first_and_filters_by_path():
    mod = _load()
    stuck = {"project": "D:/work/stuck", "root_dir": "D:/work/stuck", "file_count": 0,
             "last_indexed_at": "2026-01-01T00:00:00",
             "code_snapshot_state": "preparing", "code_active_index_job_id": "job-7",
             "code_previous_completed_snapshot_available": False, "dirs": []}
    moved = dict(_completed_scope("D:/work/alias", 2), root_dir="E:/mounted/alias")
    hosts = {"box": [_completed_scope(f"C:/repo/{i}", 3) for i in range(60)] + [stuck, moved]}
    cap = _Capture({"/api/index/inventory": _inventory(hosts)})
    _wire(mod, cap)

    out = asyncio.run(mod.index_inventory(limit=5))
    assert out.splitlines()[3].startswith("  · D:/work/stuck")
    assert "code snapshot: preparing; job=job-7" in out

    filtered = asyncio.run(mod.index_inventory(project="d:\\WORK"))
    assert "2 project(s) match" in filtered and "Listing" not in filtered
    assert "C:/repo/" not in filtered
    assert "D:/work/alias (1 files)" in filtered and "[root E:/mounted/alias]" in filtered


def test_index_inventory_formatter_matches_repo_server():
    import mcp_server.backend as backend

    mod = _load()
    for helper in ("_project_freshness", "_code_snapshot_inventory_note",
                   "_same_scope_path", "_format_inventory_summary"):
        assert inspect.getsource(getattr(mod, helper)) == inspect.getsource(
            getattr(backend, helper)), f"standalone plugin drifted in {helper}"

def test_known_code_hashes_requests_detail_tree():
    """The incremental diff baseline MUST request detail=True — the summary inventory omits
    the per-file content_hash tree it diffs against, so without detail it would re-index all."""
    mod = _load()
    cap = _Capture({"/api/index/inventory": {"users": []}})
    _wire(mod, cap)
    asyncio.run(mod._known_code_hashes("C:/repo"))
    path, params = cap.gets[0]
    assert path == "/api/index/inventory"
    assert params.get("detail") is True and params.get("kind") == "code"


# --------------------------------------------------------------------------- #
# Efficiency: reuse the HTTP client + AST builder instead of rebuilding per call
# --------------------------------------------------------------------------- #

def test_http_client_is_reused_across_calls():
    # Every _post/_get previously opened a fresh httpx.AsyncClient (new TCP+TLS per
    # call, 20+ per session). The client must be a lazily-built singleton carrying
    # base_url + auth headers.
    mod = _load()
    c1 = mod._client()
    c2 = mod._client()
    assert c1 is c2, "expected a reused singleton httpx client"
    assert str(c1.base_url).rstrip("/") == mod.BASE_URL.rstrip("/")


def test_chonkie_imported_eagerly_not_inside_tool():
    # Regression (W155c): `from chonkie import CodeChunker` inside _builder() — i.e.
    # inside asyncio.to_thread while FastMCP's loop is live — deadlocks on the import
    # lock and index_code hangs forever. The import must be at module top level.
    import ast, pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mcp_plugin" / "retrieval_mcp.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)}
    assert "chonkie" in top_level, "chonkie must be imported eagerly at module top"
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom) and sub.module == "chonkie":
                    raise AssertionError(
                        f"lazy chonkie import inside {node.name}() — W155c deadlock")


def test_journal_index_dir_skips_image_and_binary(tmp_path):
    """W161 (plugin mirror): a broad pattern still never indexes rendered-image/binary
    files (SVG/PNG/pickle) as prose; text data files (.json) are NOT blocked."""
    mod = _load()
    cap = _Capture()
    _wire(mod, cap)
    (tmp_path / "notes.md").write_text("# doc\n", encoding="utf-8")
    (tmp_path / "data.json").write_text('{"k": 1}\n', encoding="utf-8")
    (tmp_path / "plot.svg").write_text('<svg><path d="M0 0"/></svg>', encoding="utf-8")
    (tmp_path / "chart.PNG").write_text("\x89PNG", encoding="utf-8")

    out = asyncio.run(mod.journal_index_dir(str(tmp_path), pattern="*"))

    titles = {p["title"] for path, p in cap.posts if path == "/api/journal/entries"}
    assert titles == {"notes.md", "data.json"}
    assert "binary/image skipped" in out


def test_journal_index_dir_respects_gitignore(tmp_path):
    # Plugin mirror: a gitignored doc must not be indexed into the journal by default.
    mod = _load()
    cap = _Capture()
    _wire(mod, cap)
    git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("draft.md\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# readme\n", encoding="utf-8")
    (tmp_path / "draft.md").write_text("# draft\n", encoding="utf-8")

    out = asyncio.run(mod.journal_index_dir(str(tmp_path)))
    titles = {p["title"] for path, p in cap.posts if path == "/api/journal/entries"}
    assert titles == {"readme.md"}          # draft.md gitignored
    assert "1 gitignored" in out            # surfaced to the caller


def test_journal_index_dir_default_pattern_matches_markdown_case_insensitively(tmp_path, monkeypatch):
    mod = _load()
    cap = _Capture()
    _wire(mod, cap)
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    (tmp_path / "ENV.MD").write_text("# env\n", encoding="utf-8")
    (tmp_path / "Plan.Md").write_text("# plan\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me\n", encoding="utf-8")

    out = asyncio.run(mod.journal_index_dir(str(tmp_path)))

    titles = {p["title"] for path, p in cap.posts if path == "/api/journal/entries"}
    assert titles == {"README.md", "ENV.MD", "Plan.Md"}
    assert "3 file(s)" in out


def _child_dir_with_file(tmp_path, filename, content):
    child = tmp_path / "child"
    child.mkdir()
    (child / filename).write_text(content, encoding="utf-8")
    return child


def test_journal_index_dir_allows_scope_when_parent_does_not_cover_it(tmp_path):
    # No overlap conflict from the server → the child adds real coverage → proceeds +
    # reconciles its own snapshot. (The overlap decision itself is covered server-side by
    # tests/test_index_catalog.py::test_scope_overlap_no_block_when_parent_has_no_file_under_child.)
    mod = _load()
    child = _child_dir_with_file(tmp_path, "a.md", "# child\n")
    cap = _Capture({"/api/index/scope-overlap": {"ancestor_blocks": [], "child_warnings": []},
        "/api/index/report": {"total_files": 1},
        "/api/journal/docs/reconcile": {"removed_entries": 1, "skipped": False},
    })
    _wire(mod, cap)

    out = asyncio.run(mod.journal_index_dir(str(child)))

    titles = {p["title"] for path, p in cap.posts if path == "/api/journal/entries"}
    assert titles == {"a.md"}
    report = next(p for path, p in cap.posts if path == "/api/index/report")
    assert report["replace"] is True
    reconcile = next(p for path, p in cap.posts if path == "/api/journal/docs/reconcile")
    assert reconcile == {"project_slug": mod._scope_path(str(child)),
                         "live_titles": ["a.md"], "max_depth": 0, "patterns": ["*.md"]}
    assert "1 stale doc(s) removed" in out


def test_journal_index_dir_blocks_child_when_parent_scope_overlap_covers_it(tmp_path):
    # The parent/child overlap is now computed server-side (GET /api/index/scope-overlap):
    # an ancestor_block means an existing parent already covers this child → refuse.
    mod = _load()
    child = _child_dir_with_file(tmp_path, "a.md", "# child\n")
    cap = _Capture({"/api/index/scope-overlap": {
        "ancestor_blocks": [{"project": mod._scope_path(str(tmp_path)), "host": "h1"}],
        "child_warnings": []}})
    _wire(mod, cap)

    out = asyncio.run(mod.journal_index_dir(str(child)))

    assert "already covered by ancestor scope" in out
    assert "not indexed" in out
    assert not [p for path, p in cap.posts if path == "/api/journal/entries"]
    assert not [p for path, p in cap.posts if path == "/api/index/report"]


def test_journal_index_dir_allows_parent_over_child_scope_with_warning(tmp_path):
    mod = _load()
    child = _child_dir_with_file(tmp_path, "a.md", "# child\n")
    (tmp_path / "root.md").write_text("# root\n", encoding="utf-8")
    cap = _Capture({"/api/index/scope-overlap": {
        "ancestor_blocks": [],
        "child_warnings": [{"project": mod._scope_path(str(child)), "host": "h1"}]},
        "/api/index/report": {"total_files": 1}})
    _wire(mod, cap)

    out = asyncio.run(mod.journal_index_dir(str(tmp_path)))

    assert "warning" in out.lower()
    assert "child scope" in out
    titles = {p["title"] for path, p in cap.posts if path == "/api/journal/entries"}
    assert titles == {"root.md"}


def test_journal_index_dir_keeps_catalog_and_skips_reconcile_when_a_file_fails(tmp_path):
    # Finding #3 (plugin mirror): a failed file must NOT trigger a destructive catalog
    # replace, and the stale-doc reconcile must be skipped.
    mod = _load()
    (tmp_path / "a.md").write_text("# alpha\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# beta\n", encoding="utf-8")

    class _FailingCapture(_Capture):
        async def post(self, path, payload):
            if path == "/api/journal/entries" and payload.get("title") == "b.md":
                raise RuntimeError("simulated indexing failure")
            return await super().post(path, payload)

    cap = _FailingCapture({"/api/index/report": {"total_files": 1}})
    _wire(mod, cap)

    out = asyncio.run(mod.journal_index_dir(str(tmp_path)))

    assert "1 failed" in out
    report = next(p for path, p in cap.posts if path == "/api/index/report")
    assert report["replace"] is False
    assert not [p for path, p in cap.posts if path == "/api/journal/docs/reconcile"]


def test_journal_index_dir_reconciles_only_within_its_pattern_envelope(tmp_path):
    # W169b (plugin mirror): a "*.txt" re-run must sweep only within the *.txt envelope, so a
    # prior "*.md" index in the same scope is never treated as a "disappeared file".
    mod = _load()
    (tmp_path / "current.txt").write_text("txt body\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# md the txt scan never looks for\n", encoding="utf-8")
    cap = _Capture({"/api/index/report": {"total_files": 1},
                    "/api/journal/docs/reconcile": {"removed_entries": 0, "skipped": False}})
    _wire(mod, cap)

    asyncio.run(mod.journal_index_dir(str(tmp_path), pattern="*.txt"))

    titles = {p["title"] for path, p in cap.posts if path == "/api/journal/entries"}
    assert titles == {"current.txt"}
    reconcile = next(p for path, p in cap.posts if path == "/api/journal/docs/reconcile")
    assert reconcile["patterns"] == ["*.txt"]
    assert reconcile["live_titles"] == ["current.txt"]


def test_journal_index_dir_brace_pattern_covers_multiple_extensions(tmp_path):
    mod = _load()
    (tmp_path / "a.md").write_text("# md\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("txt\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("print(1)\n", encoding="utf-8")  # outside both globs
    cap = _Capture({"/api/index/report": {"total_files": 2},
                    "/api/journal/docs/reconcile": {"removed_entries": 0, "skipped": False}})
    _wire(mod, cap)

    asyncio.run(mod.journal_index_dir(str(tmp_path), pattern="*.{md,txt}"))

    titles = {p["title"] for path, p in cap.posts if path == "/api/journal/entries"}
    assert titles == {"a.md", "b.txt"}
    reconcile = next(p for path, p in cap.posts if path == "/api/journal/docs/reconcile")
    assert reconcile["patterns"] == ["*.md", "*.txt"]


def test_expand_brace_patterns():
    mod = _load()
    assert mod._expand_brace_patterns("*.md") == ["*.md"]
    assert mod._expand_brace_patterns("*.{md,txt}") == ["*.md", "*.txt"]
    assert mod._expand_brace_patterns("{a,b}.md") == ["a.md", "b.md"]
    assert mod._expand_brace_patterns("*") == ["*"]
    assert mod._expand_brace_patterns("*.{md") == ["*.{md"]
    assert mod._expand_brace_patterns("{a,{b,c}}.md") == ["{a,{b,c}}.md"]
    assert mod._expand_brace_patterns("{a,b}.{c,d}") == ["{a,b}.{c,d}"]
    assert mod._expand_brace_patterns("*.{}") == ["*.{}"]


def test_journal_index_dir_titles_nested_docs_by_relative_path(tmp_path):
    mod = _load()
    cap = _Capture()
    _wire(mod, cap)
    (tmp_path / "api").mkdir()
    (tmp_path / "user").mkdir()
    (tmp_path / "api" / "README.md").write_text("# API\n", encoding="utf-8")
    (tmp_path / "user" / "README.md").write_text("# User\n", encoding="utf-8")

    asyncio.run(mod.journal_index_dir(str(tmp_path), recursive=True))

    titles = {p["title"] for path, p in cap.posts if path == "/api/journal/entries"}
    assert titles == {"api/README.md", "user/README.md"}


def test_journal_search_formats_returned_body_chunks():
    # The installed plugin is what Codex actually calls. It must surface the
    # returned body chunk, not only the chunk-0 doc summary.
    mod = _load()
    cap = _Capture({"/api/journal/search": {
        "query": "ICML 2026",
        "project_slug": mod._scope_path("."),
        "mode": "keyword",
        "total": 1,
        "hits": [{
            "score": 0.91,
            "entry": {
                "entry_id": "journal_entry_doc",
                "project_slug": mod._scope_path("."),
                "client": "mcp",
                "source_kind": "doc",
                "title": "journal.md",
                "content_hash": "abc",
            },
            "best_chunk": {
                "chunk_id": "summary",
                "chunk_index": 0,
                "content": "journal.md source_kind: doc client: mcp",
            },
            "chunks": [
                {
                    "chunk_id": "summary",
                    "chunk_index": 0,
                    "content": "journal.md source_kind: doc client: mcp",
                },
                {
                    "chunk_id": "body",
                    "chunk_index": "body-12",
                    "heading": "ICML 2026",
                    "content": "ICML 2026 CDP manifest PDF download paragraph with concrete steps.",
                },
            ],
        }],
    }})
    _wire(mod, cap)

    out = asyncio.run(mod.journal_search(query="ICML 2026"))

    path, payload = cap.posts[0]
    assert path == "/api/journal/search"
    assert payload["include_chunks"] is True
    assert payload["chunk_limit"] == 3
    assert "ICML 2026 CDP manifest PDF download paragraph" in out


def test_journal_search_keeps_real_summary_heading_chunk():
    mod = _load()
    cap = _Capture({"/api/journal/search": {
        "query": "real summary",
        "project_slug": mod._scope_path("."),
        "mode": "keyword",
        "total": 1,
        "hits": [{
            "score": 0.91,
            "entry": {
                "entry_id": "journal_entry_doc",
                "project_slug": mod._scope_path("."),
                "client": "mcp",
                "source_kind": "doc",
                "title": "summary-section.md",
                "content_hash": "abc",
            },
            "best_chunk": {
                "chunk_id": "body-summary",
                "chunk_index": 1,
                "heading": "Summary",
                "content": "REAL SUMMARY MATCH from the user's Markdown section.",
            },
            "chunks": [
                {
                    "chunk_id": "body-summary",
                    "chunk_index": 1,
                    "heading": "Summary",
                    "content": "REAL SUMMARY MATCH from the user's Markdown section.",
                },
                {
                    "chunk_id": "details",
                    "chunk_index": 2,
                    "heading": "Details",
                    "content": "Unrelated details should not replace the matched summary section.",
                },
            ],
        }],
    }})
    _wire(mod, cap)

    out = asyncio.run(mod.journal_search(query="real summary"))

    assert "REAL SUMMARY MATCH" in out


def test_journal_tools_accept_explicit_scope_dir(tmp_path):
    mod = _load()
    cap = _Capture({
        "/api/journal/entries": {
            "entry": {
                "entry_id": "journal_entry_scoped",
                "project_slug": mod._scope_path(str(tmp_path)),
                "client": "mcp",
                "source_kind": "memory",
                "title": "Scoped note",
                "content_hash": "abc",
            },
            "chunk_count": 1,
            "indexed": True,
        },
        "/api/journal/search": {
            "query": "scoped docs",
            "project_slug": mod._scope_path(str(tmp_path)),
            "mode": "keyword",
            "total": 0,
            "hits": [],
        },
    })
    _wire(mod, cap)

    asyncio.run(mod.journal_record(body="Scoped note", scope_dir=str(tmp_path)))
    asyncio.run(mod.journal_search(query="scoped docs", scope_dir=str(tmp_path)))
    asyncio.run(mod.journal_recent(scope_dir=str(tmp_path)))

    scope = mod._scope_path(str(tmp_path))
    assert cap.posts[0] == ("/api/journal/entries", {
        "project_slug": scope,
        "client": "mcp",
        "source_kind": "memory",
        "body": "Scoped note",
        "tags": [],
    })
    assert cap.posts[1][0] == "/api/journal/search"
    assert cap.posts[1][1]["project_slug"] == scope
    assert cap.gets[0] == ("/api/journal/entries", {"project_slug": scope, "limit": 10})


def test_journal_search_renders_more_than_default_chunks_when_requested():
    mod = _load()
    chunks = [
        {"chunk_id": f"c{idx}", "chunk_index": idx, "content": f"body chunk {idx}"}
        for idx in range(1, 7)
    ]
    cap = _Capture({"/api/journal/search": {
        "query": "body chunk",
        "project_slug": mod._scope_path("."),
        "mode": "keyword",
        "total": 1,
        "hits": [{
            "score": 0.91,
            "entry": {
                "entry_id": "journal_entry_doc",
                "project_slug": mod._scope_path("."),
                "client": "mcp",
                "source_kind": "doc",
                "title": "large.md",
                "content_hash": "abc",
            },
            "best_chunk": chunks[0],
            "chunks": chunks,
        }],
    }})
    _wire(mod, cap)

    out = asyncio.run(mod.journal_search(query="body chunk", chunk_limit=5))

    assert "body chunk 5" in out
    assert "body chunk 6" not in out
    path, payload = cap.posts[0]
    assert path == "/api/journal/search"
    assert payload["include_chunks"] is True
    assert payload["chunk_limit"] == 5


def test_git_commit_uses_devnull_stdin():
    # W159: git is spawned from an MCP STDIO server. Without stdin=DEVNULL the child
    # inherits this process's stdin — the JSON-RPC pipe from the client — and wedges on
    # it for the full 5s timeout on EVERY index_code call (read as "index hangs").
    import inspect
    src = inspect.getsource(_load()._git_commit)
    assert "stdin=subprocess.DEVNULL" in src, "git subprocess must not inherit the stdio pipe"


def test_code_builder_is_cached_per_language():
    # Chunking a repo rebuilt the tree-sitter grammar per file. Cache per language.
    mod = _load()
    b1 = mod._builder("python")
    b2 = mod._builder("python")
    assert b1 is b2, "same language must return the cached AST builder"
    assert mod._builder("java") is not b1, "different language must be distinct"


def test_code_builder_cache_is_isolated_per_worker_thread():
    mod = _load()
    barrier = threading.Barrier(2)

    def get_builder(_):
        builder = mod._builder("c")
        barrier.wait(timeout=5)
        return builder

    with mod.ThreadPoolExecutor(max_workers=2) as executor:
        builders = list(executor.map(get_builder, range(2)))

    assert builders[0] is not builders[1]


def test_plugin_chunk_executor_restores_process_default_stack_size():
    mod = _load()
    original_size = threading.stack_size()

    with mod._chunk_executor(2) as executor:
        assert threading.stack_size() == original_size
        assert list(executor.map(lambda value: value + 1, [1, 2])) == [2, 3]


def test_plugin_file_identity_is_the_plain_file_sha256(tmp_path):
    """content_hash is the catalog's hash of the file's bytes, shared with the repo
    server; 0.5.6 briefly salted it with a chunker generation."""
    import hashlib

    from core.code_kb.chunker import file_content_hash

    mod = _load()
    path = tmp_path / "a.py"
    path.write_text("x = 1  # — 中文\n", encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert mod._file_content_hash(str(path)) == file_content_hash(str(path)) == expected


def test_plugin_chunks_match_the_server_chunker():
    """The plugin chunks private code, the server chunks public code; both drifted
    identically on byte offsets and on large files (2026-10-13), and must agree."""
    from core.code_kb import chunker
    from tests.test_code_chunker import (
        HUGE_TABLE_C, LARGE_CPP, NON_ASCII_PY, assert_chunks_cite_their_source)

    mod = _load()
    for helper in ("_char_spans", "_segments"):
        assert inspect.getsource(getattr(mod, helper)) == inspect.getsource(
            getattr(chunker, helper)), f"standalone plugin drifted in {helper}"
    for name, source, language in (("steps.py", NON_ASCII_PY, "python"),
                                   ("dlss5-feed.cpp", LARGE_CPP, "cpp"),
                                   ("dcadata.c", HUGE_TABLE_C, "c")):
        plugin = mod._chunk_source(name, source, language)
        server = chunker.chunk_code(name, source, language=language)
        assert_chunks_cite_their_source(plugin, source)
        assert [(c.start_line, c.end_line, c.code, c.label) for c in plugin] == [
            (c.start_line, c.end_line, c.code, c.label) for c in server], name


def test_plugin_chunks_huge_c_initializers_concurrently(tmp_path, monkeypatch):
    mod = _load()
    source = "const unsigned table[] = {\n" + "1,\n" * 150_000 + "};\n"
    for index in range(4):
        (tmp_path / f"table_{index}.c").write_text(source, encoding="utf-8")
    monkeypatch.setattr(mod, "_CHUNK_WORKERS", 4)

    payloads = mod._collect_code_files(str(tmp_path), respect_gitignore=False)

    assert len(payloads) == 4
    assert all(len(payload["chunks"]) > 50 for payload in payloads)
    assert all(max(len(chunk["code"]) for chunk in payload["chunks"]) < 8 * 1024
               for payload in payloads)
    assert all(payload["chunks"][0]["code"].startswith("const unsigned table[]")
               for payload in payloads)
    assert all(payload["chunks"][-1]["code"].rstrip().endswith("};")
               for payload in payloads)


# --------------------------------------------------------------------------- #
# Code KB — local chonkie chunking + REST upload/search
# --------------------------------------------------------------------------- #

def test_collect_code_files_chunks_locally_with_path_header(tmp_path):
    mod = _load()
    src = tmp_path / "sample.py"
    src.write_text(
        "def alpha(x):\n    return x + 1\n\n\nclass Beta:\n    def gamma(self):\n        return 2\n",
        encoding="utf-8",
    )
    files = mod._collect_code_files(str(tmp_path))
    assert len(files) == 1
    f = files[0]
    assert f["rel_path"] == "sample.py"       # forward slashes, repo-relative
    assert f["language"] == "python"
    assert f["chunks"], "expected at least one chunk"
    c = f["chunks"][0]
    assert c["content"].startswith("# path: sample.py\n")  # scope header
    assert c["start_line"] >= 1                             # 1-based lines
    assert c["end_line"] >= c["start_line"]


def test_collect_code_files_skips_unsupported_and_junk_dirs(tmp_path):
    mod = _load()
    (tmp_path / "keep.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# not code\n", encoding="utf-8")
    junk = tmp_path / "node_modules"
    junk.mkdir()
    (junk / "dep.js").write_text("function z(){return 1}\n", encoding="utf-8")
    rels = {f["rel_path"] for f in mod._collect_code_files(str(tmp_path))}
    assert rels == {"keep.py"}  # .md ignored, node_modules pruned


def test_plugin_file_walk_includes_cuda_sources(tmp_path):
    mod = _load()
    (tmp_path / "kernel.cu").write_text(
        "__global__ void scale(float *values) { values[threadIdx.x] *= 2; }\n",
        encoding="utf-8",
    )
    (tmp_path / "kernel.cuh").write_text(
        "void launch_scale(float *values);\n",
        encoding="utf-8",
    )

    assert {rel for _, rel in mod._iter_repo_files(str(tmp_path))} == {
        "kernel.cu",
        "kernel.cuh",
    }
    indexed_files = {item["rel_path"]: item for item in mod._collect_code_files(str(tmp_path))}
    assert set(indexed_files) == {"kernel.cu", "kernel.cuh"}
    assert indexed_files["kernel.cu"]["language"] == "cpp"
    assert indexed_files["kernel.cu"]["chunks"]


def test_plugin_indexes_web_source_under_a_frontend_directory(tmp_path):
    """The standalone MCP must not treat a source layout name as build output."""
    mod = _load()
    source_dir = tmp_path / "frontend" / "mobile-web" / "src"
    source_dir.mkdir(parents=True)
    for filename, source in {
        "publisher.ts": "export const publish = () => true;\n",
        "Page.tsx": "export const Page = () => <main />;\n",
        "legacy.js": "export const legacy = true;\n",
        "Widget.jsx": "export const Widget = () => <div />;\n",
    }.items():
        (source_dir / filename).write_text(source, encoding="utf-8")

    indexed_files = {
        item["rel_path"]: item["language"]
        for item in mod._collect_code_files(str(tmp_path))
    }
    assert indexed_files == {
        "frontend/mobile-web/src/publisher.ts": "typescript",
        "frontend/mobile-web/src/Page.tsx": "tsx",
        "frontend/mobile-web/src/legacy.js": "javascript",
        "frontend/mobile-web/src/Widget.jsx": "tsx",
    }


def test_plugin_indexes_source_under_a_vendor_directory(tmp_path):
    """The published MCP follows the same vendor-source inclusion rule as core."""
    mod = _load()
    source_file = tmp_path / "vendor" / "example.dev" / "codec" / "decoder.go"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("package codec\n", encoding="utf-8")

    indexed_files = {
        item["rel_path"]: item["language"]
        for item in mod._collect_code_files(str(tmp_path))
    }
    assert indexed_files == {
        "vendor/example.dev/codec/decoder.go": "go",
    }


def test_collect_code_files_respects_gitignore(tmp_path):
    # Plugin mirror: the standalone copy must skip gitignored files by default and
    # honor respect_gitignore=False to opt back in.
    mod = _load()
    git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("secret.py\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "secret.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    assert {f["rel_path"] for f in mod._collect_code_files(str(tmp_path))} == {"keep.py"}
    assert {f["rel_path"] for f in
            mod._collect_code_files(str(tmp_path), respect_gitignore=False)} == {"keep.py", "secret.py"}


def test_collect_code_files_applies_non_git_retrievalignore_and_explicit_excludes(tmp_path):
    mod = _load()
    (tmp_path / ".retrievalignore").write_text("generated/**\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "private.py").write_text("secret = 1\n", encoding="utf-8")
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "drop.py").write_text("y = 2\n", encoding="utf-8")

    rels = {f["rel_path"] for f in mod._collect_code_files(
        str(tmp_path), exclude_globs=["private.py"])}

    assert rels == {"keep.py"}


def test_gitignore_allowed_none_outside_repo_set_inside(tmp_path):
    mod = _load()
    assert mod._gitignore_allowed(str(tmp_path)) is None  # not a git repo → no filtering
    git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("y = 2\n", encoding="utf-8")
    allowed = mod._gitignore_allowed(str(tmp_path))
    assert allowed is not None and "keep.py" in allowed and "ignored.py" not in allowed


def test_index_code_uploads_chunks_and_bundles_report(monkeypatch):
    # W156 async + W159 true-async: index_code POSTs ONLY /api/code/index (202 job) and
    # bundles the catalog records in `report` — it never blocks/polls, and never makes a
    # separate /api/index/report round-trip (the backend writes it on completion).
    mod = _load()
    cap = _Capture({
        "/api/code/index": {"job_id": "j1", "status": "queued", "progress": 0.0,
                            "queue_position": 0},
    })
    _wire(mod, cap)
    monkeypatch.setattr(mod, "_collect_code_files", lambda root, *_a, **_k: [
        {"rel_path": "a.py", "language": "python",
         "chunks": [{"start_line": 1, "end_line": 3, "label": "a", "content": "# path: a.py\nx", "code": "x"}]},
        {"rel_path": "b.py", "language": "python",
         "chunks": [{"start_line": 1, "end_line": 2, "label": "", "content": "# path: b.py\ny", "code": "y"}]},
    ])
    out = _index_code_to_completion(mod, repo_root=".")
    paths = [p for p, _ in cap.posts]
    assert paths == ["/api/index/policy", "/api/code/index"]
    policy = cap.posts[0][1]
    assert policy["auto_refresh"] is True
    assert policy["respect_gitignore"] is True
    idx_payload = cap.posts[1][1]
    # scope is FORCED to the indexed dir's abspath — no hand-typed project (dedup fix).
    assert idx_payload["project"] == mod._scope_path(".")
    assert {f["rel_path"] for f in idx_payload["files"]} == {"a.py", "b.py"}
    report = idx_payload["report"]
    assert report["replace"] is True
    assert all(f["kind"] == "code" and f["collection"] == "code_chunks" for f in report["files"])
    assert "Indexing started" in out and "job j1" in out


def test_index_code_continues_when_policy_persistence_fails(monkeypatch):
    mod = _load()
    cap = _Capture({
        "/api/code/index": {"job_id": "j1", "status": "queued", "queue_position": 0},
    })

    async def post(path, payload):
        if path == "/api/index/policy":
            raise RuntimeError("policy endpoint down")
        return await cap.post(path, payload)

    mod._post = post
    mod._get = cap.get
    mod._delete = cap.delete
    monkeypatch.setattr(mod, "_collect_code_files", lambda root, *_a, **_k: [{
        "rel_path": "a.py", "language": "python", "chunks": [{
            "start_line": 1, "end_line": 1, "label": "", "content": "# path: a.py\nx",
            "code": "x",
        }],
    }])

    out = _index_code_to_completion(mod, repo_root=".")

    assert any(path == "/api/code/index" for path, _ in cap.posts)
    assert "auto-refresh policy could not be saved" in out


def test_index_code_blocks_child_when_parent_scope_overlap_covers_it(tmp_path, monkeypatch):
    mod = _load()
    child = _child_dir_with_file(tmp_path, "a.py", "x = 1\n")
    cap = _Capture({
        "/api/index/scope-overlap": {
            "ancestor_blocks": [{"project": mod._scope_path(str(tmp_path)), "host": "h1"}],
            "child_warnings": []},
        "/api/code/index": {"job_id": "j1", "status": "queued", "progress": 0.0,
                            "queue_position": 0},
    })
    _wire(mod, cap)
    monkeypatch.setattr(mod, "_collect_code_files", lambda root, *_a, **_k: [
        {"rel_path": "a.py", "language": "python", "chunks": [
            {"start_line": 1, "end_line": 1, "label": "", "content": "# path: a.py\nx", "code": "x"}]}])

    out = _index_code_to_completion(mod, repo_root=str(child))

    assert "already covered by ancestor scope" in out
    assert "not indexed" in out
    assert not any(path == "/api/code/index" for path, _ in cap.posts)


def test_index_code_allows_parent_when_child_scope_exists_with_warning(tmp_path, monkeypatch):
    mod = _load()
    child = _child_dir_with_file(tmp_path, "a.py", "x = 1\n")
    (tmp_path / "root.py").write_text("root = 1\n", encoding="utf-8")
    cap = _Capture({
        "/api/index/scope-overlap": {
            "ancestor_blocks": [],
            "child_warnings": [{"project": mod._scope_path(str(child)), "host": "h1"}]},
        "/api/code/index": {"job_id": "j1", "status": "queued", "progress": 0.0,
                            "queue_position": 0},
    })
    _wire(mod, cap)
    monkeypatch.setattr(mod, "_collect_code_files", lambda root, *_a, **_k: [
        {"rel_path": "root.py", "language": "python", "chunks": [
            {"start_line": 1, "end_line": 1, "label": "", "content": "# path: root.py\nx", "code": "x"}]},
        {"rel_path": "child/a.py", "language": "python", "chunks": [
            {"start_line": 1, "end_line": 1, "label": "", "content": "# path: child/a.py\nx", "code": "x"}]},
    ])

    out = _index_code_to_completion(mod, repo_root=str(tmp_path))

    assert "Indexing started" in out
    assert "warning" in out.lower()
    assert "child scope" in out
    idx_payload = next(pl for p, pl in cap.posts if p == "/api/code/index")
    assert idx_payload["project"] == mod._scope_path(str(tmp_path))


def test_index_code_returns_job_id_immediately_without_blocking(monkeypatch):
    """W159: index_code must NEVER block/poll — submit and hand back the job_id + how to
    poll, right away, regardless of repo size."""
    mod = _load()
    cap = _Capture({
        "/api/code/index": {"job_id": "jq", "status": "queued", "progress": 0.0,
                            "queue_position": 2},
    })
    _wire(mod, cap)
    monkeypatch.setattr(mod, "_collect_code_files", lambda root, *_a, **_k: [
        {"rel_path": "a.py", "language": "python", "chunks": [
            {"start_line": 1, "end_line": 1, "label": "", "content": "# path: a.py\nx", "code": "x"}]}])
    async def scenario():
        out = await mod.index_code(repo_root=".")
        match = re.search(r"job ([0-9a-f]+)", out)
        assert match and "[preparing]" in out and "index_code_status" in out
        assert cap.posts == [], (
            "the tool reply must precede preflight, local collection, and submission")
        task = mod._CODE_PREPARATION_TASKS[match.group(1)]
        await task

    asyncio.run(scenario())
    assert [p for p, _ in cap.posts] == ["/api/index/policy", "/api/code/index"]


def test_index_code_returns_while_local_chunking_is_still_running(monkeypatch):
    """The MCP call itself must not wait for the client-side repository walk/chunker."""
    mod = _load()
    cap = _Capture({
        "/api/code/index": {"job_id": "backend-1", "status": "queued", "progress": 0.0},
        "/api/code/jobs/backend-1": {
            "job_id": "backend-1", "status": "completed", "progress": 1.0,
            "indexed": 1,
        },
    })
    _wire(mod, cap)
    collection_started = threading.Event()
    release_collection = threading.Event()

    def collect(*_args, **_kwargs):
        collection_started.set()
        release_collection.wait(timeout=5)
        return [{
            "rel_path": "a.py", "language": "python", "chunks": [{
                "start_line": 1, "end_line": 1, "label": "",
                "content": "# path: a.py\nx", "code": "x",
            }],
        }]

    monkeypatch.setattr(mod, "_collect_code_files", collect)

    async def scenario():
        call = asyncio.create_task(mod.index_code(repo_root=".", auto_refresh=False))
        while not collection_started.is_set():
            await asyncio.sleep(0.01)
        try:
            reply = await asyncio.wait_for(asyncio.shield(call), timeout=0.1)
            local_job_id = re.search(r"job ([0-9a-f]+)", reply).group(1)
            status = await mod.index_code_status(local_job_id)
            assert "[preparing]" in status
            assert not any(path == "/api/code/index" for path, _ in cap.posts)
        finally:
            release_collection.set()
            await call

        while not any(path == "/api/code/index" for path, _ in cap.posts):
            await asyncio.sleep(0.01)
        while "[completed]" not in await mod.index_code_status(local_job_id):
            await asyncio.sleep(0.01)

    asyncio.run(scenario())


def test_index_code_returns_while_scope_preflight_is_still_running(monkeypatch):
    mod = _load()
    cap = _Capture()
    _wire(mod, cap)
    preflight_started = asyncio.Event()
    release_preflight = asyncio.Event()

    async def guard(*_args, **_kwargs):
        preflight_started.set()
        await release_preflight.wait()
        return None, ""

    monkeypatch.setattr(mod, "_nested_scope_guard", guard)
    monkeypatch.setattr(mod, "_collect_code_files", lambda *_a, **_k: [])

    async def scenario():
        reply = await asyncio.wait_for(
            mod.index_code(repo_root=".", auto_refresh=False), timeout=0.1)
        local_job_id = re.search(r"job ([0-9a-f]+)", reply).group(1)
        await preflight_started.wait()
        assert "checking scope overlap" in await mod.index_code_status(local_job_id)
        release_preflight.set()
        await mod._CODE_PREPARATION_TASKS[local_job_id]

    asyncio.run(scenario())


def test_active_preparation_deduplicates_index_calls_and_suppresses_watcher(monkeypatch):
    mod = _load()
    cap = _Capture({
        "/api/code/index": {"job_id": "backend-1", "status": "completed",
                            "progress": 1.0, "indexed": 1},
        "/api/code/search": {"hits": []},
    })
    _wire(mod, cap)
    collection_started = threading.Event()
    release_collection = threading.Event()
    collection_count = 0
    watcher_count = 0

    def collect(*_args, **_kwargs):
        nonlocal collection_count
        collection_count += 1
        collection_started.set()
        release_collection.wait(timeout=5)
        return [{
            "rel_path": "a.py", "language": "python", "chunks": [{
                "start_line": 1, "end_line": 1, "label": "",
                "content": "# path: a.py\nx", "code": "x",
            }],
        }]

    async def watch_scope(**_kwargs):
        nonlocal watcher_count
        watcher_count += 1

    monkeypatch.setattr(mod, "_collect_code_files", collect)
    monkeypatch.setattr(mod, "_watch_code_scope", watch_scope)

    async def scenario():
        first = await mod.index_code(repo_root=".", auto_refresh=False)
        first_job_id = re.search(r"job ([0-9a-f]+)", first).group(1)
        while not collection_started.is_set():
            await asyncio.sleep(0.01)
        second = await mod.index_code(repo_root=".", auto_refresh=False)
        assert first_job_id in second
        assert "no duplicate scan" in second.lower()
        await mod.search_code(query="probe", repo_root=".")
        assert collection_count == 1
        assert watcher_count == 0
        release_collection.set()
        await mod._CODE_PREPARATION_TASKS[first_job_id]

    asyncio.run(scenario())


def test_delete_waits_for_inflight_index_submission(monkeypatch):
    mod = _load()
    submission_started = asyncio.Event()
    release_submission = asyncio.Event()
    operation_order = []

    async def post(path, payload):
        if path == "/api/index/policy":
            return {}
        assert path == "/api/code/index"
        operation_order.append("submit-started")
        submission_started.set()
        await release_submission.wait()
        operation_order.append("submit-finished")
        return {"job_id": payload["client_job_id"], "status": "completed",
                "progress": 1.0, "indexed": 1}

    async def delete(path, params=None):
        operation_order.append("delete-sent")
        return {"deleted": 1, "catalog_removed": 1}

    mod._post = post
    mod._get = _Capture().get
    mod._delete = delete
    monkeypatch.setattr(mod, "_collect_code_files", lambda *_a, **_k: [{
        "rel_path": "a.py", "language": "python", "chunks": [{
            "start_line": 1, "end_line": 1, "label": "",
            "content": "# path: a.py\nx", "code": "x",
        }],
    }])

    async def scenario():
        reply = await mod.index_code(repo_root=".", auto_refresh=False)
        local_job_id = re.search(r"job ([0-9a-f]+)", reply).group(1)
        await submission_started.wait()
        deletion = asyncio.create_task(mod.delete_code(repo_root="."))
        await asyncio.sleep(0.05)
        assert "delete-sent" not in operation_order
        release_submission.set()
        await deletion
        task = mod._CODE_PREPARATION_TASKS.get(local_job_id)
        if task is not None:
            await task

    asyncio.run(scenario())
    assert operation_order == ["submit-started", "submit-finished", "delete-sent"]


def test_lost_submit_response_recovers_by_durable_receipt(monkeypatch):
    mod = _load()
    receipt_queries = []

    async def post(path, payload):
        if path == "/api/index/policy":
            return {}
        raise mod.httpx.ReadTimeout(
            "response lost after backend acceptance",
            request=mod.httpx.Request("POST", "http://backend/api/code/index"),
        )

    async def get(path, params=None):
        if path == "/api/index/scope-overlap":
            return {}
        if path.startswith("/api/code/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            receipt_queries.append(job_id)
            return {"job_id": job_id, "status": "completed", "progress": 1.0,
                    "indexed": 1}
        return {}

    mod._post = post
    mod._get = get
    mod._delete = _Capture().delete
    monkeypatch.setattr(mod, "_collect_code_files", lambda *_a, **_k: [{
        "rel_path": "a.py", "language": "python", "chunks": [{
            "start_line": 1, "end_line": 1, "label": "",
            "content": "# path: a.py\nx", "code": "x",
        }],
    }])

    out = _index_code_to_completion(mod, repo_root=".", auto_refresh=False)

    local_job_id = re.search(r"job ([0-9a-f]+)", out).group(1)
    assert receipt_queries == [local_job_id]
    assert "[completed]" in out and "response lost" not in out


def test_submit_http_error_is_not_reconciled_as_an_old_receipt(monkeypatch):
    mod = _load()
    receipt_queried = False

    async def submit(_request):
        response = mod.httpx.Response(
            409,
            request=mod.httpx.Request("POST", "http://backend/api/code/index"),
        )
        raise mod.httpx.HTTPStatusError(
            "receipt collision", request=response.request, response=response)

    async def receipt(_job_id):
        nonlocal receipt_queried
        receipt_queried = True
        return {"status": "completed"}

    monkeypatch.setattr(mod, "_submit_code_index_request", submit)
    monkeypatch.setattr(mod, "_get_code_index_receipt", receipt)

    async def scenario():
        with pytest.raises(mod.httpx.HTTPStatusError):
            await mod._submit_code_index_with_receipt(
                key=("host", "project"),
                client_job_id="e" * 32,
                request={"project": "project"},
            )

    asyncio.run(scenario())
    assert receipt_queried is False


def _inv_response(files):
    """A minimal /api/index/inventory response — the incremental diff baseline.
    files: (rel_path, content_hash, chunk_count, language)."""
    return {"users": [{"hosts": [{"projects": [{"dirs": [{"files": [
        {"rel_path": r, "kind": "code", "content_hash": h, "chunk_count": cc, "language": lang}
        for (r, h, cc, lang) in files]}]}]}]}]}


def test_index_code_incremental_first_pass_flags_and_manifest(tmp_path):
    # No prior inventory → incremental still submits everything, but flagged incremental
    # with a manifest so the server reconciles instead of wiping.
    mod = _load()
    (tmp_path / "sample.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    cap = _Capture({
        "/api/index/inventory": {"users": []},
        "/api/code/index": {"job_id": "j1", "status": "completed", "indexed": 1,
                            "progress": 1.0, "queue_position": 0},
        "/api/index/report": {"total_files": 1},
    })
    _wire(mod, cap)
    out = _index_code_to_completion(mod, repo_root=str(tmp_path), incremental=True)
    idx = next(pl for p, pl in cap.posts if p == "/api/code/index")
    assert idx["incremental"] is True
    assert idx["manifest"] == ["sample.py"]
    assert {f["rel_path"] for f in idx["files"]} == {"sample.py"}
    assert all(f.get("content_hash") for f in idx["files"])
    assert "Indexing started" in out


def test_index_code_incremental_up_to_date_skips_submit(tmp_path):
    # Inventory already at the file's CURRENT hash → nothing to do, no job submitted.
    mod = _load()
    (tmp_path / "sample.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    h = mod._file_content_hash(str(tmp_path / "sample.py"))
    cap = _Capture({"/api/index/inventory": _inv_response([("sample.py", h, 1, "python")])})
    _wire(mod, cap)
    out = _index_code_to_completion(mod, repo_root=str(tmp_path), incremental=True)
    assert "up to date" in out
    assert not any(p == "/api/code/index" for p, _ in cap.posts)


def test_index_code_incremental_only_sends_changed(tmp_path):
    # a.py unchanged (real hash), b.py changed (stale hash) → only b.py re-embedded;
    # manifest keeps both; the catalog is re-reported as a full snapshot (replace=True).
    mod = _load()
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    ah = mod._file_content_hash(str(tmp_path / "a.py"))
    cap = _Capture({
        "/api/index/inventory": _inv_response([("a.py", ah, 1, "python"),
                                               ("b.py", "stale", 1, "python")]),
        "/api/code/index": {"job_id": "j1", "status": "completed", "indexed": 1,
                            "progress": 1.0, "queue_position": 0},
        "/api/index/report": {"total_files": 2},
    })
    _wire(mod, cap)
    _index_code_to_completion(mod, repo_root=str(tmp_path), incremental=True)
    idx = next(pl for p, pl in cap.posts if p == "/api/code/index")
    assert {f["rel_path"] for f in idx["files"]} == {"b.py"}
    assert set(idx["manifest"]) == {"a.py", "b.py"}
    rep = idx["report"]  # bundled full snapshot, written by the backend on completion
    assert rep["replace"] is True
    assert {f["rel_path"] for f in rep["files"]} == {"a.py", "b.py"}


def test_auto_refresh_non_git_repo_submits_changed_hashes(tmp_path):
    mod = _load()
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    project = mod._scope_path(str(tmp_path))
    cap = _Capture({
        "/api/index/policy": {
            "host": "desktop", "project": project, "kind": "code",
            "root_dir": str(tmp_path), "auto_refresh": True,
            "respect_gitignore": True, "exclude_globs": [],
            "min_check_interval_s": 0,
        },
        "/api/index/inventory": _inv_response([("a.py", "stale", 1, "python")]),
        "/api/code/index": {"job_id": "refresh-1", "status": "queued"},
    })
    _wire(mod, cap)

    result = asyncio.run(mod._auto_refresh_code(
        project=project, host="desktop", repo_root=str(tmp_path)))

    assert result == {
        "state": "refresh_started", "job_id": "refresh-1", "status": "queued",
        "changed": 1, "removed": 0,
    }
    payload = next(body for path, body in cap.posts if path == "/api/code/index")
    assert payload["incremental"] is True
    assert payload["manifest"] == ["a.py"]
    assert payload["files"][0]["content_hash"] != "stale"


def test_auto_refresh_active_job_short_circuits_before_policy_or_scan(tmp_path, monkeypatch):
    mod = _load()
    project = mod._scope_path(str(tmp_path))
    cap = _Capture({
        "/api/code/jobs": {"jobs": [{
            "job_id": "active-1", "project": project, "status": "running",
        }]},
    })
    _wire(mod, cap)
    monkeypatch.setattr(
        mod, "_collect_incremental",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not scan")),
    )

    result = asyncio.run(mod._auto_refresh_code(
        project=project, host="desktop", repo_root=str(tmp_path)))

    assert result == {"state": "refresh_running", "job_id": "active-1", "status": "running"}
    assert [path for path, _ in cap.gets] == ["/api/code/jobs"]


def test_auto_refresh_missing_repo_preserves_completed_index(tmp_path):
    mod = _load()
    missing = tmp_path / "gone"
    project = mod._scope_path(str(missing))
    cap = _Capture({
        "/api/index/policy": {
            "host": "desktop", "project": project, "kind": "code",
            "root_dir": str(missing), "auto_refresh": True,
            "respect_gitignore": True, "exclude_globs": [],
            "min_check_interval_s": 0,
        },
    })
    _wire(mod, cap)

    result = asyncio.run(mod._auto_refresh_code(
        project=project, host="desktop", repo_root=str(missing)))

    assert result["state"] == "unavailable"
    assert not any(path == "/api/code/index" for path, _ in cap.posts)


def test_auto_refresh_never_scans_server_supplied_different_root(tmp_path, monkeypatch):
    mod = _load()
    requested = tmp_path / "requested"
    requested.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    project = mod._scope_path(str(requested))
    cap = _Capture({
        "/api/code/jobs": {"jobs": []},
        "/api/index/policy": {
            "host": "desktop", "project": project, "kind": "code",
            "root_dir": str(other), "auto_refresh": True,
            "respect_gitignore": True, "exclude_globs": [], "min_check_interval_s": 0,
        },
        "/api/index/inventory": _inv_response([("a.py", "old", 1, "python")]),
    })
    _wire(mod, cap)
    seen = []

    def collect(root, *_args):
        seen.append(mod._scope_path(root))
        return [], ["a.py"], {"a.py": "old"}

    monkeypatch.setattr(mod, "_collect_incremental", collect)
    result = asyncio.run(mod._auto_refresh_code(
        project=project, host="desktop", repo_root=str(requested)))

    assert result["state"] == "current"
    assert seen == [project]


def test_watched_paths_submit_only_touched_files_and_explicit_deletions(tmp_path):
    mod = _load()
    changed = tmp_path / "changed.py"
    changed.write_text("value = 2\n", encoding="utf-8")
    deleted = tmp_path / "deleted.py"
    project = mod._scope_path(str(tmp_path))
    cap = _Capture({
        "/api/code/index": {"job_id": "watch-1", "status": "queued"},
    })
    _wire(mod, cap)
    known = {
        "changed.py": {"content_hash": "old", "chunk_count": 1, "language": "python"},
        "deleted.py": {"content_hash": "old", "chunk_count": 1, "language": "python"},
        "untouched.py": {"content_hash": "same", "chunk_count": 1, "language": "python"},
    }

    result = asyncio.run(mod._submit_watched_code_paths(
        project=project,
        host="desktop",
        repo_root=str(tmp_path),
        changed_paths={str(changed), str(deleted)},
        known=known,
        respect_gitignore=False,
        exclude_globs=[],
    ))

    assert result.status["state"] == "refresh_started"
    path, payload = next(item for item in cap.posts if item[0] == "/api/code/index")
    assert path == "/api/code/index"
    assert {item["rel_path"] for item in payload["files"]} == {"changed.py"}
    assert payload["removed_paths"] == ["deleted.py"]
    assert "manifest" not in payload
    assert payload["report"]["replace"] is False
    assert {item["rel_path"] for item in payload["report"]["files"]} == {"changed.py"}
    assert known["changed.py"]["content_hash"] == "old"
    assert "deleted.py" in known

    mod._apply_watch_submission(known, result)
    assert known["changed.py"]["content_hash"] != "old"
    assert "deleted.py" not in known


def test_watched_paths_ignore_files_outside_selected_repo_root(tmp_path):
    mod = _load()
    selected = tmp_path / "selected"
    sibling = tmp_path / "sibling"
    selected.mkdir()
    sibling.mkdir()
    inside = selected / "inside.py"
    outside = sibling / "outside.py"
    inside.write_text("inside = True\n", encoding="utf-8")
    outside.write_text("outside = True\n", encoding="utf-8")

    payloads, removed, _current, reconcile = mod._collect_watched_code_changes(
        str(selected), {str(inside), str(outside)}, {}, False, [])

    assert {payload["rel_path"] for payload in payloads} == {"inside.py"}
    assert removed == []
    assert reconcile is False


def test_search_code_reuses_one_watcher_and_never_runs_full_scan(tmp_path, monkeypatch):
    mod = _load()
    cap = _Capture({"/api/code/search": {"hits": []}})
    _wire(mod, cap)
    started = 0
    release = asyncio.Event()

    async def watch_scope(**_kwargs):
        nonlocal started
        started += 1
        await release.wait()

    monkeypatch.setattr(mod, "_watch_code_scope", watch_scope)
    monkeypatch.setattr(
        mod, "_collect_incremental",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("search must not scan")),
    )

    async def scenario():
        await mod.search_code(query="first", repo_root=str(tmp_path))
        await mod.search_code(query="second", repo_root=str(tmp_path))
        await asyncio.sleep(0)
        assert started == 1
        release.set()
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_real_filesystem_watcher_indexes_create_and_delete_events(tmp_path, monkeypatch):
    mod = _load()
    project = mod._scope_path(str(tmp_path))
    cap = _Capture({
        "/api/index/policy": {
            "host": "desktop", "project": project, "kind": "code",
            "root_dir": str(tmp_path), "auto_refresh": True,
            "respect_gitignore": True, "exclude_globs": [],
            "min_check_interval_s": 3600,
        },
        "/api/index/inventory": {"users": []},
        "/api/code/index": {"job_id": "watch-real", "status": "queued"},
        "/api/code/jobs/watch-real": {
            "job_id": "watch-real", "status": "completed", "error": None,
        },
    })
    _wire(mod, cap)
    monkeypatch.setattr(
        mod, "_collect_incremental",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("watch event must not scan")),
    )

    async def wait_until(predicate, timeout_seconds=8.0):
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("filesystem watcher did not observe the event in time")
            await asyncio.sleep(0.05)

    async def scenario():
        task = mod._ensure_code_watcher(
            project=project, host="desktop", repo_root=str(tmp_path))
        assert task is not None
        await wait_until(
            lambda: mod._CODE_WATCH_STATUS.get(("desktop", project), {}).get("state")
            == "watching")

        source = tmp_path / "watched.py"
        source.write_text("value = 1\n", encoding="utf-8")
        await wait_until(lambda: len([
            body for path, body in cap.posts if path == "/api/code/index"
        ]) >= 1)
        first = [body for path, body in cap.posts if path == "/api/code/index"][0]
        assert {item["rel_path"] for item in first["files"]} == {"watched.py"}
        assert first["removed_paths"] == []

        source.unlink()
        await wait_until(lambda: len([
            body for path, body in cap.posts if path == "/api/code/index"
        ]) >= 2)
        second = [body for path, body in cap.posts if path == "/api/code/index"][1]
        assert second["files"] == []
        assert second["removed_paths"] == ["watched.py"]

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_disabled_watcher_policy_is_cached_across_searches(tmp_path):
    mod = _load()
    cap = _Capture({
        "/api/code/search": {"hits": []},
        "/api/index/policy": {"auto_refresh": False},
    })
    _wire(mod, cap)

    async def scenario():
        await mod.search_code(query="first", repo_root=str(tmp_path))
        await asyncio.sleep(0)
        await mod.search_code(query="second", repo_root=str(tmp_path))
        await asyncio.sleep(0)

    asyncio.run(scenario())

    policy_gets = [path for path, _params in cap.gets if path == "/api/index/policy"]
    assert policy_gets == ["/api/index/policy"]


def test_watcher_coalesces_events_while_one_index_job_is_running(tmp_path, monkeypatch):
    mod = _load()
    project = mod._scope_path(str(tmp_path))
    event_batches: asyncio.Queue = asyncio.Queue()
    submitted: list[set[str]] = []
    first_job_started = asyncio.Event()
    release_first_job = asyncio.Event()

    async def policy(**_kwargs):
        return {"auto_refresh": True, "respect_gitignore": False, "exclude_globs": []}

    async def watch_changes(_root):
        while True:
            yield await event_batches.get()

    async def submit(**kwargs):
        submitted.append(set(kwargs["changed_paths"]))
        return mod._WatchSubmission(
            status={"state": "refresh_started", "job_id": f"job-{len(submitted)}",
                    "status": "queued"},
            updated_nodes={},
            removed_paths=[],
        )

    async def wait_job(job_id):
        if job_id == "job-1":
            first_job_started.set()
            await release_first_job.wait()
        return {"job_id": job_id, "status": "completed"}

    monkeypatch.setattr(mod, "_get_code_policy", policy)
    monkeypatch.setattr(mod, "_known_code_hashes", lambda _project: asyncio.sleep(0, result={}))
    monkeypatch.setattr(mod, "_watch_changes", watch_changes)
    monkeypatch.setattr(mod, "_submit_watched_code_paths", submit)
    monkeypatch.setattr(mod, "_wait_code_job", wait_job)

    async def scenario():
        task = asyncio.create_task(mod._watch_code_scope(
            project=project, host="desktop", repo_root=str(tmp_path)))
        while mod._CODE_WATCH_STATUS.get(("desktop", project), {}).get("state") != "watching":
            await asyncio.sleep(0)
        await event_batches.put({(1, str(tmp_path / "a.py"))})
        await first_job_started.wait()
        await event_batches.put({(2, str(tmp_path / "b.py"))})
        await event_batches.put({(2, str(tmp_path / "c.py"))})
        await asyncio.sleep(0.05)
        assert len(submitted) == 1
        release_first_job.set()
        while len(submitted) < 2:
            await asyncio.sleep(0)
        assert submitted[1] == {str(tmp_path / "b.py"), str(tmp_path / "c.py")}
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_watcher_queues_deletion_until_bootstrap_index_completes(tmp_path, monkeypatch):
    mod = _load()
    project = mod._scope_path(str(tmp_path))
    deleted = tmp_path / "deleted.py"
    event_batches: asyncio.Queue = asyncio.Queue()
    bootstrap_started = asyncio.Event()
    release_bootstrap = asyncio.Event()
    cap = _Capture({
        "/api/code/index": {"job_id": "delete-job", "status": "queued"},
    })
    _wire(mod, cap)

    async def policy(**_kwargs):
        return {"auto_refresh": True, "respect_gitignore": False, "exclude_globs": []}

    async def watch_changes(_root):
        while True:
            yield await event_batches.get()

    async def wait_job(job_id):
        if job_id == "bootstrap-job":
            bootstrap_started.set()
            await release_bootstrap.wait()
        return {"job_id": job_id, "status": "completed"}

    monkeypatch.setattr(mod, "_get_code_policy", policy)
    monkeypatch.setattr(mod, "_watch_changes", watch_changes)
    monkeypatch.setattr(mod, "_wait_code_job", wait_job)

    async def scenario():
        task = asyncio.create_task(mod._watch_code_scope(
            project=project,
            host="desktop",
            repo_root=str(tmp_path),
            bootstrap_job_id="bootstrap-job",
            initial_known={
                "deleted.py": {
                    "rel_path": "deleted.py", "content_hash": "old",
                    "chunk_count": 1, "language": "python",
                },
            },
        ))
        await bootstrap_started.wait()
        await event_batches.put({(3, str(deleted))})
        await asyncio.sleep(0.05)
        assert not any(path == "/api/code/index" for path, _body in cap.posts)
        release_bootstrap.set()
        while not any(path == "/api/code/index" for path, _body in cap.posts):
            await asyncio.sleep(0)
        payload = next(body for path, body in cap.posts if path == "/api/code/index")
        assert payload["files"] == []
        assert payload["removed_paths"] == ["deleted.py"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_failed_watcher_job_does_not_advance_known_hash(tmp_path, monkeypatch):
    mod = _load()
    project = mod._scope_path(str(tmp_path))
    event_batches: asyncio.Queue = asyncio.Queue()
    observed_hashes: list[str] = []

    async def policy(**_kwargs):
        return {"auto_refresh": True, "respect_gitignore": False, "exclude_globs": []}

    async def watch_changes(_root):
        while True:
            yield await event_batches.get()

    async def submit(**kwargs):
        observed_hashes.append(kwargs["known"]["a.py"]["content_hash"])
        return mod._WatchSubmission(
            status={"state": "refresh_started", "job_id": "failed-job",
                    "status": "queued"},
            updated_nodes={"a.py": {"rel_path": "a.py", "content_hash": "new"}},
            removed_paths=[],
        )

    async def failed_job(_job_id):
        return {"job_id": "failed-job", "status": "failed", "error": "embed failed"}

    monkeypatch.setattr(mod, "_get_code_policy", policy)
    monkeypatch.setattr(mod, "_watch_changes", watch_changes)
    monkeypatch.setattr(mod, "_submit_watched_code_paths", submit)
    monkeypatch.setattr(mod, "_wait_code_job", failed_job)

    async def scenario():
        task = asyncio.create_task(mod._watch_code_scope(
            project=project,
            host="desktop",
            repo_root=str(tmp_path),
            initial_known={"a.py": {"rel_path": "a.py", "content_hash": "old"}},
        ))
        while mod._CODE_WATCH_STATUS.get(("desktop", project), {}).get("state") != "watching":
            await asyncio.sleep(0)
        event = {(2, str(tmp_path / "a.py"))}
        await event_batches.put(event)
        while len(observed_hashes) < 1:
            await asyncio.sleep(0)
        await event_batches.put(event)
        while len(observed_hashes) < 2:
            await asyncio.sleep(0)
        assert observed_hashes == ["old", "old"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_failed_active_job_does_not_drop_pending_ignore_reconcile(tmp_path, monkeypatch):
    mod = _load()
    project = mod._scope_path(str(tmp_path))
    event_batches: asyncio.Queue = asyncio.Queue()
    submissions = 0

    async def policy(**_kwargs):
        return {"auto_refresh": True, "respect_gitignore": True, "exclude_globs": []}

    async def watch_changes(_root):
        while True:
            yield await event_batches.get()

    async def submit(**_kwargs):
        nonlocal submissions
        submissions += 1
        if submissions == 1:
            return mod._WatchSubmission(
                status={"state": "refresh_running", "job_id": "active-job",
                        "status": "running"},
                updated_nodes={}, removed_paths=[], refresh_inventory=True)
        return mod._WatchSubmission(
            status={"state": "current", "files": 1},
            updated_nodes={}, removed_paths=[], refresh_inventory=True)

    async def failed_job(_job_id):
        return {"job_id": "active-job", "status": "failed", "error": "upsert failed"}

    monkeypatch.setattr(mod, "_get_code_policy", policy)
    monkeypatch.setattr(mod, "_known_code_hashes", lambda _project: asyncio.sleep(0, result={}))
    monkeypatch.setattr(mod, "_watch_changes", watch_changes)
    monkeypatch.setattr(mod, "_submit_watched_code_paths", submit)
    monkeypatch.setattr(mod, "_wait_code_job", failed_job)

    async def scenario():
        task = asyncio.create_task(mod._watch_code_scope(
            project=project, host="desktop", repo_root=str(tmp_path)))
        while mod._CODE_WATCH_STATUS.get(("desktop", project), {}).get("state") != "watching":
            await asyncio.sleep(0)
        await event_batches.put({(2, str(tmp_path / ".gitignore"))})
        while submissions < 2:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_index_code_incremental_keeps_transiently_unreadable_file(tmp_path, monkeypatch):
    # CF1 (plugin mirror): b.py is already indexed but momentarily unreadable this run
    # (hash reads None) → it must NOT be reconciled away. It stays in the manifest and
    # its stored hash is preserved in the catalog report.
    mod = _load()
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")  # changed
    (tmp_path / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")  # unreadable
    real_hash = mod._file_content_hash

    def flaky_hash(path):
        return None if path.replace("\\", "/").endswith("/b.py") else real_hash(path)
    monkeypatch.setattr(mod, "_file_content_hash", flaky_hash)

    cap = _Capture({
        "/api/index/inventory": _inv_response([("a.py", "stale", 1, "python"),
                                               ("b.py", "bhash", 1, "python")]),
        "/api/code/index": {"job_id": "j1", "status": "completed", "indexed": 1,
                            "progress": 1.0, "queue_position": 0},
        "/api/index/report": {"total_files": 2},
    })
    _wire(mod, cap)
    _index_code_to_completion(mod, repo_root=str(tmp_path), incremental=True)
    idx = next(pl for p, pl in cap.posts if p == "/api/code/index")
    assert {f["rel_path"] for f in idx["files"]} == {"a.py"}      # only a.py re-embedded
    assert set(idx["manifest"]) == {"a.py", "b.py"}              # unreadable b.py kept
    reported = {f["rel_path"]: f for f in idx["report"]["files"]}
    assert reported["b.py"]["content_hash"] == "bhash"          # stored hash preserved


def test_index_code_status_formats_single_job_and_list():
    mod = _load()
    cap = _Capture({
        "/api/code/jobs/j1": {"job_id": "j1", "status": "running", "project": "p",
                              "chunks_done": 50, "chunks_total": 100, "progress": 0.5,
                              "queue_position": 0, "indexed": 0},
        "/api/code/jobs": {"jobs": [
            {"job_id": "j1", "status": "completed", "project": "p", "chunks_done": 3,
             "chunks_total": 3, "progress": 1.0, "indexed": 3}]},
    })
    _wire(mod, cap)
    one = asyncio.run(mod.index_code_status(job_id="j1"))
    assert "j1" in one and "50/100" in one and "50%" in one
    many = asyncio.run(mod.index_code_status())
    assert "j1" in many and "completed" in many


def test_index_code_status_refreshes_live_backend_progress_for_local_job():
    mod = _load()
    cap = _Capture({
        "/api/code/jobs/backend-job": {
            "job_id": "backend-job", "status": "running", "project": "p",
            "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
            "queue_position": 0, "indexed": 0,
        },
    })
    _wire(mod, cap)
    mod._CODE_PREPARATION_STATUS["local-job"] = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 0, "chunks_total": 1024, "progress": 0.0,
    }

    out = asyncio.run(mod.index_code_status(job_id="local-job"))

    assert "384/1024 chunks (38%)" in out
    assert mod._CODE_PREPARATION_STATUS["local-job"]["chunks_done"] == 384
    assert cap.gets == [("/api/code/jobs/backend-job", None)]


def test_index_code_status_list_merges_backend_progress_into_local_jobs():
    mod = _load()
    cap = _Capture({
        "/api/code/jobs": {"jobs": [{
            "job_id": "backend-job", "status": "running", "project": "p",
            "chunks_done": 640, "chunks_total": 1024, "progress": 0.625,
            "queue_position": 0, "indexed": 0,
        }]},
    })
    _wire(mod, cap)
    mod._CODE_PREPARATION_STATUS["local-job"] = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 0, "chunks_total": 1024, "progress": 0.0,
    }

    out = asyncio.run(mod.index_code_status())

    assert "640/1024 chunks (62%)" in out
    assert out.count("backend-job") == 1
    assert cap.gets == [("/api/code/jobs", {"limit": 100})]


def test_index_code_status_keeps_cached_local_progress_when_backend_job_is_lost():
    mod = _load()
    response = mod.httpx.Response(
        404, request=mod.httpx.Request("GET", "http://backend/api/code/jobs/backend-job"))

    async def missing_job(_job_id):
        raise mod.httpx.HTTPStatusError(
            "job not found", request=response.request, response=response)

    mod._get_code_index_receipt = missing_job
    mod._CODE_PREPARATION_STATUS["local-job"] = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
    }

    out = asyncio.run(mod.index_code_status(job_id="local-job"))

    assert "384/1024 chunks (38%)" in out
    assert "backend restarted" in out
    assert "showing cached local status" in out


def test_index_code_status_keeps_cached_progress_during_transient_backend_outage():
    mod = _load()
    request = mod.httpx.Request("GET", "http://backend/api/code/jobs/backend-job")

    async def unavailable_job(_job_id):
        raise mod.httpx.ConnectError("backend offline", request=request)

    mod._get_code_index_receipt = unavailable_job
    mod._CODE_PREPARATION_STATUS["local-job"] = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
    }

    out = asyncio.run(mod.index_code_status(job_id="local-job"))

    assert "384/1024 chunks (38%)" in out
    assert "progress refresh unavailable" in out
    assert "showing cached local status" in out


def test_index_code_status_keeps_cached_progress_during_retryable_gateway_error():
    mod = _load()
    response = mod.httpx.Response(
        503, request=mod.httpx.Request("GET", "http://backend/api/code/jobs/backend-job"))

    async def unavailable_job(_job_id):
        raise mod.httpx.HTTPStatusError(
            "service unavailable", request=response.request, response=response)

    mod._get_code_index_receipt = unavailable_job
    mod._CODE_PREPARATION_STATUS["local-job"] = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
    }

    out = asyncio.run(mod.index_code_status(job_id="local-job"))

    assert "384/1024 chunks (38%)" in out
    assert "progress refresh unavailable" in out
    assert "showing cached local status" in out


def test_index_code_status_list_keeps_cached_progress_during_backend_outage():
    mod = _load()
    request = mod.httpx.Request("GET", "http://backend/api/code/jobs")

    async def unavailable_jobs(path, params=None):
        assert path == "/api/code/jobs"
        raise mod.httpx.ConnectError("backend offline", request=request)

    mod._get = unavailable_jobs
    mod._CODE_PREPARATION_STATUS["local-job"] = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
    }

    out = asyncio.run(mod.index_code_status())

    assert "384/1024 chunks (38%)" in out
    assert "progress refresh unavailable" in out
    assert "showing cached local status" in out


def test_index_code_status_list_keeps_cached_progress_during_gateway_restart():
    mod = _load()
    response = mod.httpx.Response(
        503, request=mod.httpx.Request("GET", "http://backend/api/code/jobs"))

    async def unavailable_jobs(path, params=None):
        assert path == "/api/code/jobs"
        raise mod.httpx.HTTPStatusError(
            "service unavailable", request=response.request, response=response)

    mod._get = unavailable_jobs
    mod._CODE_PREPARATION_STATUS["local-job"] = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
    }

    out = asyncio.run(mod.index_code_status())

    assert "384/1024 chunks (38%)" in out
    assert "progress refresh unavailable" in out
    assert "showing cached local status" in out


def test_index_code_status_list_warns_when_active_backend_job_disappears():
    mod = _load()
    cap = _Capture({"/api/code/jobs": {"jobs": []}})
    _wire(mod, cap)
    mod._CODE_PREPARATION_STATUS["local-job"] = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
    }

    out = asyncio.run(mod.index_code_status())

    assert "384/1024 chunks (38%)" in out
    assert "backend may have restarted" in out
    assert "showing cached local status" in out


def test_backend_progress_refresh_cannot_regress_a_terminal_local_job():
    mod = _load()
    status = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "chunks_done": 384,
        "chunks_total": 1024, "progress": 0.375,
    }

    async def stale_running(_job_id):
        status.update({"status": "completed", "chunks_done": 1024, "progress": 1.0})
        return {
            "job_id": "backend-job", "status": "running", "project": "p",
            "chunks_done": 512, "chunks_total": 1024, "progress": 0.5,
        }

    mod._get_code_index_receipt = stale_running

    asyncio.run(mod._refresh_preparation_from_backend(status))

    assert status["status"] == "completed"
    assert status["chunks_done"] == 1024
    assert status["progress"] == 1.0


def test_backend_progress_refresh_cannot_regress_running_progress():
    mod = _load()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    call_count = 0
    status = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "chunks_done": 128,
        "chunks_total": 1024, "progress": 0.125,
    }

    async def out_of_order(_job_id):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_started.set()
            await release_first.wait()
            done = 384
        else:
            done = 640
        return {
            "job_id": "backend-job", "status": "running", "project": "p",
            "chunks_done": done, "chunks_total": 1024, "progress": done / 1024,
        }

    mod._get_code_index_receipt = out_of_order

    async def scenario():
        first = asyncio.create_task(mod._refresh_preparation_from_backend(status))
        await first_started.wait()
        await mod._refresh_preparation_from_backend(status)
        release_first.set()
        await first

    asyncio.run(scenario())

    assert status["status"] == "running"
    assert status["chunks_done"] == 640
    assert status["progress"] == 0.625


def test_wait_code_job_retries_transient_network_and_gateway_failures():
    mod = _load()
    request = mod.httpx.Request("GET", "http://backend/api/code/jobs/backend-job")
    response = mod.httpx.Response(503, request=request)
    outcomes = [
        mod.httpx.ConnectError("backend offline", request=request),
        mod.httpx.HTTPStatusError(
            "service unavailable", request=request, response=response),
        {"job_id": "backend-job", "status": "completed", "progress": 1.0},
    ]

    async def get_job(path, params=None):
        assert path == "/api/code/jobs/backend-job"
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    mod._get = get_job
    mod._CODE_JOB_POLL_SECONDS = 0.0
    mod._CODE_JOB_RETRY_MAX_SECONDS = 0.0

    job = asyncio.run(mod._wait_code_job("backend-job"))

    assert job["status"] == "completed"
    assert outcomes == []


def test_search_code_sends_project_and_formats_full_code():
    mod = _load()
    body = "\n".join(f"    step_{i}()" for i in range(30))
    code = f"def foo():\n{body}\n    return 1"
    cap = _Capture({"/api/code/search": {"hits": [
        {"path": "core/x.py", "start_line": 10, "end_line": 42, "label": "foo",
         "language": "python", "code": code, "score": 0.87},
    ]}})
    _wire(mod, cap)
    out = asyncio.run(mod.search_code(query="what does foo do", top_k=3))
    path, payload = cap.posts[0]
    assert path == "/api/code/search"
    # scope is FORCED to the cwd abspath — the same path index_code(repo_root=".") wrote.
    assert payload == {"query": "what does foo do", "top_k": 3, "project": mod._scope_path("."),
                       "mode": "hybrid", "rerank": False}
    assert "core/x.py:10-42" in out and "[foo]" in out and "0.870" in out
    # Wave-1 #2: full fenced chunk, indentation preserved, not a 200-char preview.
    assert "```python" in out and "    step_29()" in out


def test_plugin_search_code_recovery_required_guides_full_reindex_without_old_poll(tmp_path):
    mod = _load()
    cap = _Capture({"/api/code/search": {
        "hits": [], "total": 0,
        "scope_snapshot_state": "recovery_required",
        "recovery_required": True,
        "active_index_job_id": "failed-write-job",
        "notice": "Full recovery index required after interrupted write.",
    }})
    _wire(mod, cap)

    out = asyncio.run(mod.search_code(query="SendBPS", repo_root=str(tmp_path)))

    assert "full recovery index" in out.lower()
    assert "index_code(repo_root=" in out
    assert "Poll index_code_status" not in out
    assert "failed-write-job" not in out
    assert "No matching code" not in out


def test_plugin_search_code_catalog_unavailable_guides_retry_without_poll(tmp_path):
    mod = _load()
    cap = _Capture({"/api/code/search": {
        "hits": [], "total": 0,
        "scope_snapshot_state": "unavailable",
        "recovery_required": False,
        "notice": "Index Catalog visibility is unavailable; retry shortly.",
    }})
    _wire(mod, cap)

    out = asyncio.run(mod.search_code(query="SendBPS", repo_root=str(tmp_path)))

    assert "catalog" in out.lower()
    assert "retry" in out.lower()
    assert "Poll index_code_status" not in out
    assert "No matching code" not in out


def test_plugin_format_code_hits_caps_giant_chunk():
    mod = _load()
    giant = "y" * (mod._CODE_HIT_CODE_CHARS + 3000)
    text = mod._fmt_code_hits(
        [{"path": "huge.html", "start_line": 5, "end_line": 5, "label": "",
          "score": 0.4, "language": "html", "code": giant}],
        "Code matches for: 'q'",
    )
    assert len(text) < len(giant)
    assert "truncated" in text.lower()
    assert "huge.html:5" in text


def test_plugin_search_code_times_out_instead_of_hanging():
    mod = _load()
    _wire(mod, _Capture())

    async def slow_post(path, payload):
        await asyncio.sleep(5)
        return {"hits": []}

    mod._post = slow_post
    mod._SEARCH_CODE_TIMEOUT_S = 0.05
    out = asyncio.run(mod.search_code(query="q", top_k=2))
    assert "timed out" in out.lower()


def test_plugin_search_papers_times_out_instead_of_hanging():
    """Interactive paper search must fail-fast; concurrent host tools previously
    left one search_papers call without a completed result for minutes."""
    mod = _load()
    _wire(mod, _Capture())

    async def slow_post(path, payload):
        await asyncio.sleep(5)
        return {"papers": [], "results": []}

    mod._post = slow_post
    mod._SEARCH_PAPERS_TIMEOUT_S = 0.05
    out = asyncio.run(mod.search_papers(
        query="masked diffusion language model unmasking order", top_k=20))
    assert "timed out" in out.lower()
    assert "0.05" in out or "seconds" in out.lower()


def test_plugin_http_timeout_is_layered():
    mod = _load()
    timeout = mod._http_timeout()
    assert isinstance(timeout, mod.httpx.Timeout)
    assert timeout.connect <= 10.0
    assert timeout.pool <= 10.0
    assert timeout.read >= 30.0


def test_search_code_forwards_filter_params():
    # Wave-1 #3/#6: language / path_glob / max_per_file reach the REST payload.
    mod = _load()
    cap = _Capture({"/api/code/search": {"hits": []}})
    _wire(mod, cap)
    asyncio.run(mod.search_code(query="q", top_k=4, mode="keyword",
                                rerank=True, language="rust", path_glob="src/peers/",
                                max_per_file=2))
    _, payload = cap.posts[0]
    assert payload["mode"] == "keyword"
    assert payload["rerank"] is True
    assert payload["language"] == "rust"
    assert payload["path_glob"] == "src/peers/"
    assert payload["max_per_file"] == 2


def test_search_code_empty_gives_detailed_scope_and_remote_machine_guidance(monkeypatch):
    mod = _load()
    remote_scope = r"e:\JETB\Pythons\Playground\Octopus\AxonHub\LLM"
    watcher_calls = []
    monkeypatch.setattr(mod, "_local_repo_root", lambda _path: None)
    monkeypatch.setattr(
        mod, "_ensure_code_watcher", lambda **kwargs: watcher_calls.append(kwargs))
    _wire(mod, _Capture({"/api/code/search": {
        "hits": [],
        "requested_mode": "keyword",
        "effective_mode": "unavailable",
        "notice": "keyword search is unavailable because sparse lexical retrieval is unavailable; it does not substitute dense results.",
        "path_glob_matched_files": 0,
        "indexed_file_count": 583,
    }}))
    out = asyncio.run(mod.search_code(
        query="anything", repo_root=remote_scope, mode="keyword", path_glob="conversion\\"))
    scope = "E:/jetb/pythons/playground/octopus/axonhub/llm"

    assert f"scope {scope!r}" in out
    assert "does not prove that repo_root is wrong" in out
    assert "repo_root may be missing or wrong" not in out
    assert f"search_code(query='anything', repo_root={scope!r})" in out
    assert "index_inventory()" in out
    assert f"index_code(repo_root={scope!r})" in out
    assert "index_code_status" in out
    assert "scope-relative" in out
    assert "another machine" in out
    assert "does not substitute dense" in out
    assert watcher_calls == []


def test_search_code_rejects_absolute_windows_path_glob_before_http():
    mod = _load()
    cap = _Capture()
    _wire(mod, cap)

    with pytest.raises(ValueError, match="scope-relative"):
        asyncio.run(mod.search_code(
            query="anything", repo_root=r"E:\JETB\repo",
            path_glob=r"E:\JETB\repo\conversion"))

    assert cap.posts == []


def test_delete_code_forces_abspath_and_issues_delete(tmp_path):
    # W165 (plugin mirror): delete_code forces the scope to the dir's abspath and DELETEs
    # /api/code?project=<abspath> — clearing both vectors and inventory rows.
    mod = _load()
    cap = _Capture({"/api/code": {"project": "p", "deleted": 3, "catalog_removed": 1}})
    _wire(mod, cap)
    out = asyncio.run(mod.delete_code(repo_root=str(tmp_path)))
    assert cap.deletes, "delete_code must issue a DELETE"
    path, params = cap.deletes[0]
    assert path == "/api/code"
    assert params == {"project": mod._scope_path(str(tmp_path))}
    assert "Deleted" in out and "3" in out and "1" in out


def test_delete_code_reports_legacy_windows_scopes_deleted_together(tmp_path):
    mod = _load()
    legacy = "E:/JETB/Pythons/Playground/Octopus/AxonHub/LLM"
    cap = _Capture({"/api/code": {
        "project": "E:/jetb/pythons/playground/octopus/axonhub/llm",
        "deleted": 3,
        "catalog_removed": 1,
        "scope_aliases": [legacy],
    }})
    _wire(mod, cap)

    out = asyncio.run(mod.delete_code(repo_root=str(tmp_path)))

    assert "Included legacy Windows scope spelling(s)" in out
    assert legacy in out


def test_delete_journal_docs_forces_abspath_and_issues_delete(tmp_path):
    # doc-scope clear (plugin mirror), symmetric to delete_code: DELETE /api/journal/docs
    # ?project_slug=<abspath> clears doc entries+vectors + inventory rows, keeps memories.
    mod = _load()
    cap = _Capture({"/api/journal/docs": {"removed_entries": 3, "deleted_vectors": 9,
                                          "catalog_removed": 2}})
    _wire(mod, cap)
    out = asyncio.run(mod.delete_journal_docs(dir=str(tmp_path)))
    assert cap.deletes, "delete_journal_docs must issue a DELETE"
    path, params = cap.deletes[0]
    assert path == "/api/journal/docs"
    assert params == {"project_slug": mod._scope_path(str(tmp_path))}
    assert "Deleted doc scope" in out and "3" in out and "kept" in out.lower()


def test_delete_journal_docs_reports_nothing_when_already_clear(tmp_path):
    mod = _load()
    cap = _Capture({"/api/journal/docs": {"removed_entries": 0, "deleted_vectors": 0,
                                          "catalog_removed": 0}})
    _wire(mod, cap)
    out = asyncio.run(mod.delete_journal_docs(dir=str(tmp_path)))
    assert "Nothing to delete" in out


def test_index_code_full_zero_files_hints_delete(tmp_path, monkeypatch):
    # W165 (plugin mirror): a full reindex yielding 0 files keeps the existing index and
    # hints delete_code — it never submits a pointless wipe job.
    mod = _load()
    cap = _Capture()
    _wire(mod, cap)
    monkeypatch.setattr(mod, "_collect_code_files", lambda root, *_a, **_k: [])
    out = _index_code_to_completion(mod, repo_root=str(tmp_path))
    assert not any(p == "/api/code/index" for p, _ in cap.posts)  # no job submitted
    assert "delete_code" in out and "kept" in out.lower()


# --------------------------------------------------------------------------- #
# ACE playbook — drop-in ace-mcp signatures over /api/playbook/*
# --------------------------------------------------------------------------- #

def test_ace_smart_reflect_records_bullet():
    mod = _load()
    cap = _Capture({"/api/playbook/bullets": {
        "id": "b1", "section": "insights", "content": "wait=True on upsert",
        "helpful": 0, "harmful": 0, "deduped": False}})
    _wire(mod, cap)
    out = asyncio.run(mod.ace_smart_reflect(code="wait=True on upsert", section="insights"))
    path, payload = cap.posts[0]
    assert path == "/api/playbook/bullets"
    assert payload["content"] == "wait=True on upsert" and payload["section"] == "insights"
    assert "added new" in out and "wait=True on upsert" in out


def test_ace_smart_reflect_reports_reinforced_on_dedupe():
    mod = _load()
    _wire(mod, _Capture({"/api/playbook/bullets": {
        "id": "b1", "section": "insights", "content": "x", "deduped": True}}))
    out = asyncio.run(mod.ace_smart_reflect(code="x"))
    assert "reinforced existing" in out


def test_ace_smart_reflect_rejects_empty():
    mod = _load()
    out = asyncio.run(mod.ace_smart_reflect(code="   "))
    assert "Nothing to reflect" in out


def test_ace_context_aware_searches_and_formats():
    mod = _load()
    cap = _Capture({"/api/playbook/search": {"bullets": [
        {"section": "patterns", "domain": "qdrant", "content": "use RRF",
         "helpful": 2, "harmful": 0, "score": 0.5}]}})
    _wire(mod, cap)
    out = asyncio.run(mod.ace_context_aware(query="how to fuse", domain="qdrant", top_k=5))
    path, payload = cap.posts[0]
    assert path == "/api/playbook/search"
    assert payload["query"] == "how to fuse" and payload["domain"] == "qdrant"
    assert "[patterns][qdrant] use RRF" in out and "sim=0.500" in out


def test_ace_context_aware_empty_hints_reflect():
    mod = _load()
    _wire(mod, _Capture({"/api/playbook/search": {"bullets": []}}))
    out = asyncio.run(mod.ace_context_aware(query="nothing"))
    assert "record lessons with ace_smart_reflect" in out


def test_ace_playbook_lists_via_get_with_count():
    mod = _load()
    cap = _Capture({"/api/playbook/bullets": {
        "bullets": [{"section": "insights", "content": "a"},
                    {"section": "patterns", "content": "b"}], "total": 2}})
    _wire(mod, cap)
    out = asyncio.run(mod.ace_playbook(limit=50))
    assert cap.gets and cap.gets[0][0] == "/api/playbook/bullets"
    assert "Playbook (2 bullets)" in out


def test_ace_enhance_prompt_appends_guidelines():
    mod = _load()
    _wire(mod, _Capture({"/api/playbook/search": {"bullets": [
        {"section": "insights", "content": "always retry upserts"}]}}))
    out = asyncio.run(mod.ace_enhance_prompt(prompt="do the thing"))
    assert out.startswith("do the thing")
    assert "Playbook guidelines" in out and "always retry upserts" in out


def test_ace_enhance_prompt_passthrough_when_empty():
    mod = _load()
    _wire(mod, _Capture({"/api/playbook/search": {"bullets": []}}))
    out = asyncio.run(mod.ace_enhance_prompt(prompt="untouched"))
    assert out == "untouched"


def test_ace_smart_generate_returns_guidance_and_enhanced_prompt():
    mod = _load()
    _wire(mod, _Capture({"/api/playbook/search": {"bullets": [
        {"section": "insights", "content": "prefer library over hand-roll"}]}}))
    out = asyncio.run(mod.ace_smart_generate(prompt="write a parser"))
    assert "Apply these learned playbook bullets" in out
    assert "enhanced prompt for your model" in out
    assert "prefer library over hand-roll" in out


# --------------------------- local chunking parallelism (async-ification) --------------------------- #

def test_collect_code_files_chunks_files_concurrently(tmp_path, monkeypatch):
    """Plugin mirror: local AST chunking runs files in parallel, not serially."""
    import threading
    import time
    mod = _load()
    for i in range(4):
        (tmp_path / f"f{i}.py").write_text(f"def f{i}(x):\n    return x + {i}\n", encoding="utf-8")
    monkeypatch.setattr(mod, "_CHUNK_WORKERS", 4)
    state = {"cur": 0, "max": 0}
    lock = threading.Lock()
    real = mod._chunk_file_payload

    def probe(path, rel, repo_root, content_hash=None):
        with lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.1)
        try:
            return real(path, rel, repo_root, content_hash)
        finally:
            with lock:
                state["cur"] -= 1

    monkeypatch.setattr(mod, "_chunk_file_payload", probe)
    files = mod._collect_code_files(str(tmp_path))
    assert len(files) == 4
    assert state["max"] >= 2, f"expected concurrent chunking, saw peak {state['max']}"


def test_journal_index_dir_posts_entries_concurrently(tmp_path, monkeypatch):
    """journal_index_dir must POST entries concurrently (bounded), not one-at-a-time."""
    import asyncio
    mod = _load()
    for i in range(4):
        (tmp_path / f"n{i}.md").write_text(f"# doc {i}\n\nbody {i}\n", encoding="utf-8")
    monkeypatch.setattr(mod, "_JOURNAL_INDEX_CONCURRENCY", 4)
    state = {"cur": 0, "max": 0}

    async def fake_post(path, payload):
        if path == "/api/journal/entries":
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
            await asyncio.sleep(0.05)
            state["cur"] -= 1
            return {"entry_id": "e", "chunk_count": 1, "deduped": False}
        return {"total_files": 0, "removed_entries": 0}  # report + reconcile

    async def fake_get(path, params=None):
        return {}  # nested-scope guard: empty inventory → no block

    monkeypatch.setattr(mod, "_post", fake_post)
    monkeypatch.setattr(mod, "_get", fake_get)
    out = asyncio.run(mod.journal_index_dir(dir=str(tmp_path), pattern="*.md"))
    assert "Indexed 4 file(s)" in out
    assert state["max"] >= 2, f"entries must POST concurrently, peak {state['max']}"


def test_journal_index_dir_reports_cross_scope_synced(tmp_path, monkeypatch):
    """方案2: journal_index_dir sums duplicates_updated from each entry POST and reports it."""
    mod = _load()
    for i in range(2):
        (tmp_path / f"n{i}.md").write_text(f"# doc {i}\n\nbody {i}\n", encoding="utf-8")

    async def fake_post(path, payload):
        if path == "/api/journal/entries":
            return {"entry_id": "e", "chunk_count": 1, "deduped": False,
                    "duplicates_updated": 1}  # each file synced 1 other-scope copy
        return {"total_files": 0, "removed_entries": 0}

    async def fake_get(path, params=None):
        return {}

    monkeypatch.setattr(mod, "_post", fake_post)
    monkeypatch.setattr(mod, "_get", fake_get)
    out = asyncio.run(mod.journal_index_dir(dir=str(tmp_path), pattern="*.md"))
    assert "2 cross-scope synced" in out


def test_fmt_job_line_surfaces_cross_scope_synced():
    """index_code_status must show how many other-scope copies a completed job synced."""
    mod = _load()
    line = mod._fmt_job_line({
        "job_id": "abc", "status": "completed", "project": "/p",
        "chunks_done": 10, "chunks_total": 10, "progress": 1.0,
        "indexed": 10, "duplicates_updated": 3,
    })
    assert "3 cross-scope" in line
    # No dup → no note.
    quiet = mod._fmt_job_line({
        "job_id": "abc", "status": "completed", "project": "/p",
        "chunks_done": 10, "chunks_total": 10, "progress": 1.0,
        "indexed": 10, "duplicates_updated": 0,
    })
    assert "cross-scope" not in quiet
def test_job_line_describes_delete_only_refresh():
    mod = _load()
    line = mod._fmt_job_line({
        "job_id": "delete-job", "status": "completed", "project": "C:/repo",
        "chunks_done": 0, "chunks_total": 0, "progress": 1.0,
        "indexed": 0, "removed_total": 1,
    })

    assert "removed 1 path" in line


class TestIndexingAGitignoredSubtreeUnderAnIndexedParent:
    """The guard must not refuse a child that reaches what the parent's filter skipped.

    Driven end to end: the real FastAPI route, the real IndexCatalogService and a real
    SQLite database, with only the network hop replaced by an in-process ASGI transport.
    The query params the guard sends, the route's parsing of them, the overlap
    computation, the ancestor policy lookup and the message formatting are all the
    shipping code paths.
    """

    @staticmethod
    def _kb(files, *, parent_respects_gitignore=True, record_policy=True,
            child_files=None):
        """A catalog holding one parent scope that indexed ``files``.

        ``child_files`` additionally records the /repo/erp/data scope as already
        indexed, which is what every refresh of that scope sees.
        """
        from contextlib import contextmanager

        from fastapi import FastAPI
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from app.api.routes import index_catalog as ic_routes
        from app.services.index_catalog.repository import IndexCatalogRepository
        from app.services.index_catalog.schemas import (
            IndexFileRecord, IndexReportRequest, IndexScopePolicyUpsert,
        )
        from app.services.index_catalog.service import IndexCatalogService
        from core.db.database import Base

        engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                               poolclass=StaticPool)
        Base.metadata.create_all(engine)
        session_local = sessionmaker(bind=engine, expire_on_commit=False)

        @contextmanager
        def session_factory():
            session = session_local()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        svc = IndexCatalogService(
            repository=IndexCatalogRepository(session_factory=session_factory))
        svc.report(IndexReportRequest(
            host="h1", project="/repo/erp", root_dir="/repo/erp", replace=True,
            files=[IndexFileRecord(rel_path=p, kind="code", collection="code_chunks",
                                   chunk_count=3) for p in files],
        ), owner_user_id="u1")
        if record_policy:
            svc.upsert_policy(IndexScopePolicyUpsert(
                host="h1", project="/repo/erp", root_dir="/repo/erp", kind="code",
                respect_gitignore=parent_respects_gitignore), owner_user_id="u1")
        if child_files:
            svc.report(IndexReportRequest(
                host="h1", project="/repo/erp/data", root_dir="/repo/erp/data",
                replace=True,
                files=[IndexFileRecord(rel_path=p, kind="code",
                                       collection="code_chunks", chunk_count=3)
                       for p in child_files],
            ), owner_user_id="u1")

        ic_routes.get_index_catalog_service = lambda: svc
        app = FastAPI()
        app.include_router(ic_routes.router, prefix="/api/index")
        return app

    @staticmethod
    def _guard(mod, app, *, respect_gitignore):
        """Run the real guard against the real route over ASGI."""
        import httpx

        async def scenario():
            async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://kb",
                    headers={"X-Retrival-User-Id": "u1"}) as client:

                async def _get(path, params=None, *, headers=None):
                    response = await client.get(path, params=params or {}, headers=headers)
                    response.raise_for_status()
                    return response.json()

                mod._get = _get  # type: ignore[assignment]
                return await mod._nested_scope_guard(
                    "/repo/erp/data", "h1", "code", respect_gitignore)

        return asyncio.run(scenario())

    def test_a_wholly_ignored_subtree_indexes_without_a_word(self):
        # The parent skipped every file under data/, so it holds no coverage there.
        mod = _load()
        app = self._kb(["src/main.py", "README.md"])
        block, warning = self._guard(mod, app, respect_gitignore=False)
        assert block is None
        assert warning == ""

    def test_a_partly_ignored_subtree_indexes_with_a_warning_naming_the_overlap(self):
        # data/README.md is tracked (the parent has it) and data/corpus.db is ignored
        # (the parent does not). Refusing here would leave "delete the parent scope
        # first" as the only route to those files.
        mod = _load()
        app = self._kb(["src/main.py", "data/README.md"])
        block, warning = self._guard(mod, app, respect_gitignore=False)
        assert block is None
        assert "/repo/erp" in warning
        assert "respect_gitignore=False" in warning
        assert "both code scopes" in warning

    def test_keeping_the_filter_on_is_still_refused(self):
        # Nothing new would be reached, so the child really is redundant.
        mod = _load()
        app = self._kb(["src/main.py", "data/README.md"])
        block, warning = self._guard(mod, app, respect_gitignore=True)
        assert block is not None
        assert "already covered by ancestor scope '/repo/erp'" in block
        assert warning == ""

    def test_a_parent_that_itself_ignored_nothing_still_refuses(self):
        # The parent already holds the ignored files, so widening adds nothing.
        mod = _load()
        app = self._kb(["src/main.py", "data/README.md"],
                       parent_respects_gitignore=False)
        block, _ = self._guard(mod, app, respect_gitignore=False)
        assert block is not None

    def test_a_parent_indexed_before_policies_were_recorded_is_read_as_filtered(self):
        # No policy row exists for scopes indexed before policies were stored. The
        # indexer default was respect_gitignore=True, which is what actually ran.
        mod = _load()
        app = self._kb(["src/main.py", "data/README.md"], record_policy=False)
        block, warning = self._guard(mod, app, respect_gitignore=False)
        assert block is None
        assert "/repo/erp" in warning

    def test_the_child_scope_stays_refreshable_once_it_exists(self):
        """The follow-on run, which is where the guard used to strand the scope.

        Having been allowed to create /repo/erp/data, a plain index_code() re-run
        sends the default respect_gitignore=True and would be refused by the
        coverage test — the scope could never be updated again, and an already
        published client, which sends no filter at all, could never update it
        either. Refusing a scope that exists cannot un-create it.
        """
        mod = _load()
        app = self._kb(["src/main.py", "data/README.md"],
                       child_files=["README.md", "corpus.db"])

        refreshed_with_filter = self._guard(mod, app, respect_gitignore=True)
        assert refreshed_with_filter == (None, "")

        # And the widening warning must not repeat on every auto-refresh tick,
        # which replays the scope's stored respect_gitignore=False forever.
        assert self._guard(mod, app, respect_gitignore=False) == (None, "")


@pytest.mark.parametrize("arguments, message", [
    ({"sources": "everything"}, "search_code.sources must be one of private, public, all"),
    ({"source_ids": ["pion"]}, "source_ids/domains filter the public corpus"),
    ({"domains": ["networking"]}, "source_ids/domains filter the public corpus"),
    ({"sources": "public", "path_glob": "src/"}, "path_glob/max_per_file filter your private code"),
    ({"sources": "public", "max_per_file": 2}, "path_glob/max_per_file filter your private code"),
])
def test_merged_search_code_rejects_contradictory_arguments(arguments, message):
    """search_code absorbed search_all_code and search_public_code (0.5.0); arguments
    that only make sense for the other side fail explicitly instead of being ignored."""
    mod = _load()
    with pytest.raises(ValueError, match=message):
        asyncio.run(mod.search_code(query="x", **arguments))
