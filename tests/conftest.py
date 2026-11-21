"""Global pytest config — applies before any test imports requests/httpx.

Reason: this dev machine runs an HTTP proxy on 127.0.0.1:10808 and exports
HTTP_PROXY/HTTPS_PROXY in the shell env. Without NO_PROXY=localhost, tests that
hit the FastAPI server at http://127.0.0.1:8000 get silently routed through the
proxy, occasionally timing out at the proxy layer (not at FastAPI).

This file ensures every test process sees NO_PROXY set, regardless of which
test file runs first.
"""
from __future__ import annotations

import os
import socket
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pytest


# The backend's start-up waits for the embedding service for up to three minutes
# (settings.embedding.service_wait_s); a test that enters main.app's lifespan must not, whatever runs on 18200.
os.environ.setdefault("EMBEDDING__SERVICE_WAIT_S", "0")


def pytest_configure(config) -> None:
    """Avoid pytest's broken global `pytest-current` reparse point on this Windows host.

    A per-process fixed base also keeps concurrent test runs isolated and, unlike a
    repo-local temp directory, does not make gitignore-aware index tests hide fixtures.
    """
    if config.option.basetemp is None:
        config.option.basetemp = str(
            Path(tempfile.gettempdir()) / f"retrival-pytest-{os.getpid()}")


def git_init_repo(path) -> None:
    """Init a throwaway git repo at `path` (no commit — `git ls-files` reads the worktree
    + index). Shared by the gitignore-filter tests across the code-KB test modules."""
    subprocess.run(["git", "init", "-q", str(path)], check=True,
                   stdin=subprocess.DEVNULL, capture_output=True)


def case_sensitive_markdown_glob(root):
    """Fake Linux-style Markdown glob results for journal_index_dir regression tests."""
    by_pattern = {
        "*.md": ["README.md"],
        "*.MD": ["ENV.MD"],
        "*.Md": ["Plan.Md"],
        "*.mD": [],
    }

    def glob(pattern, recursive=False):
        suffix = str(pattern).replace("\\", "/").rsplit("/", 1)[-1]
        return [str(root / name) for name in by_pattern.get(suffix, [])]

    return glob


# Set BEFORE pytest collects tests, so requests/httpx pick it up at first import.
_existing = os.environ.get("NO_PROXY", "")
_required = {"localhost", "127.0.0.1", "::1"}
_current = {p.strip() for p in _existing.split(",") if p.strip()}
if not _required.issubset(_current):
    os.environ["NO_PROXY"] = ",".join(sorted(_current | _required))
    os.environ["no_proxy"] = os.environ["NO_PROXY"]  # lowercase for libcurl-style libs

# Load embedder/reranker/tokenizer weights from the local HF cache instead of
# hitting the network for an update check. The BGE-M3 + reranker snapshots are
# cached on dev machines; without this, from_pretrained() blocks on a network
# call and stalls the whole suite when the machine is offline/proxied.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _server_up(url: str) -> bool:
    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


# Integration tests that drive a REAL GPU embed of real corpus papers (not the
# in-memory/hermetic ones). They need a free GPU + populated Qdrant, so they hang
# or fail on a plain dev box. Opt in with RUN_LIVE_TESTS=1 (e.g. on the 4090).
_LIVE_GPU_MODULES = {"test_batch_preindex"}
_RUN_LIVE = os.environ.get("RUN_LIVE_TESTS") == "1"


def pytest_collection_modifyitems(config, items):
    """Skip (don't fail) live-infra tests when the infra isn't available.

    Two cases: (1) live-server tests declaring a module-level HTTP base
    (``API`` / ``BASE_URL`` / ``BASE``, e.g. env ``RETRIVAL_API``) are skipped when
    that backend is unreachable; (2) real-GPU integration modules are skipped unless
    ``RUN_LIVE_TESTS=1``. Both keep ``pytest tests/`` green on a plain dev box while
    still running when pointed at live infra (e.g. the 4090)."""
    reachable: dict[str, bool] = {}
    for item in items:
        module = getattr(item, "module", None)
        mod_name = getattr(module, "__name__", "").rsplit(".", 1)[-1]
        if not _RUN_LIVE and mod_name in _LIVE_GPU_MODULES:
            item.add_marker(pytest.mark.skip(reason="real-GPU integration test (set RUN_LIVE_TESTS=1)"))
            continue
        base = None
        for attr in ("API", "BASE_URL", "BASE"):
            value = getattr(module, attr, None)
            if isinstance(value, str) and value.startswith("http"):
                base = value
                break
        if base is None:
            continue
        if base not in reachable:
            reachable[base] = _server_up(base)
        if not reachable[base]:
            item.add_marker(pytest.mark.skip(reason=f"live server {base} unreachable"))


@pytest.fixture(autouse=True)
def _fresh_stats_cache():
    """The statistics endpoints cache for minutes; no test may see another test's database."""
    from core.cache import stats_cache

    stats_cache.clear()
    yield
    stats_cache.clear()
