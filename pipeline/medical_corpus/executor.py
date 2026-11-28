"""Where medical PDF bytes may be fetched, checked before the first request.

The Task service runs only on compute .88. The standalone CLI runs there too, or on
a GitHub Actions runner that declares itself (since 2026-11-12 the bulk downloads
run on runners instead of .88's proxy). This module sits below the service so the
CLI can apply the same gate without importing the Task machinery.
"""
from __future__ import annotations

import os
from pathlib import Path

from pipeline.medical_corpus.prepare import require_absolute_medical_data_root
from pipeline.medical_corpus.proxy import DEFAULT_PROXY_URL, MedicalProxyConfig

MEDICAL_PRODUCTION_DATA_ROOT = Path("/mnt/lsb/retrival/medical")
MEDICAL_ACTIONS_EXECUTOR = "github-actions"


def designated_medical_executor(
    env=None, os_name: str | None = None, *, allow_actions: bool = True,
) -> str:
    """Which executor may fetch medical PDFs here: "compute88" or "github-actions".

    A runner must both be one (GITHUB_ACTIONS=true, set by GitHub) and opt in
    (MEDICAL_DOWNLOAD_HOST=github-actions).
    """
    env = os.environ if env is None else env
    if (os.name if os_name is None else os_name) == "nt":
        raise RuntimeError("medical PDF execution is disabled on Windows; use compute .88")
    host = env.get("MEDICAL_DOWNLOAD_HOST", "").strip()
    if host == "10.100.100.88":
        return "compute88"
    if allow_actions and host == MEDICAL_ACTIONS_EXECUTOR and env.get("GITHUB_ACTIONS") == "true":
        return MEDICAL_ACTIONS_EXECUTOR
    raise RuntimeError(
        "medical PDF execution is disabled outside compute .88; set "
        "MEDICAL_DOWNLOAD_HOST=10.100.100.88 only in its service environment"
    )


def require_designated_medical_download_host() -> None:
    """The only Task-service executor is the Linux Retrieval service on .88."""
    designated_medical_executor(allow_actions=False)


def require_designated_medical_cli_execution(
    data_root: Path,
    proxy: MedicalProxyConfig,
) -> Path:
    """Fail closed before the standalone CLI can fetch a single PDF byte.

    The service runner allows explicitly injected temporary roots for isolated
    tests.  The human-invoked executable is stricter: it has no task-owned
    deployment configuration, so on .88 real transfers may use only the artifact
    mount, and everywhere only the local proxy (a GitHub runner forwards
    127.0.0.1:10808 to its own exit). A runner's root is its own work disk, whose
    files then go to the shared volume, never to .88's mount directly.
    """
    executor = designated_medical_executor()
    root = require_absolute_medical_data_root(data_root)
    if executor == "compute88" and root != MEDICAL_PRODUCTION_DATA_ROOT:
        raise RuntimeError(
            "medical CLI execution requires MEDICAL_DATA_ROOT=/mnt/lsb/retrival/medical"
        )
    if proxy.http_proxy != DEFAULT_PROXY_URL or proxy.https_proxy != DEFAULT_PROXY_URL:
        raise RuntimeError(
            f"medical CLI execution requires proxy {DEFAULT_PROXY_URL}"
        )
    return root
