"""Real-network acceptance for the two-hop conference PDF downloader.

No origin, proxy, HTTP client, PDF bytes, or downloader is faked here.  The
test is intentionally environment-bound: it runs only where the operator's
runtime proxy pool and the local Xray first hop are present.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import aiohttp
import pytest
from aiohttp_socks import ProxyConnector

from pipeline.download_control import ChainedProxyPool
from pipeline.paper_crawler.ccf_a_main_corpus import (
    CorpusPaper,
    MainConferenceCorpusBuilder,
)
from pipeline.paper_crawler.runtime_files import is_valid_pdf_artifact


REAL_ACL_PDF = "https://aclanthology.org/2025.acl-long.1.pdf"


def _runtime_pool_path() -> Path:
    configured = os.environ.get("RETRIVAL_PROXY_POOL_FILE")
    return Path(configured) if configured else Path.home() / "Desktop" / "a.txt"


def _runtime_xray_path() -> Path:
    configured = os.environ.get("RETRIVAL_XRAY_EXECUTABLE")
    if configured:
        return Path(configured)
    discovered = shutil.which("xray")
    if discovered:
        return Path(discovered)
    return Path(r"C:\Arc\zz_v2rayN-With-Core-SelfContained\bin\xray\xray.exe")


async def _egress_identity(proxy_url: str) -> str:
    connector = ProxyConnector.from_url(proxy_url)
    timeout = aiohttp.ClientTimeout(total=20, sock_connect=8, sock_read=8)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        trust_env=False,
    ) as session:
        async with session.get("https://api.ipify.org") as response:
            response.raise_for_status()
            value = (await response.text()).strip()
    assert value
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def test_real_two_hop_pool_downloads_and_validates_an_acl_pdf() -> None:
    pool_path = _runtime_pool_path()
    if not pool_path.is_file():
        pytest.skip(f"real proxy pool is not present: {pool_path}")

    first_hop = os.environ.get(
        "RETRIVAL_FIRST_HOP_PROXY", "socks5://127.0.0.1:10808",
    )
    pool = ChainedProxyPool.from_file(
        pool_path,
        first_hop_proxy=first_hop,
    )

    async def run_smoke(corpus_root: Path) -> tuple[int, Path]:
        probe = await pool.probe_pdf(
            REAL_ACL_PDF,
            required_healthy=1,
            max_candidates=48,
            concurrency=6,
        )
        assert probe.healthy_count >= 1

        paper = CorpusPaper(
            paper_id="acl-real-two-hop-smoke",
            title="Real two-hop download smoke",
            authors=[],
            conference="ACL",
            year=2025,
            pdf_url=REAL_ACL_PDF,
            paper_page_url="https://aclanthology.org/2025.acl-long.1/",
            source_name="acl-anthology-real-smoke",
        )
        builder = MainConferenceCorpusBuilder(
            corpus_root=corpus_root,
            proxy_url=first_hop,
            years=(2025,),
            conferences=(),
        )
        completed = await builder.download_papers(
            [paper],
            max_concurrent=1,
            max_bytes_per_second=0,
            chained_proxy_pool=pool,
            proxy_attempts_per_url=24,
        )
        artifact = Path(completed[0].pdf_path or "")
        assert completed[0].download_error is None
        assert is_valid_pdf_artifact(artifact)
        return artifact.stat().st_size, artifact

    with tempfile.TemporaryDirectory(prefix="retrival-real-chain-") as temp_dir:
        size, artifact = asyncio.run(run_smoke(Path(temp_dir) / "corpus"))
        assert size > 100_000
        assert artifact.is_file()

    assert not artifact.exists()


def test_real_xray_browser_bridge_uses_a_landing_exit_after_the_first_hop() -> None:
    pool_path = _runtime_pool_path()
    xray_path = _runtime_xray_path()
    if not pool_path.is_file():
        pytest.skip(f"real proxy pool is not present: {pool_path}")
    if not xray_path.is_file():
        pytest.skip(f"real Xray executable is not present: {xray_path}")

    first_hop = os.environ.get(
        "RETRIVAL_FIRST_HOP_PROXY", "socks5://127.0.0.1:10808",
    )
    pool = ChainedProxyPool.from_file(pool_path, first_hop_proxy=first_hop)

    async def run_smoke() -> None:
        probe = await pool.probe_pdf(
            REAL_ACL_PDF,
            required_healthy=1,
            max_candidates=48,
            concurrency=6,
        )
        assert probe.healthy_count >= 1
        first_hop_identity = await _egress_identity(first_hop)
        async with pool.open_browser_proxy(
            xray_executable=xray_path,
        ) as lease:
            assert lease.proxy_url.startswith("socks5://127.0.0.1:")
            landing_identity = await _egress_identity(lease.proxy_url)
        assert landing_identity != first_hop_identity

    asyncio.run(run_smoke())
