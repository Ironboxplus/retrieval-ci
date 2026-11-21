"""Shared transfer controls for proxy-routed corpus downloads.

The byte clock is shared by every worker in one batch, so ``concurrency`` only
controls how many files are in flight; it never multiplies the configured
aggregate bandwidth.  Both synchronous (medical) and asynchronous (conference)
downloaders use the same reservation algorithm.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from urllib.parse import unquote, urlsplit

import aiohttp
from aiohttp_socks import ChainProxyConnector

from pipeline.paper_crawler.runtime_files import remove_file_if_present


DEFAULT_DOWNLOAD_BYTES_PER_SECOND = 2_000_000
DEFAULT_STREAM_CHUNK_BYTES = 64 * 1024
_SUPPORTED_PROXY_SCHEMES = {"http", "https", "socks5", "socks5h"}
_SUPPORTED_CHAIN_EXIT_SCHEMES = {"socks5", "socks5h"}


def require_explicit_proxy(proxy_url: str | None) -> str:
    """Return a usable explicit proxy URL or fail before any network request."""
    value = (proxy_url or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in _SUPPORTED_PROXY_SCHEMES or not parsed.hostname:
        raise RuntimeError(
            "conference PDF downloads require an explicit HTTP proxy "
            "(HTTP/SOCKS URL); "
            "refusing direct or malformed proxy configuration"
        )
    return value


def safe_proxy_label(proxy_url: str) -> str:
    """Describe a proxy in logs/task state without exposing user-info secrets."""
    parsed = urlsplit(require_explicit_proxy(proxy_url))
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme.lower()}://{host}{port}"


def require_reachable_proxy(proxy_url: str, *, timeout_seconds: float = 3.0) -> None:
    """Fail the whole batch before per-paper retries when the proxy is down."""
    value = require_explicit_proxy(proxy_url)
    parsed = urlsplit(value)
    defaults = {"http": 80, "https": 443, "socks5": 1080, "socks5h": 1080}
    try:
        port = parsed.port or defaults[parsed.scheme.lower()]
        with socket.create_connection(
            (parsed.hostname or "", port), timeout=max(0.1, timeout_seconds),
        ):
            return
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"conference PDF proxy is unreachable at {safe_proxy_label(value)}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


class AggregateDownloadRateLimiter:
    """Paced aggregate byte reader shared by concurrent transfer workers.

    A worker reserves its next socket read before asking the HTTP client for
    bytes.  This avoids turning the proxy/client buffers into an unbounded burst
    and intentionally grants no free first-chunk allowance.
    """

    def __init__(
        self,
        bytes_per_second: int = DEFAULT_DOWNLOAD_BYTES_PER_SECOND,
        *,
        initial_quantum_bytes: int = DEFAULT_STREAM_CHUNK_BYTES,
    ) -> None:
        if bytes_per_second <= 0:
            raise ValueError("bytes_per_second must be positive")
        if initial_quantum_bytes <= 0:
            raise ValueError("initial_quantum_bytes must be positive")
        self.bytes_per_second = int(bytes_per_second)
        self._lock = threading.Lock()
        self._next_read_at = time.monotonic() + (
            initial_quantum_bytes / self.bytes_per_second
        )

    def _reserve_delay(self, requested_bytes: int) -> float:
        if requested_bytes <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_read_at)
            self._next_read_at = scheduled + (
                requested_bytes / self.bytes_per_second
            )
        return max(0.0, scheduled - now)

    def wait_for_next_read(
        self, requested_bytes: int = DEFAULT_STREAM_CHUNK_BYTES,
    ) -> None:
        delay = self._reserve_delay(requested_bytes)
        if delay > 0:
            time.sleep(delay)

    async def wait_for_next_read_async(
        self, requested_bytes: int = DEFAULT_STREAM_CHUNK_BYTES,
    ) -> None:
        delay = self._reserve_delay(requested_bytes)
        if delay > 0:
            await asyncio.sleep(delay)


class ChainedTargetHTTPError(RuntimeError):
    """A terminal response from the paper origin, not an exit-node failure."""

    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(f"paper origin returned HTTP {status_code} for {url}")
        self.status_code = status_code


class ChainedProxyDownloadError(RuntimeError):
    """Every permitted exit failed without falling back to a direct request."""


@dataclass(frozen=True)
class ProxyPoolProbeReport:
    configured_count: int
    attempted_count: int
    healthy_count: int
    elapsed_seconds: float


@dataclass
class _ProxyExitState:
    url: str
    ordinal: int
    verified: bool = False
    ewma_latency_seconds: float | None = None
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    in_flight: int = 0
    uses: int = 0


@dataclass(frozen=True, slots=True)
class BrowserProxyLease:
    """Credential-free handle for one local browser-facing two-hop proxy."""

    proxy_url: str
    exit_ordinal: int


def _xray_socks_server(proxy_url: str) -> dict[str, object]:
    parsed = urlsplit(require_explicit_proxy(proxy_url))
    if parsed.scheme.lower() not in _SUPPORTED_CHAIN_EXIT_SCHEMES or parsed.port is None:
        raise RuntimeError("Xray chain members must be explicit SOCKS5 URLs")
    server: dict[str, object] = {
        "address": parsed.hostname or "",
        "port": parsed.port,
    }
    if parsed.username is not None or parsed.password is not None:
        server["users"] = [{
            "user": unquote(parsed.username or ""),
            "pass": unquote(parsed.password or ""),
        }]
    return server


class _XrayBrowserBridge:
    """Expose one Xray SOCKS listener whose only route is first-hop -> exit."""

    def __init__(
        self,
        *,
        xray_executable: Path,
        first_hop_proxy: str,
        exit_proxy: str,
        startup_timeout_seconds: float,
    ) -> None:
        self._xray_executable = Path(xray_executable)
        self._first_hop_proxy = first_hop_proxy
        self._exit_proxy = exit_proxy
        self._startup_timeout_seconds = max(1.0, startup_timeout_seconds)
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._port: int | None = None

    @property
    def proxy_url(self) -> str:
        if self._port is None:
            raise RuntimeError("Xray browser bridge has not started")
        return f"socks5://127.0.0.1:{self._port}"

    @staticmethod
    def _free_loopback_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    def start(self) -> None:
        if not self._xray_executable.is_file():
            raise RuntimeError(f"Xray executable does not exist: {self._xray_executable}")
        self._port = self._free_loopback_port()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="retrival-xray-chain-")
        config_path = Path(self._temp_dir.name) / "config.json"
        config = {
            "log": {"loglevel": "warning"},
            "inbounds": [{
                "tag": "browser-chain",
                "listen": "127.0.0.1",
                "port": self._port,
                "protocol": "socks",
                "settings": {"auth": "noauth", "udp": False},
            }],
            "outbounds": [
                {
                    "tag": "landing",
                    "protocol": "socks",
                    "settings": {"servers": [_xray_socks_server(self._exit_proxy)]},
                    "proxySettings": {"tag": "first-hop", "transportLayer": True},
                },
                {
                    "tag": "first-hop",
                    "protocol": "socks",
                    "settings": {"servers": [_xray_socks_server(self._first_hop_proxy)]},
                },
                {"tag": "blocked", "protocol": "blackhole"},
            ],
            "routing": {
                "domainStrategy": "AsIs",
                "rules": [{
                    "type": "field",
                    "inboundTag": ["browser-chain"],
                    "outboundTag": "landing",
                }],
            },
        }
        config_path.write_text(
            json.dumps(config, ensure_ascii=False),
            encoding="utf-8",
        )
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            [str(self._xray_executable), "run", "-c", str(config_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=creationflags,
        )
        deadline = time.monotonic() + self._startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self.stop()
                raise RuntimeError("Xray browser bridge exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", self._port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        self.stop()
        raise RuntimeError("Xray browser bridge did not open its local listener")

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
        self._port = None


def _proxy_endpoint(proxy_url: str, *, schemes: set[str]) -> tuple[str, int]:
    value = require_explicit_proxy(proxy_url)
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    if scheme not in schemes or parsed.port is None:
        raise RuntimeError(
            f"proxy must use one of {sorted(schemes)} and include an explicit port"
        )
    return (parsed.hostname or "").lower(), parsed.port


def _load_allowed_endpoints(path: Path) -> set[tuple[str, int]]:
    endpoints: set[tuple[str, int]] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1,
    ):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        candidate = value if "://" in value else f"socks5://{value}"
        try:
            endpoints.add(_proxy_endpoint(candidate, schemes=_SUPPORTED_CHAIN_EXIT_SCHEMES))
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid exit allowlist entry at {path}:{line_number}: {exc}"
            ) from exc
    if not endpoints:
        raise RuntimeError(f"exit allowlist contains no endpoints: {path}")
    return endpoints


class ChainedProxyPool:
    """Credentialed SOCKS exits reached strictly through one local first hop.

    Exit reachability is tested with the same two-hop connector used for real
    transfers.  There is deliberately no direct-connect branch and no raw
    credential logging surface.
    """

    def __init__(self, first_hop_proxy: str, exit_urls: list[str]) -> None:
        self.first_hop_proxy = require_explicit_proxy(first_hop_proxy)
        _proxy_endpoint(
            self.first_hop_proxy,
            schemes=_SUPPORTED_CHAIN_EXIT_SCHEMES,
        )
        if not exit_urls:
            raise RuntimeError("chained proxy pool contains no permitted exits")
        states = [
            _ProxyExitState(url=url, ordinal=index)
            for index, url in enumerate(exit_urls)
        ]
        random.SystemRandom().shuffle(states)
        self._states = states
        self._lock = asyncio.Lock()

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        first_hop_proxy: str,
        allowlist_path: Path | None = None,
    ) -> "ChainedProxyPool":
        if not path.is_file():
            raise RuntimeError(f"proxy pool file does not exist: {path}")
        allowed = _load_allowed_endpoints(allowlist_path) if allowlist_path else None
        first_endpoint = _proxy_endpoint(
            first_hop_proxy,
            schemes=_SUPPORTED_CHAIN_EXIT_SCHEMES,
        )
        exits_by_endpoint: dict[tuple[str, int], str] = {}
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1,
        ):
            value = raw_line.strip()
            if not value:
                continue
            try:
                endpoint = _proxy_endpoint(
                    value,
                    schemes=_SUPPORTED_CHAIN_EXIT_SCHEMES,
                )
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid SOCKS exit at {path}:{line_number}: {exc}"
                ) from exc
            if endpoint == first_endpoint:
                raise RuntimeError("an exit endpoint duplicates the local first hop")
            if allowed is not None and endpoint not in allowed:
                continue
            exits_by_endpoint.setdefault(endpoint, value)
        if allowed is not None:
            missing = allowed.difference(exits_by_endpoint)
            if missing:
                raise RuntimeError(
                    f"{len(missing)} allowlisted exit endpoint(s) are absent from {path}"
                )
        return cls(first_hop_proxy, list(exits_by_endpoint.values()))

    @property
    def configured_count(self) -> int:
        return len(self._states)

    @property
    def healthy_count(self) -> int:
        return sum(state.verified for state in self._states)

    @staticmethod
    def _headers(extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,*/*;q=0.8",
            "Accept-Encoding": "identity",
        }
        if extra:
            headers.update(extra)
        return headers

    def _connector(self, exit_url: str) -> ChainProxyConnector:
        return ChainProxyConnector.from_urls(
            [self.first_hop_proxy, exit_url],
            limit=1,
        )

    async def _record_success(
        self, state: _ProxyExitState, elapsed_seconds: float,
    ) -> None:
        async with self._lock:
            state.in_flight = max(0, state.in_flight - 1)
            state.verified = True
            state.consecutive_failures = 0
            state.cooldown_until = 0.0
            if state.ewma_latency_seconds is None:
                state.ewma_latency_seconds = elapsed_seconds
            else:
                state.ewma_latency_seconds = (
                    state.ewma_latency_seconds * 0.7 + elapsed_seconds * 0.3
                )

    async def _record_failure(self, state: _ProxyExitState) -> None:
        async with self._lock:
            state.in_flight = max(0, state.in_flight - 1)
            state.verified = False
            state.consecutive_failures += 1
            cooldown = min(300.0, float(2 ** min(state.consecutive_failures, 8)))
            state.cooldown_until = time.monotonic() + cooldown

    async def _release(self, state: _ProxyExitState) -> None:
        async with self._lock:
            state.in_flight = max(0, state.in_flight - 1)

    async def _acquire(
        self,
        excluded_ordinals: set[int],
        *,
        verified_only: bool = False,
    ) -> _ProxyExitState | None:
        async with self._lock:
            now = time.monotonic()
            candidates = [
                state for state in self._states
                if state.ordinal not in excluded_ordinals
                and state.cooldown_until <= now
                and (state.verified or not verified_only)
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda state: (
                not state.verified,
                state.in_flight,
                state.uses,
                state.consecutive_failures,
                state.ewma_latency_seconds
                if state.ewma_latency_seconds is not None else float("inf"),
            ))
            state = candidates[0]
            state.in_flight += 1
            state.uses += 1
            return state

    async def _probe_one(
        self,
        state: _ProxyExitState,
        url: str,
        timeout_seconds: float,
    ) -> bool:
        state.in_flight += 1
        started = time.monotonic()
        timeout = aiohttp.ClientTimeout(
            total=timeout_seconds,
            sock_connect=min(4.0, timeout_seconds),
            sock_read=min(4.0, timeout_seconds),
        )
        try:
            async with aiohttp.ClientSession(
                connector=self._connector(state.url),
                timeout=timeout,
                trust_env=False,
            ) as session:
                async with session.get(
                    url,
                    headers=self._headers({"Range": "bytes=0-4095"}),
                    allow_redirects=True,
                ) as response:
                    prefix = await response.content.read(5)
                    if response.status not in {200, 206} or prefix != b"%PDF-":
                        raise RuntimeError(f"probe returned HTTP {response.status} or non-PDF bytes")
            await self._record_success(state, time.monotonic() - started)
            return True
        except asyncio.CancelledError:
            await self._release(state)
            raise
        except Exception:
            await self._record_failure(state)
            return False

    @asynccontextmanager
    async def open_browser_proxy(
        self,
        *,
        xray_executable: Path,
        excluded_ordinals: set[int] | None = None,
        startup_timeout_seconds: float = 8.0,
    ):
        """Lease a local SOCKS endpoint backed only by first-hop -> one exit.

        The browser never receives the landing credential. A failed browser
        operation penalizes that exit; a normal context exit marks it healthy.
        """
        excluded = excluded_ordinals if excluded_ordinals is not None else set()
        state = await self._acquire(excluded, verified_only=True)
        if state is None:
            raise ChainedProxyDownloadError("no eligible landing exit for browser session")
        bridge = _XrayBrowserBridge(
            xray_executable=Path(xray_executable),
            first_hop_proxy=self.first_hop_proxy,
            exit_proxy=state.url,
            startup_timeout_seconds=startup_timeout_seconds,
        )
        started = time.monotonic()
        try:
            await asyncio.to_thread(bridge.start)
            yield BrowserProxyLease(
                proxy_url=bridge.proxy_url,
                exit_ordinal=state.ordinal,
            )
        except asyncio.CancelledError:
            await self._release(state)
            raise
        except Exception:
            await self._record_failure(state)
            raise
        else:
            await self._record_success(state, time.monotonic() - started)
        finally:
            await asyncio.to_thread(bridge.stop)

    async def probe_pdf(
        self,
        url: str,
        *,
        required_healthy: int,
        max_candidates: int,
        concurrency: int,
        timeout_seconds: float = 8.0,
    ) -> ProxyPoolProbeReport:
        if required_healthy <= 0 or concurrency <= 0 or max_candidates <= 0:
            raise ValueError("probe counts must be positive")
        await asyncio.to_thread(require_reachable_proxy, self.first_hop_proxy)
        started = time.monotonic()
        candidates = self._states[: min(max_candidates, len(self._states))]
        attempted = 0
        for offset in range(0, len(candidates), concurrency):
            batch = candidates[offset : offset + concurrency]
            attempted += len(batch)
            await asyncio.gather(*(
                self._probe_one(state, url, timeout_seconds) for state in batch
            ))
            if self.healthy_count >= required_healthy:
                break
        return ProxyPoolProbeReport(
            configured_count=len(self._states),
            attempted_count=attempted,
            healthy_count=self.healthy_count,
            elapsed_seconds=time.monotonic() - started,
        )

    async def download_to_path(
        self,
        url: str,
        destination: Path,
        *,
        max_attempts: int,
        validator: Callable[[Path], bool],
        rate_limiter: AggregateDownloadRateLimiter | None = None,
        headers: Mapping[str, str] | None = None,
        cookies: Mapping[str, str] | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        used: set[int] = set()
        failures: Counter[str] = Counter()
        for _ in range(min(max_attempts, len(self._states))):
            state = await self._acquire(used)
            if state is None:
                break
            used.add(state.ordinal)
            started = time.monotonic()
            timeout = aiohttp.ClientTimeout(
                total=timeout_seconds,
                sock_connect=min(8.0, timeout_seconds),
                sock_read=min(45.0, timeout_seconds),
            )
            try:
                async with aiohttp.ClientSession(
                    connector=self._connector(state.url),
                    timeout=timeout,
                    trust_env=False,
                    cookies=cookies,
                ) as session:
                    async with session.get(
                        url,
                        headers=self._headers(headers),
                        allow_redirects=True,
                    ) as response:
                        if response.status in {404, 410}:
                            await self._record_success(
                                state, time.monotonic() - started,
                            )
                            raise ChainedTargetHTTPError(response.status, url)
                        if response.status < 200 or response.status >= 300:
                            raise RuntimeError(f"HTTP {response.status}")
                        with destination.open("wb") as output_file:
                            while True:
                                if rate_limiter is not None:
                                    await rate_limiter.wait_for_next_read_async()
                                chunk = await response.content.read(
                                    DEFAULT_STREAM_CHUNK_BYTES,
                                )
                                if not chunk:
                                    break
                                output_file.write(chunk)
                if not validator(destination):
                    raise RuntimeError("downloaded payload failed PDF validation")
                await self._record_success(state, time.monotonic() - started)
                return
            except ChainedTargetHTTPError:
                remove_file_if_present(destination)
                raise
            except asyncio.CancelledError:
                await self._release(state)
                remove_file_if_present(destination)
                raise
            except Exception as exc:
                failures[type(exc).__name__] += 1
                await self._record_failure(state)
                remove_file_if_present(destination)
        detail = ", ".join(
            f"{name}={count}" for name, count in sorted(failures.items())
        ) or "no eligible exit"
        raise ChainedProxyDownloadError(
            f"two-hop download failed after {len(used)} distinct exit(s): {detail}"
        )
