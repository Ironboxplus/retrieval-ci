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
from aiohttp_socks import ChainProxyConnector, ProxyConnector

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
    url: str | None  # None = the pool's single exit-less route (first hop only)
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

    With no exits the pool degrades to one route that goes through the first
    hop only (never direct).  That route is retried in place, cooldown
    included, instead of failing over to another exit.
    """

    def __init__(self, first_hop_proxy: str, exit_urls: list[str]) -> None:
        self.first_hop_proxy = require_explicit_proxy(first_hop_proxy)
        _proxy_endpoint(
            self.first_hop_proxy,
            schemes=_SUPPORTED_CHAIN_EXIT_SCHEMES,
        )
        self._exitless = not exit_urls
        states = [
            _ProxyExitState(url=url, ordinal=index)
            for index, url in enumerate(exit_urls or [None])
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
        if not exits_by_endpoint:
            # An exit-less pool is a deliberate choice (``cls(first_hop, [])``),
            # not what an empty or fully filtered pool file should turn into.
            raise RuntimeError(f"proxy pool file holds no permitted exits: {path}")
        return cls(first_hop_proxy, list(exits_by_endpoint.values()))

    @property
    def configured_count(self) -> int:
        return 0 if self._exitless else len(self._states)

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

    def _connector(self, exit_url: str | None) -> ChainProxyConnector | ProxyConnector:
        if exit_url is None:
            return ProxyConnector.from_url(self.first_hop_proxy, limit=1)
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
        """Lease the best exit; wait out a cooldown rather than fail on it.

        ``None`` only when ``excluded_ordinals``/``verified_only`` leave no
        exit that could ever become available.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                eligible = [
                    state for state in self._states
                    if state.ordinal not in excluded_ordinals
                    and (state.verified or not verified_only)
                ]
                candidates = [s for s in eligible if s.cooldown_until <= now]
                if candidates:
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
                if not eligible:
                    return None
                wait = min(s.cooldown_until for s in eligible) - now
            await asyncio.sleep(wait)

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
        if state is None or state.url is None:
            if state is not None:
                await self._release(state)
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
        attempted = 0
        # Exits are tried once each; the exit-less route has nowhere to fail
        # over to, so it is retried in place (``_acquire`` waits out cooldown).
        budget = max_attempts if self._exitless else min(max_attempts, len(self._states))
        for _ in range(budget):
            state = await self._acquire(used)
            if state is None:
                break
            attempted += 1
            if not self._exitless:
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
            f"two-hop download failed after {attempted} attempt(s) "
            f"over {len(used)} distinct exit(s): {detail}"
        )


# Responses that mean "this host is refusing us right now" (Anubis/IEEE answer
# 202 with a challenge page, Cloudflare 403/503, rate limits 429). They are never
# retried at once: the host is cooled down and the queue moves to another host.
BLOCKING_STATUS_CODES = frozenset({202, 403, 429, 503})
# Bodies that are an anti-bot interstitial rather than the requested file:
# Springer/Akamai "Client Challenge", Cloudflare, Anubis (DBLP).
CHALLENGE_PAGE_MARKERS = (
    b"<title>Client Challenge</title>",
    b"<title>Just a moment...</title>",
    b"challenges.cloudflare.com",
    b"Attention Required! | Cloudflare",
    b"Making sure you&#39;re not a bot",
    b"Making sure you're not a bot",
)


class ChallengePageError(ValueError):
    """A response whose body is an anti-bot challenge page, not the file."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def is_challenge_page(head: bytes) -> bool:
    return any(marker in head for marker in CHALLENGE_PAGE_MARKERS)


def interleave_by_key(items, key, *, secondary_key=None) -> list:
    """Order ``items`` so two consecutive entries never share ``key`` while another
    key still has work.

    Each step takes the head of the key queue with the most remaining items among
    those different from the previous step's key, so the run ends with the
    shortest possible single-key tail. ``secondary_key`` breaks ties toward a head
    whose secondary value also differs from the previous one (two venues behind
    one host still alternate). Order inside one key is preserved.
    """
    queues: dict = {}
    for item in items:
        queues.setdefault(key(item), []).append(item)
    for queue in queues.values():
        queue.reverse()  # pop() from the end keeps the original order
    ordered: list = []
    last_key: object = object()
    last_secondary: object = object()
    while queues:
        candidates = [k for k in queues if k != last_key] or list(queues)

        def rank(k):
            head = queues[k][-1]
            differs = secondary_key is None or secondary_key(head) != last_secondary
            return (1 if differs else 0, len(queues[k]))

        chosen = max(candidates, key=rank)
        item = queues[chosen].pop()
        if not queues[chosen]:
            del queues[chosen]
        ordered.append(item)
        last_key = chosen
        last_secondary = secondary_key(item) if secondary_key is not None else None
    return ordered


class HostTurnQueue:
    """Serial download queue that takes turns between hosts and backs off blocks.

    ``next()`` never returns two consecutive items from one host while another
    host has ready work, preferring the host with the most remaining items (and,
    among those, a different venue than the previous item). ``block(host)``
    cools a host down -- ``base_cooldown_seconds`` doubling per consecutive block,
    at least the server's Retry-After -- and after ``max_strikes`` consecutive
    blocks the host is abandoned for this run; its items move to ``abandoned``.
    ``ok(host)`` clears the strike count. A blocked item may be ``requeue``d once
    to retry after its host's cooldown.
    """

    def __init__(
        self,
        items,
        *,
        host_of: Callable,
        venue_of: Callable | None = None,
        clock: Callable[[], float] = time.monotonic,
        base_cooldown_seconds: float = 300.0,
        max_cooldown_seconds: float = 3600.0,
        max_strikes: int = 4,
        max_requeues: int = 1,
    ) -> None:
        self._host_of = host_of
        self._venue_of = venue_of
        self._clock = clock
        self._base = float(base_cooldown_seconds)
        self._max = float(max_cooldown_seconds)
        self._max_strikes = int(max_strikes)
        self._max_requeues = int(max_requeues)
        self._queues: dict[str, list] = {}
        for item in items:
            self._queues.setdefault(host_of(item), []).append(item)
        for host, queue in self._queues.items():
            if venue_of is not None:
                # Venues sharing one host (TPAMI and IJCV on arxiv.org) alternate too.
                queue[:] = interleave_by_key(queue, key=venue_of)
            queue.reverse()
        self._cool_until: dict[str, float] = {}
        self._strikes: Counter = Counter()
        self._requeues: Counter = Counter()
        self._last_host: str | None = None
        self._last_venue: object = object()
        self.abandoned: list = []
        self.abandoned_hosts: set[str] = set()

    def remaining(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    def next(self):
        """``(item, 0.0)`` when one is ready, ``(None, seconds)`` while every host
        with work is cooling down, ``(None, 0.0)`` when drained."""
        if not self._queues:
            return None, 0.0
        now = self._clock()
        ready = [host for host in self._queues if self._cool_until.get(host, 0.0) <= now]
        if not ready:
            return None, max(0.0, min(self._cool_until[host] for host in self._queues) - now)
        candidates = [host for host in ready if host != self._last_host] or ready

        def rank(host):
            head = self._queues[host][-1]
            differs = self._venue_of is None or self._venue_of(head) != self._last_venue
            return (1 if differs else 0, len(self._queues[host]))

        host = max(candidates, key=rank)
        item = self._queues[host].pop()
        if not self._queues[host]:
            del self._queues[host]
        self._last_host = host
        self._last_venue = self._venue_of(item) if self._venue_of is not None else None
        return item, 0.0

    def ok(self, host: str) -> None:
        self._strikes.pop(host, None)

    def block(self, host: str, retry_after_seconds: float | None = None) -> None:
        self._strikes[host] += 1
        strikes = self._strikes[host]
        cooldown = min(self._max, self._base * (2 ** (strikes - 1)))
        cooldown = max(cooldown, float(retry_after_seconds or 0.0))
        self._cool_until[host] = self._clock() + cooldown
        if strikes >= self._max_strikes:
            self.abandoned_hosts.add(host)
            queue = self._queues.pop(host, [])
            self.abandoned.extend(reversed(queue))

    def requeue(self, item) -> bool:
        """Put a blocked item back at the head of its host queue, once."""
        host = self._host_of(item)
        if host in self.abandoned_hosts:
            self.abandoned.append(item)
            return False
        ident = id(item)
        if self._requeues[ident] >= self._max_requeues:
            return False
        self._requeues[ident] += 1
        self._queues.setdefault(host, []).append(item)
        return True


def blocking_status(exc: BaseException) -> int | None:
    """HTTP status of a host-refusal error, else None."""
    if isinstance(exc, ChallengePageError):
        return exc.status_code  # a challenge is a refusal whatever its status
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    return status if status in BLOCKING_STATUS_CODES else None
