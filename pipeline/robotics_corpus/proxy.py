"""Mandatory local proxy for robotics network I/O (no silent direct fallback)."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PROXY_URL = "http://127.0.0.1:10808"


@dataclass(frozen=True, slots=True)
class RoboticsProxyConfig:
    http_proxy: str = DEFAULT_PROXY_URL
    https_proxy: str = DEFAULT_PROXY_URL
    require_proxy: bool = True

    def as_httpx_proxy(self) -> str:
        if self.require_proxy and not self.http_proxy:
            raise RuntimeError("robotics corpus requires http_proxy")
        return self.http_proxy


def load_robotics_proxy_from_settings() -> RoboticsProxyConfig:
    try:
        from config import settings

        http_proxy = getattr(settings.proxy, "http_proxy", None) or DEFAULT_PROXY_URL
        https_proxy = getattr(settings.proxy, "https_proxy", None) or http_proxy
        return RoboticsProxyConfig(http_proxy=http_proxy, https_proxy=https_proxy)
    except Exception:
        return RoboticsProxyConfig()
