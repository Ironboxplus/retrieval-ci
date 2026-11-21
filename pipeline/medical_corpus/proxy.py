"""Mandatory local proxy for medical network I/O (no silent direct fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_PROXY_URL = "http://127.0.0.1:10808"


@dataclass(frozen=True, slots=True)
class MedicalProxyConfig:
    """Always-on proxy settings for Arc-side medical collection."""

    http_proxy: str = DEFAULT_PROXY_URL
    https_proxy: str = DEFAULT_PROXY_URL
    require_proxy: bool = True

    def as_requests_dict(self) -> dict[str, str]:
        if self.require_proxy and not (self.http_proxy and self.https_proxy):
            raise RuntimeError(
                "medical corpus requires a local proxy; refusing empty proxy config"
            )
        return {"http": self.http_proxy, "https": self.https_proxy}

    def as_httpx_proxy(self) -> str:
        """httpx accepts a single proxy URL for both schemes in recent versions."""
        if self.require_proxy and not self.http_proxy:
            raise RuntimeError("medical corpus requires http_proxy")
        return self.http_proxy


def load_medical_proxy_from_settings() -> MedicalProxyConfig:
    """Prefer project settings; fall back to the hard 10808 default."""
    try:
        from config import settings
        http_proxy = getattr(settings.proxy, "http_proxy", None) or DEFAULT_PROXY_URL
        https_proxy = getattr(settings.proxy, "https_proxy", None) or http_proxy
        return MedicalProxyConfig(http_proxy=http_proxy, https_proxy=https_proxy)
    except Exception:
        return MedicalProxyConfig()
