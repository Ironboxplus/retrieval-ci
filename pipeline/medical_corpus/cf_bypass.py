"""Cloudflare dual-path solver adapter (TurnstileBypass, minimal headless).

Uses Ironboxplus/TurnstileBypass vendored at ``vendor/TurnstileBypass``:

- soft  → session cookie_header (often no cf_clearance)
- widget → cf-turnstile-response token
- managed → cf_clearance + cookie_header for subsequent HTTP

Default FO host is **minimal** (system Chrome headless via selenium-driverless),
not headed click-theater and not captcha SaaS.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pipeline.medical_corpus.proxy import DEFAULT_PROXY_URL, MedicalProxyConfig, load_medical_proxy_from_settings

logger = logging.getLogger(__name__)

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "TurnstileBypass"


def _ensure_vendor_on_path() -> None:
    root = str(_VENDOR_ROOT)
    if _VENDOR_ROOT.is_dir() and root not in sys.path:
        sys.path.insert(0, root)


@dataclass(frozen=True, slots=True)
class CfBypassResult:
    ok: bool
    path: str  # soft | widget | managed | ""
    page_url: str
    final_url: str = ""
    cookie_header: str = ""
    cf_clearance: str = ""
    token: str = ""
    user_agent: str = ""
    error: str = ""
    ms: float = 0.0
    backend: str = ""
    verified: bool = False
    still_challenged: bool = False
    cookies: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_httpx_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.user_agent:
            headers["User-Agent"] = self.user_agent
        if self.cookie_header:
            headers["Cookie"] = self.cookie_header
        return headers


def turnstile_available() -> bool:
    """True when the vendored package imports (Chrome FO still needs selenium-driverless)."""
    try:
        _ensure_vendor_on_path()
        import turnstile_bypass  # noqa: F401
        return True
    except Exception:
        return False


def minimal_backend_available() -> bool:
    try:
        _ensure_vendor_on_path()
        from turnstile_bypass import is_backend_available
        return any(
            bool(is_backend_available(name))
            for name in ("camoufox", "driverless", "browser", "minimal")
        )
    except Exception:
        return False


async def solve_cloudflare_for_url(
    url: str,
    *,
    proxy: MedicalProxyConfig | str | None = None,
    path: str = "auto",
    require_clearance: bool = True,
    headless: bool = True,
    verify: bool = True,
    timeout_s: float = 120.0,
    overall_timeout_s: float | None = None,
    backend: str | None = None,
    force_challenge_urls: list[str] | None = None,
    verified_session_callback: Any | None = None,
) -> CfBypassResult:
    """Run TurnstileBypass dual-path against ``url`` through the medical proxy.

    Parameters mirror ``solve_cf_dual_path``; ``minimal`` headless is preferred.
    """
    _ensure_vendor_on_path()
    try:
        import turnstile_bypass as turnstile_sdk
    except ImportError as exc:
        return CfBypassResult(
            ok=False, path="", page_url=url,
            error=(
                f"turnstile_bypass not importable ({exc}). "
                f"Expected vendor at {_VENDOR_ROOT}. "
                "Clone https://github.com/Ironboxplus/TurnstileBypass.git into vendor/TurnstileBypass "
                "and pip install selenium-driverless for minimal FO."
            ),
        )

    if isinstance(proxy, MedicalProxyConfig):
        proxy_url = proxy.http_proxy
    elif isinstance(proxy, str) and proxy.strip():
        proxy_url = proxy.strip()
    elif proxy is None:
        proxy_url = load_medical_proxy_from_settings().http_proxy
    else:
        proxy_url = DEFAULT_PROXY_URL

    if not proxy_url:
        return CfBypassResult(
            ok=False, path="", page_url=url,
            error="proxy required for CF bypass (medical corpus forbids direct egress)",
        )

    default_ua = ""
    try:
        from turnstile_bypass import BROWSER_DEFAULT_UA

        default_ua = BROWSER_DEFAULT_UA
    except (ImportError, AttributeError):
        default_ua = getattr(turnstile_sdk, "BROWSER_DEFAULT_UA", "")

    try:
        solver_retries = max(1, int(os.environ.get("TURNSTILE_MANAGED_RETRIES", "2")))
        overall_timeout = (
            float(overall_timeout_s)
            if overall_timeout_s is not None
            else timeout_s * solver_retries + 10
        )
        if (backend or "").strip().lower() == "camoufox":
            solver = turnstile_sdk.CamoufoxManagedChallengeSolver(
                proxy=proxy_url,
                timeout_ms=int(timeout_s * 1000),
                headless=headless,
                require_clearance=require_clearance,
                verify_url=url if verify else None,
                humanize=True,
                geoip=True,
                allow_click=True,
                verified_session_callback=verified_session_callback,
            )
            res = await asyncio.wait_for(
                solver.solve_detailed(url),
                timeout=overall_timeout,
            )
            sdk_generation = "managed"
        elif hasattr(turnstile_sdk, "solve_cf_dual_path"):
            # Ironboxplus 0.4 exposes the stable dual-path API.  It keeps the
            # challenge classification, site-domain cookie selection and UA
            # binding in the SDK instead of reimplementing them here.
            res = await asyncio.wait_for(
                turnstile_sdk.solve_cf_dual_path(
                    url,
                    proxy=proxy_url,
                    path=path,
                    headless=headless,
                    require_clearance=require_clearance,
                    verify_url=url if verify else None,
                    force_challenge_urls=force_challenge_urls,
                    verified_session_callback=verified_session_callback,
                    timeout_ms=int(timeout_s * 1000),
                ),
                timeout=overall_timeout,
            )
            sdk_generation = "dual_path"
        else:
            # Compatibility for a managed-only SDK generation.
            res = await asyncio.wait_for(
                turnstile_sdk.solve_managed_challenge(
                    url,
                    proxy=proxy_url,
                    headless=headless,
                    require_clearance=require_clearance,
                    verify_url=url if verify else None,
                    force_challenge_urls=force_challenge_urls,
                    verified_session_callback=verified_session_callback,
                    timeout_ms=int(timeout_s * 1000),
                ),
                timeout=overall_timeout,
            )
            sdk_generation = "managed"
    except asyncio.TimeoutError:
        return CfBypassResult(
            ok=False, path="", page_url=url,
            error=f"solve_cf_dual_path timed out after {overall_timeout:.0f}s",
            user_agent=default_ua,
        )
    except Exception as exc:
        logger.exception("solve_cf_dual_path failed for %s", url)
        return CfBypassResult(
            ok=False, path="", page_url=url,
            error=f"{type(exc).__name__}: {exc}",
            user_agent=default_ua,
        )

    ua = getattr(res, "user_agent", None) or default_ua
    cookie_header = getattr(res, "cookie_header", None) or ""
    clearance = getattr(res, "cf_clearance", None) or ""
    token = getattr(res, "token", None) or ""
    res_path = str(getattr(res, "path", "") or ("managed" if sdk_generation == "managed" else ""))
    ok = bool(getattr(res, "ok", False))
    error = str(getattr(res, "error", "") or "")
    ms = float(getattr(res, "ms", 0.0) or 0.0)
    backend = str(getattr(res, "backend", "") or "")
    final_url = str(getattr(res, "final_url", "") or url)
    page_url = str(getattr(res, "page_url", "") or url)
    still = bool(getattr(res, "still_challenged", False))
    try:
        normalized_cookies = tuple(
            dict(cookie)
            for cookie in turnstile_sdk.normalize_cookie_list(
                getattr(res, "cookies", None)
            )
        )
    except Exception:
        normalized_cookies = ()

    verified = bool(getattr(res, "verified", False))
    if (
        ok
        and not verified
        and sdk_generation == "dual_path"
        and res_path == "managed"
        and clearance
        and verify
    ):
        try:
            proof = await asyncio.to_thread(
                turnstile_sdk.verify_clearance_http,
                page_url,
                clearance,
                cookie_header=cookie_header,
                user_agent=ua,
                proxy=proxy_url,
            )
            verified = bool(proof.get("ok")) and not bool(proof.get("challenged"))
            if not verified:
                error = error or f"clearance verify failed: {proof}"
                ok = False
        except Exception as exc:
            error = error or f"verify_clearance_http: {exc}"
            ok = False

    return CfBypassResult(
        ok=ok,
        path=res_path,
        page_url=page_url,
        final_url=final_url,
        cookie_header=cookie_header,
        cf_clearance=clearance,
        token=token,
        user_agent=ua,
        error=error,
        ms=ms,
        backend=backend,
        verified=verified,
        still_challenged=still,
        cookies=normalized_cookies,
        raw={
            **dict(getattr(res, "raw", {}) or {}),
            "sdk_generation": sdk_generation,
            "session_api": getattr(res, "session_api", None),
            "sitekey": getattr(res, "sitekey", None),
        },
    )


def solve_cloudflare_for_url_sync(url: str, **kwargs: Any) -> CfBypassResult:
    """Sync wrapper for CLI / non-async callers."""
    return asyncio.run(solve_cloudflare_for_url(url, **kwargs))
