"""Shield / bot-wall / rate-limit classification for medical HTTP responses.

Medical OA hubs are mostly API-friendly, but publisher PDF edges and some
preprint sites return Cloudflare/403/429 bodies. We classify them so the
pipeline can backoff, mark blocked, or hand off to TurnstileBypass (minimal
headless) — not brute-force.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from pipeline.crawler.captcha_detector import CaptchaDetector, CaptchaType
from pipeline.http_retry import exponential_backoff_seconds, retry_after_seconds

# A server may name an hour; a download thread waits at most this long on its say-so
# (the same ceiling the arXiv OAI walk applies to its 503 Retry-After).
MAX_RETRY_AFTER_SECONDS = 600.0


class ShieldKind(str, Enum):
    NONE = "none"
    RATE_LIMIT = "rate_limit"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    CAPTCHA = "captcha"
    CLOUDFLARE = "cloudflare"
    HTML_INSTEAD_OF_PDF = "html_instead_of_pdf"
    EMPTY = "empty"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ShieldDecision:
    kind: ShieldKind
    retryable: bool
    message: str
    captcha_type: Optional[str] = None
    confidence: float = 0.0
    retry_after_s: Optional[float] = None  # the server's own Retry-After, when it sent a usable one

    @property
    def blocked(self) -> bool:
        return self.kind in {
            ShieldKind.FORBIDDEN,
            ShieldKind.CAPTCHA,
            ShieldKind.CLOUDFLARE,
            ShieldKind.HTML_INSTEAD_OF_PDF,
        }

    @property
    def needs_cf_bypass(self) -> bool:
        """True when TurnstileBypass dual-path should be attempted."""
        return self.kind in {
            ShieldKind.CLOUDFLARE,
            ShieldKind.CAPTCHA,
            ShieldKind.FORBIDDEN,
            ShieldKind.HTML_INSTEAD_OF_PDF,
        }


_CAPTCHA = CaptchaDetector()


def _body_text(body: bytes | str | None, limit: int = 8000) -> str:
    if body is None:
        return ""
    if isinstance(body, bytes):
        return body[:limit].decode("utf-8", errors="replace")
    return body[:limit]


def _detect_captcha(html: str, status_code: int = 200) -> object:
    return _CAPTCHA.detect_from_html(html, status_code=status_code)


def classify_http_shield(
    *,
    status_code: int | None,
    content_type: str | None = None,
    body: bytes | str | None = None,
    expect_pdf: bool = False,
    headers: Mapping[str, str] | None = None,
) -> ShieldDecision:
    """Classify a response for retry / block / CF-bypass decisions."""
    headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    text = _body_text(body)
    ctype = (content_type or headers.get("content-type") or "").lower()
    code = int(status_code) if status_code is not None else None

    if code is None:
        return ShieldDecision(ShieldKind.TRANSIENT, True, "no status (transport failure)")

    if code == 429:
        retry_after = headers.get("retry-after")
        msg = "HTTP 429 rate limited" + (f" retry-after={retry_after}" if retry_after else "")
        return ShieldDecision(
            ShieldKind.RATE_LIMIT, True, msg,
            retry_after_s=retry_after_seconds(retry_after, default=None),
        )

    if code in {408, 500, 502, 503, 504}:
        return ShieldDecision(
            ShieldKind.TRANSIENT, True, f"HTTP {code} transient",
            retry_after_s=(
                retry_after_seconds(headers.get("retry-after"), default=None) if code == 503 else None
            ),
        )

    if code == 404:
        return ShieldDecision(ShieldKind.NOT_FOUND, False, "HTTP 404 not found")

    if code in {401, 403}:
        captcha = _detect_captcha(text, status_code=code)
        if captcha.detected and captcha.captcha_type in {
            CaptchaType.CLOUDFLARE, CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3,
            CaptchaType.HCAPTCHA,
        }:
            return ShieldDecision(
                ShieldKind.CLOUDFLARE if captcha.captcha_type == CaptchaType.CLOUDFLARE
                else ShieldKind.CAPTCHA,
                False,
                f"HTTP {code} with {captcha.captcha_type.value}",
                captcha_type=captcha.captcha_type.value,
                confidence=captcha.confidence,
            )
        # Many CF edges return bare 403 without classic markers — still try dual-path.
        return ShieldDecision(ShieldKind.FORBIDDEN, False, f"HTTP {code} forbidden")

    if code >= 400:
        return ShieldDecision(ShieldKind.UNKNOWN, code >= 500, f"HTTP {code}")

    if expect_pdf:
        raw = body if isinstance(body, (bytes, bytearray)) else None
        if raw is not None and len(raw) == 0:
            return ShieldDecision(ShieldKind.EMPTY, True, "empty body while expecting PDF")
        if raw is not None and not raw.startswith(b"%PDF"):
            captcha = _detect_captcha(text, status_code=code)
            if captcha.detected:
                kind = (
                    ShieldKind.CLOUDFLARE
                    if captcha.captcha_type == CaptchaType.CLOUDFLARE
                    else ShieldKind.CAPTCHA
                )
                return ShieldDecision(
                    kind, False,
                    f"non-PDF body with {captcha.captcha_type.value}",
                    captcha_type=captcha.captcha_type.value,
                    confidence=captcha.confidence,
                )
            if "html" in ctype or text.lstrip().lower().startswith("<!doctype") or "<html" in text.lower():
                return ShieldDecision(
                    ShieldKind.HTML_INSTEAD_OF_PDF, False,
                    f"HTML body instead of PDF (content-type={ctype or 'unknown'})",
                )
            return ShieldDecision(
                ShieldKind.HTML_INSTEAD_OF_PDF, False,
                "body does not start with %PDF magic",
            )
        if raw is not None and raw.startswith(b"%PDF"):
            return ShieldDecision(ShieldKind.NONE, False, "PDF magic ok")

    captcha = _detect_captcha(text, status_code=code or 200)
    if captcha.detected and captcha.confidence >= 0.85:
        kind = (
            ShieldKind.CLOUDFLARE
            if captcha.captcha_type == CaptchaType.CLOUDFLARE
            else ShieldKind.CAPTCHA
        )
        return ShieldDecision(
            kind, False,
            f"page looks like {captcha.captcha_type.value}",
            captcha_type=captcha.captcha_type.value,
            confidence=captcha.confidence,
        )

    return ShieldDecision(ShieldKind.NONE, False, "ok")


def retry_delay_seconds(decision: ShieldDecision, attempt: int) -> float:
    """Bounded delay for retryable shield decisions; never shorter than the
    server's own Retry-After (capped at ``MAX_RETRY_AFTER_SECONDS``)."""
    if not decision.retryable:
        return 0.0
    if decision.kind == ShieldKind.RATE_LIMIT:
        delay = float(min(120.0, 30.0 * attempt))
    else:
        delay = exponential_backoff_seconds(attempt)
    if decision.retry_after_s is not None:
        delay = max(delay, min(decision.retry_after_s, MAX_RETRY_AFTER_SECONDS))
    return delay
