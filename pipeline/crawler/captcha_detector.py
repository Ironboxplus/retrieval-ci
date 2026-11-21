"""
Captcha检测器 - 检测网页中的各类验证码
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import re


class CaptchaType(Enum):
    """验证码类型枚举"""
    NONE = "none"
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    CLOUDFLARE = "cloudflare"
    IMAGE = "image"
    SLIDER = "slider"
    UNKNOWN = "unknown"


@dataclass
class CaptchaDetectionResult:
    """Captcha检测结果"""
    detected: bool
    captcha_type: CaptchaType
    confidence: float  # 0.0 - 1.0
    details: Optional[str] = None
    matched_patterns: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.detected:
            self.captcha_type = CaptchaType.NONE
            self.confidence = 0.0


class CaptchaDetector:
    """检测页面中的Captcha验证码"""

    # 各类验证码的检测模式
    PATTERNS: Dict[CaptchaType, List[Tuple[str, float]]] = {
        CaptchaType.RECAPTCHA_V2: [
            (r'g-recaptcha', 0.9),
            (r'grecaptcha\.render', 0.95),
            (r'www\.google\.com/recaptcha/api2', 0.95),
            (r'data-sitekey', 0.7),
            (r'recaptcha-checkbox', 0.9),
            (r'recaptcha/api\.js', 0.85),
        ],
        CaptchaType.RECAPTCHA_V3: [
            (r'grecaptcha\.execute', 0.95),
            (r'recaptcha/api\.js\?render=', 0.95),
            (r'recaptcha-v3', 0.9),
            (r'grecaptcha\.ready', 0.8),
        ],
        CaptchaType.HCAPTCHA: [
            (r'h-captcha', 0.95),
            (r'hcaptcha\.com', 0.95),
            (r'data-hcaptcha', 0.9),
            (r'hcaptcha-checkbox', 0.9),
            (r'hcaptcha\.render', 0.95),
        ],
        CaptchaType.CLOUDFLARE: [
            (r'cf-turnstile', 0.95),
            (r'challenges\.cloudflare\.com', 0.95),
            (r'Just a moment\.\.\.', 0.85),
            (r'Checking your browser', 0.8),
            (r'cf-browser-verification', 0.9),
            (r'__cf_bm', 0.7),
            (r'cf_clearance', 0.8),
            (r'Attention Required! \| Cloudflare', 0.95),
            (r'Enable JavaScript and cookies', 0.75),
            (r'DDoS protection by Cloudflare', 0.9),
            (r'Ray ID:', 0.6),
        ],
        CaptchaType.IMAGE: [
            (r'captcha.*image', 0.7),
            (r'verification.*image', 0.6),
            (r'verify.*image', 0.6),
            (r'captcha\.php', 0.8),
            (r'captcha\.aspx', 0.8),
            (r'captcha\.jsp', 0.8),
            (r'getCaptcha', 0.85),
            (r'refreshCaptcha', 0.85),
        ],
        CaptchaType.SLIDER: [
            (r'slider.*captcha', 0.85),
            (r'slide.*verify', 0.8),
            (r'geetest', 0.95),
            (r'gt_captcha', 0.95),
            (r'TencentCaptcha', 0.95),
            (r'slide-to-unlock', 0.8),
            (r'drag.*slider', 0.7),
        ],
    }

    # HTTP状态码相关的验证码检测
    CAPTCHA_STATUS_CODES = {
        403: 0.5,  # Forbidden - 可能触发验证
        429: 0.7,  # Too Many Requests - 频率限制
        503: 0.6,  # Service Unavailable - 可能是Cloudflare
    }

    # 响应头中的验证码指示
    CAPTCHA_HEADERS = {
        'cf-ray': (CaptchaType.CLOUDFLARE, 0.5),
        'cf-cache-status': (CaptchaType.CLOUDFLARE, 0.4),
        'server': {
            'cloudflare': (CaptchaType.CLOUDFLARE, 0.6),
        },
    }

    def detect_from_html(
        self,
        html: str,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None
    ) -> CaptchaDetectionResult:
        """
        从HTML内容检测Captcha

        Args:
            html: HTML页面内容
            status_code: HTTP状态码
            headers: HTTP响应头

        Returns:
            CaptchaDetectionResult: 检测结果
        """
        if not html:
            return CaptchaDetectionResult(
                detected=False,
                captcha_type=CaptchaType.NONE,
                confidence=0.0
            )

        best_match: Optional[CaptchaType] = None
        best_confidence: float = 0.0
        all_matched_patterns: List[str] = []
        type_confidences: Dict[CaptchaType, float] = {}

        # 检测HTML内容中的模式
        for captcha_type, patterns in self.PATTERNS.items():
            max_confidence_for_type = 0.0
            matched = []

            for pattern, confidence in patterns:
                if re.search(pattern, html, re.IGNORECASE):
                    matched.append(pattern)
                    max_confidence_for_type = max(max_confidence_for_type, confidence)

            if matched:
                type_confidences[captcha_type] = max_confidence_for_type
                all_matched_patterns.extend(matched)

                if max_confidence_for_type > best_confidence:
                    best_confidence = max_confidence_for_type
                    best_match = captcha_type

        # 考虑HTTP状态码
        if status_code in self.CAPTCHA_STATUS_CODES:
            status_confidence = self.CAPTCHA_STATUS_CODES[status_code]
            # 如果已检测到验证码类型，增强置信度
            if best_match:
                best_confidence = min(1.0, best_confidence + status_confidence * 0.2)
            # 如果是503且有Cloudflare迹象
            elif status_code == 503 and headers:
                if 'cf-ray' in {k.lower() for k in headers.keys()}:
                    best_match = CaptchaType.CLOUDFLARE
                    best_confidence = 0.85

        # 检查响应头
        if headers:
            header_result = self._check_headers(headers)
            if header_result and not best_match:
                best_match, header_confidence = header_result
                best_confidence = max(best_confidence, header_confidence)

        # 确定是否检测到验证码
        detected = best_confidence >= 0.5

        if detected and best_match:
            return CaptchaDetectionResult(
                detected=True,
                captcha_type=best_match,
                confidence=best_confidence,
                details=f"Detected {best_match.value} captcha",
                matched_patterns=all_matched_patterns
            )

        return CaptchaDetectionResult(
            detected=False,
            captcha_type=CaptchaType.NONE,
            confidence=0.0
        )

    def _check_headers(
        self,
        headers: Dict[str, str]
    ) -> Optional[Tuple[CaptchaType, float]]:
        """检查HTTP响应头中的验证码指示"""
        headers_lower = {k.lower(): v for k, v in headers.items()}

        # 检查特定头
        for header, value in self.CAPTCHA_HEADERS.items():
            if header in headers_lower:
                if isinstance(value, tuple):
                    return value
                elif isinstance(value, dict):
                    header_value = headers_lower[header].lower()
                    for pattern, result in value.items():
                        if pattern in header_value:
                            return result

        return None

    async def detect_from_page(self, page) -> CaptchaDetectionResult:
        """
        从Playwright页面检测Captcha

        Args:
            page: Playwright Page对象

        Returns:
            CaptchaDetectionResult: 检测结果
        """
        try:
            # 获取页面内容
            html = await page.content()

            # 获取响应状态（如果可用）
            # Playwright页面没有直接的status，需要从response获取
            # 这里我们主要依赖HTML内容检测

            # 基本HTML检测
            result = self.detect_from_html(html)

            if result.detected:
                return result

            # 额外的动态检测：检查iframe
            iframe_check = await self._check_captcha_iframes(page)
            if iframe_check.detected:
                return iframe_check

            # 检查特定元素的存在
            element_check = await self._check_captcha_elements(page)
            if element_check.detected:
                return element_check

            return result

        except Exception as e:
            return CaptchaDetectionResult(
                detected=False,
                captcha_type=CaptchaType.UNKNOWN,
                confidence=0.0,
                details=f"Detection error: {str(e)}"
            )

    async def _check_captcha_iframes(self, page) -> CaptchaDetectionResult:
        """检查页面中的captcha iframe"""
        captcha_iframe_patterns = [
            ('recaptcha', CaptchaType.RECAPTCHA_V2, 0.9),
            ('hcaptcha', CaptchaType.HCAPTCHA, 0.9),
            ('cloudflare', CaptchaType.CLOUDFLARE, 0.9),
            ('turnstile', CaptchaType.CLOUDFLARE, 0.95),
            ('geetest', CaptchaType.SLIDER, 0.9),
        ]

        try:
            iframes = await page.query_selector_all('iframe')

            for iframe in iframes:
                src = await iframe.get_attribute('src') or ''
                name = await iframe.get_attribute('name') or ''
                title = await iframe.get_attribute('title') or ''

                combined = f"{src} {name} {title}".lower()

                for pattern, captcha_type, confidence in captcha_iframe_patterns:
                    if pattern in combined:
                        return CaptchaDetectionResult(
                            detected=True,
                            captcha_type=captcha_type,
                            confidence=confidence,
                            details=f"Found captcha iframe: {pattern}",
                            matched_patterns=[pattern]
                        )

        except Exception:
            pass

        return CaptchaDetectionResult(
            detected=False,
            captcha_type=CaptchaType.NONE,
            confidence=0.0
        )

    async def _check_captcha_elements(self, page) -> CaptchaDetectionResult:
        """检查页面中的captcha元素"""
        element_selectors = [
            ('.g-recaptcha', CaptchaType.RECAPTCHA_V2, 0.95),
            ('#g-recaptcha', CaptchaType.RECAPTCHA_V2, 0.95),
            ('.h-captcha', CaptchaType.HCAPTCHA, 0.95),
            ('#h-captcha', CaptchaType.HCAPTCHA, 0.95),
            ('.cf-turnstile', CaptchaType.CLOUDFLARE, 0.95),
            ('#cf-turnstile', CaptchaType.CLOUDFLARE, 0.95),
            ('[data-sitekey]', CaptchaType.RECAPTCHA_V2, 0.8),
            ('[data-hcaptcha-sitekey]', CaptchaType.HCAPTCHA, 0.9),
            ('.geetest_holder', CaptchaType.SLIDER, 0.95),
            ('#TencentCaptcha', CaptchaType.SLIDER, 0.95),
        ]

        try:
            for selector, captcha_type, confidence in element_selectors:
                element = await page.query_selector(selector)
                if element:
                    return CaptchaDetectionResult(
                        detected=True,
                        captcha_type=captcha_type,
                        confidence=confidence,
                        details=f"Found captcha element: {selector}",
                        matched_patterns=[selector]
                    )

        except Exception:
            pass

        return CaptchaDetectionResult(
            detected=False,
            captcha_type=CaptchaType.NONE,
            confidence=0.0
        )

    def is_blocked_response(
        self,
        html: str,
        status_code: int = 200
    ) -> Tuple[bool, Optional[str]]:
        """
        检查是否被阻止访问

        Args:
            html: HTML内容
            status_code: HTTP状态码

        Returns:
            (是否被阻止, 阻止原因)
        """
        block_patterns = [
            (r'Access Denied', "Access Denied"),
            (r'403 Forbidden', "HTTP 403 Forbidden"),
            (r'You have been blocked', "IP blocked"),
            (r'Too many requests', "Rate limited"),
            (r'Rate limit exceeded', "Rate limit exceeded"),
            (r'Please try again later', "Temporarily blocked"),
            (r'Your IP address has been blocked', "IP blocked"),
            (r'Suspicious Activity Detected', "Suspicious activity"),
        ]

        if status_code in (403, 429):
            for pattern, reason in block_patterns:
                if re.search(pattern, html, re.IGNORECASE):
                    return True, reason
            # 即使没有匹配到特定模式，状态码本身也表示被阻止
            if status_code == 403:
                return True, "HTTP 403 Forbidden"
            elif status_code == 429:
                return True, "HTTP 429 Too Many Requests"

        # 检查Cloudflare特殊情况
        if status_code == 503:
            if 'cloudflare' in html.lower() or 'cf-ray' in html.lower():
                return True, "Cloudflare challenge"

        return False, None
