"""
测试增强版爬虫功能

使用方法:
    python test_enhanced_crawler.py
"""
import asyncio
import sys
import os

# 确保项目路径在搜索路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.crawler import (
    CaptchaDetector,
    CaptchaType,
    CaptchaDetectionResult,
    RetryHandler,
    RetryConfig,
    RateLimiter,
    BackoffStrategy,
    EnhancedCrawler,
    CrawlerConfig,
    CrawlStatusType,
)


def test_captcha_detector():
    """测试Captcha检测器"""
    print("\n=== Testing CaptchaDetector ===")

    detector = CaptchaDetector()

    # 测试reCAPTCHA检测
    html_recaptcha = '''
    <html>
        <head><title>Test Page</title></head>
        <body>
            <div class="g-recaptcha" data-sitekey="xxxx"></div>
            <script src="https://www.google.com/recaptcha/api.js"></script>
        </body>
    </html>
    '''
    result = detector.detect_from_html(html_recaptcha)
    print(f"reCAPTCHA test: detected={result.detected}, type={result.captcha_type.value}, confidence={result.confidence:.2f}")
    assert result.detected, "Should detect reCAPTCHA"
    assert result.captcha_type == CaptchaType.RECAPTCHA_V2

    # 测试Cloudflare检测
    html_cloudflare = '''
    <html>
        <head><title>Just a moment...</title></head>
        <body>
            Checking your browser before accessing...
            <noscript>Enable JavaScript and cookies to continue</noscript>
        </body>
    </html>
    '''
    result = detector.detect_from_html(html_cloudflare)
    print(f"Cloudflare test: detected={result.detected}, type={result.captcha_type.value}, confidence={result.confidence:.2f}")
    assert result.detected, "Should detect Cloudflare"
    assert result.captcha_type == CaptchaType.CLOUDFLARE

    # 测试hCaptcha检测
    html_hcaptcha = '''
    <html>
        <body>
            <div class="h-captcha" data-sitekey="xxxx"></div>
            <script src="https://hcaptcha.com/1/api.js"></script>
        </body>
    </html>
    '''
    result = detector.detect_from_html(html_hcaptcha)
    print(f"hCaptcha test: detected={result.detected}, type={result.captcha_type.value}, confidence={result.confidence:.2f}")
    assert result.detected, "Should detect hCaptcha"
    assert result.captcha_type == CaptchaType.HCAPTCHA

    # 测试无验证码页面
    html_normal = '''
    <html>
        <head><title>Normal Page</title></head>
        <body>
            <h1>Welcome</h1>
            <p>This is a normal page without any captcha.</p>
        </body>
    </html>
    '''
    result = detector.detect_from_html(html_normal)
    print(f"Normal page test: detected={result.detected}, type={result.captcha_type.value}")
    assert not result.detected, "Should not detect captcha on normal page"

    # 测试阻止检测
    is_blocked, reason = detector.is_blocked_response(
        "Access Denied - Your IP has been blocked", 403
    )
    print(f"Block detection test: blocked={is_blocked}, reason={reason}")
    assert is_blocked, "Should detect blocked response"

    print("All CaptchaDetector tests passed!")


def test_retry_handler():
    """测试重试处理器"""
    print("\n=== Testing RetryHandler ===")

    # 测试延迟计算
    config = RetryConfig(
        max_retries=3,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        initial_delay=1.0,
        exponential_base=2.0
    )
    handler = RetryHandler(config)

    delays = [handler.calculate_delay(i) for i in range(1, 5)]
    print(f"Exponential delays: {delays}")
    assert delays[0] == 1.0, "First delay should be 1.0"
    assert delays[1] == 2.0, "Second delay should be 2.0"
    assert delays[2] == 4.0, "Third delay should be 4.0"

    # 测试重试判断
    class TestException(Exception):
        pass

    assert handler.should_retry(ConnectionError("test"), 1), "Should retry on ConnectionError"
    assert handler.should_retry(asyncio.TimeoutError(), 1), "Should retry on TimeoutError"
    assert not handler.should_retry(TestException("test"), 1), "Should not retry on unknown exception"
    assert not handler.should_retry(ConnectionError("test"), 5), "Should not retry after max retries"

    # 测试状态码重试
    assert handler.should_retry(Exception("test"), 1, status_code=503), "Should retry on 503"
    assert handler.should_retry(Exception("test"), 1, status_code=429), "Should retry on 429"

    print("All RetryHandler tests passed!")


def test_rate_limiter():
    """测试频率限制器"""
    print("\n=== Testing RateLimiter ===")

    limiter = RateLimiter(requests_per_second=10.0, burst_size=3)

    async def run() -> tuple[float, float]:
        # 前3个请求应该不需要等待（burst）
        import time
        start = time.time()
        for _ in range(3):
            await limiter.acquire()
        return time.time() - start, await limiter.acquire()

    elapsed, wait_time = asyncio.run(run())
    print(f"3 burst requests took: {elapsed:.3f}s")
    assert elapsed < 0.1, "Burst requests should be immediate"

    # 第4个请求应该需要等待
    print(f"4th request waited: {wait_time:.3f}s")
    assert wait_time > 0, "The request after the burst must wait"

    print("All RateLimiter tests passed!")


def test_crawler_config():
    """测试爬虫配置"""
    print("\n=== Testing CrawlerConfig ===")

    config = CrawlerConfig(
        headless=True,
        timeout=30000,
        max_concurrent=5,
        rate_limit_enabled=True,
        requests_per_second=2.0,
    )

    print(f"Config: headless={config.headless}, timeout={config.timeout}")
    print(f"Rate limit: enabled={config.rate_limit_enabled}, rps={config.requests_per_second}")
    print(f"Retry: max_retries={config.retry_config.max_retries}")

    # 测试代理配置
    proxy_config = config.proxy_config
    proxy_config.enabled = True
    proxy_config.proxies = ["http://proxy1:8080", "http://proxy2:8080", "http://proxy3:8080"]

    # 测试轮询
    proxies = [proxy_config.get_next_proxy() for _ in range(6)]
    print(f"Proxy rotation: {proxies}")
    assert proxies[0] == proxies[3], "Round-robin should cycle"

    # 测试失败标记
    proxy_config.mark_failed("http://proxy1:8080")
    proxy_config.mark_failed("http://proxy1:8080")
    proxy_config.mark_failed("http://proxy1:8080")
    assert "http://proxy1:8080" in proxy_config.failed_proxies, "Proxy should be marked as failed"

    print("All CrawlerConfig tests passed!")


def test_enhanced_crawler_init():
    """测试增强版爬虫初始化"""
    async def _test():
        config = CrawlerConfig(
            headless=True,
            max_concurrent=3,
            detect_captcha=True,
        )

        crawler = EnhancedCrawler(config)

        assert not crawler.is_cancelled()
        crawler.cancel()
        assert crawler.is_cancelled()

        crawler.reset_stats()
        stats = crawler.get_stats()
        assert stats["total_requests"] == 0

    asyncio.run(_test())


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Enhanced Crawler Test Suite")
    print("=" * 60)

    try:
        test_captcha_detector()
        test_retry_handler()
        test_rate_limiter()
        test_crawler_config()
        asyncio.run(test_enhanced_crawler_init())

        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\nTest failed: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
