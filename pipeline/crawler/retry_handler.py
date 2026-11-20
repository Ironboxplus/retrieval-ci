"""
重试处理器 - 提供智能重试机制和指数退避策略
"""
import asyncio
import random
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Set, Any, Awaitable
import logging

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """退避策略枚举"""
    FIXED = "fixed"                 # 固定延迟
    LINEAR = "linear"               # 线性增长
    EXPONENTIAL = "exponential"     # 指数增长
    EXPONENTIAL_JITTER = "exponential_jitter"  # 指数增长 + 随机抖动


class RetryableError(Exception):
    """可重试的错误基类"""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class NetworkError(RetryableError):
    """网络错误"""
    pass


class TimeoutError(RetryableError):
    """超时错误"""
    pass


class RateLimitError(RetryableError):
    """频率限制错误"""
    pass


class CaptchaError(RetryableError):
    """验证码错误"""
    pass


class BlockedError(RetryableError):
    """IP被阻止错误"""
    pass


@dataclass
class RetryConfig:
    """重试配置"""
    # 最大重试次数
    max_retries: int = 3

    # 退避策略
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER

    # 初始延迟（秒）
    initial_delay: float = 1.0

    # 最大延迟（秒）
    max_delay: float = 60.0

    # 指数退避的乘数
    exponential_base: float = 2.0

    # 抖动因子 (0.0 - 1.0)
    jitter_factor: float = 0.1

    # 可重试的HTTP状态码
    retryable_status_codes: Set[int] = field(default_factory=lambda: {
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
        520,  # Cloudflare Unknown Error
        521,  # Cloudflare Web Server Down
        522,  # Cloudflare Connection Timed Out
        523,  # Cloudflare Origin Unreachable
        524,  # Cloudflare A Timeout Occurred
    })

    # 可重试的异常类型
    retryable_exceptions: tuple = (
        NetworkError,
        TimeoutError,
        RateLimitError,
        ConnectionError,
        asyncio.TimeoutError,
    )

    # 不可重试的异常类型（优先级高于retryable_exceptions）
    non_retryable_exceptions: tuple = (
        CaptchaError,
        BlockedError,
    )

    # 是否在验证码错误时重试
    retry_on_captcha: bool = False

    # 是否在被阻止时重试
    retry_on_blocked: bool = False

    # 重试回调函数
    on_retry: Optional[Callable[[int, Exception, float], None]] = None


@dataclass
class RetryState:
    """重试状态跟踪"""
    attempt: int = 0
    total_delay: float = 0.0
    errors: List[Exception] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


class RetryHandler:
    """重试处理器"""

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        初始化重试处理器

        Args:
            config: 重试配置，如果为None则使用默认配置
        """
        self.config = config or RetryConfig()

    def calculate_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """
        计算下次重试的延迟时间

        Args:
            attempt: 当前重试次数（从1开始）
            retry_after: 服务器建议的重试等待时间

        Returns:
            延迟时间（秒）
        """
        # 如果服务器指定了retry-after，优先使用
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self.config.max_delay)

        strategy = self.config.backoff_strategy
        initial = self.config.initial_delay
        base = self.config.exponential_base

        if strategy == BackoffStrategy.FIXED:
            delay = initial

        elif strategy == BackoffStrategy.LINEAR:
            delay = initial * attempt

        elif strategy == BackoffStrategy.EXPONENTIAL:
            delay = initial * (base ** (attempt - 1))

        elif strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            delay = initial * (base ** (attempt - 1))
            # 添加随机抖动
            jitter_range = delay * self.config.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)

        else:
            delay = initial

        # 确保延迟在合理范围内
        return max(0, min(delay, self.config.max_delay))

    def should_retry(
        self,
        error: Exception,
        attempt: int,
        status_code: Optional[int] = None
    ) -> bool:
        """
        判断是否应该重试

        Args:
            error: 发生的异常
            attempt: 当前尝试次数
            status_code: HTTP状态码

        Returns:
            是否应该重试
        """
        # 超过最大重试次数
        if attempt >= self.config.max_retries:
            return False

        # 检查不可重试的异常
        if isinstance(error, self.config.non_retryable_exceptions):
            # 特殊情况：配置允许重试
            if isinstance(error, CaptchaError) and self.config.retry_on_captcha:
                return True
            if isinstance(error, BlockedError) and self.config.retry_on_blocked:
                return True
            return False

        # 检查可重试的异常
        if isinstance(error, self.config.retryable_exceptions):
            return True

        # 检查HTTP状态码
        if status_code and status_code in self.config.retryable_status_codes:
            return True

        # 检查错误消息中的关键字
        error_msg = str(error).lower()
        retryable_keywords = [
            'timeout', 'timed out', 'connection',
            'network', 'temporary', 'unavailable',
            'retry', 'too many requests', 'rate limit'
        ]

        if any(keyword in error_msg for keyword in retryable_keywords):
            return True

        return False

    async def execute_with_retry(
        self,
        func: Callable[[], Awaitable[Any]],
        cancelled_check: Optional[Callable[[], bool]] = None
    ) -> Any:
        """
        执行函数并在失败时自动重试

        Args:
            func: 要执行的异步函数
            cancelled_check: 检查是否取消的函数

        Returns:
            函数执行结果

        Raises:
            Exception: 重试次数用尽后的最后一个异常
        """
        state = RetryState()

        while True:
            state.attempt += 1

            # 检查是否取消
            if cancelled_check and cancelled_check():
                raise asyncio.CancelledError("Operation was cancelled")

            try:
                return await func()

            except Exception as e:
                state.errors.append(e)
                status_code = getattr(e, 'status_code', None)
                retry_after = getattr(e, 'retry_after', None)

                if not self.should_retry(e, state.attempt, status_code):
                    logger.warning(
                        f"Non-retryable error after {state.attempt} attempts: {e}"
                    )
                    raise

                delay = self.calculate_delay(state.attempt, retry_after)
                state.total_delay += delay

                logger.info(
                    f"Retry {state.attempt}/{self.config.max_retries} "
                    f"after {delay:.2f}s delay. Error: {e}"
                )

                # 触发重试回调
                if self.config.on_retry:
                    try:
                        self.config.on_retry(state.attempt, e, delay)
                    except Exception:
                        pass

                # 检查是否取消
                if cancelled_check and cancelled_check():
                    raise asyncio.CancelledError("Operation was cancelled")

                # 等待延迟
                await asyncio.sleep(delay)


class RateLimiter:
    """请求频率限制器"""

    def __init__(
        self,
        requests_per_second: float = 1.0,
        burst_size: int = 1
    ):
        """
        初始化频率限制器

        Args:
            requests_per_second: 每秒允许的请求数
            burst_size: 突发请求的最大数量
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """
        获取请求许可

        Returns:
            实际等待的时间（秒）
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # 补充令牌
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return 0.0
            else:
                # 计算需要等待的时间
                wait_time = (1 - self.tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 0
                self.last_update = time.time()
                return wait_time
