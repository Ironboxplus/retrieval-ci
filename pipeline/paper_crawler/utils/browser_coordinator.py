"""
Browser Coordinator - 浏览器协调器

解决多个 subagent 同时操作 Chrome 浏览器的竞争问题。

设计思路：
1. 使用文件锁机制 (filelock) 确保同一时间只有一个进程操作浏览器
2. 使用异步队列管理浏览器操作请求
3. 提供统一的浏览器操作接口
4. 支持超时控制和优雅降级

使用示例：
    # 方式一：上下文管理器（推荐）
    async with BrowserCoordinator() as browser:
        snapshot = await browser.navigate_and_snapshot("https://example.com")

    # 方式二：手动管理锁
    coordinator = BrowserCoordinator()
    try:
        await coordinator.acquire(timeout=30.0)
        result = await coordinator.execute_with_lock(some_operation)
    finally:
        coordinator.release()

    # 方式三：使用装饰器
    @browser_lock
    async def my_browser_operation():
        # 执行浏览器操作
        pass
"""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import filelock

# 尝试导入项目日志模块，如果失败则使用标准日志
try:
    # 直接从 logger 模块导入，避免触发 paper_crawler 包的初始化
    import sys
    from pathlib import Path as _Path
    _utils_dir = _Path(__file__).parent
    if str(_utils_dir) not in sys.path:
        sys.path.insert(0, str(_utils_dir))
    from logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str = "browser_coordinator"):
        """获取标准日志记录器（降级方案）"""
        _logger = logging.getLogger(name)
        if not _logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
        return _logger

# 类型定义
T = TypeVar("T")
AsyncCallable = Callable[..., Any]  # 异步可调用类型

# 日志记录器
logger = get_logger("browser_coordinator")


class LockStatus(Enum):
    """锁状态枚举"""
    IDLE = "idle"           # 空闲
    WAITING = "waiting"     # 等待获取锁
    ACQUIRED = "acquired"   # 已获取锁
    RELEASED = "released"   # 已释放锁
    TIMEOUT = "timeout"     # 超时
    ERROR = "error"         # 错误


@dataclass
class LockInfo:
    """锁信息"""
    holder_id: str                          # 持有者ID
    acquired_at: Optional[datetime] = None  # 获取时间
    status: LockStatus = LockStatus.IDLE    # 状态
    wait_time: float = 0.0                  # 等待时间（秒）
    operation: str = ""                     # 操作描述


@dataclass
class OperationResult:
    """操作结果"""
    success: bool                           # 是否成功
    data: Any = None                        # 返回数据
    error: Optional[str] = None             # 错误信息
    duration: float = 0.0                   # 执行时长（秒）
    lock_wait_time: float = 0.0             # 锁等待时间（秒）


@dataclass
class QueuedOperation:
    """队列中的操作"""
    id: str                                 # 操作ID
    operation: Callable                     # 操作函数
    priority: int = 0                       # 优先级（数字越大优先级越高）
    created_at: datetime = field(default_factory=datetime.now)
    timeout: float = 60.0                   # 超时时间


class BrowserCoordinator:
    """
    浏览器协调器 - 确保串行访问

    使用文件锁机制确保多个进程/线程安全地访问浏览器资源。
    支持异步操作、超时控制、操作队列和优雅降级。

    Attributes:
        LOCK_FILE: 锁文件路径
        DEFAULT_TIMEOUT: 默认超时时间（秒）
        POLL_INTERVAL: 锁轮询间隔（秒）
    """

    # 锁文件路径（使用项目data目录）
    LOCK_FILE = Path("E:/JETB/pythons/sf/retrival/data/.browser.lock")

    # 默认超时时间
    DEFAULT_TIMEOUT = 60.0

    # 锁轮询间隔
    POLL_INTERVAL = 0.1

    # 单例实例
    _instance: Optional["BrowserCoordinator"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """单例模式 - 确保全局只有一个协调器实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        lock_file: Optional[Union[str, Path]] = None,
        default_timeout: float = DEFAULT_TIMEOUT,
        enable_queue: bool = True,
    ):
        """
        初始化浏览器协调器

        Args:
            lock_file: 锁文件路径，默认使用 LOCK_FILE
            default_timeout: 默认锁超时时间（秒）
            enable_queue: 是否启用操作队列
        """
        # 避免重复初始化
        if BrowserCoordinator._initialized:
            return

        self.lock_file = Path(lock_file) if lock_file else self.LOCK_FILE
        self.default_timeout = default_timeout
        self.enable_queue = enable_queue

        # 确保锁文件目录存在
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        # 创建文件锁（用于跨进程协调）
        self._file_lock = filelock.FileLock(str(self.lock_file))

        # 创建异步锁（用于同一进程内的协程协调）
        self._async_lock = asyncio.Lock()

        # 锁持有计数（用于跟踪当前是否持有锁）
        self._lock_held = False

        # 当前锁信息
        self._lock_info: Optional[LockInfo] = None

        # 操作队列
        self._operation_queue: asyncio.Queue[QueuedOperation] = asyncio.Queue()

        # 队列处理任务
        self._queue_processor: Optional[asyncio.Task] = None

        # 统计信息
        self._stats = {
            "total_acquisitions": 0,
            "total_releases": 0,
            "total_timeouts": 0,
            "total_operations": 0,
            "total_wait_time": 0.0,
        }

        # 实例标识
        self._instance_id = str(uuid.uuid4())[:8]

        BrowserCoordinator._initialized = True
        logger.info(f"浏览器协调器初始化完成 | 实例ID: {self._instance_id} | 锁文件: {self.lock_file}")

    @property
    def is_locked(self) -> bool:
        """检查当前是否持有锁"""
        return self._lock_held

    @property
    def lock_info(self) -> Optional[LockInfo]:
        """获取当前锁信息"""
        return self._lock_info

    @property
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()

    async def acquire(
        self,
        timeout: Optional[float] = None,
        operation: str = "",
    ) -> bool:
        """
        获取浏览器操作权

        使用双重锁机制：
        1. asyncio.Lock - 用于同一进程内的协程协调
        2. FileLock - 用于跨进程协调

        Args:
            timeout: 超时时间（秒），None 表示使用默认超时
            operation: 操作描述（用于日志）

        Returns:
            bool: 是否成功获取锁

        Raises:
            TimeoutError: 获取锁超时
        """
        if timeout is None:
            timeout = self.default_timeout

        holder_id = f"{self._instance_id}-{int(time.time() * 1000)}"

        # 使用局部变量记录锁信息，避免多个协程共享时的竞态条件
        lock_info = LockInfo(
            holder_id=holder_id,
            status=LockStatus.WAITING,
            operation=operation,
        )

        start_time = time.monotonic()
        logger.debug(f"尝试获取浏览器锁 | 持有者: {holder_id} | 超时: {timeout}s | 操作: {operation}")

        try:
            # 第一步：获取异步锁（用于进程内协程协调）
            remaining_timeout = timeout
            try:
                await asyncio.wait_for(
                    self._async_lock.acquire(),
                    timeout=remaining_timeout
                )
            except asyncio.TimeoutError:
                lock_info.status = LockStatus.TIMEOUT
                lock_info.wait_time = time.monotonic() - start_time
                self._stats["total_timeouts"] += 1
                logger.warning(f"获取异步锁超时 | 持有者: {holder_id} | 等待: {lock_info.wait_time:.2f}s")
                return False

            # 更新剩余超时时间
            elapsed = time.monotonic() - start_time
            remaining_timeout = timeout - elapsed

            if remaining_timeout <= 0:
                # 异步锁获取成功但超时了，释放并返回
                self._async_lock.release()
                lock_info.status = LockStatus.TIMEOUT
                lock_info.wait_time = elapsed
                self._stats["total_timeouts"] += 1
                logger.warning(f"获取浏览器锁超时 | 持有者: {holder_id} | 等待: {elapsed:.2f}s")
                return False

            # 第二步：获取文件锁（用于跨进程协调）
            try:
                # 在线程池中执行文件锁获取（避免阻塞事件循环）
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._file_lock.acquire(timeout=remaining_timeout)
                    ),
                    timeout=remaining_timeout + 0.1  # 留一点余量
                )
            except (asyncio.TimeoutError, filelock.Timeout):
                # 文件锁获取失败，释放异步锁
                self._async_lock.release()
                lock_info.status = LockStatus.TIMEOUT
                lock_info.wait_time = time.monotonic() - start_time
                self._stats["total_timeouts"] += 1
                logger.warning(f"获取文件锁超时 | 持有者: {holder_id} | 等待: {lock_info.wait_time:.2f}s")
                return False

            # 成功获取两个锁，现在可以安全地更新共享状态
            lock_info.status = LockStatus.ACQUIRED
            lock_info.acquired_at = datetime.now()
            lock_info.wait_time = time.monotonic() - start_time

            self._lock_held = True
            self._lock_info = lock_info  # 只在成功时更新共享的锁信息

            self._stats["total_acquisitions"] += 1
            self._stats["total_wait_time"] += lock_info.wait_time

            logger.info(
                f"成功获取浏览器锁 | 持有者: {holder_id} | "
                f"等待: {lock_info.wait_time:.2f}s | 操作: {operation}"
            )
            return True

        except Exception as e:
            # 发生异常时确保释放已获取的锁
            if self._async_lock.locked():
                try:
                    self._async_lock.release()
                except RuntimeError:
                    pass  # 锁可能不是由当前任务持有
            lock_info.status = LockStatus.ERROR
            logger.error(f"获取浏览器锁时发生错误 | 持有者: {holder_id} | 错误: {e}")
            raise

    def release(self) -> None:
        """
        释放浏览器操作权

        释放异步锁和文件锁，允许其他协程/进程获取锁。
        """
        if not self._lock_held:
            logger.warning("尝试释放未持有的锁")
            return

        holder_id = self._lock_info.holder_id if self._lock_info else "unknown"

        try:
            # 释放文件锁
            if self._file_lock.is_locked:
                self._file_lock.release()

            # 释放异步锁
            if self._async_lock.locked():
                self._async_lock.release()

            self._lock_held = False

            if self._lock_info:
                self._lock_info.status = LockStatus.RELEASED

            self._stats["total_releases"] += 1
            logger.info(f"释放浏览器锁 | 持有者: {holder_id}")

        except Exception as e:
            logger.error(f"释放浏览器锁时发生错误 | 持有者: {holder_id} | 错误: {e}")
            raise
        finally:
            self._lock_info = None

    async def execute_with_lock(
        self,
        operation: Callable[..., T],
        *args,
        timeout: Optional[float] = None,
        operation_name: str = "",
        **kwargs,
    ) -> OperationResult:
        """
        在锁保护下执行操作

        自动获取锁、执行操作、释放锁。支持同步和异步操作。

        Args:
            operation: 要执行的操作函数
            *args: 操作函数的位置参数
            timeout: 锁超时时间
            operation_name: 操作名称（用于日志）
            **kwargs: 操作函数的关键字参数

        Returns:
            OperationResult: 操作结果
        """
        if not operation_name:
            operation_name = getattr(operation, "__name__", "anonymous")

        lock_start = time.monotonic()

        # 获取锁
        acquired = await self.acquire(timeout=timeout, operation=operation_name)

        if not acquired:
            return OperationResult(
                success=False,
                error="获取浏览器锁超时",
                lock_wait_time=time.monotonic() - lock_start,
            )

        lock_wait_time = time.monotonic() - lock_start
        operation_start = time.monotonic()

        try:
            # 执行操作
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                # 同步操作在线程池中执行
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: operation(*args, **kwargs))

            self._stats["total_operations"] += 1

            return OperationResult(
                success=True,
                data=result,
                duration=time.monotonic() - operation_start,
                lock_wait_time=lock_wait_time,
            )

        except Exception as e:
            logger.error(f"执行操作时发生错误 | 操作: {operation_name} | 错误: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                duration=time.monotonic() - operation_start,
                lock_wait_time=lock_wait_time,
            )

        finally:
            self.release()

    async def navigate_and_snapshot(
        self,
        url: str,
        timeout: float = 30.0,
        wait_for_load: bool = True,
    ) -> Dict[str, Any]:
        """
        导航并获取页面快照（原子操作）

        在锁保护下执行：导航到URL -> 等待加载 -> 获取快照。
        这是一个原子操作，确保在整个过程中浏览器不会被其他进程干扰。

        Args:
            url: 目标URL
            timeout: 操作超时时间（秒）
            wait_for_load: 是否等待页面加载完成

        Returns:
            dict: 包含快照信息的字典
                - success: bool, 是否成功
                - url: str, 导航的URL
                - snapshot: dict, 页面快照（如果成功）
                - error: str, 错误信息（如果失败）
                - duration: float, 操作耗时
        """
        async def _navigate_and_snapshot_impl():
            """内部实现：导航并获取快照"""
            # 这里需要与实际的 MCP Chrome DevTools 集成
            # 以下是模拟实现，实际使用时需要调用真正的浏览器API

            result = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "snapshot": None,
            }

            try:
                # 模拟导航操作
                # 实际实现应该调用 mcp__chrome-devtools__navigate_page
                logger.info(f"导航到页面 | URL: {url}")
                await asyncio.sleep(0.1)  # 模拟网络延迟

                if wait_for_load:
                    # 等待页面加载
                    # 实际实现应该调用 mcp__chrome-devtools__wait_for
                    await asyncio.sleep(0.1)

                # 获取快照
                # 实际实现应该调用 mcp__chrome-devtools__take_snapshot
                result["snapshot"] = {
                    "title": f"Page at {url}",
                    "elements_count": 0,
                    "captured_at": datetime.now().isoformat(),
                }

                return result

            except Exception as e:
                logger.error(f"导航并获取快照失败 | URL: {url} | 错误: {e}")
                raise

        operation_result = await self.execute_with_lock(
            _navigate_and_snapshot_impl,
            timeout=timeout,
            operation_name=f"navigate_and_snapshot({url})",
        )

        return {
            "success": operation_result.success,
            "url": url,
            "snapshot": operation_result.data.get("snapshot") if operation_result.data else None,
            "error": operation_result.error,
            "duration": operation_result.duration,
            "lock_wait_time": operation_result.lock_wait_time,
        }

    async def queue_operation(
        self,
        operation: Callable,
        priority: int = 0,
        timeout: float = 60.0,
    ) -> str:
        """
        将操作加入队列

        Args:
            operation: 要执行的操作
            priority: 优先级（数字越大优先级越高）
            timeout: 超时时间

        Returns:
            str: 操作ID
        """
        if not self.enable_queue:
            raise RuntimeError("操作队列未启用")

        op_id = str(uuid.uuid4())[:8]

        queued_op = QueuedOperation(
            id=op_id,
            operation=operation,
            priority=priority,
            timeout=timeout,
        )

        await self._operation_queue.put(queued_op)
        logger.debug(f"操作已加入队列 | ID: {op_id} | 优先级: {priority}")

        return op_id

    async def start_queue_processor(self) -> None:
        """启动队列处理器"""
        if self._queue_processor is not None:
            logger.warning("队列处理器已在运行")
            return

        async def _process_queue():
            logger.info("队列处理器启动")
            while True:
                try:
                    queued_op = await self._operation_queue.get()

                    logger.debug(f"处理队列操作 | ID: {queued_op.id}")

                    await self.execute_with_lock(
                        queued_op.operation,
                        timeout=queued_op.timeout,
                        operation_name=f"queued_{queued_op.id}",
                    )

                    self._operation_queue.task_done()

                except asyncio.CancelledError:
                    logger.info("队列处理器停止")
                    break
                except Exception as e:
                    logger.error(f"处理队列操作时发生错误 | 错误: {e}")

        self._queue_processor = asyncio.create_task(_process_queue())

    async def stop_queue_processor(self) -> None:
        """停止队列处理器"""
        if self._queue_processor is None:
            return

        self._queue_processor.cancel()
        try:
            await self._queue_processor
        except asyncio.CancelledError:
            pass
        self._queue_processor = None

    def get_queue_size(self) -> int:
        """获取队列中待处理的操作数量"""
        return self._operation_queue.qsize()

    async def __aenter__(self) -> "BrowserCoordinator":
        """异步上下文管理器入口"""
        acquired = await self.acquire(operation="context_manager")
        if not acquired:
            raise TimeoutError("无法在超时时间内获取浏览器锁")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口"""
        self.release()

    @classmethod
    def reset(cls) -> None:
        """重置单例实例（用于测试）

        警告：此方法仅用于测试目的。在生产环境中不应调用。

        注意：由于 Windows 文件锁的限制，在同一进程中重置后可能无法立即
        重新获取锁。跨进程的锁定功能不受影响。如果需要在测试中多次创建
        新实例，建议使用 subprocess 或在不同的测试进程中运行。
        """
        if cls._instance is not None:
            instance = cls._instance

            # 重置状态标志
            if hasattr(instance, '_lock_held'):
                instance._lock_held = False
            if hasattr(instance, '_lock_info'):
                instance._lock_info = None

            # 释放异步锁
            try:
                if hasattr(instance, '_async_lock') and instance._async_lock.locked():
                    instance._async_lock.release()
            except Exception:
                pass

            # 释放文件锁
            try:
                if hasattr(instance, '_file_lock') and instance._file_lock.is_locked:
                    instance._file_lock.release()
            except Exception:
                pass

            # 清理单例引用（保留锁对象以便重用）
            cls._initialized = False

            # 注意：不删除 cls._instance，因为在 Windows 上同一进程需要
            # 重用相同的 FileLock 对象才能正常工作

            logger.info("浏览器协调器状态已重置")

    @classmethod
    def force_reset(cls) -> None:
        """强制完全重置（仅用于测试，可能导致锁状态不一致）

        警告：此方法会完全销毁单例实例。在 Windows 上，这可能导致
        同一进程无法再次获取文件锁，直到进程重启。
        """
        if cls._instance is not None:
            instance = cls._instance

            # 释放所有锁
            try:
                if hasattr(instance, '_async_lock') and instance._async_lock.locked():
                    instance._async_lock.release()
            except Exception:
                pass

            try:
                if hasattr(instance, '_file_lock') and instance._file_lock.is_locked:
                    instance._file_lock.release()
            except Exception:
                pass

            # 完全销毁实例
            cls._instance = None
            cls._initialized = False

            # 强制垃圾回收
            import gc
            gc.collect()

            logger.info("浏览器协调器已强制重置")


def browser_lock(
    timeout: Optional[float] = None,
    operation_name: Optional[str] = None,
):
    """
    浏览器锁装饰器

    用于装饰需要浏览器访问权限的异步函数。

    Args:
        timeout: 锁超时时间
        operation_name: 操作名称

    Usage:
        @browser_lock(timeout=30.0)
        async def my_browser_operation():
            # 执行浏览器操作
            pass
    """
    def decorator(func: AsyncCallable) -> AsyncCallable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            coordinator = BrowserCoordinator()
            name = operation_name or func.__name__

            result = await coordinator.execute_with_lock(
                func,
                *args,
                timeout=timeout,
                operation_name=name,
                **kwargs,
            )

            if not result.success:
                raise RuntimeError(f"浏览器操作失败: {result.error}")

            return result.data

        return wrapper
    return decorator


@asynccontextmanager
async def browser_session(
    timeout: float = 60.0,
    operation: str = "",
):
    """
    浏览器会话上下文管理器

    提供一个更简洁的方式来管理浏览器锁。

    Args:
        timeout: 锁超时时间
        operation: 操作描述

    Usage:
        async with browser_session(timeout=30.0, operation="scrape_page") as acquired:
            if acquired:
                # 执行浏览器操作
                pass
    """
    coordinator = BrowserCoordinator()
    acquired = await coordinator.acquire(timeout=timeout, operation=operation)

    try:
        yield acquired
    finally:
        if acquired:
            coordinator.release()


# 便捷函数
async def with_browser_lock(
    operation: Callable[..., T],
    *args,
    timeout: float = 60.0,
    **kwargs,
) -> OperationResult:
    """
    在浏览器锁保护下执行操作（便捷函数）

    Args:
        operation: 要执行的操作
        *args: 位置参数
        timeout: 超时时间
        **kwargs: 关键字参数

    Returns:
        OperationResult: 操作结果
    """
    coordinator = BrowserCoordinator()
    return await coordinator.execute_with_lock(
        operation,
        *args,
        timeout=timeout,
        **kwargs,
    )


def get_coordinator() -> BrowserCoordinator:
    """获取全局浏览器协调器实例"""
    return BrowserCoordinator()


# 模块导出
__all__ = [
    # 主类
    "BrowserCoordinator",
    # 数据类
    "LockStatus",
    "LockInfo",
    "OperationResult",
    "QueuedOperation",
    # 装饰器
    "browser_lock",
    # 上下文管理器
    "browser_session",
    # 便捷函数
    "with_browser_lock",
    "get_coordinator",
]
