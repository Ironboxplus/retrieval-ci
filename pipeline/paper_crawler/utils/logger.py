"""
Logger - 日志工具模块

使用 loguru 库提供统一的日志配置和管理。
支持：
- 按日期轮转日志文件
- 不同级别日志 (DEBUG, INFO, WARNING, ERROR)
- 格式化输出，便于监控
- 爬取进度、下载状态、错误信息记录
"""

import sys
from pathlib import Path
from typing import Optional, Union
from functools import wraps
from datetime import datetime

from loguru import logger

# 默认日志目录
DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / "data" / "logs"

# 日志格式配置
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# 爬虫专用格式（包含额外上下文信息）
CRAWLER_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{extra[module]: <12} | "
    "{message}"
)


class CrawlerLogger:
    """论文爬取模块的日志管理器

    提供统一的日志配置，支持：
    - 控制台彩色输出
    - 按日期轮转的日志文件
    - 不同级别日志分离
    - 爬取进度追踪
    - 下载状态记录
    - 错误信息详细记录
    """

    _instance: Optional["CrawlerLogger"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """单例模式，确保全局只有一个日志管理器实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        level: str = "INFO",
        rotation: str = "00:00",  # 每天午夜轮转
        retention: str = "30 days",  # 保留30天
        compression: str = "zip",  # 压缩旧日志
        console: bool = True,
        file: bool = True,
    ):
        """初始化日志管理器

        Args:
            log_dir: 日志文件目录，默认为 data/logs/
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            rotation: 日志轮转时间，默认每天午夜
            retention: 日志保留时间，默认30天
            compression: 旧日志压缩格式
            console: 是否输出到控制台
            file: 是否输出到文件
        """
        # 避免重复初始化
        if CrawlerLogger._initialized:
            return

        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.level = level.upper()
        self.rotation = rotation
        self.retention = retention
        self.compression = compression

        # 移除默认处理器
        logger.remove()

        # 添加控制台处理器
        if console:
            logger.add(
                sys.stderr,
                format=CONSOLE_FORMAT,
                level=self.level,
                colorize=True,
                backtrace=True,
                diagnose=True,
            )

        # 添加文件处理器
        if file:
            self._setup_file_handlers()

        CrawlerLogger._initialized = True
        logger.info(f"日志系统初始化完成 | 目录: {self.log_dir} | 级别: {self.level}")

    def _setup_file_handlers(self):
        """设置文件日志处理器"""

        # 主日志文件 - 所有级别
        logger.add(
            self.log_dir / "paper_crawler_{time:YYYY-MM-DD}.log",
            format=FILE_FORMAT,
            level=self.level,
            rotation=self.rotation,
            retention=self.retention,
            compression=self.compression,
            encoding="utf-8",
            enqueue=True,  # 异步写入，提高性能
        )

        # 错误日志文件 - 只记录 ERROR 及以上
        logger.add(
            self.log_dir / "error_{time:YYYY-MM-DD}.log",
            format=FILE_FORMAT,
            level="ERROR",
            rotation=self.rotation,
            retention=self.retention,
            compression=self.compression,
            encoding="utf-8",
            enqueue=True,
            filter=lambda record: record["level"].no >= logger.level("ERROR").no,
        )

        # 爬取进度日志
        logger.add(
            self.log_dir / "crawl_progress_{time:YYYY-MM-DD}.log",
            format=CRAWLER_FORMAT,
            level="INFO",
            rotation=self.rotation,
            retention=self.retention,
            compression=self.compression,
            encoding="utf-8",
            enqueue=True,
            filter=lambda record: record["extra"].get("module") in [
                "scraper", "downloader", "progress"
            ],
        )

        # 下载状态日志
        logger.add(
            self.log_dir / "download_{time:YYYY-MM-DD}.log",
            format=CRAWLER_FORMAT,
            level="DEBUG",
            rotation=self.rotation,
            retention=self.retention,
            compression=self.compression,
            encoding="utf-8",
            enqueue=True,
            filter=lambda record: record["extra"].get("module") == "downloader",
        )

    @staticmethod
    def get_logger(module: str = "general"):
        """获取带有模块上下文的日志记录器

        Args:
            module: 模块名称，用于日志过滤和分类

        Returns:
            带有模块上下文的 logger 实例
        """
        return logger.bind(module=module)


# 便捷函数
def setup_logger(
    log_dir: Optional[Union[str, Path]] = None,
    level: str = "INFO",
    rotation: str = "00:00",
    retention: str = "30 days",
    compression: str = "zip",
    console: bool = True,
    file: bool = True,
) -> CrawlerLogger:
    """设置并返回日志管理器

    Args:
        log_dir: 日志文件目录
        level: 日志级别
        rotation: 日志轮转时间
        retention: 日志保留时间
        compression: 压缩格式
        console: 是否输出到控制台
        file: 是否输出到文件

    Returns:
        CrawlerLogger 实例
    """
    return CrawlerLogger(
        log_dir=log_dir,
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        console=console,
        file=file,
    )


def get_logger(module: str = "general"):
    """获取带有模块上下文的日志记录器

    Args:
        module: 模块名称

    Returns:
        带有模块上下文的 logger 实例
    """
    # 确保日志系统已初始化
    if not CrawlerLogger._initialized:
        CrawlerLogger()
    return CrawlerLogger.get_logger(module)


# 爬取进度记录器
class ProgressLogger:
    """爬取进度记录器

    专门用于记录爬取任务的进度信息。
    """

    def __init__(self, task_name: str, total: int = 0):
        """初始化进度记录器

        Args:
            task_name: 任务名称
            total: 总任务数
        """
        self.task_name = task_name
        self.total = total
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = datetime.now()
        self.logger = get_logger("progress")

        self.logger.info(
            f"[{self.task_name}] 任务开始 | 总数: {self.total}"
        )

    def update(
        self,
        completed: int = 0,
        failed: int = 0,
        skipped: int = 0,
        message: str = "",
    ):
        """更新进度

        Args:
            completed: 新完成的数量
            failed: 新失败的数量
            skipped: 新跳过的数量
            message: 额外信息
        """
        self.completed += completed
        self.failed += failed
        self.skipped += skipped

        progress = (self.completed + self.failed + self.skipped) / max(self.total, 1) * 100

        self.logger.info(
            f"[{self.task_name}] 进度: {progress:.1f}% | "
            f"完成: {self.completed} | 失败: {self.failed} | 跳过: {self.skipped}"
            + (f" | {message}" if message else "")
        )

    def finish(self, message: str = ""):
        """完成任务

        Args:
            message: 完成信息
        """
        elapsed = datetime.now() - self.start_time

        self.logger.info(
            f"[{self.task_name}] 任务完成 | "
            f"耗时: {elapsed} | "
            f"完成: {self.completed} | 失败: {self.failed} | 跳过: {self.skipped}"
            + (f" | {message}" if message else "")
        )

    def log_error(self, error: Exception, context: str = ""):
        """记录错误

        Args:
            error: 异常对象
            context: 错误上下文
        """
        self.logger.error(
            f"[{self.task_name}] 错误 | {context} | {type(error).__name__}: {error}"
        )


# 下载状态记录器
class DownloadLogger:
    """下载状态记录器

    专门用于记录文件下载的状态信息。
    """

    def __init__(self):
        self.logger = get_logger("downloader")

    def start(self, url: str, target: str):
        """记录下载开始

        Args:
            url: 下载URL
            target: 目标路径
        """
        self.logger.info(f"开始下载 | URL: {url} | 目标: {target}")

    def progress(self, url: str, downloaded: int, total: int):
        """记录下载进度

        Args:
            url: 下载URL
            downloaded: 已下载字节数
            total: 总字节数
        """
        if total > 0:
            progress = downloaded / total * 100
            self.logger.debug(
                f"下载进度 | URL: {url} | {downloaded}/{total} ({progress:.1f}%)"
            )

    def success(self, url: str, target: str, size: int, elapsed: float):
        """记录下载成功

        Args:
            url: 下载URL
            target: 目标路径
            size: 文件大小（字节）
            elapsed: 耗时（秒）
        """
        size_mb = size / (1024 * 1024)
        speed = size_mb / max(elapsed, 0.001)
        self.logger.info(
            f"下载成功 | URL: {url} | "
            f"大小: {size_mb:.2f}MB | 耗时: {elapsed:.2f}s | 速度: {speed:.2f}MB/s"
        )

    def failure(self, url: str, error: Exception, retry_count: int = 0):
        """记录下载失败

        Args:
            url: 下载URL
            error: 异常对象
            retry_count: 已重试次数
        """
        self.logger.error(
            f"下载失败 | URL: {url} | 重试: {retry_count} | "
            f"错误: {type(error).__name__}: {error}"
        )

    def skip(self, url: str, reason: str):
        """记录下载跳过

        Args:
            url: 下载URL
            reason: 跳过原因
        """
        self.logger.info(f"跳过下载 | URL: {url} | 原因: {reason}")


# 装饰器：记录函数执行时间
def log_execution_time(func):
    """装饰器：记录函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        elapsed = datetime.now() - start
        logger.debug(f"{func.__name__} 执行完成 | 耗时: {elapsed}")
        return result
    return wrapper


# 装饰器：记录异步函数执行时间
def log_async_execution_time(func):
    """装饰器：记录异步函数执行时间"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = datetime.now()
        result = await func(*args, **kwargs)
        elapsed = datetime.now() - start
        logger.debug(f"{func.__name__} 执行完成 | 耗时: {elapsed}")
        return result
    return wrapper


# 导出的日志实例
scraper_logger = get_logger("scraper")
downloader_logger = get_logger("downloader")
converter_logger = get_logger("converter")
workflow_logger = get_logger("workflow")


# 模块级别的便捷函数
def debug(msg: str, **kwargs):
    """记录 DEBUG 级别日志"""
    logger.debug(msg, **kwargs)


def info(msg: str, **kwargs):
    """记录 INFO 级别日志"""
    logger.info(msg, **kwargs)


def warning(msg: str, **kwargs):
    """记录 WARNING 级别日志"""
    logger.warning(msg, **kwargs)


def error(msg: str, **kwargs):
    """记录 ERROR 级别日志"""
    logger.error(msg, **kwargs)


def critical(msg: str, **kwargs):
    """记录 CRITICAL 级别日志"""
    logger.critical(msg, **kwargs)


def exception(msg: str, **kwargs):
    """记录异常（自动包含堆栈跟踪）"""
    logger.exception(msg, **kwargs)
