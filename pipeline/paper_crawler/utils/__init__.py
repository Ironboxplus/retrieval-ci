"""
Utils - 工具函数模块

提供日志、配置读取、浏览器协调等通用工具函数。
"""

from pipeline.paper_crawler.utils.logger import (
    # 日志管理器
    CrawlerLogger,
    setup_logger,
    get_logger,
    # 专用日志记录器
    ProgressLogger,
    DownloadLogger,
    # 预配置的日志实例
    scraper_logger,
    downloader_logger,
    converter_logger,
    workflow_logger,
    # 便捷函数
    debug,
    info,
    warning,
    error,
    critical,
    exception,
    # 装饰器
    log_execution_time,
    log_async_execution_time,
)

from pipeline.paper_crawler.utils.browser_coordinator import (
    # 主类
    BrowserCoordinator,
    # 数据类
    LockStatus,
    LockInfo,
    OperationResult,
    QueuedOperation,
    # 装饰器
    browser_lock,
    # 上下文管理器
    browser_session,
    # 便捷函数
    with_browser_lock,
    get_coordinator,
)

__all__ = [
    # 日志管理器
    "CrawlerLogger",
    "setup_logger",
    "get_logger",
    # 专用日志记录器
    "ProgressLogger",
    "DownloadLogger",
    # 预配置的日志实例
    "scraper_logger",
    "downloader_logger",
    "converter_logger",
    "workflow_logger",
    # 便捷函数
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
    # 装饰器
    "log_execution_time",
    "log_async_execution_time",
    # 浏览器协调器
    "BrowserCoordinator",
    "LockStatus",
    "LockInfo",
    "OperationResult",
    "QueuedOperation",
    "browser_lock",
    "browser_session",
    "with_browser_lock",
    "get_coordinator",
]
