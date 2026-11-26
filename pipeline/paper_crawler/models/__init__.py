"""
Models - Data Models Module

Defines structured models for papers, collections, download status, and crawl tasks.
"""

from pipeline.paper_crawler.models.paper import (
    # Enumerations
    Conference,
    PaperType,
    TaskStatus,
    CrawlStatus,  # Alias for TaskStatus
    SortField,
    SortOrder,
    PaperSource,
    # Models
    Author,
    PaperMetadata,
    Paper,
    PaperFilter,
    PaperSort,
    PaperCollection,
    DownloadStatus,
    CrawlProgress,
    CrawlTask,
    # Convenience functions
    create_paper,
    create_crawl_task,
)

__all__ = [
    # Enumerations
    "Conference",
    "PaperType",
    "TaskStatus",
    "CrawlStatus",
    "SortField",
    "SortOrder",
    "PaperSource",
    # Models
    "Author",
    "PaperMetadata",
    "Paper",
    "PaperFilter",
    "PaperSort",
    "PaperCollection",
    "DownloadStatus",
    "CrawlProgress",
    "CrawlTask",
    # Convenience functions
    "create_paper",
    "create_crawl_task",
]
