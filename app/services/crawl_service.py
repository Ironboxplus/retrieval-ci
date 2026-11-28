"""
爬虫服务 - 封装网页爬取的业务逻辑
"""
import asyncio
import re
import logging
from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime
from urllib.parse import urljoin, urlparse

from sqlalchemy.orm import Session
from bs4 import BeautifulSoup

from core.db import get_db_session
from core.db.models import (
    CrawlRecord, Task, Document,
    CrawlStatus as DBCrawlStatus, TaskStatus as DBTaskStatus,
    SourceType, DataSource, DocumentStatus
)
from config import settings
from pipeline.crawler.browser_crawler import BrowserCrawler, CrawlResult
from pipeline.crawler.enhanced_crawler import (
    EnhancedCrawler, EnhancedCrawlResult, CrawlerConfig,
    CrawlStatusType, ProxyConfig, batch_crawl
)
from pipeline.crawler.captcha_detector import CaptchaType
from pipeline.crawler.retry_handler import RetryConfig, BackoffStrategy

logger = logging.getLogger(__name__)


def _extracted_text_length(content: str) -> int:
    """Character count of extracted plain text, stripping HTML tags."""
    if not content:
        return 0
    return len(BeautifulSoup(content, "html.parser").get_text(separator=" ", strip=True))


class CrawlService:
    """爬虫服务类"""

    def __init__(
        self,
        use_proxy: bool = False,
        use_enhanced: bool = True,
        cancelled_check: Optional[Callable[[], bool]] = None
    ):
        """
        初始化爬虫服务

        Args:
            use_proxy: 是否使用代理
            use_enhanced: 是否使用增强版爬虫（带Captcha检测和重试）
            cancelled_check: 取消检查回调函数
        """
        self._use_proxy = use_proxy
        self._use_enhanced = use_enhanced
        self._cancelled_check = cancelled_check
        self._crawler = None
        self._enhanced_crawler = None

    @property
    def crawler(self) -> BrowserCrawler:
        """懒加载基础爬虫"""
        if self._crawler is None:
            self._crawler = BrowserCrawler(use_proxy=self._use_proxy)
        return self._crawler

    @property
    def enhanced_crawler(self) -> EnhancedCrawler:
        """懒加载增强版爬虫"""
        if self._enhanced_crawler is None:
            config = CrawlerConfig(
                headless=settings.crawler.headless,
                timeout=settings.crawler.timeout,
                max_concurrent=settings.crawler.max_concurrent,
                delay_between_requests=settings.crawler.delay_between_requests,
                retry_config=RetryConfig(
                    max_retries=settings.crawler.retry_times,
                    backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
                    initial_delay=2.0,
                    max_delay=30.0,
                ),
            )
            config.proxy_config.enabled = self._use_proxy
            self._enhanced_crawler = EnhancedCrawler(config)
        return self._enhanced_crawler

    def set_cancelled_check(self, callback: Callable[[], bool]):
        """设置取消检查回调"""
        self._cancelled_check = callback

    def cancel(self):
        """取消爬取操作"""
        if self._enhanced_crawler:
            self._enhanced_crawler.cancel()

    def _index_crawled_document(
        self,
        *,
        doc_id: int,
        result,
        metadata: Dict[str, Any],
        db: Session = None,
    ) -> Dict[str, Any]:
        """Index crawled content through one explicit result.url-based path."""
        from app.services.index_service import get_index_service

        return get_index_service(chunker_type="html").index_document(
            doc_id=doc_id,
            content=result.content,
            title=result.title or result.url,
            source=result.url,
            metadata=metadata,
            db=db,
            source_type="webpage",
            data_source=DataSource.system.value,
            content_type="html",
        )

    def _upsert_crawl_record(
        self,
        session: Session,
        *,
        url: str,
        status: DBCrawlStatus,
        task_id: int = None,
        title: str = None,
        content_length: int = None,
        doc_id: int = None,
        error: str = None,
    ) -> CrawlRecord:
        """Reuse the pre-created crawl row for task-driven crawls instead of duplicating it."""
        record = None
        if task_id is not None:
            record = session.query(CrawlRecord).filter(
                CrawlRecord.task_id == task_id,
                CrawlRecord.url == url,
            ).first()

        if record is None:
            record = CrawlRecord(task_id=task_id, url=url)
            session.add(record)

        record.status = status
        record.title = title
        record.content_length = content_length
        record.doc_id = doc_id
        record.error_message = error
        session.flush()
        return record

    def crawl_url(
        self,
        url: str,
        save_to_db: bool = True,
        auto_index: bool = False,
        task_id: int = None,
        db: Session = None,
        use_enhanced: bool = None
    ) -> Dict[str, Any]:
        """
        爬取单个URL

        Args:
            url: 目标URL
            save_to_db: 是否保存到数据库
            auto_index: 是否自动建立索引
            task_id: 关联的任务ID
            db: 数据库会话
            use_enhanced: 是否使用增强版爬虫（None则使用默认设置）

        Returns:
            爬取结果字典
        """
        use_enhanced = use_enhanced if use_enhanced is not None else self._use_enhanced

        # 运行异步爬取
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            if use_enhanced:
                result = loop.run_until_complete(
                    self.enhanced_crawler.crawl_single(url)
                )
                return self._process_enhanced_result(
                    result, save_to_db, auto_index, task_id, db
                )
            else:
                result = loop.run_until_complete(self.crawler.crawl_single(url))
                return self._process_basic_result(
                    result, save_to_db, auto_index, task_id, db
                )
        finally:
            loop.close()

    def _process_enhanced_result(
        self,
        result: EnhancedCrawlResult,
        save_to_db: bool,
        auto_index: bool,
        task_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """处理增强版爬虫的结果"""
        if result.status != CrawlStatusType.SUCCESS:
            # 记录失败
            if save_to_db:
                self._save_crawl_record(
                    url=result.url,
                    status=self._convert_status(result.status),
                    error=result.error,
                    task_id=task_id,
                    db=db,
                    extra_data={
                        "captcha_detected": result.captcha_detected,
                        "captcha_type": result.captcha_type.value if result.captcha_type else None,
                        "retry_count": result.retry_count,
                        "crawl_time": result.crawl_time,
                    }
                )

            return {
                "success": False,
                "url": result.url,
                "error": result.error,
                "status": result.status.value,
                "captcha_detected": result.captcha_detected,
                "captcha_type": result.captcha_type.value if result.captcha_type else None,
                "retry_count": result.retry_count,
            }

        # 保存结果
        doc_id = None
        if save_to_db:
            doc = self._save_enhanced_crawl_result(result, task_id, db)
            if doc:
                doc_id = doc.id

        # 自动索引
        indexed = False
        if auto_index and doc_id and result.content:
            index_result = self._index_crawled_document(
                doc_id=doc_id,
                result=result,
                metadata={
                    "crawled_at": result.crawled_at.isoformat() if result.crawled_at else None,
                    "retry_count": result.retry_count,
                },
                db=db,
            )
            indexed = index_result.get("success", False)

        return {
            "success": True,
            "url": result.url,
            "title": result.title,
            "content_length": _extracted_text_length(result.content),
            "doc_id": doc_id,
            "indexed": indexed,
            "crawled_at": result.crawled_at.isoformat() if result.crawled_at else None,
            "retry_count": result.retry_count,
            "crawl_time": result.crawl_time,
        }

    def _process_basic_result(
        self,
        result: CrawlResult,
        save_to_db: bool,
        auto_index: bool,
        task_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """处理基础爬虫的结果"""

        if result.status != "success":
            # 记录失败
            if save_to_db:
                self._save_crawl_record(
                    url=result.url,
                    status=DBCrawlStatus.failed,
                    error=result.error,
                    task_id=task_id,
                    db=db
                )

            return {
                "success": False,
                "url": result.url,
                "error": result.error
            }

        # 保存结果
        doc_id = None
        if save_to_db:
            doc = self._save_crawl_result(result, task_id, db)
            if doc:
                doc_id = doc.id

        # 自动索引
        indexed = False
        if auto_index and doc_id and result.content:
            index_result = self._index_crawled_document(
                doc_id=doc_id,
                result=result,
                metadata={"crawled_at": result.crawled_at.isoformat() if result.crawled_at else None},
                db=db,
            )
            indexed = index_result.get("success", False)

        return {
            "success": True,
            "url": result.url,
            "title": result.title,
            "content_length": _extracted_text_length(result.content),
            "doc_id": doc_id,
            "indexed": indexed,
            "crawled_at": result.crawled_at.isoformat() if result.crawled_at else None
        }

    def _convert_status(self, status: CrawlStatusType) -> DBCrawlStatus:
        """转换爬取状态到数据库状态"""
        status_map = {
            CrawlStatusType.PENDING: DBCrawlStatus.pending,
            CrawlStatusType.CRAWLING: DBCrawlStatus.pending,
            CrawlStatusType.SUCCESS: DBCrawlStatus.success,
            CrawlStatusType.FAILED: DBCrawlStatus.failed,
            CrawlStatusType.CAPTCHA: DBCrawlStatus.failed,
            CrawlStatusType.BLOCKED: DBCrawlStatus.failed,
            CrawlStatusType.CANCELLED: DBCrawlStatus.skipped,
        }
        return status_map.get(status, DBCrawlStatus.failed)

    def crawl_urls(
        self,
        urls: List[str],
        max_concurrent: int = 5,
        save_to_db: bool = True,
        auto_index: bool = False,
        task_id: int = None,
        db: Session = None,
        use_enhanced: bool = None
    ) -> Dict[str, Any]:
        """
        批量爬取URL

        Args:
            urls: URL列表
            max_concurrent: 最大并发数
            save_to_db: 是否保存到数据库
            auto_index: 是否自动建立索引
            task_id: 关联的任务ID
            db: 数据库会话
            use_enhanced: 是否使用增强版爬虫

        Returns:
            批量爬取结果
        """
        if not urls:
            return {
                "success": True,
                "total": 0,
                "success_count": 0,
                "failed_count": 0,
                "results": []
            }

        use_enhanced = use_enhanced if use_enhanced is not None else self._use_enhanced

        # 运行异步爬取
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            if use_enhanced:
                results = loop.run_until_complete(
                    self.enhanced_crawler.crawl_urls(urls, max_concurrent=max_concurrent)
                )
                return self._process_enhanced_batch_results(
                    results, save_to_db, auto_index, task_id, db
                )
            else:
                results = loop.run_until_complete(
                    self.crawler.crawl_urls(urls, max_concurrent=max_concurrent)
                )
                return self._process_basic_batch_results(
                    results, save_to_db, auto_index, task_id, db
                )
        finally:
            loop.close()

    def _process_enhanced_batch_results(
        self,
        results: List[EnhancedCrawlResult],
        save_to_db: bool,
        auto_index: bool,
        task_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """处理增强版爬虫的批量结果"""
        success_results = []
        failed_results = []
        captcha_results = []
        blocked_results = []

        for result in results:
            if result.status == CrawlStatusType.SUCCESS:
                doc_id = None
                if save_to_db:
                    doc = self._save_enhanced_crawl_result(result, task_id, db)
                    if doc:
                        doc_id = doc.id

                # 自动索引
                indexed = False
                if auto_index and doc_id and result.content:
                    index_result = self._index_crawled_document(
                        doc_id=doc_id,
                        result=result,
                        metadata={
                            "crawled_at": result.crawled_at.isoformat() if result.crawled_at else None
                        },
                        db=db,
                    )
                    indexed = index_result.get("success", False)

                success_results.append({
                    "url": result.url,
                    "title": result.title,
                    "content_length": _extracted_text_length(result.content),
                    "doc_id": doc_id,
                    "indexed": indexed,
                    "retry_count": result.retry_count,
                    "crawl_time": result.crawl_time,
                })

            elif result.status == CrawlStatusType.CAPTCHA:
                if save_to_db:
                    self._save_crawl_record(
                        url=result.url,
                        status=DBCrawlStatus.failed,
                        error=f"Captcha: {result.captcha_type.value if result.captcha_type else 'unknown'}",
                        task_id=task_id,
                        db=db
                    )
                captcha_results.append({
                    "url": result.url,
                    "captcha_type": result.captcha_type.value if result.captcha_type else None,
                    "error": result.error,
                })

            elif result.status == CrawlStatusType.BLOCKED:
                if save_to_db:
                    self._save_crawl_record(
                        url=result.url,
                        status=DBCrawlStatus.failed,
                        error=f"Blocked: {result.error}",
                        task_id=task_id,
                        db=db
                    )
                blocked_results.append({
                    "url": result.url,
                    "error": result.error,
                })

            else:
                if save_to_db:
                    self._save_crawl_record(
                        url=result.url,
                        status=self._convert_status(result.status),
                        error=result.error,
                        task_id=task_id,
                        db=db
                    )
                failed_results.append({
                    "url": result.url,
                    "error": result.error,
                    "retry_count": result.retry_count,
                })

        # 获取统计信息
        stats = self.enhanced_crawler.get_stats() if self._enhanced_crawler else {}

        return {
            "success": True,
            "total": len(results),
            "success_count": len(success_results),
            "failed_count": len(failed_results),
            "captcha_count": len(captcha_results),
            "blocked_count": len(blocked_results),
            "success_results": success_results,
            "failed_results": failed_results,
            "captcha_results": captcha_results,
            "blocked_results": blocked_results,
            "stats": stats,
        }

    def _process_basic_batch_results(
        self,
        results: List[CrawlResult],
        save_to_db: bool,
        auto_index: bool,
        task_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """处理基础爬虫的批量结果"""
        success_results = []
        failed_results = []

        for result in results:
            if result.status == "success":
                doc_id = None
                if save_to_db:
                    doc = self._save_crawl_result(result, task_id, db)
                    if doc:
                        doc_id = doc.id

                # 自动索引
                indexed = False
                if auto_index and doc_id and result.content:
                    index_result = self._index_crawled_document(
                        doc_id=doc_id,
                        result=result,
                        metadata={"crawled_at": result.crawled_at.isoformat() if result.crawled_at else None},
                        db=db,
                    )
                    indexed = index_result.get("success", False)

                success_results.append({
                    "url": result.url,
                    "title": result.title,
                    "content_length": _extracted_text_length(result.content),
                    "doc_id": doc_id,
                    "indexed": indexed
                })
            else:
                if save_to_db:
                    self._save_crawl_record(
                        url=result.url,
                        status=DBCrawlStatus.failed,
                        error=result.error,
                        task_id=task_id,
                        db=db
                    )

                failed_results.append({
                    "url": result.url,
                    "error": result.error
                })

        return {
            "success": True,
            "total": len(results),
            "success_count": len(success_results),
            "failed_count": len(failed_results),
            "success_results": success_results,
            "failed_results": failed_results
        }

    def _save_crawl_result(
        self,
        result: CrawlResult,
        task_id: int = None,
        db: Session = None
    ) -> Optional[Document]:
        """保存爬取结果到数据库"""
        session = db
        use_context = session is None

        def _save_impl(session: Session) -> Document:
            # 创建文档
            doc = Document(
                title=result.title or result.url,
                source=result.url,
                source_type=SourceType.webpage,
                data_source=DataSource.system,
                url=result.url,
                status=DocumentStatus.pending,
                metadata_json={
                    "crawled_at": result.crawled_at.isoformat() if result.crawled_at else None,
                    "content_length": _extracted_text_length(result.content)
                }
            )
            session.add(doc)
            session.flush()

            self._upsert_crawl_record(
                session,
                url=result.url,
                status=DBCrawlStatus.success,
                task_id=task_id,
                title=result.title,
                content_length=_extracted_text_length(result.content),
                doc_id=doc.id,
                error=None,
            )

            return doc

        if use_context:
            with get_db_session() as session:
                return _save_impl(session)
        else:
            return _save_impl(session)

    def _save_enhanced_crawl_result(
        self,
        result: EnhancedCrawlResult,
        task_id: int = None,
        db: Session = None
    ) -> Optional[Document]:
        """保存增强版爬取结果到数据库"""
        session = db
        use_context = session is None

        def _save_impl(session: Session) -> Document:
            # 创建文档
            doc = Document(
                title=result.title or result.url,
                source=result.url,
                source_type=SourceType.webpage,
                data_source=DataSource.system,
                url=result.url,
                status=DocumentStatus.pending,
                metadata_json={
                    "crawled_at": result.crawled_at.isoformat() if result.crawled_at else None,
                    "content_length": _extracted_text_length(result.content),
                    "retry_count": result.retry_count,
                    "crawl_time": result.crawl_time,
                    "proxy_used": result.proxy_used,
                }
            )
            session.add(doc)
            session.flush()

            self._upsert_crawl_record(
                session,
                url=result.url,
                status=DBCrawlStatus.success,
                task_id=task_id,
                title=result.title,
                content_length=_extracted_text_length(result.content),
                doc_id=doc.id,
                error=None,
            )

            return doc

        if use_context:
            with get_db_session() as session:
                return _save_impl(session)
        else:
            return _save_impl(session)

    def _save_crawl_record(
        self,
        url: str,
        status: DBCrawlStatus,
        error: str = None,
        task_id: int = None,
        db: Session = None,
        extra_data: Dict[str, Any] = None
    ):
        """保存爬取记录"""
        session = db
        use_context = session is None

        def _save_impl(session: Session):
            self._upsert_crawl_record(
                session,
                url=url,
                status=status,
                task_id=task_id,
                error=error,
            )

        if use_context:
            with get_db_session() as session:
                _save_impl(session)
        else:
            _save_impl(session)

    def parse_html_links(
        self,
        html_content: str,
        base_url: str = None,
        same_domain_only: bool = False
    ) -> List[Dict[str, str]]:
        """
        解析HTML中的链接

        Args:
            html_content: HTML内容
            base_url: 基础URL（用于解析相对链接）
            same_domain_only: 是否只返回同域名链接

        Returns:
            链接列表
        """
        soup = BeautifulSoup(html_content, 'lxml')
        links = []

        base_domain = None
        if base_url:
            parsed = urlparse(base_url)
            base_domain = parsed.netloc

        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '').strip()

            # 跳过锚点和javascript链接
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue

            # 跳过邮件和电话链接
            if href.startswith('mailto:') or href.startswith('tel:'):
                continue

            # 处理相对URL
            if base_url and not href.startswith(('http://', 'https://')):
                href = urljoin(base_url, href)

            # 只保留http/https链接
            if not href.startswith(('http://', 'https://')):
                continue

            # 同域名过滤
            if same_domain_only and base_domain:
                parsed = urlparse(href)
                if parsed.netloc != base_domain:
                    continue

            link_info = {
                "url": href,
                "text": a_tag.get_text(strip=True) or href,
                "title": a_tag.get('title', '')
            }

            # 去重
            if not any(l["url"] == href for l in links):
                links.append(link_info)

        return links

    def get_crawl_records(
        self,
        task_id: int = None,
        status: DBCrawlStatus = None,
        url_pattern: str = None,
        offset: int = 0,
        limit: int = 50,
        db: Session = None
    ) -> Tuple[List[CrawlRecord], int]:
        """
        获取爬取记录

        Args:
            task_id: 按任务ID筛选
            status: 按状态筛选
            url_pattern: URL模式匹配
            offset: 偏移量
            limit: 限制数量
            db: 数据库会话

        Returns:
            (记录列表, 总数)
        """
        session = db
        use_context = session is None

        def _get_impl(session: Session) -> Tuple[List[CrawlRecord], int]:
            query = session.query(CrawlRecord)

            if task_id is not None:
                query = query.filter(CrawlRecord.task_id == task_id)
            if status is not None:
                query = query.filter(CrawlRecord.status == status)
            if url_pattern:
                query = query.filter(CrawlRecord.url.ilike(f"%{url_pattern}%"))

            total = query.count()
            records = query.order_by(
                CrawlRecord.created_at.desc()
            ).offset(offset).limit(limit).all()

            return records, total

        if use_context:
            with get_db_session() as session:
                return _get_impl(session)
        else:
            return _get_impl(session)

    def get_crawl_stats(
        self,
        task_id: int = None,
        db: Session = None
    ) -> Dict[str, int]:
        """
        获取爬取统计

        Args:
            task_id: 任务ID筛选
            db: 数据库会话

        Returns:
            统计信息字典
        """
        session = db
        use_context = session is None

        def _stats_impl(session: Session) -> Dict[str, int]:
            base_query = session.query(CrawlRecord)

            if task_id is not None:
                base_query = base_query.filter(CrawlRecord.task_id == task_id)

            total = base_query.count()
            success = base_query.filter(CrawlRecord.status == DBCrawlStatus.success).count()
            failed = base_query.filter(CrawlRecord.status == DBCrawlStatus.failed).count()
            pending = base_query.filter(CrawlRecord.status == DBCrawlStatus.pending).count()
            skipped = base_query.filter(CrawlRecord.status == DBCrawlStatus.skipped).count()

            return {
                "total": total,
                "success": success,
                "failed": failed,
                "pending": pending,
                "skipped": skipped
            }

        if use_context:
            with get_db_session() as session:
                return _stats_impl(session)
        else:
            return _stats_impl(session)

    def check_url_crawled(
        self,
        url: str,
        db: Session = None
    ) -> bool:
        """
        检查URL是否已爬取

        Args:
            url: URL
            db: 数据库会话

        Returns:
            是否已爬取
        """
        session = db
        use_context = session is None

        def _check_impl(session: Session) -> bool:
            record = session.query(CrawlRecord).filter(
                CrawlRecord.url == url,
                CrawlRecord.status == DBCrawlStatus.success
            ).first()
            return record is not None

        if use_context:
            with get_db_session() as session:
                return _check_impl(session)
        else:
            return _check_impl(session)

    def filter_uncrawled_urls(
        self,
        urls: List[str],
        db: Session = None
    ) -> List[str]:
        """
        过滤出未爬取的URL

        Args:
            urls: URL列表
            db: 数据库会话

        Returns:
            未爬取的URL列表
        """
        if not urls:
            return []

        session = db
        use_context = session is None

        def _filter_impl(session: Session) -> List[str]:
            # 查询已成功爬取的URL
            crawled = session.query(CrawlRecord.url).filter(
                CrawlRecord.url.in_(urls),
                CrawlRecord.status == DBCrawlStatus.success
            ).all()

            crawled_set = {r.url for r in crawled}

            return [u for u in urls if u not in crawled_set]

        if use_context:
            with get_db_session() as session:
                return _filter_impl(session)
        else:
            return _filter_impl(session)
