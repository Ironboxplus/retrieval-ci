"""
PDFDownloader - 论文 PDF 异步批量下载器

功能：
1. 异步批量下载 PDF
2. 支持断点续传
3. 自动重试失败的下载
4. 使用代理（可选）
5. 下载进度日志
6. 文件完整性校验（文件大小/MD5）
7. 按会议/年份组织存储
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiofiles
import aiohttp
from tqdm.asyncio import tqdm

from pipeline.paper_crawler.config import Config
from pipeline.paper_crawler.models.paper import Paper
from pipeline.paper_crawler.utils.logger import get_logger, setup_logger


class DownloadStatus(Enum):
    """下载状态枚举"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # 文件已存在且校验通过


@dataclass
class DownloadTask:
    """下载任务数据类"""
    paper: Paper
    status: DownloadStatus = DownloadStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None
    downloaded_bytes: int = 0
    total_bytes: int = 0
    file_path: Optional[Path] = None
    md5_hash: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于持久化"""
        return {
            "paper_id": self.paper.id,
            "paper_title": self.paper.title,
            "pdf_url": self.paper.pdf_url,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "file_path": str(self.file_path) if self.file_path else None,
            "md5_hash": self.md5_hash,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class DownloadConfig:
    """下载器配置"""
    # 并发配置
    max_concurrent: int = 5

    # 超时配置（秒）
    connect_timeout: int = 30
    read_timeout: int = 300  # PDF 可能较大，给予充足时间
    total_timeout: int = 600

    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0  # 重试间隔基数（秒）
    retry_backoff: float = 2.0  # 退避因子

    # 代理配置
    proxy: Optional[str] = None

    # 下载配置
    chunk_size: int = 8192  # 8KB
    verify_integrity: bool = True

    # 用户代理
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # 请求头
    headers: Dict[str, str] = field(default_factory=lambda: {
        "Accept": "application/pdf,*/*",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })


class PDFDownloader:
    """
    论文 PDF 异步批量下载器

    支持：
    - 异步批量下载
    - 断点续传
    - 自动重试
    - 代理支持
    - 进度日志
    - 文件完整性校验
    - 按会议/年份组织存储

    使用示例:
        async with PDFDownloader(config) as downloader:
            results = await downloader.download_batch(papers)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        download_config: Optional[DownloadConfig] = None,
    ):
        """
        初始化下载器

        Args:
            config: 项目配置
            download_config: 下载器配置
        """
        self.config = config or Config()
        self.download_config = download_config or DownloadConfig()

        # HTTP 会话
        self.session: Optional[aiohttp.ClientSession] = None

        # 信号量控制并发
        self._semaphore: Optional[asyncio.Semaphore] = None

        # 下载状态记录文件
        self._state_file = self.config.log_dir / "download_state.json"

        # 下载任务记录
        self._tasks: Dict[str, DownloadTask] = {}

        # 日志记录器 - use get_logger instead of setup_logger
        self.logger = get_logger("pdf_downloader")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self._close_session()

    async def _create_session(self):
        """创建 HTTP 会话"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(
                connect=self.download_config.connect_timeout,
                sock_read=self.download_config.read_timeout,
                total=self.download_config.total_timeout,
            )

            connector = aiohttp.TCPConnector(
                limit=self.download_config.max_concurrent * 2,
                limit_per_host=self.download_config.max_concurrent,
                force_close=False,
                enable_cleanup_closed=True,
            )

            headers = {
                **self.download_config.headers,
                "User-Agent": self.download_config.user_agent,
            }

            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=headers,
            )

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.download_config.max_concurrent)

    async def _close_session(self):
        """关闭 HTTP 会话"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def _get_save_path(self, paper: Paper) -> Path:
        """
        获取论文 PDF 的存储路径

        路径结构: data/papers/pdf/{conference}/{year}/{paper_id}.pdf

        Args:
            paper: 论文对象

        Returns:
            PDF 存储路径
        """
        # 获取会议名称（从 venue 或 source 提取）
        conference = "unknown"
        if paper.metadata and paper.metadata.venue:
            # 清理会议名称，移除特殊字符
            conference = paper.metadata.venue.lower()
            conference = "".join(c if c.isalnum() or c in "-_" else "_" for c in conference)
            conference = conference.strip("_")
        elif paper.source:
            conference = paper.source.value

        # 获取年份
        year = "unknown"
        if paper.metadata and paper.metadata.year:
            year = str(paper.metadata.year)

        # 清理论文 ID
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in paper.id)
        # 限制文件名长度
        safe_id = safe_id[:200]

        # 构建路径
        save_dir = self.config.pdf_dir / conference / year
        save_dir.mkdir(parents=True, exist_ok=True)

        return save_dir / f"{safe_id}.pdf"

    async def _calculate_md5(self, file_path: Path) -> str:
        """
        异步计算文件的 MD5 哈希值

        Args:
            file_path: 文件路径

        Returns:
            MD5 哈希值（十六进制字符串）
        """
        md5_hash = hashlib.md5()
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(self.download_config.chunk_size):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    async def _verify_file(
        self,
        file_path: Path,
        expected_size: Optional[int] = None,
        expected_md5: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        验证文件完整性

        Args:
            file_path: 文件路径
            expected_size: 预期文件大小
            expected_md5: 预期 MD5 哈希值

        Returns:
            (是否验证通过, 错误信息)
        """
        if not file_path.exists():
            return False, "File does not exist"

        actual_size = file_path.stat().st_size

        # 文件大小为 0 肯定无效
        if actual_size == 0:
            return False, "File is empty"

        # 验证文件大小
        if expected_size is not None and actual_size != expected_size:
            return False, f"Size mismatch: expected {expected_size}, got {actual_size}"

        # 验证 MD5
        if expected_md5 is not None:
            actual_md5 = await self._calculate_md5(file_path)
            if actual_md5 != expected_md5:
                return False, f"MD5 mismatch: expected {expected_md5}, got {actual_md5}"

        # 基本的 PDF 格式检查
        try:
            async with aiofiles.open(file_path, "rb") as f:
                header = await f.read(5)
                if header != b"%PDF-":
                    return False, "Invalid PDF format: missing PDF header"
        except Exception as e:
            return False, f"Failed to read file header: {e}"

        return True, ""

    async def _get_remote_file_info(
        self,
        url: str,
    ) -> Tuple[Optional[int], Optional[str], bool]:
        """
        获取远程文件信息

        Args:
            url: 文件 URL

        Returns:
            (文件大小, ETag/MD5, 是否支持断点续传)
        """
        try:
            async with self.session.head(
                url,
                proxy=self.download_config.proxy,
                allow_redirects=True,
            ) as response:
                if response.status != 200:
                    return None, None, False

                content_length = response.headers.get("Content-Length")
                file_size = int(content_length) if content_length else None

                etag = response.headers.get("ETag")
                content_md5 = response.headers.get("Content-MD5")
                identifier = content_md5 or etag

                accept_ranges = response.headers.get("Accept-Ranges", "")
                supports_resume = accept_ranges.lower() == "bytes"

                return file_size, identifier, supports_resume

        except Exception as e:
            self.logger.warning(f"Failed to get remote file info for {url}: {e}")
            return None, None, False

    async def download_paper(
        self,
        paper: Paper,
        force: bool = False,
    ) -> bool:
        """
        下载单篇论文的 PDF

        Args:
            paper: 论文对象
            force: 是否强制重新下载（即使文件已存在）

        Returns:
            下载是否成功
        """
        if not paper.pdf_url:
            self.logger.warning(f"Paper {paper.id} has no PDF URL")
            return False

        # 获取或创建任务
        task = self._tasks.get(paper.id) or DownloadTask(paper=paper)
        self._tasks[paper.id] = task

        # 获取保存路径
        save_path = self._get_save_path(paper)
        task.file_path = save_path

        # 检查文件是否已存在
        if save_path.exists() and not force:
            # 验证现有文件
            is_valid, error = await self._verify_file(save_path)
            if is_valid:
                task.status = DownloadStatus.SKIPPED
                task.md5_hash = await self._calculate_md5(save_path)
                paper.pdf_path = str(save_path)
                self.logger.info(f"Skipped (already exists): {paper.title[:50]}...")
                return True
            else:
                self.logger.warning(f"Existing file invalid ({error}), re-downloading: {paper.id}")

        # 确保会话已创建
        await self._create_session()

        # 使用信号量控制并发
        async with self._semaphore:
            return await self._download_with_retry(task)

    async def _download_with_retry(self, task: DownloadTask) -> bool:
        """
        带重试机制的下载

        Args:
            task: 下载任务

        Returns:
            下载是否成功
        """
        paper = task.paper
        save_path = task.file_path

        task.start_time = time.time()
        task.status = DownloadStatus.DOWNLOADING

        # 获取远程文件信息
        remote_size, remote_id, supports_resume = await self._get_remote_file_info(
            paper.pdf_url
        )
        task.total_bytes = remote_size or 0

        for attempt in range(self.download_config.max_retries + 1):
            task.retry_count = attempt

            try:
                # 检查是否可以断点续传
                resume_from = 0
                temp_path = save_path.with_suffix(".pdf.part")

                if supports_resume and temp_path.exists():
                    resume_from = temp_path.stat().st_size
                    self.logger.info(
                        f"Resuming download from {resume_from} bytes: {paper.id}"
                    )

                # 构建请求头
                headers = {}
                if resume_from > 0:
                    headers["Range"] = f"bytes={resume_from}-"

                # 发起请求
                async with self.session.get(
                    paper.pdf_url,
                    proxy=self.download_config.proxy,
                    headers=headers,
                    allow_redirects=True,
                ) as response:
                    # 检查响应状态
                    if response.status == 416:  # Range Not Satisfiable
                        # 文件可能已完整下载
                        if temp_path.exists():
                            temp_path.rename(save_path)
                            task.status = DownloadStatus.COMPLETED
                            task.downloaded_bytes = save_path.stat().st_size
                            paper.pdf_path = str(save_path)
                            return True

                    if response.status not in (200, 206):
                        raise aiohttp.ClientError(
                            f"HTTP {response.status}: {response.reason}"
                        )

                    # 获取实际文件大小
                    if response.status == 200:
                        resume_from = 0  # 服务器不支持断点续传
                        content_length = response.headers.get("Content-Length")
                        if content_length:
                            task.total_bytes = int(content_length)
                    elif response.status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if "/" in content_range:
                            total = content_range.split("/")[-1]
                            if total.isdigit():
                                task.total_bytes = int(total)

                    # 写入文件
                    mode = "ab" if resume_from > 0 else "wb"
                    task.downloaded_bytes = resume_from

                    async with aiofiles.open(temp_path, mode) as f:
                        async for chunk in response.content.iter_chunked(
                            self.download_config.chunk_size
                        ):
                            await f.write(chunk)
                            task.downloaded_bytes += len(chunk)

                # 下载完成，重命名文件
                if temp_path.exists():
                    # 先删除可能存在的旧文件
                    if save_path.exists():
                        save_path.unlink()
                    temp_path.rename(save_path)

                # 验证文件完整性
                if self.download_config.verify_integrity:
                    is_valid, error = await self._verify_file(
                        save_path,
                        expected_size=remote_size if remote_size else None,
                    )
                    if not is_valid:
                        raise ValueError(f"File verification failed: {error}")

                # 计算 MD5
                task.md5_hash = await self._calculate_md5(save_path)

                # 更新状态
                task.status = DownloadStatus.COMPLETED
                task.end_time = time.time()
                paper.pdf_path = str(save_path)

                duration = task.end_time - task.start_time
                size_mb = task.downloaded_bytes / (1024 * 1024)
                speed = size_mb / duration if duration > 0 else 0

                self.logger.info(
                    f"Downloaded: {paper.title[:50]}... "
                    f"({size_mb:.2f}MB in {duration:.1f}s, {speed:.2f}MB/s)"
                )

                return True

            except asyncio.CancelledError:
                raise

            except Exception as e:
                task.error_message = str(e)

                if attempt < self.download_config.max_retries:
                    delay = self.download_config.retry_delay * (
                        self.download_config.retry_backoff ** attempt
                    )
                    self.logger.warning(
                        f"Download failed (attempt {attempt + 1}/{self.download_config.max_retries + 1}): "
                        f"{paper.id} - {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    task.status = DownloadStatus.FAILED
                    task.end_time = time.time()
                    self.logger.error(
                        f"Download failed after {attempt + 1} attempts: {paper.id} - {e}"
                    )
                    return False

        return False

    async def download_batch(
        self,
        papers: List[Paper],
        force: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, bool]:
        """
        批量下载论文 PDF

        Args:
            papers: 论文列表
            force: 是否强制重新下载
            show_progress: 是否显示进度条

        Returns:
            下载结果字典 {paper_id: success}
        """
        if not papers:
            return {}

        self.logger.info(f"Starting batch download of {len(papers)} papers")

        # 确保会话已创建
        await self._create_session()

        # 创建下载任务
        async def download_one(paper: Paper) -> Tuple[str, bool]:
            result = await self.download_paper(paper, force=force)
            return paper.id, result

        # 执行下载
        if show_progress:
            tasks = [download_one(paper) for paper in papers]
            results = []
            with tqdm(total=len(tasks), desc="Downloading PDFs", unit="file") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)
        else:
            tasks = [download_one(paper) for paper in papers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        result_dict = {}
        success_count = 0
        failed_count = 0

        for item in results:
            if isinstance(item, Exception):
                failed_count += 1
            elif isinstance(item, tuple):
                paper_id, success = item
                result_dict[paper_id] = success
                if success:
                    success_count += 1
                else:
                    failed_count += 1

        self.logger.info(
            f"Batch download completed: {success_count} succeeded, "
            f"{failed_count} failed"
        )

        # 保存状态
        await self._save_state()

        return result_dict

    async def _save_state(self):
        """保存下载状态到文件"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "tasks": {
                paper_id: task.to_dict()
                for paper_id, task in self._tasks.items()
            },
        }

        try:
            async with aiofiles.open(self._state_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(state, ensure_ascii=False, indent=2))
            self.logger.debug(f"State saved to {self._state_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save state: {e}")

    async def _load_state(self) -> Dict[str, Any]:
        """加载下载状态"""
        if not self._state_file.exists():
            return {}

        try:
            async with aiofiles.open(self._state_file, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            self.logger.warning(f"Failed to load state: {e}")
            return {}

    async def resume_download(
        self,
        papers: Optional[List[Paper]] = None,
    ) -> Dict[str, bool]:
        """
        恢复未完成的下载

        Args:
            papers: 论文列表（如果提供，只恢复列表中的论文）

        Returns:
            下载结果字典 {paper_id: success}
        """
        state = await self._load_state()

        if not state or "tasks" not in state:
            self.logger.info("No previous download state found")
            return {}

        tasks_state = state["tasks"]

        # 找出需要恢复的任务
        papers_to_resume = []

        if papers:
            # 只恢复指定的论文
            paper_dict = {p.id: p for p in papers}
            for paper_id, task_data in tasks_state.items():
                if paper_id in paper_dict:
                    status = DownloadStatus(task_data["status"])
                    if status in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING, DownloadStatus.FAILED):
                        papers_to_resume.append(paper_dict[paper_id])
        else:
            # 恢复所有未完成的任务
            # 需要从元数据恢复 Paper 对象
            self.logger.warning(
                "Resuming without paper list - some papers may not be recoverable"
            )
            for paper_id, task_data in tasks_state.items():
                status = DownloadStatus(task_data["status"])
                if status in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING, DownloadStatus.FAILED):
                    # 尝试从元数据目录恢复
                    metadata_file = self.config.metadata_dir / f"{paper_id}.json"
                    if metadata_file.exists():
                        try:
                            async with aiofiles.open(metadata_file, "r", encoding="utf-8") as f:
                                content = await f.read()
                                paper = Paper.from_json(content)
                                papers_to_resume.append(paper)
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to restore paper {paper_id}: {e}"
                            )

        if not papers_to_resume:
            self.logger.info("No downloads to resume")
            return {}

        self.logger.info(f"Resuming {len(papers_to_resume)} downloads")

        return await self.download_batch(papers_to_resume)

    async def get_download_stats(self) -> Dict[str, Any]:
        """
        获取下载统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "total": len(self._tasks),
            "completed": 0,
            "failed": 0,
            "pending": 0,
            "skipped": 0,
            "downloading": 0,
            "total_bytes": 0,
            "downloaded_bytes": 0,
        }

        for task in self._tasks.values():
            stats["total_bytes"] += task.total_bytes
            stats["downloaded_bytes"] += task.downloaded_bytes

            if task.status == DownloadStatus.COMPLETED:
                stats["completed"] += 1
            elif task.status == DownloadStatus.FAILED:
                stats["failed"] += 1
            elif task.status == DownloadStatus.PENDING:
                stats["pending"] += 1
            elif task.status == DownloadStatus.SKIPPED:
                stats["skipped"] += 1
            elif task.status == DownloadStatus.DOWNLOADING:
                stats["downloading"] += 1

        return stats

    def get_failed_papers(self) -> List[Paper]:
        """
        获取下载失败的论文列表

        Returns:
            失败的论文列表
        """
        return [
            task.paper
            for task in self._tasks.values()
            if task.status == DownloadStatus.FAILED
        ]

    async def cleanup_partial_downloads(self):
        """清理未完成的下载（.part 文件）"""
        count = 0
        for part_file in self.config.pdf_dir.rglob("*.pdf.part"):
            try:
                part_file.unlink()
                count += 1
            except Exception as e:
                self.logger.warning(f"Failed to delete {part_file}: {e}")

        self.logger.info(f"Cleaned up {count} partial download files")

    # ==================== Sync wrappers ====================

    def download(
        self,
        paper: Paper,
        output_dir: Optional[Path] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Synchronous wrapper for downloading a single paper's PDF.

        Args:
            paper: Paper object to download
            output_dir: Optional output directory (uses default if None)
            force: Force re-download even if file exists

        Returns:
            Path to downloaded file, or None if failed
        """
        import asyncio

        if not paper.pdf_url:
            self.logger.warning(f"Paper {paper.id} has no PDF URL")
            return None

        # Determine save path
        if output_dir:
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in paper.id)
            safe_id = safe_id[:200]
            save_path = output_dir / f"{safe_id}.pdf"
        else:
            save_path = self._get_save_path_sync(paper)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists
        if save_path.exists() and not force:
            # Verify existing file
            if self._verify_file_sync(save_path):
                self.logger.info(f"Skipped (already exists): {paper.title[:50]}...")
                return save_path

        # Download using sync HTTP
        try:
            import httpx

            headers = {
                "User-Agent": self.download_config.user_agent,
                **self.download_config.headers,
            }

            with httpx.Client(
                timeout=self.download_config.read_timeout,
                follow_redirects=True,
            ) as client:
                response = client.get(str(paper.pdf_url), headers=headers)
                response.raise_for_status()

                with open(save_path, "wb") as f:
                    f.write(response.content)

            # Verify downloaded file
            if self._verify_file_sync(save_path):
                self.logger.info(f"Downloaded: {paper.title[:50]}...")
                return save_path
            else:
                self.logger.error(f"Downloaded file invalid: {paper.id}")
                return None

        except Exception as e:
            self.logger.error(f"Download failed for {paper.id}: {e}")
            return None

    def _get_save_path_sync(self, paper: Paper) -> Path:
        """Synchronous version of _get_save_path"""
        # Get conference name from direct field
        conference = "unknown"
        if hasattr(paper, 'conference') and paper.conference:
            if hasattr(paper.conference, 'value'):
                conference = paper.conference.value.lower()
            else:
                conference = str(paper.conference).lower()
        conference = "".join(c if c.isalnum() or c in "-_" else "_" for c in conference)
        conference = conference.strip("_")

        # Get year from direct field
        year = "unknown"
        if hasattr(paper, 'year') and paper.year:
            year = str(paper.year)

        # Clean paper ID
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in paper.id)
        safe_id = safe_id[:200]

        # Build path
        save_dir = self.config.pdf_dir / conference / year
        save_dir.mkdir(parents=True, exist_ok=True)

        return save_dir / f"{safe_id}.pdf"

    def _verify_file_sync(self, file_path: Path) -> bool:
        """Synchronous file verification"""
        if not file_path.exists():
            return False

        if file_path.stat().st_size == 0:
            return False

        # Basic PDF format check
        try:
            with open(file_path, "rb") as f:
                header = f.read(5)
                if header != b"%PDF-":
                    return False
        except Exception:
            return False

        return True


# 便捷函数
async def download_papers(
    papers: List[Paper],
    config: Optional[Config] = None,
    download_config: Optional[DownloadConfig] = None,
    force: bool = False,
    show_progress: bool = True,
) -> Dict[str, bool]:
    """
    便捷函数：下载论文列表

    Args:
        papers: 论文列表
        config: 项目配置
        download_config: 下载器配置
        force: 是否强制重新下载
        show_progress: 是否显示进度条

    Returns:
        下载结果字典 {paper_id: success}
    """
    async with PDFDownloader(config, download_config) as downloader:
        return await downloader.download_batch(
            papers,
            force=force,
            show_progress=show_progress,
        )


# 测试和示例代码
async def _test_downloader():
    """测试下载器功能"""
    from pipeline.paper_crawler.models.paper import Author, PaperMetadata, PaperSource

    # 创建测试论文
    test_papers = [
        Paper(
            id="arxiv-2301.00001",
            title="Test Paper 1: Introduction to Machine Learning",
            abstract="This is a test abstract for paper 1.",
            authors=[Author(name="John Doe")],
            source=PaperSource.ARXIV,
            url="https://arxiv.org/abs/2301.00001",
            pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
            metadata=PaperMetadata(venue="ICML", year=2023),
        ),
    ]

    # 配置下载器
    download_config = DownloadConfig(
        max_concurrent=3,
        max_retries=2,
        # proxy="http://127.0.0.1:7890",  # 可选代理
    )

    # 执行下载
    async with PDFDownloader(download_config=download_config) as downloader:
        results = await downloader.download_batch(test_papers)

        # 打印结果
        for paper_id, success in results.items():
            print(f"{paper_id}: {'Success' if success else 'Failed'}")

        # 打印统计
        stats = await downloader.get_download_stats()
        print(f"\nDownload Statistics:")
        print(f"  Total: {stats['total']}")
        print(f"  Completed: {stats['completed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Skipped: {stats['skipped']}")


if __name__ == "__main__":
    asyncio.run(_test_downloader())
