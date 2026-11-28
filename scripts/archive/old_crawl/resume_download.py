"""
Resume Download Script - 断点续传下载脚本

功能:
1. 读取 top_1000_papers.json 获取全部论文元数据
2. 扫描 pdf 目录获取已下载的 PDF
3. 计算待下载的论文列表
4. 使用 PDFDownloader 进行异步批量下载
5. 支持断点续传和并发控制
6. 写入下载日志和状态文件
7. 生成下载报告
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paper_crawler.config import Config
from pipeline.paper_crawler.downloaders.pdf_downloader import (
    DownloadConfig,
    DownloadStatus,
    PDFDownloader,
)
from pipeline.paper_crawler.models.paper import Conference, Paper, PaperType
from pipeline.paper_crawler.utils.logger import get_logger, setup_logger


class ResumeDownloader:
    """断点续传下载管理器"""

    def __init__(
        self,
        max_concurrent: int = 5,
        request_delay: float = 1.0,
    ):
        """
        初始化下载管理器

        Args:
            max_concurrent: 最大并发下载数
            request_delay: 请求间隔（秒）
        """
        self.config = Config()
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay

        # 设置日志
        log_dir = self.config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(log_dir=log_dir, level="INFO")
        self.logger = get_logger("resume_download")

        # 路径配置
        self.metadata_file = self.config.metadata_dir / "top_1000_papers.json"
        self.pdf_dir = self.config.pdf_dir
        self.state_file = self.config.log_dir / "download_state.json"
        self.report_file = self.config.metadata_dir / "download_report.json"

        # 下载统计
        self.stats = {
            "total": 0,
            "already_downloaded": 0,
            "to_download": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }
        self.failed_papers: List[Dict] = []

    def load_papers(self) -> List[Paper]:
        """
        从 JSON 文件加载论文元数据

        Returns:
            Paper 对象列表
        """
        if not self.metadata_file.exists():
            self.logger.error(f"Metadata file not found: {self.metadata_file}")
            return []

        self.logger.info(f"Loading papers from {self.metadata_file}")

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        papers = []
        for item in data:
            try:
                # 从 JSON 构建 Paper 对象
                # 处理 conference 字段
                conference_str = item.get("conference", "Other")
                try:
                    conference = Conference(conference_str)
                except ValueError:
                    conference = Conference.OTHER

                # 处理 paper_type 字段
                paper_type_str = item.get("paper_type", "unknown")
                try:
                    paper_type = PaperType(paper_type_str)
                except ValueError:
                    paper_type = PaperType.UNKNOWN

                paper = Paper(
                    id=item["id"],
                    title=item["title"],
                    authors=item.get("authors", []),
                    abstract=item.get("abstract"),
                    year=item.get("year", 2024),
                    conference=conference,
                    paper_type=paper_type,
                    pdf_url=item.get("pdf_url"),
                    paper_page_url=item.get("url"),
                    keywords=item.get("keywords", []),
                )
                papers.append(paper)
            except Exception as e:
                self.logger.warning(f"Failed to parse paper {item.get('id', 'unknown')}: {e}")
                continue

        self.logger.info(f"Loaded {len(papers)} papers from metadata file")
        return papers

    def get_downloaded_ids(self) -> Set[str]:
        """
        获取已下载的论文 ID 集合

        Returns:
            已下载的论文 ID 集合
        """
        downloaded = set()

        if not self.pdf_dir.exists():
            return downloaded

        # 递归扫描所有 PDF 文件
        for pdf_file in self.pdf_dir.rglob("*.pdf"):
            # 文件名（不含扩展名）即为论文 ID
            paper_id = pdf_file.stem
            downloaded.add(paper_id)

        self.logger.info(f"Found {len(downloaded)} already downloaded PDFs")
        return downloaded

    def filter_pending_papers(self, papers: List[Paper], downloaded_ids: Set[str]) -> List[Paper]:
        """
        过滤出待下载的论文

        Args:
            papers: 全部论文列表
            downloaded_ids: 已下载的论文 ID 集合

        Returns:
            待下载的论文列表
        """
        pending = []
        no_pdf_url = 0

        for paper in papers:
            # 检查是否已下载
            if paper.id in downloaded_ids:
                continue

            # 检查是否有 PDF URL
            if not paper.pdf_url:
                no_pdf_url += 1
                continue

            pending.append(paper)

        self.logger.info(
            f"Filtered papers: {len(pending)} to download, "
            f"{len(downloaded_ids)} already downloaded, "
            f"{no_pdf_url} without PDF URL"
        )

        return pending

    async def download_with_delay(self, downloader: PDFDownloader, paper: Paper) -> bool:
        """
        带延迟的单篇论文下载

        Args:
            downloader: PDF 下载器实例
            paper: 论文对象

        Returns:
            下载是否成功
        """
        try:
            result = await downloader.download_paper(paper)
            # 添加请求间隔
            await asyncio.sleep(self.request_delay)
            return result
        except Exception as e:
            self.logger.error(f"Download error for {paper.id}: {e}")
            return False

    async def run_download(self, papers: List[Paper]) -> Dict[str, bool]:
        """
        执行批量下载

        Args:
            papers: 待下载的论文列表

        Returns:
            下载结果字典 {paper_id: success}
        """
        if not papers:
            self.logger.info("No papers to download")
            return {}

        self.logger.info(f"Starting download of {len(papers)} papers")
        self.logger.info(f"Concurrent: {self.max_concurrent}, Delay: {self.request_delay}s")

        # 配置下载器
        download_config = DownloadConfig(
            max_concurrent=self.max_concurrent,
            max_retries=3,
            retry_delay=self.request_delay,
            retry_backoff=2.0,
            verify_integrity=True,
        )

        # 执行下载
        async with PDFDownloader(
            config=self.config,
            download_config=download_config,
        ) as downloader:
            results = await downloader.download_batch(
                papers,
                force=False,
                show_progress=True,
            )

            # 获取下载统计
            stats = await downloader.get_download_stats()
            self.logger.info(f"Download stats: {stats}")

            # 收集失败的论文信息
            for paper in downloader.get_failed_papers():
                task = downloader._tasks.get(paper.id)
                self.failed_papers.append({
                    "id": paper.id,
                    "title": paper.title,
                    "pdf_url": str(paper.pdf_url) if paper.pdf_url else None,
                    "error": task.error_message if task else "Unknown error",
                })

        return results

    def generate_report(self, results: Dict[str, bool]) -> Dict:
        """
        生成下载报告

        Args:
            results: 下载结果字典

        Returns:
            报告字典
        """
        success_count = sum(1 for v in results.values() if v)
        failed_count = sum(1 for v in results.values() if not v)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_papers": self.stats["total"],
                "already_downloaded": self.stats["already_downloaded"],
                "attempted": len(results),
                "success": success_count,
                "failed": failed_count,
                "success_rate": f"{success_count / max(len(results), 1) * 100:.1f}%",
            },
            "failed_papers": self.failed_papers,
        }

        # 保存报告
        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Download report saved to {self.report_file}")
        return report

    async def run(self):
        """主运行入口"""
        self.logger.info("=" * 60)
        self.logger.info("Resume Download Script Started")
        self.logger.info("=" * 60)

        # 1. 加载论文元数据
        papers = self.load_papers()
        if not papers:
            self.logger.error("No papers loaded, exiting")
            return

        self.stats["total"] = len(papers)

        # 2. 获取已下载的论文
        downloaded_ids = self.get_downloaded_ids()
        self.stats["already_downloaded"] = len(downloaded_ids)

        # 3. 过滤待下载论文
        pending_papers = self.filter_pending_papers(papers, downloaded_ids)
        self.stats["to_download"] = len(pending_papers)

        if not pending_papers:
            self.logger.info("All papers already downloaded!")
            return

        self.logger.info(f"Papers to download: {len(pending_papers)}")

        # 4. 执行下载
        results = await self.run_download(pending_papers)

        # 5. 更新统计
        self.stats["success"] = sum(1 for v in results.values() if v)
        self.stats["failed"] = sum(1 for v in results.values() if not v)

        # 6. 生成报告
        report = self.generate_report(results)

        # 7. 打印最终统计
        self.logger.info("=" * 60)
        self.logger.info("Download Complete!")
        self.logger.info(f"Total papers: {self.stats['total']}")
        self.logger.info(f"Already downloaded: {self.stats['already_downloaded']}")
        self.logger.info(f"Attempted: {len(results)}")
        self.logger.info(f"Success: {self.stats['success']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info("=" * 60)

        return report


async def main():
    """主函数"""
    downloader = ResumeDownloader(
        max_concurrent=5,
        request_delay=1.0,  # 1秒间隔避免限流
    )
    await downloader.run()


if __name__ == "__main__":
    asyncio.run(main())
