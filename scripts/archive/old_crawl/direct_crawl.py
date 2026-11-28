"""
直接执行爬取脚本 - 绕过Huey队列

功能:
1. 读取SQLite中pending的crawl任务
2. 直接调用crawler执行爬取
3. 使用代理 127.0.0.1:10808
4. 更新任务状态
5. 带超时机制和错误处理
6. 支持断点续爬
"""
import asyncio
import sys
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import traceback

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import and_
from sqlalchemy.orm import Session

from core.db.database import get_db_session, init_db
from core.db.models import CrawlRecord, CrawlStatus, Task, TaskStatus


# 配置常量
PROXY_SERVER = "127.0.0.1:10808"
TIMEOUT_SECONDS = 30
MAX_CONCURRENT = 3
MAX_RETRIES = 1


class DirectCrawler:
    """直接爬取器 - 使用Playwright进行网页爬取"""

    def __init__(self, proxy: str = PROXY_SERVER, timeout: int = TIMEOUT_SECONDS):
        self.proxy = proxy
        self.timeout = timeout
        self._stop_flag = False

    def stop(self):
        """设置停止标志"""
        self._stop_flag = True
        print("\n[!] 收到停止信号，等待当前任务完成...")

    async def crawl_single(self, url: str) -> dict:
        """
        爬取单个URL

        Args:
            url: 目标URL

        Returns:
            包含爬取结果的字典
        """
        from playwright.async_api import async_playwright

        result = {
            "url": url,
            "title": "",
            "content": "",
            "status": "failed",
            "error": None,
            "content_length": 0
        }

        try:
            async with async_playwright() as p:
                # 配置浏览器选项
                launch_options = {
                    "headless": True,
                }

                # 代理配置
                if self.proxy:
                    launch_options["proxy"] = {
                        "server": f"http://{self.proxy}"
                    }

                browser = await p.chromium.launch(**launch_options)

                try:
                    context = await browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                    page = await context.new_page()

                    # 设置超时 (毫秒)
                    page.set_default_timeout(self.timeout * 1000)

                    # 导航到页面
                    await page.goto(url, wait_until="domcontentloaded")

                    # 等待一小段时间确保内容加载
                    await asyncio.sleep(1)

                    # 提取内容
                    title = await page.title()
                    content = await self._extract_content(page)

                    result["title"] = title or ""
                    result["content"] = content or ""
                    result["content_length"] = len(content) if content else 0
                    result["status"] = "success"

                except Exception as e:
                    result["error"] = str(e)
                    result["status"] = "failed"
                finally:
                    await browser.close()

        except Exception as e:
            result["error"] = str(e)
            result["status"] = "failed"

        return result

    async def _extract_content(self, page) -> str:
        """从页面提取主要文本内容"""
        content = await page.evaluate("""
            () => {
                // 移除脚本和样式
                const elementsToRemove = document.querySelectorAll('script, style, nav, footer, header, aside, .ad, .advertisement, .sidebar');
                elementsToRemove.forEach(el => el.remove());

                // 尝试找到主内容区域
                const mainContent = document.querySelector('main, article, .content, .post, .entry, #content, #main')
                    || document.body;

                // 提取文本
                const text = mainContent.innerText || mainContent.textContent;

                // 清理空白
                return text.replace(/\\s+/g, ' ').trim();
            }
        """)
        return content


def run_crawl_with_timeout(crawler: DirectCrawler, url: str, timeout: int) -> dict:
    """
    使用超时机制运行爬取

    Args:
        crawler: 爬取器实例
        url: 目标URL
        timeout: 超时秒数

    Returns:
        爬取结果字典
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(
            asyncio.wait_for(
                crawler.crawl_single(url),
                timeout=timeout
            )
        )
        return result
    except asyncio.TimeoutError:
        return {
            "url": url,
            "title": "",
            "content": "",
            "status": "failed",
            "error": f"爬取超时 ({timeout}秒)",
            "content_length": 0
        }
    except Exception as e:
        return {
            "url": url,
            "title": "",
            "content": "",
            "status": "failed",
            "error": str(e),
            "content_length": 0
        }
    finally:
        loop.close()


def get_pending_records(db: Session, limit: int = 100) -> List[CrawlRecord]:
    """获取待处理的爬取记录"""
    return db.query(CrawlRecord).filter(
        CrawlRecord.status == CrawlStatus.pending
    ).limit(limit).all()


def update_record_status(
    db: Session,
    record: CrawlRecord,
    status: CrawlStatus,
    title: str = None,
    content_length: int = None,
    error: str = None
):
    """更新爬取记录状态"""
    record.status = status
    if title is not None:
        record.title = title
    if content_length is not None:
        record.content_length = content_length
    if error is not None:
        record.error_message = error
    db.commit()


def process_single_record(
    crawler: DirectCrawler,
    record_id: int,
    url: str,
    timeout: int,
    max_retries: int
) -> dict:
    """
    处理单条爬取记录 (在线程池中运行)

    Args:
        crawler: 爬取器实例
        record_id: 记录ID
        url: 目标URL
        timeout: 超时秒数
        max_retries: 最大重试次数

    Returns:
        包含record_id和结果的字典
    """
    result = None
    attempts = 0

    while attempts <= max_retries:
        attempts += 1
        result = run_crawl_with_timeout(crawler, url, timeout)

        if result["status"] == "success":
            break
        elif attempts <= max_retries:
            print(f"    [!] 重试 {attempts}/{max_retries}: {url[:60]}...")

    return {
        "record_id": record_id,
        "result": result
    }


def run_direct_crawl(
    batch_size: int = 100,
    max_concurrent: int = MAX_CONCURRENT,
    timeout: int = TIMEOUT_SECONDS,
    max_retries: int = MAX_RETRIES
):
    """
    主执行函数 - 读取pending任务并执行爬取

    Args:
        batch_size: 每批处理的记录数
        max_concurrent: 最大并发数
        timeout: 单个URL超时秒数
        max_retries: 失败重试次数
    """
    print("=" * 60)
    print("Direct Crawl - 直接爬取脚本")
    print("=" * 60)
    print(f"配置: 代理={PROXY_SERVER}, 超时={timeout}s, 并发={max_concurrent}, 重试={max_retries}")
    print("-" * 60)

    # 初始化数据库
    init_db()

    # 创建爬取器
    crawler = DirectCrawler(proxy=PROXY_SERVER, timeout=timeout)

    # 设置信号处理
    stop_requested = [False]

    def signal_handler(signum, frame):
        stop_requested[0] = True
        crawler.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    total_processed = 0
    total_success = 0
    total_failed = 0

    while not stop_requested[0]:
        # 获取待处理记录
        with get_db_session() as db:
            pending_records = get_pending_records(db, batch_size)

            if not pending_records:
                print("\n[*] 没有待处理的爬取任务")
                break

            print(f"\n[*] 找到 {len(pending_records)} 条待处理记录")

            # 收集记录信息 (在session内)
            records_info = [
                {"id": r.id, "url": r.url}
                for r in pending_records
            ]

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []

            for info in records_info:
                if stop_requested[0]:
                    break

                future = executor.submit(
                    process_single_record,
                    crawler,
                    info["id"],
                    info["url"],
                    timeout,
                    max_retries
                )
                futures.append(future)

            # 收集结果并更新数据库
            for i, future in enumerate(futures):
                if stop_requested[0]:
                    break

                try:
                    data = future.result(timeout=timeout + 10)  # 额外10秒缓冲
                    record_id = data["record_id"]
                    result = data["result"]

                    # 更新数据库
                    with get_db_session() as db:
                        record = db.query(CrawlRecord).filter(
                            CrawlRecord.id == record_id
                        ).first()

                        if record:
                            if result["status"] == "success":
                                update_record_status(
                                    db, record,
                                    CrawlStatus.success,
                                    title=result["title"],
                                    content_length=result["content_length"]
                                )
                                total_success += 1
                                status_icon = "[OK]"
                            else:
                                update_record_status(
                                    db, record,
                                    CrawlStatus.failed,
                                    error=result["error"]
                                )
                                total_failed += 1
                                status_icon = "[X]"

                            total_processed += 1

                            # 输出进度
                            url_display = result["url"][:50] + "..." if len(result["url"]) > 50 else result["url"]
                            print(f"  {status_icon} [{total_processed}] {url_display}")
                            if result["error"]:
                                print(f"       Error: {result['error'][:80]}")

                except FuturesTimeoutError:
                    print(f"  [X] 任务超时")
                    total_failed += 1
                    total_processed += 1
                except Exception as e:
                    print(f"  [X] 处理异常: {e}")
                    total_failed += 1
                    total_processed += 1

    # 输出统计
    print("\n" + "=" * 60)
    print("爬取完成统计")
    print("=" * 60)
    print(f"  总处理: {total_processed}")
    print(f"  成功: {total_success}")
    print(f"  失败: {total_failed}")
    if total_processed > 0:
        print(f"  成功率: {total_success / total_processed * 100:.1f}%")
    print("=" * 60)


def show_status():
    """显示当前爬取任务状态"""
    init_db()

    with get_db_session() as db:
        pending = db.query(CrawlRecord).filter(
            CrawlRecord.status == CrawlStatus.pending
        ).count()

        success = db.query(CrawlRecord).filter(
            CrawlRecord.status == CrawlStatus.success
        ).count()

        failed = db.query(CrawlRecord).filter(
            CrawlRecord.status == CrawlStatus.failed
        ).count()

        total = pending + success + failed

    print("=" * 40)
    print("爬取任务状态")
    print("=" * 40)
    print(f"  待处理 (pending): {pending}")
    print(f"  成功 (success):   {success}")
    print(f"  失败 (failed):    {failed}")
    print(f"  总计:             {total}")
    print("=" * 40)


def reset_failed():
    """将失败的任务重置为pending状态"""
    init_db()

    with get_db_session() as db:
        count = db.query(CrawlRecord).filter(
            CrawlRecord.status == CrawlStatus.failed
        ).update({CrawlRecord.status: CrawlStatus.pending})
        db.commit()

    print(f"[*] 已将 {count} 条失败记录重置为pending状态")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="直接执行爬取脚本")
    parser.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=["run", "status", "reset"],
        help="执行动作: run=执行爬取, status=查看状态, reset=重置失败任务"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=100,
        help="每批处理的记录数 (默认: 100)"
    )
    parser.add_argument(
        "--concurrent", "-c",
        type=int,
        default=MAX_CONCURRENT,
        help=f"最大并发数 (默认: {MAX_CONCURRENT})"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=TIMEOUT_SECONDS,
        help=f"单个URL超时秒数 (默认: {TIMEOUT_SECONDS})"
    )
    parser.add_argument(
        "--retries", "-r",
        type=int,
        default=MAX_RETRIES,
        help=f"失败重试次数 (默认: {MAX_RETRIES})"
    )

    args = parser.parse_args()

    if args.action == "status":
        show_status()
    elif args.action == "reset":
        reset_failed()
    else:
        run_direct_crawl(
            batch_size=args.batch,
            max_concurrent=args.concurrent,
            timeout=args.timeout,
            max_retries=args.retries
        )
