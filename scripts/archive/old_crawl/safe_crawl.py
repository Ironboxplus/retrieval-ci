#!/usr/bin/env python
"""
安全爬取脚本 - 独立运行，不依赖Huey队列
使用multiprocessing实现超时机制，避免卡死
"""
import sys
import os
import time
import json
import signal
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from multiprocessing import Process, Queue
import requests

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置
PROXY = "http://127.0.0.1:10808"
TIMEOUT = 30  # 秒
MAX_CONCURRENT = 3
OUTPUT_DIR = PROJECT_ROOT / "data" / "crawled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def crawl_url_simple(url: str, use_proxy: bool = True) -> dict:
    """使用requests简单爬取URL"""
    try:
        proxies = {"http": PROXY, "https": PROXY} if use_proxy else None
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        resp = requests.get(url, proxies=proxies, headers=headers, timeout=TIMEOUT, verify=False)

        return {
            "url": url,
            "status": "success",
            "status_code": resp.status_code,
            "content_length": len(resp.text),
            "title": extract_title(resp.text),
            "content": resp.text[:50000]  # 限制内容大小
        }
    except Exception as e:
        return {
            "url": url,
            "status": "failed",
            "error": str(e)
        }

def extract_title(html: str) -> str:
    """提取HTML标题"""
    import re
    match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def save_result(result: dict):
    """保存爬取结果"""
    url = result.get("url", "")
    # 生成文件名
    filename = url.replace("://", "_").replace("/", "_").replace("?", "_")[:100]
    filename = "".join(c for c in filename if c.isalnum() or c in "_-.")
    filepath = OUTPUT_DIR / f"{filename}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def load_pending_urls() -> list:
    """从数据库加载pending的URL"""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        db_path = PROJECT_ROOT / "data" / "db" / "retrieval.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=engine)
        session = Session()

        # 查询pending的爬取记录
        from core.db.models import CrawlRecord, CrawlStatus
        records = session.query(CrawlRecord).filter(
            CrawlRecord.status == CrawlStatus.pending
        ).limit(500).all()

        urls = [r.url for r in records]
        session.close()
        return urls
    except Exception as e:
        print(f"Error loading URLs from DB: {e}")
        return []

def update_record_status(url: str, status: str, error: str = None):
    """更新数据库记录状态"""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        db_path = PROJECT_ROOT / "data" / "db" / "retrieval.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=engine)
        session = Session()

        from core.db.models import CrawlRecord, CrawlStatus
        record = session.query(CrawlRecord).filter(CrawlRecord.url == url).first()
        if record:
            record.status = CrawlStatus.success if status == "success" else CrawlStatus.failed
            record.error_message = error
            record.crawled_at = datetime.utcnow()
            session.commit()
        session.close()
    except Exception as e:
        print(f"Error updating record: {e}")

def main():
    print("=" * 60)
    print("Safe Crawl Script - 安全爬取脚本")
    print(f"Proxy: {PROXY}")
    print(f"Timeout: {TIMEOUT}s")
    print(f"Concurrent: {MAX_CONCURRENT}")
    print("=" * 60)

    # 禁用SSL警告
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # 加载URL
    urls = load_pending_urls()
    if not urls:
        print("No pending URLs found")
        return

    print(f"\nFound {len(urls)} pending URLs")

    # 统计
    success_count = 0
    failed_count = 0

    # 使用线程池爬取
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {}

        for url in urls:
            future = executor.submit(crawl_url_simple, url, True)
            futures[future] = url

        for i, future in enumerate(futures):
            url = futures[future]
            try:
                result = future.result(timeout=TIMEOUT + 10)

                if result["status"] == "success":
                    success_count += 1
                    save_result(result)
                    update_record_status(url, "success")
                    print(f"[{i+1}/{len(urls)}] OK: {url[:60]}")
                else:
                    failed_count += 1
                    update_record_status(url, "failed", result.get("error"))
                    print(f"[{i+1}/{len(urls)}] FAIL: {url[:60]} - {result.get('error', '')[:50]}")

            except FuturesTimeout:
                failed_count += 1
                update_record_status(url, "failed", "Timeout")
                print(f"[{i+1}/{len(urls)}] TIMEOUT: {url[:60]}")
            except Exception as e:
                failed_count += 1
                update_record_status(url, "failed", str(e))
                print(f"[{i+1}/{len(urls)}] ERROR: {url[:60]} - {e}")

    print("\n" + "=" * 60)
    print(f"Completed: {success_count} success, {failed_count} failed")
    print("=" * 60)

if __name__ == "__main__":
    main()
