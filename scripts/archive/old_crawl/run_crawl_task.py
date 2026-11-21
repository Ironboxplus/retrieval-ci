#!/usr/bin/env python
"""
直接执行爬取任务的脚本
"""
import asyncio
import sqlite3
import json
import sys
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.crawler import BrowserCrawler
from config import settings

def update_db_progress(task_id: int, progress: int, total: int, status: str = "running"):
    """更新数据库中的任务进度"""
    conn = sqlite3.connect('data/db/retrieval.db')
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE tasks SET progress = ?, status = ? WHERE id = ?",
        (progress, status, task_id)
    )
    conn.commit()
    conn.close()

def get_task_urls(task_id: int):
    """从数据库获取任务URL列表"""
    conn = sqlite3.connect('data/db/retrieval.db')
    cursor = conn.cursor()
    cursor.execute('SELECT params_json FROM tasks WHERE id = ?', (task_id,))
    row = cursor.fetchone()
    conn.close()

    if row and row[0]:
        params = json.loads(row[0])
        return params.get('urls', [])
    return []

async def run_crawl(task_id: int, max_concurrent: int = 5, batch_size: int = 50):
    """
    执行爬取任务

    Args:
        task_id: 任务ID
        max_concurrent: 最大并发数
        batch_size: 每批处理的URL数量
    """
    urls = get_task_urls(task_id)
    if not urls:
        print("No URLs found")
        return

    total = len(urls)
    print(f"Starting crawl task {task_id} with {total} URLs")
    print(f"Max concurrent: {max_concurrent}, Batch size: {batch_size}")

    crawler = BrowserCrawler(use_proxy=False)

    success_count = 0
    failed_count = 0

    # 分批处理
    for i in range(0, total, batch_size):
        batch_urls = urls[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\nBatch {batch_num}/{total_batches}: Processing {len(batch_urls)} URLs...")

        try:
            results = await crawler.crawl_urls(batch_urls, max_concurrent=max_concurrent)

            for result in results:
                if result.status == "success":
                    success_count += 1
                else:
                    failed_count += 1
                    print(f"  Failed: {result.url[:60]}... - {result.error}")

            # 更新进度
            progress = i + len(batch_urls)
            update_db_progress(task_id, progress, total)
            print(f"  Progress: {progress}/{total} (Success: {success_count}, Failed: {failed_count})")

        except Exception as e:
            print(f"  Batch error: {e}")
            failed_count += len(batch_urls)

    # 完成任务
    conn = sqlite3.connect('data/db/retrieval.db')
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE tasks SET
           status = 'completed',
           progress = ?,
           completed_at = datetime('now'),
           result_json = ?
           WHERE id = ?""",
        (total, json.dumps({
            "success_count": success_count,
            "failed_count": failed_count,
            "total": total
        }), task_id)
    )
    conn.commit()
    conn.close()

    print(f"\nTask completed!")
    print(f"Total: {total}, Success: {success_count}, Failed: {failed_count}")

if __name__ == "__main__":
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    max_concurrent = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    asyncio.run(run_crawl(task_id, max_concurrent, batch_size))
