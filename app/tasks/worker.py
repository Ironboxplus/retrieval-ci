#!/usr/bin/env python
"""
任务队列Worker启动脚本
增强版：包含启动清理和信号处理
"""
import sys
import signal
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.tasks.task_queue import huey
from huey.consumer import Consumer

# 导入所有任务以注册
from app.tasks import indexing_tasks, crawl_tasks
from app.tasks.base import on_worker_start, cleanup_zombie_tasks


def signal_handler(signum, frame):
    """处理终止信号"""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    sys.exit(0)


if __name__ == '__main__':
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Starting Huey consumer...")
    print(f"Database: {huey.storage.filename}")
    print(f"Registered tasks:")
    for name in huey._registry._registry.keys():
        print(f"  - {name}")
    print()

    # 启动时清理遗留任务
    print("Performing startup cleanup...")
    on_worker_start()
    print()

    consumer = Consumer(huey, workers=4, worker_type='thread')
    consumer.run()
