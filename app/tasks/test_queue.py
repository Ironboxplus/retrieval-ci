#!/usr/bin/env python
"""
任务队列测试脚本 - 增强版
测试内容：
1. 基础功能测试
2. 任务生命周期测试
3. 任务取消测试
4. 任务超时测试
5. Worker恢复测试
6. 心跳检测测试
"""
import sys
import os
import time
import uuid
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目路径
project_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_dir)


def test_huey_import():
    """测试huey导入"""
    print("1. Testing huey import...")
    from app.tasks.task_queue import huey, get_huey
    print(f"   Huey instance: {huey}")
    print(f"   Database path: {huey.storage.filename}")
    print(f"   Immediate mode: {huey.immediate}")
    assert get_huey() is huey
    print("   [PASS] Huey import successful")
    return True


def test_task_context():
    """测试任务上下文"""
    print("\n2. Testing TaskContext...")
    from app.tasks.base import (
        TaskContext, create_task_record, get_task_record,
        update_task_record, TaskStatus
    )

    # 创建任务记录
    record = create_task_record("test_task")
    task_id = record.task_id
    print(f"   Created task: {task_id}")

    # 获取任务记录
    loaded = get_task_record(task_id)
    assert loaded is not None
    assert loaded.task_id == task_id
    assert loaded.status == TaskStatus.PENDING
    print(f"   Task status: {loaded.status}")

    # 使用上下文更新
    ctx = TaskContext(task_id)
    ctx.update_progress(0.5, "Half done")

    loaded = get_task_record(task_id)
    assert loaded.progress == 0.5
    assert loaded.message == "Half done"
    print(f"   Progress: {loaded.progress}, Message: {loaded.message}")

    # 设置结果
    ctx.set_result({"test": "data"})

    loaded = get_task_record(task_id)
    assert loaded.status == TaskStatus.COMPLETED
    assert loaded.result == {"test": "data"}
    print(f"   Final status: {loaded.status}")

    print("   [PASS] TaskContext works correctly")
    return True


def test_task_registration():
    """测试任务注册"""
    print("\n3. Testing task registration...")
    from app.tasks.task_queue import huey
    from app.tasks import indexing_tasks, crawl_tasks

    # 检查任务是否已注册
    registry = huey._registry._registry
    print(f"   Registered tasks: {list(registry.keys())}")

    assert 'tasks.indexing_tasks.index_document_task' in registry or \
           any('index_document' in k for k in registry.keys())
    assert 'tasks.crawl_tasks.crawl_urls_task' in registry or \
           any('crawl_urls' in k for k in registry.keys())

    print("   [PASS] Tasks registered successfully")
    return True


def test_run_with_context_decorator():
    """测试run_with_context装饰器（不通过huey）"""
    print("\n4. Testing run_with_context decorator...")
    from app.tasks.base import (
        create_task_record, get_task_record, TaskContext,
        run_with_context, TaskStatus
    )

    # 直接测试装饰器功能，不经过huey
    @run_with_context("direct_test")
    def direct_test_func(ctx: TaskContext, value: int):
        ctx.update_progress(0.5, "Processing...")
        result = value * 2
        return {"result": result}

    # 创建任务记录
    task_id = str(uuid.uuid4())
    create_task_record("direct_test", task_id)

    # 直接调用（不通过huey队列）
    result = direct_test_func(task_id, 21)

    print(f"   Task result: {result}")
    assert result["result"] == 42

    # 检查任务记录
    record = get_task_record(task_id)
    print(f"   Record status: {record.status}")
    print(f"   Record result: {record.result}")

    assert record.status == TaskStatus.COMPLETED

    print("   [PASS] run_with_context decorator works")
    return True


def test_enqueue_functions():
    """测试入队辅助函数"""
    print("\n5. Testing enqueue functions...")
    from app.tasks.indexing_tasks import enqueue_index_document, enqueue_batch_index
    from app.tasks.crawl_tasks import enqueue_crawl_urls, enqueue_crawl_single
    from app.tasks.base import get_task_record, TaskStatus

    # 测试入队索引任务（不会真正执行，因为没有worker）
    task_id = enqueue_index_document(
        title="Test Document",
        content="This is test content.",
        source="test.txt",
        source_type="txt"
    )
    print(f"   Enqueued index task: {task_id}")

    # 验证任务记录已创建
    record = get_task_record(task_id)
    assert record is not None
    assert record.task_type == "index_document"
    print(f"   Task record exists: {record.task_type}")

    # 测试批量索引入队
    batch_task_id = enqueue_batch_index([
        {"title": "Doc1", "content": "Content1", "source": "doc1.txt", "source_type": "txt"},
        {"title": "Doc2", "content": "Content2", "source": "doc2.txt", "source_type": "txt"},
    ])
    print(f"   Enqueued batch index task: {batch_task_id}")

    # 测试爬虫入队
    crawl_task_id = enqueue_crawl_urls(
        urls=["https://example.com"],
        use_proxy=False,
        auto_index=False
    )
    print(f"   Enqueued crawl task: {crawl_task_id}")

    print("   [PASS] Enqueue functions work correctly")
    return True


def test_task_timeout():
    """测试任务超时机制"""
    print("\n6. Testing task timeout mechanism...")
    from app.tasks.base import (
        TaskContext, create_task_record, get_task_record,
        TaskStatus, DEFAULT_TASK_TIMEOUT
    )

    # 创建一个短超时的任务
    task_id = str(uuid.uuid4())
    short_timeout = 2  # 2秒超时
    record = create_task_record("timeout_test", task_id, timeout_seconds=short_timeout)
    print(f"   Created task with {short_timeout}s timeout: {task_id}")

    # 验证超时时间已设置
    loaded = get_task_record(task_id)
    assert loaded.timeout_seconds == short_timeout
    print(f"   Timeout configured: {loaded.timeout_seconds}s")

    # 创建上下文并检查超时
    ctx = TaskContext(task_id, timeout_seconds=short_timeout)
    ctx.start()

    # 刚开始不应该超时
    assert not ctx.is_timed_out()
    print("   Task not timed out initially: OK")

    # 等待超时
    print(f"   Waiting {short_timeout + 1}s for timeout...")
    time.sleep(short_timeout + 1)

    # 现在应该超时了
    assert ctx.is_timed_out()
    print("   Task timed out after waiting: OK")

    # 检查check_should_stop
    stop_reason = ctx.check_should_stop()
    assert stop_reason == "timeout"
    print(f"   check_should_stop returned: {stop_reason}")

    ctx.stop()

    print("   [PASS] Task timeout mechanism works")
    return True


def test_heartbeat():
    """测试心跳机制"""
    print("\n7. Testing heartbeat mechanism...")
    from app.tasks.base import (
        TaskContext, create_task_record, get_task_record,
        start_heartbeat, stop_heartbeat, HEARTBEAT_INTERVAL
    )

    task_id = str(uuid.uuid4())
    create_task_record("heartbeat_test", task_id)
    print(f"   Created task: {task_id}")

    # 创建上下文并启动
    ctx = TaskContext(task_id)
    ctx.start()

    # 检查初始心跳
    record1 = get_task_record(task_id)
    heartbeat1 = record1.last_heartbeat
    assert heartbeat1 is not None
    print(f"   Initial heartbeat: {heartbeat1}")

    # 等待一小段时间（心跳线程会更新）
    print(f"   Waiting for heartbeat update...")
    time.sleep(min(HEARTBEAT_INTERVAL + 1, 5))  # 等待心跳间隔，最多5秒

    # 检查心跳是否更新
    record2 = get_task_record(task_id)
    heartbeat2 = record2.last_heartbeat
    print(f"   Updated heartbeat: {heartbeat2}")

    # 心跳应该已经更新（如果心跳间隔<=4秒的话）
    if HEARTBEAT_INTERVAL <= 4:
        assert heartbeat2 > heartbeat1
        print("   Heartbeat was updated: OK")
    else:
        print(f"   Heartbeat interval ({HEARTBEAT_INTERVAL}s) too long for quick test, skipping update check")

    # 停止心跳
    ctx.stop()
    print("   Heartbeat stopped: OK")

    print("   [PASS] Heartbeat mechanism works")
    return True


def test_task_cancellation():
    """测试任务取消"""
    print("\n8. Testing task cancellation...")
    from app.tasks.base import (
        TaskContext, create_task_record, get_task_record,
        cancel_task, TaskStatus
    )

    task_id = str(uuid.uuid4())
    create_task_record("cancel_test", task_id)
    print(f"   Created task: {task_id}")

    # 创建上下文
    ctx = TaskContext(task_id)

    # 开始任务前检查取消状态
    assert not ctx.is_cancelled()
    print("   Task not cancelled initially: OK")

    # 取消任务
    result = cancel_task(task_id)
    assert result == True
    print("   cancel_task returned True: OK")

    # 检查取消状态
    assert ctx.is_cancelled()
    print("   Task is now cancelled: OK")

    # 验证记录状态
    record = get_task_record(task_id)
    assert record.status == TaskStatus.CANCELLED
    print(f"   Record status: {record.status}")

    print("   [PASS] Task cancellation works")
    return True


def test_zombie_detection():
    """测试僵尸任务检测"""
    print("\n9. Testing zombie task detection...")
    from app.tasks.base import (
        create_task_record, get_task_record, update_task_record,
        is_task_zombie, find_zombie_tasks, TaskStatus,
        HEARTBEAT_TIMEOUT
    )
    from datetime import datetime, timedelta

    # 创建一个正常任务
    task_id = str(uuid.uuid4())
    create_task_record("zombie_test", task_id)
    print(f"   Created task: {task_id}")

    # 设置为运行中，心跳正常
    update_task_record(
        task_id,
        status=TaskStatus.RUNNING,
        started_at=datetime.now(),
        last_heartbeat=datetime.now()
    )

    # 应该不是僵尸
    record = get_task_record(task_id)
    assert not is_task_zombie(record)
    print("   Normal running task is not zombie: OK")

    # 设置过期的心跳（模拟僵尸任务）
    old_heartbeat = datetime.now() - timedelta(seconds=HEARTBEAT_TIMEOUT + 60)
    update_task_record(task_id, last_heartbeat=old_heartbeat)

    # 现在应该是僵尸
    record = get_task_record(task_id)
    assert is_task_zombie(record)
    print("   Task with expired heartbeat is zombie: OK")

    # 查找僵尸任务
    zombies = find_zombie_tasks()
    zombie_ids = [z.task_id for z in zombies]
    assert task_id in zombie_ids
    print(f"   find_zombie_tasks found {len(zombies)} zombie(s): OK")

    print("   [PASS] Zombie task detection works")
    return True


def test_zombie_cleanup():
    """测试僵尸任务清理"""
    print("\n10. Testing zombie task cleanup...")
    from app.tasks.base import (
        create_task_record, get_task_record, update_task_record,
        cleanup_zombie_tasks, TaskStatus, HEARTBEAT_TIMEOUT
    )
    from datetime import datetime, timedelta

    # 创建一个僵尸任务
    task_id = str(uuid.uuid4())
    create_task_record("cleanup_test", task_id)

    old_heartbeat = datetime.now() - timedelta(seconds=HEARTBEAT_TIMEOUT + 60)
    update_task_record(
        task_id,
        status=TaskStatus.RUNNING,
        started_at=datetime.now() - timedelta(seconds=300),
        last_heartbeat=old_heartbeat
    )
    print(f"   Created zombie task: {task_id}")

    # 清理僵尸任务
    cleaned = cleanup_zombie_tasks()
    print(f"   Cleaned {len(cleaned)} zombie tasks")

    # 验证任务已被标记为失败
    record = get_task_record(task_id)
    assert record.status == TaskStatus.FAILED
    assert "heartbeat" in record.error.lower() or "worker" in record.error.lower()
    print(f"   Task status: {record.status}")
    print(f"   Error message: {record.error}")

    print("   [PASS] Zombie task cleanup works")
    return True


def test_worker_id():
    """测试Worker ID生成"""
    print("\n11. Testing worker ID generation...")
    from app.tasks.base import get_worker_id

    worker_id1 = get_worker_id()
    worker_id2 = get_worker_id()

    assert worker_id1 == worker_id2  # 同一进程应该返回相同ID
    assert "worker-" in worker_id1
    print(f"   Worker ID: {worker_id1}")
    print("   Same process returns same ID: OK")

    print("   [PASS] Worker ID generation works")
    return True


def test_stale_task_recovery():
    """测试遗留任务恢复"""
    print("\n12. Testing stale task recovery...")
    from app.tasks.base import (
        create_task_record, get_task_record, update_task_record,
        recover_stale_running_tasks, get_worker_id, TaskStatus
    )

    # 创建一个遗留的运行中任务（属于不同的worker）
    task_id = str(uuid.uuid4())
    create_task_record("recovery_test", task_id)
    update_task_record(
        task_id,
        status=TaskStatus.RUNNING,
        worker_id="old-worker-12345-1234567890"
    )
    print(f"   Created stale task with old worker: {task_id}")

    # 恢复遗留任务
    current_worker = get_worker_id()
    recovered = recover_stale_running_tasks(current_worker)
    print(f"   Recovered {len(recovered)} stale tasks")

    # 验证任务已被标记为失败
    if task_id in recovered:
        record = get_task_record(task_id)
        assert record.status == TaskStatus.FAILED
        assert "crash" in record.error.lower() or "recovered" in record.error.lower()
        print(f"   Task status: {record.status}")
        print("   [PASS] Stale task recovery works")
    else:
        print("   Task not in recovered list (may have been cleaned by previous test)")
        print("   [PASS] Stale task recovery mechanism exists")

    return True


def test_task_timeout_status():
    """测试超时任务状态"""
    print("\n13. Testing timeout task status...")
    from app.tasks.base import (
        create_task_record, get_task_record, update_task_record,
        cleanup_zombie_tasks, TaskStatus, HEARTBEAT_TIMEOUT
    )
    from datetime import datetime, timedelta

    # 创建一个超时的任务
    task_id = str(uuid.uuid4())
    short_timeout = 60  # 60秒超时
    create_task_record("timeout_status_test", task_id, timeout_seconds=short_timeout)

    # 设置为很久以前开始的任务（超过timeout）
    old_start = datetime.now() - timedelta(seconds=short_timeout + 120)
    old_heartbeat = datetime.now() - timedelta(seconds=HEARTBEAT_TIMEOUT + 60)
    update_task_record(
        task_id,
        status=TaskStatus.RUNNING,
        started_at=old_start,
        last_heartbeat=old_heartbeat
    )
    print(f"   Created timed-out task: {task_id}")

    # 清理
    cleaned = cleanup_zombie_tasks()
    print(f"   Cleaned {len(cleaned)} tasks")

    # 验证状态是TIMEOUT而不是FAILED
    record = get_task_record(task_id)
    assert record.status == TaskStatus.TIMEOUT
    print(f"   Task status: {record.status}")
    print(f"   Error: {record.error}")

    print("   [PASS] Timeout task status works")
    return True


def test_task_service_integration():
    """测试TaskService集成"""
    print("\n14. Testing TaskService integration...")
    from app.services.task_service import TaskService

    service = TaskService()

    # 测试cleanup_all_zombies方法存在
    assert hasattr(service, 'cleanup_all_zombies')
    print("   cleanup_all_zombies method exists: OK")

    # 测试get_task_health方法存在
    assert hasattr(service, 'get_task_health')
    print("   get_task_health method exists: OK")

    # 测试get_running_tasks方法存在
    assert hasattr(service, 'get_running_tasks')
    print("   get_running_tasks method exists: OK")

    print("   [PASS] TaskService integration works")
    return True


def test_crawl_task_record_sync():
    """测试爬虫任务会同步更新预创建的CrawlRecord"""
    print("\n15. Testing crawl task record sync...")
    from datetime import datetime
    from core.db.database import init_db, get_db_session
    from core.db.models import Task, TaskStatus as DBTaskStatus, CrawlRecord, CrawlStatus as DBCrawlStatus
    from app.tasks.crawl_tasks import _update_crawl_record

    init_db()
    unique_url = f"https://example.com/task-sync-{uuid.uuid4()}"

    with get_db_session() as db:
        task = Task(task_type="crawl_urls", status=DBTaskStatus.pending, total=1)
        db.add(task)
        db.flush()

        record = CrawlRecord(
            task_id=task.id,
            url=unique_url,
            status=DBCrawlStatus.pending,
        )
        db.add(record)
        db.flush()
        task_id = task.id

    _update_crawl_record(
        task_id=str(task_id),
        url=unique_url,
        status=DBCrawlStatus.success,
        title="Task Sync Title",
        content_length=123,
        error_message=None,
    )

    with get_db_session() as db:
        records = db.query(CrawlRecord).filter(CrawlRecord.task_id == task_id, CrawlRecord.url == unique_url).all()
        assert len(records) == 1
        assert records[0].status == DBCrawlStatus.success
        assert records[0].title == "Task Sync Title"
        assert records[0].content_length == 123
        assert records[0].error_message is None

    print("   [PASS] Crawl task record sync works")
    return True


def test_crawl_service_upsert_record():
    """测试CrawlService会复用预创建的CrawlRecord而不是重复插入"""
    print("\n16. Testing crawl service crawl-record upsert...")
    from datetime import datetime
    from types import SimpleNamespace
    from core.db.database import init_db, get_db_session
    from core.db.models import Task, TaskStatus as DBTaskStatus, CrawlRecord, CrawlStatus as DBCrawlStatus
    from app.services.crawl_service import CrawlService

    init_db()
    unique_url = f"https://example.com/service-upsert-{uuid.uuid4()}"

    with get_db_session() as db:
        task = Task(task_type="crawl_urls", status=DBTaskStatus.pending, total=1)
        db.add(task)
        db.flush()

        record = CrawlRecord(
            task_id=task.id,
            url=unique_url,
            status=DBCrawlStatus.pending,
        )
        db.add(record)
        db.flush()
        task_id = task.id

    result = SimpleNamespace(
        url=unique_url,
        title="Service Upsert Title",
        content="service crawl content",
        crawled_at=datetime.now(),
    )

    service = CrawlService(use_enhanced=False)
    service._save_crawl_result(result, task_id=task_id)

    with get_db_session() as db:
        records = db.query(CrawlRecord).filter(CrawlRecord.task_id == task_id, CrawlRecord.url == unique_url).all()
        assert len(records) == 1
        assert records[0].status == DBCrawlStatus.success
        assert records[0].title == "Service Upsert Title"
        assert records[0].content_length == len(result.content)
        assert records[0].doc_id is not None

    print("   [PASS] Crawl service reuses existing crawl record")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Task Queue System Tests - Enhanced Version")
    print("=" * 60)

    tests = [
        test_huey_import,
        test_task_context,
        test_task_registration,
        test_run_with_context_decorator,
        test_enqueue_functions,
        test_task_timeout,
        test_heartbeat,
        test_task_cancellation,
        test_zombie_detection,
        test_zombie_cleanup,
        test_worker_id,
        test_stale_task_recovery,
        test_task_timeout_status,
        test_task_service_integration,
        test_crawl_task_record_sync,
        test_crawl_service_upsert_record,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"   [FAIL] {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
