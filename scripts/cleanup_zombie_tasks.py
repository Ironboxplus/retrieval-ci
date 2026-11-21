"""
清理僵尸任务脚本

用于清理现有的僵尸任务，修复JSON文件和SQLite数据库之间的状态不一致问题。

使用方法:
    python scripts/cleanup_zombie_tasks.py [--dry-run] [--fix-running]

参数:
    --dry-run: 只显示需要清理的任务，不实际执行清理
    --fix-running: 强制修复所有RUNNING状态的任务到FAILED状态
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.db import get_db_session
from core.db.models import Task, TaskStatus as DBTaskStatus
from app.tasks.base import (
    TaskStatus, TaskRecord,
    get_task_record, update_task_record,
    find_zombie_tasks, cleanup_zombie_tasks,
    TASKS_DIR, HEARTBEAT_TIMEOUT
)


def list_all_running_tasks():
    """列出所有RUNNING状态的任务"""
    print("\n=== JSON文件中的RUNNING任务 ===")
    json_running = []
    for task_file in TASKS_DIR.glob("*.json"):
        task_id = task_file.stem
        record = get_task_record(task_id)
        if record and record.status == TaskStatus.RUNNING:
            json_running.append(record)
            heartbeat_age = "N/A"
            if record.last_heartbeat:
                heartbeat_age = f"{(datetime.now() - record.last_heartbeat).total_seconds():.0f}s"
            print(f"  - Task {task_id}: type={record.task_type}, "
                  f"worker={record.worker_id}, heartbeat_age={heartbeat_age}")

    print(f"\n  总计: {len(json_running)} 个RUNNING任务\n")

    print("=== SQLite数据库中的RUNNING任务 ===")
    with get_db_session() as session:
        db_running = session.query(Task).filter(
            Task.status == DBTaskStatus.running
        ).all()

        for task in db_running:
            heartbeat_age = "N/A"
            if task.last_heartbeat:
                heartbeat_age = f"{(datetime.now() - task.last_heartbeat).total_seconds():.0f}s"
            print(f"  - Task {task.id}: type={task.task_type}, "
                  f"worker={task.worker_id}, heartbeat_age={heartbeat_age}")

        print(f"\n  总计: {len(db_running)} 个RUNNING任务\n")

    return json_running, db_running


def check_sync_status():
    """检查JSON和SQLite之间的同步状态"""
    print("\n=== 检查同步状态 ===")
    out_of_sync = []

    with get_db_session() as session:
        # 获取所有数据库任务
        db_tasks = session.query(Task).all()

        for db_task in db_tasks:
            task_id = str(db_task.id)
            json_record = get_task_record(task_id)

            if json_record is None:
                print(f"  [WARN] Task {task_id}: 存在于数据库但没有JSON文件")
                continue

            # 比较状态
            json_status = json_record.status.value
            db_status = db_task.status.value

            if json_status != db_status:
                out_of_sync.append({
                    'task_id': task_id,
                    'json_status': json_status,
                    'db_status': db_status,
                    'db_task': db_task
                })
                print(f"  [OUT OF SYNC] Task {task_id}: "
                      f"JSON={json_status}, DB={db_status}")

    print(f"\n  不同步的任务数: {len(out_of_sync)}\n")
    return out_of_sync


def fix_out_of_sync_tasks(out_of_sync: list, dry_run: bool = False):
    """
    修复不同步的任务

    以JSON文件状态为准，同步到数据库
    """
    print("\n=== 修复不同步的任务 ===")

    if dry_run:
        print("  [DRY RUN] 以下任务需要修复:")
        for item in out_of_sync:
            print(f"    - Task {item['task_id']}: {item['db_status']} -> {item['json_status']}")
        return

    fixed = 0
    with get_db_session() as session:
        for item in out_of_sync:
            try:
                task = session.query(Task).filter(Task.id == int(item['task_id'])).first()
                if task:
                    task.status = DBTaskStatus(item['json_status'])
                    fixed += 1
                    print(f"  [FIXED] Task {item['task_id']}: {item['db_status']} -> {item['json_status']}")
            except Exception as e:
                print(f"  [ERROR] Task {item['task_id']}: {e}")

    print(f"\n  已修复 {fixed} 个任务\n")


def force_fix_running_tasks(dry_run: bool = False):
    """
    强制修复所有RUNNING状态的任务

    将所有RUNNING任务标记为FAILED
    """
    print("\n=== 强制修复所有RUNNING任务 ===")

    now = datetime.now()

    # 1. 修复JSON文件中的RUNNING任务
    print("\n  处理JSON文件...")
    json_fixed = 0
    for task_file in TASKS_DIR.glob("*.json"):
        task_id = task_file.stem
        record = get_task_record(task_id)
        if record and record.status == TaskStatus.RUNNING:
            if dry_run:
                print(f"    [DRY RUN] Would fix: Task {task_id}")
            else:
                update_task_record(
                    task_id,
                    status=TaskStatus.FAILED,
                    error="Force cleaned by cleanup script",
                    completed_at=now
                )
                print(f"    [FIXED] Task {task_id}")
            json_fixed += 1

    print(f"\n  JSON文件: 已修复 {json_fixed} 个任务")

    # 2. 修复数据库中的RUNNING任务 (以防有些没有JSON文件)
    print("\n  处理SQLite数据库...")
    db_fixed = 0
    with get_db_session() as session:
        running_tasks = session.query(Task).filter(
            Task.status == DBTaskStatus.running
        ).all()

        for task in running_tasks:
            if dry_run:
                print(f"    [DRY RUN] Would fix: Task {task.id}")
            else:
                task.status = DBTaskStatus.failed
                task.error_message = "Force cleaned by cleanup script"
                task.completed_at = now
                print(f"    [FIXED] Task {task.id}")
            db_fixed += 1

    print(f"\n  SQLite数据库: 已修复 {db_fixed} 个任务\n")


def main():
    parser = argparse.ArgumentParser(description='清理僵尸任务')
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示需要清理的任务，不实际执行清理')
    parser.add_argument('--fix-running', action='store_true',
                        help='强制修复所有RUNNING状态的任务到FAILED状态')
    args = parser.parse_args()

    print("=" * 60)
    print("  僵尸任务清理工具")
    print("=" * 60)

    # 1. 列出所有RUNNING任务
    json_running, db_running = list_all_running_tasks()

    # 2. 检查同步状态
    out_of_sync = check_sync_status()

    # 3. 使用内置的僵尸任务清理
    print("\n=== 运行内置僵尸任务清理 ===")
    if args.dry_run:
        zombies = find_zombie_tasks()
        print(f"  [DRY RUN] 发现 {len(zombies)} 个僵尸任务:")
        for z in zombies:
            print(f"    - Task {z.task_id}: {z.task_type}")
    else:
        cleaned = cleanup_zombie_tasks()
        print(f"  已清理 {len(cleaned)} 个僵尸任务")

    # 4. 修复不同步的任务
    if out_of_sync:
        fix_out_of_sync_tasks(out_of_sync, dry_run=args.dry_run)

    # 5. 强制修复所有RUNNING任务
    if args.fix_running:
        force_fix_running_tasks(dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("  清理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
