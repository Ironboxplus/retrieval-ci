"""
任务基础设施 - 任务上下文、状态管理、装饰器
增强版：包含超时机制、心跳检测、Worker崩溃恢复
修复版：确保JSON文件和SQLite数据库双存储同步
"""
import uuid
import json
import functools
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any, Callable, Dict, List
from enum import Enum
from dataclasses import dataclass, field, asdict
import threading
import atexit

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"  # 新增：超时状态


# 默认任务超时时间(秒)
DEFAULT_TASK_TIMEOUT = 3600  # 1小时
# 心跳间隔(秒)
HEARTBEAT_INTERVAL = 30
# 心跳超时(秒) - 超过此时间没有心跳则认为任务僵死
HEARTBEAT_TIMEOUT = 120
# 资源采样间隔(秒)
RESOURCE_SAMPLE_INTERVAL = 3


@dataclass
class TaskRecord:
    """任务记录 - 增强版"""
    task_id: str
    task_type: str
    status: TaskStatus
    progress: float = 0.0
    message: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None  # 新增：任务开始时间
    completed_at: Optional[datetime] = None
    # 新增字段
    worker_id: Optional[str] = None  # Worker进程标识
    last_heartbeat: Optional[datetime] = None  # 最后心跳时间
    timeout_seconds: int = DEFAULT_TASK_TIMEOUT  # 任务超时时间

    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        if self.last_heartbeat:
            data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data


# 任务记录存储目录
TASKS_DIR = Path(settings.data_dir) / "tasks" / "records"
TASKS_DIR.mkdir(parents=True, exist_ok=True)

# 内存中的取消标记
_cancelled_tasks: set = set()
_cancelled_lock = threading.Lock()

# Worker标识 - 每个Worker进程唯一
_worker_id: Optional[str] = None
_worker_id_lock = threading.Lock()

# 心跳线程相关
_heartbeat_threads: Dict[str, threading.Thread] = {}
_heartbeat_stop_events: Dict[str, threading.Event] = {}
_heartbeat_lock = threading.Lock()
_resource_threads: Dict[str, threading.Thread] = {}
_resource_stop_events: Dict[str, threading.Event] = {}
_resource_lock = threading.Lock()


def _sync_task_to_db(task_id: str, **kwargs) -> bool:
    """
    同步任务状态到SQLite数据库

    这是核心同步函数，确保JSON文件和SQLite数据库保持一致

    Args:
        task_id: 任务ID (可能是字符串格式的整数)
        **kwargs: 要更新的字段，支持:
            - status: TaskStatus 枚举
            - progress: float (0.0-1.0)
            - message: str
            - result: Any (会转为JSON)
            - error: str
            - started_at: datetime
            - completed_at: datetime
            - worker_id: str
            - last_heartbeat: datetime

    Returns:
        是否成功更新数据库
    """
    try:
        # 尝试将 task_id 转换为整数 (数据库使用整数ID)
        try:
            db_task_id = int(task_id)
        except (ValueError, TypeError):
            # 如果task_id不是整数格式，可能是UUID，数据库中可能不存在
            return False

        # 延迟导入避免循环依赖
        from core.db import get_db_session
        from core.db.models import Task, TaskStatus as DBTaskStatus

        with get_db_session() as session:
            task = session.query(Task).filter(Task.id == db_task_id).first()
            if task is None:
                return False

            # 映射并更新字段
            if 'status' in kwargs:
                status_value = kwargs['status']
                if isinstance(status_value, TaskStatus):
                    # 将 base.TaskStatus 映射到 db.models.TaskStatus
                    task.status = DBTaskStatus(status_value.value)
                elif isinstance(status_value, str):
                    task.status = DBTaskStatus(status_value)

            if 'progress' in kwargs:
                # JSON中是0.0-1.0，数据库中是整数百分比
                progress_float = kwargs['progress']
                if task.total and task.total > 0:
                    task.progress = int(progress_float * task.total)
                else:
                    # 如果没有total，用百分比
                    task.progress = int(progress_float * 100)

            if 'result' in kwargs:
                task.result_json = kwargs['result']

            if 'error' in kwargs:
                task.error_message = kwargs['error']

            if 'started_at' in kwargs:
                task.started_at = kwargs['started_at']

            if 'completed_at' in kwargs:
                task.completed_at = kwargs['completed_at']

            if 'worker_id' in kwargs:
                task.worker_id = kwargs['worker_id']

            if 'last_heartbeat' in kwargs:
                task.last_heartbeat = kwargs['last_heartbeat']

            # session会在with块结束时自动commit
            return True

    except Exception as e:
        print(f"Error syncing task {task_id} to database: {e}")
        return False


def _db_task_exists(task_id: str) -> bool:
    """
    Return True when a numeric task id still has a parent SQLite task row.

    Non-numeric task ids belong to file-only/background flows and are treated as
    valid because they do not participate in the relational runtime tables.
    """
    try:
        db_task_id = int(task_id)
    except (ValueError, TypeError):
        return True

    try:
        from core.db import get_db_session
        from core.db.models import Task

        with get_db_session() as session:
            return session.query(Task.id).filter(Task.id == db_task_id).first() is not None
    except Exception:
        return False


def get_worker_id() -> str:
    """获取当前Worker的唯一ID"""
    global _worker_id
    with _worker_id_lock:
        if _worker_id is None:
            # 使用进程ID和时间戳生成唯一ID
            _worker_id = f"worker-{os.getpid()}-{int(time.time())}"
        return _worker_id


def _get_task_file(task_id: str) -> Path:
    """获取任务记录文件路径"""
    return TASKS_DIR / f"{task_id}.json"


def create_task_record(task_type: str, task_id: str = None,
                       timeout_seconds: int = DEFAULT_TASK_TIMEOUT) -> TaskRecord:
    """
    创建任务记录

    Args:
        task_type: 任务类型
        task_id: 可选的任务ID，如不提供则自动生成
        timeout_seconds: 任务超时时间(秒)

    Returns:
        创建的任务记录
    """
    if task_id is None:
        task_id = str(uuid.uuid4())

    record = TaskRecord(
        task_id=task_id,
        task_type=task_type,
        status=TaskStatus.PENDING,
        timeout_seconds=timeout_seconds
    )

    # 保存到文件
    task_file = _get_task_file(task_id)
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)

    return record


def get_task_record(task_id: str) -> Optional[TaskRecord]:
    """
    获取任务记录

    Args:
        task_id: 任务ID

    Returns:
        任务记录，如不存在则返回None
    """
    task_file = _get_task_file(task_id)
    if not task_file.exists():
        return None

    try:
        with open(task_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return TaskRecord(
            task_id=data['task_id'],
            task_type=data['task_type'],
            status=TaskStatus(data['status']),
            progress=data.get('progress', 0.0),
            message=data.get('message', ''),
            result=data.get('result'),
            error=data.get('error'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            worker_id=data.get('worker_id'),
            last_heartbeat=datetime.fromisoformat(data['last_heartbeat']) if data.get('last_heartbeat') else None,
            timeout_seconds=data.get('timeout_seconds', DEFAULT_TASK_TIMEOUT)
        )
    except Exception as e:
        print(f"Error loading task record {task_id}: {e}")
        return None


def update_task_record(task_id: str, **kwargs) -> Optional[TaskRecord]:
    """
    更新任务记录 (同时更新JSON文件和SQLite数据库)

    Args:
        task_id: 任务ID
        **kwargs: 要更新的字段

    Returns:
        更新后的任务记录
    """
    record = get_task_record(task_id)
    if record is None:
        return None

    for key, value in kwargs.items():
        if hasattr(record, key):
            setattr(record, key, value)

    record.updated_at = datetime.now()

    # 1. 更新JSON文件
    task_file = _get_task_file(task_id)
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)

    # 2. 同步更新SQLite数据库
    _sync_task_to_db(task_id, **kwargs)

    return record


def cancel_task(task_id: str) -> bool:
    """
    取消任务

    Args:
        task_id: 任务ID

    Returns:
        是否成功标记为取消
    """
    with _cancelled_lock:
        _cancelled_tasks.add(task_id)

    # 停止心跳线程
    stop_heartbeat(task_id)

    record = get_task_record(task_id)
    if record and record.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        update_task_record(task_id, status=TaskStatus.CANCELLED,
                          completed_at=datetime.now())
        return True
    return False


def start_heartbeat(task_id: str):
    """
    启动任务心跳线程

    Args:
        task_id: 任务ID
    """
    with _heartbeat_lock:
        if task_id in _heartbeat_threads:
            return  # 已经有心跳线程

        stop_event = threading.Event()
        _heartbeat_stop_events[task_id] = stop_event

        def heartbeat_loop():
            while not stop_event.is_set():
                try:
                    update_task_record(task_id, last_heartbeat=datetime.now())
                except Exception as e:
                    print(f"Heartbeat error for task {task_id}: {e}")
                stop_event.wait(HEARTBEAT_INTERVAL)

        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
        _heartbeat_threads[task_id] = thread


def stop_heartbeat(task_id: str):
    """
    停止任务心跳线程

    Args:
        task_id: 任务ID
    """
    with _heartbeat_lock:
        if task_id in _heartbeat_stop_events:
            _heartbeat_stop_events[task_id].set()
            del _heartbeat_stop_events[task_id]
        if task_id in _heartbeat_threads:
            del _heartbeat_threads[task_id]


def start_resource_sampler(task_id: str, phase: str = "task.running"):
    """
    启动任务资源采样线程。

    仅对有数据库任务ID的任务生效，UUID形式的离线任务会自动跳过。
    """
    with _resource_lock:
        if task_id in _resource_threads:
            return

        stop_event = threading.Event()
        _resource_stop_events[task_id] = stop_event

        def sample_loop():
            from app.services.task_runtime import record_task_resource_sample

            while not stop_event.is_set():
                try:
                    record_task_resource_sample(task_id, phase=phase)
                except Exception as e:
                    print(f"Resource sampler error for task {task_id}: {e}")
                stop_event.wait(RESOURCE_SAMPLE_INTERVAL)

        thread = threading.Thread(target=sample_loop, daemon=True)
        thread.start()
        _resource_threads[task_id] = thread


def stop_resource_sampler(task_id: str):
    """停止任务资源采样线程。"""
    with _resource_lock:
        if task_id in _resource_stop_events:
            _resource_stop_events[task_id].set()
            del _resource_stop_events[task_id]
        if task_id in _resource_threads:
            del _resource_threads[task_id]


def is_task_zombie(record: TaskRecord) -> bool:
    """
    检查任务是否为僵尸任务

    Args:
        record: 任务记录

    Returns:
        是否为僵尸任务
    """
    if record.status != TaskStatus.RUNNING:
        return False

    now = datetime.now()

    # 检查心跳超时
    if record.last_heartbeat:
        heartbeat_age = (now - record.last_heartbeat).total_seconds()
        if heartbeat_age > HEARTBEAT_TIMEOUT:
            return True

    # 检查任务超时
    if record.started_at:
        task_age = (now - record.started_at).total_seconds()
        if task_age > record.timeout_seconds:
            return True

    return False


def find_zombie_tasks() -> List[TaskRecord]:
    """
    查找所有僵尸任务

    Returns:
        僵尸任务列表
    """
    zombies = []
    for task_file in TASKS_DIR.glob("*.json"):
        try:
            task_id = task_file.stem
            record = get_task_record(task_id)
            if record and is_task_zombie(record):
                zombies.append(record)
        except Exception as e:
            print(f"Error checking task {task_file}: {e}")
    return zombies


def cleanup_zombie_tasks() -> List[str]:
    """
    清理僵尸任务 - 将其标记为失败

    Returns:
        被清理的任务ID列表
    """
    cleaned = []
    zombies = find_zombie_tasks()

    for record in zombies:
        try:
            # 检查是否超时
            is_timeout = False
            if record.started_at:
                task_age = (datetime.now() - record.started_at).total_seconds()
                is_timeout = task_age > record.timeout_seconds

            if is_timeout:
                update_task_record(
                    record.task_id,
                    status=TaskStatus.TIMEOUT,
                    error=f"Task timed out after {record.timeout_seconds} seconds",
                    completed_at=datetime.now()
                )
            else:
                update_task_record(
                    record.task_id,
                    status=TaskStatus.FAILED,
                    error="Worker crashed or task became unresponsive (heartbeat timeout)",
                    completed_at=datetime.now()
                )
            cleaned.append(record.task_id)
            print(f"Cleaned zombie task: {record.task_id} ({record.task_type})")
        except Exception as e:
            print(f"Error cleaning zombie task {record.task_id}: {e}")

    return cleaned


def recover_stale_running_tasks(current_worker_id: str = None) -> List[str]:
    """
    在Worker启动时恢复/清理遗留的RUNNING任务

    将不属于当前Worker的RUNNING任务标记为FAILED

    Args:
        current_worker_id: 当前Worker的ID

    Returns:
        被恢复的任务ID列表
    """
    if current_worker_id is None:
        current_worker_id = get_worker_id()

    recovered = []
    for task_file in TASKS_DIR.glob("*.json"):
        try:
            task_id = task_file.stem
            record = get_task_record(task_id)

            if record and record.status == TaskStatus.RUNNING:
                # 如果任务的worker_id不是当前worker，说明是遗留任务
                if record.worker_id != current_worker_id:
                    update_task_record(
                        task_id,
                        status=TaskStatus.FAILED,
                        error=f"Recovered from worker crash (previous worker: {record.worker_id})",
                        completed_at=datetime.now()
                    )
                    recovered.append(task_id)
                    print(f"Recovered stale task: {task_id} (worker: {record.worker_id})")
        except Exception as e:
            print(f"Error recovering task {task_file}: {e}")

    return recovered


class TaskContext:
    """
    任务上下文 - 提供任务状态更新和取消检查的接口
    增强版：包含心跳和超时检测
    """

    def __init__(self, task_id: str, timeout_seconds: int = DEFAULT_TASK_TIMEOUT):
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds
        self._start_time: Optional[datetime] = None

    def start(self):
        """标记任务开始"""
        from app.services.task_runtime import record_task_event, record_task_resource_sample

        self._start_time = datetime.now()
        worker_id = get_worker_id()
        update_task_record(
            self.task_id,
            status=TaskStatus.RUNNING,
            started_at=self._start_time,
            worker_id=worker_id,
            last_heartbeat=self._start_time
        )
        # 启动心跳
        start_heartbeat(self.task_id)
        start_resource_sampler(self.task_id)
        record_task_event(
            self.task_id,
            event_type="task.started",
            message="Task started",
            stage="task.started",
            payload={
                "worker_id": worker_id,
                "timeout_seconds": self.timeout_seconds,
            },
        )
        record_task_resource_sample(self.task_id, phase="task.started")

    def stop(self):
        """停止心跳线程"""
        stop_heartbeat(self.task_id)
        stop_resource_sampler(self.task_id)

    def update_progress(self, progress: float, message: str = ""):
        """
        更新任务进度

        Args:
            progress: 进度值 (0.0 - 1.0)
            message: 进度消息
        """
        update_task_record(
            self.task_id,
            progress=min(max(progress, 0.0), 1.0),
            message=message,
            last_heartbeat=datetime.now()  # 更新进度也算心跳
        )
        try:
            from app.services.task_runtime import record_task_event, record_task_resource_sample

            record_task_event(
                self.task_id,
                event_type="task.progress",
                message=message or "Task progress updated",
                stage="task.progress",
                payload={"progress": min(max(progress, 0.0), 1.0)},
            )
            record_task_resource_sample(
                self.task_id,
                phase="task.progress",
                extra={"progress": min(max(progress, 0.0), 1.0)},
            )
        except Exception:
            pass

    def set_status(self, status: TaskStatus, message: str = ""):
        """
        设置任务状态

        Args:
            status: 任务状态
            message: 状态消息
        """
        kwargs = {"status": status, "message": message}
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                     TaskStatus.CANCELLED, TaskStatus.TIMEOUT):
            kwargs["completed_at"] = datetime.now()
            self.stop()  # 停止心跳
        update_task_record(self.task_id, **kwargs)
        try:
            from app.services.task_runtime import record_task_event, record_task_resource_sample

            record_task_event(
                self.task_id,
                event_type="task.status",
                message=message or f"Task status -> {status.value}",
                stage=f"task.{status.value}",
                payload={"status": status.value},
                level="error" if status in (TaskStatus.FAILED, TaskStatus.TIMEOUT) else "info",
            )
            record_task_resource_sample(
                self.task_id,
                phase=f"task.{status.value}",
                extra={"status": status.value},
            )
        except Exception:
            pass

    def set_result(self, result: Any):
        """
        设置任务结果

        Args:
            result: 任务结果数据
        """
        self.stop()  # 停止心跳
        update_task_record(
            self.task_id,
            result=result,
            status=TaskStatus.COMPLETED,
            progress=1.0,
            completed_at=datetime.now()
        )
        try:
            from app.services.task_runtime import record_task_event, record_task_resource_sample

            record_task_event(
                self.task_id,
                event_type="task.completed",
                message="Task completed",
                stage="task.completed",
                payload={"result": result if isinstance(result, dict) else {"value": str(result)}},
            )
            record_task_resource_sample(self.task_id, phase="task.completed")
        except Exception:
            pass

    def set_error(self, error: str):
        """
        设置任务错误

        Args:
            error: 错误信息
        """
        self.stop()  # 停止心跳
        update_task_record(
            self.task_id,
            error=error,
            status=TaskStatus.FAILED,
            completed_at=datetime.now()
        )
        try:
            from app.services.task_runtime import record_task_event, record_task_resource_sample

            record_task_event(
                self.task_id,
                event_type="task.failed",
                message=error,
                stage="task.failed",
                payload={"error": error},
                level="error",
            )
            record_task_resource_sample(
                self.task_id,
                phase="task.failed",
                extra={"error": error},
            )
        except Exception:
            pass

    def emit_event(
        self,
        event_type: str,
        message: str = "",
        *,
        payload: Optional[Dict[str, Any]] = None,
        level: str = "info",
        stage: Optional[str] = None,
        subtask_id: Optional[int] = None,
    ):
        """向任务事件流写入一条自定义事件。"""
        from app.services.task_runtime import record_task_event

        record_task_event(
            self.task_id,
            event_type=event_type,
            message=message,
            payload=payload,
            level=level,
            stage=stage,
            subtask_id=subtask_id,
        )

    def sample_resources(
        self,
        *,
        phase: Optional[str] = None,
        subtask_id: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """手动触发一次资源采样。"""
        from app.services.task_runtime import record_task_resource_sample

        record_task_resource_sample(
            self.task_id,
            phase=phase,
            subtask_id=subtask_id,
            extra=extra,
        )

    def upsert_subtask(
        self,
        *,
        subtask_key: str,
        subtask_type: str,
        name: Optional[str] = None,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        total: Optional[int] = None,
        retry_count: Optional[int] = None,
        max_retries: Optional[int] = None,
        message: Optional[str] = None,
        stage: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """创建或更新子任务快照。"""
        from app.services.task_runtime import upsert_task_subtask

        return upsert_task_subtask(
            self.task_id,
            subtask_key=subtask_key,
            subtask_type=subtask_type,
            name=name,
            status=status,
            progress=progress,
            total=total,
            retry_count=retry_count,
            max_retries=max_retries,
            message=message,
            stage=stage,
            result=result,
            error=error,
            metadata=metadata,
        )

    def upsert_edge(
        self,
        *,
        edge_key: str,
        from_key: str,
        to_key: str,
        status: str = "active",
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """创建或更新工作流边。"""
        from app.services.task_runtime import upsert_workflow_edge

        return upsert_workflow_edge(
            self.task_id,
            edge_key=edge_key,
            from_key=from_key,
            to_key=to_key,
            status=status,
            label=label,
            metadata=metadata,
        )

    def is_cancelled(self) -> bool:
        """
        检查任务是否已被取消

        Returns:
            是否已取消
        """
        with _cancelled_lock:
            if self.task_id in _cancelled_tasks:
                return True

        record = get_task_record(self.task_id)
        return record is not None and record.status == TaskStatus.CANCELLED

    def is_timed_out(self) -> bool:
        """
        检查任务是否已超时

        Returns:
            是否已超时
        """
        if self._start_time is None:
            return False
        elapsed = (datetime.now() - self._start_time).total_seconds()
        return elapsed > self.timeout_seconds

    def check_should_stop(self) -> Optional[str]:
        """
        检查任务是否应该停止执行

        Returns:
            停止原因，如果不需要停止则返回None
        """
        if self.is_cancelled():
            return "cancelled"
        if self.is_timed_out():
            return "timeout"
        return None


def run_with_context(task_type: str, timeout_seconds: int = DEFAULT_TASK_TIMEOUT):
    """
    装饰器：自动管理任务状态
    增强版：包含超时检测和心跳管理

    被装饰的函数应接受 TaskContext 作为第一个参数

    Args:
        task_type: 任务类型名称
        timeout_seconds: 任务超时时间(秒)

    Usage:
        @huey.task()
        @run_with_context("indexing", timeout_seconds=1800)
        def my_task(ctx: TaskContext, arg1, arg2):
            ctx.update_progress(0.5, "Processing...")
            # do work
            ctx.set_result({"success": True})
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(task_id: str, *args, **kwargs):
            if not _db_task_exists(task_id):
                print(f"Skipping orphaned queued task {task_id}: parent DB task no longer exists")
                return {"status": "orphaned", "task_id": task_id}

            # 创建或获取任务记录
            record = get_task_record(task_id)
            if record is None:
                record = create_task_record(task_type, task_id, timeout_seconds)

            # 创建上下文
            ctx = TaskContext(task_id, timeout_seconds)

            try:
                # 检查是否已取消
                if ctx.is_cancelled():
                    return {"status": "cancelled", "task_id": task_id}

                # 标记任务开始（设置running状态、worker_id、启动心跳）
                ctx.start()

                # 执行任务
                result = func(ctx, *args, **kwargs)

                # 检查是否超时
                if ctx.is_timed_out():
                    ctx.set_status(TaskStatus.TIMEOUT,
                                  f"Task timed out after {timeout_seconds} seconds")
                    return {"status": "timeout", "task_id": task_id}

                # 如果函数没有显式设置结果，自动设置
                final_record = get_task_record(task_id)
                if final_record and final_record.status == TaskStatus.RUNNING:
                    ctx.set_result(result)

                return result

            except Exception as e:
                ctx.set_error(str(e))
                raise
            finally:
                # 确保心跳线程停止
                ctx.stop()

        return wrapper
    return decorator


# Worker启动时自动清理遗留任务
def on_worker_start():
    """Worker启动时的清理操作"""
    worker_id = get_worker_id()
    print(f"Worker started: {worker_id}")

    # 清理遗留的RUNNING任务
    recovered = recover_stale_running_tasks(worker_id)
    if recovered:
        print(f"Recovered {len(recovered)} stale tasks from previous worker sessions")

    # 清理僵尸任务
    cleaned = cleanup_zombie_tasks()
    if cleaned:
        print(f"Cleaned {len(cleaned)} zombie tasks")


# 在进程退出时停止所有心跳
def _cleanup_heartbeats():
    """清理所有心跳线程"""
    with _heartbeat_lock:
        for stop_event in _heartbeat_stop_events.values():
            stop_event.set()
        _heartbeat_stop_events.clear()
        _heartbeat_threads.clear()
    with _resource_lock:
        for stop_event in _resource_stop_events.values():
            stop_event.set()
        _resource_stop_events.clear()
        _resource_threads.clear()

atexit.register(_cleanup_heartbeats)
