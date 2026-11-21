"""
任务服务 - 封装任务的增删改查业务逻辑
增强版：支持僵尸任务清理、超时任务和心跳检测
"""
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import desc

from core.db import get_db_session
from core.db.models import Task, TaskStatus as DBTaskStatus
from app.tasks.base import (
    TaskRecord, TaskStatus, TaskContext,
    create_task_record, get_task_record, update_task_record, cancel_task,
    find_zombie_tasks, cleanup_zombie_tasks, recover_stale_running_tasks,
    HEARTBEAT_TIMEOUT, DEFAULT_TASK_TIMEOUT
)
from config import settings
from app.services.task_runtime import purge_task_runtime_rows


class TaskService:
    """任务服务类 - 增强版"""

    def __init__(self, db: Session = None):
        """
        初始化任务服务

        Args:
            db: 数据库会话，如果不传则在每个方法中自动管理
        """
        self._db = db

    def _get_db(self):
        """获取数据库会话"""
        if self._db is not None:
            return self._db
        return None

    def create_task(
        self,
        task_type: str,
        params: Dict[str, Any] = None,
        priority: int = 0,
        max_retries: int = 3,
        timeout_seconds: int = DEFAULT_TASK_TIMEOUT,
        db: Session = None
    ) -> Task:
        """
        创建任务

        Args:
            task_type: 任务类型
            params: 任务参数
            priority: 优先级（数字越大优先级越高）
            max_retries: 最大重试次数
            timeout_seconds: 任务超时时间(秒)
            db: 数据库会话

        Returns:
            创建的Task对象
        """
        session = db or self._get_db()
        use_context = session is None

        if use_context:
            with get_db_session() as session:
                return self._create_task_impl(
                    session, task_type, params, priority, max_retries, timeout_seconds
                )
        else:
            return self._create_task_impl(
                session, task_type, params, priority, max_retries, timeout_seconds
            )

    def _create_task_impl(
        self,
        session: Session,
        task_type: str,
        params: Dict[str, Any],
        priority: int,
        max_retries: int,
        timeout_seconds: int
    ) -> Task:
        """创建任务实现"""
        task = Task(
            task_type=task_type,
            status=DBTaskStatus.pending,
            priority=priority,
            params_json=params or {},
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            progress=0,
            total=0
        )

        session.add(task)
        session.flush()

        # 同时创建文件记录（用于Huey队列）
        create_task_record(task_type, str(task.id), timeout_seconds)

        return task

    def get_task(
        self,
        task_id: int,
        db: Session = None
    ) -> Optional[Task]:
        """
        获取任务

        Args:
            task_id: 任务ID
            db: 数据库会话

        Returns:
            Task对象或None
        """
        session = db or self._get_db()
        use_context = session is None

        if use_context:
            with get_db_session() as session:
                return session.query(Task).filter(Task.id == task_id).first()
        else:
            return session.query(Task).filter(Task.id == task_id).first()

    def get_task_status(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        获取任务状态（包括进度信息）

        Args:
            task_id: 任务ID

        Returns:
            任务状态信息字典
        """
        # 先从文件记录获取（更实时）
        record = get_task_record(str(task_id))
        if record:
            return {
                "task_id": task_id,
                "task_type": record.task_type,
                "status": record.status.value,
                "progress": record.progress,
                "message": record.message,
                "result": record.result,
                "error": record.error,
                "worker_id": record.worker_id,
                "last_heartbeat": record.last_heartbeat.isoformat() if record.last_heartbeat else None,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "started_at": record.started_at.isoformat() if record.started_at else None,
                "completed_at": record.completed_at.isoformat() if record.completed_at else None,
                "timeout_seconds": record.timeout_seconds
            }

        # 回退到数据库
        task = self.get_task(task_id)
        if task:
            return {
                "task_id": task.id,
                "task_type": task.task_type,
                "status": task.status.value,
                "progress": task.progress,
                "total": task.total,
                "result": task.result_json,
                "error": task.error_message,
                "worker_id": task.worker_id,
                "last_heartbeat": task.last_heartbeat.isoformat() if task.last_heartbeat else None,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "timeout_seconds": task.timeout_seconds
            }

        return None

    def list_tasks(
        self,
        task_type: str = None,
        status: DBTaskStatus = None,
        offset: int = 0,
        limit: int = 50,
        db: Session = None
    ) -> Tuple[List[Task], int]:
        """
        列出任务

        Args:
            task_type: 按任务类型筛选
            status: 按状态筛选
            offset: 偏移量
            limit: 限制数量
            db: 数据库会话

        Returns:
            (任务列表, 总数)
        """
        session = db or self._get_db()
        use_context = session is None

        if use_context:
            with get_db_session() as session:
                return self._list_tasks_impl(
                    session, task_type, status, offset, limit
                )
        else:
            return self._list_tasks_impl(
                session, task_type, status, offset, limit
            )

    def _list_tasks_impl(
        self,
        session: Session,
        task_type: str,
        status: DBTaskStatus,
        offset: int,
        limit: int
    ) -> Tuple[List[Task], int]:
        """列出任务实现"""
        query = session.query(Task)

        if task_type is not None:
            query = query.filter(Task.task_type == task_type)
        if status is not None:
            query = query.filter(Task.status == status)

        total = query.count()
        tasks = query.order_by(
            desc(Task.priority),
            desc(Task.created_at)
        ).offset(offset).limit(limit).all()

        return tasks, total

    def cancel_task(
        self,
        task_id: int,
        db: Session = None
    ) -> bool:
        """
        取消任务（设置cancelled标志）

        Args:
            task_id: 任务ID
            db: 数据库会话

        Returns:
            是否成功取消
        """
        session = db or self._get_db()
        use_context = session is None

        # 更新数据库状态
        def _cancel_impl(session: Session) -> bool:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task is None:
                return cancel_task(str(task_id))

            if task.status not in (DBTaskStatus.pending, DBTaskStatus.running):
                return False

            file_cancelled = cancel_task(str(task_id))
            task.status = DBTaskStatus.cancelled
            task.completed_at = datetime.now()
            session.flush()
            return file_cancelled or True

        if use_context:
            with get_db_session() as session:
                return _cancel_impl(session)
        else:
            return _cancel_impl(session)

    def retry_task(
        self,
        task_id: int,
        db: Session = None
    ) -> Optional[Task]:
        """
        重试任务

        Args:
            task_id: 任务ID
            db: 数据库会话

        Returns:
            新创建的任务对象或None
        """
        session = db or self._get_db()
        use_context = session is None

        if use_context:
            with get_db_session() as session:
                return self._retry_task_impl(session, task_id)
        else:
            return self._retry_task_impl(session, task_id)

    def _retry_task_impl(
        self,
        session: Session,
        task_id: int
    ) -> Optional[Task]:
        """重试任务实现"""
        original_task = session.query(Task).filter(Task.id == task_id).first()
        if original_task is None:
            return None

        # 只能重试失败、取消或超时的任务
        if original_task.status not in (DBTaskStatus.failed, DBTaskStatus.cancelled,
                                        DBTaskStatus.timeout):
            return None

        # 检查重试次数
        if original_task.retry_count >= original_task.max_retries:
            return None

        # 创建新任务
        new_task = Task(
            task_type=original_task.task_type,
            status=DBTaskStatus.pending,
            priority=original_task.priority,
            params_json=original_task.params_json,
            max_retries=original_task.max_retries,
            timeout_seconds=original_task.timeout_seconds,
            retry_count=original_task.retry_count + 1,
            progress=0,
            total=original_task.total
        )

        session.add(new_task)
        session.flush()

        # 创建文件记录
        create_task_record(new_task.task_type, str(new_task.id), new_task.timeout_seconds)

        return new_task

    def delete_task(
        self,
        task_id: int,
        db: Session = None
    ) -> bool:
        """
        删除任务

        Args:
            task_id: 任务ID
            db: 数据库会话

        Returns:
            是否删除成功
        """
        session = db or self._get_db()
        use_context = session is None

        if use_context:
            with get_db_session() as session:
                return self._delete_task_impl(session, task_id)
        else:
            return self._delete_task_impl(session, task_id)

    def _delete_task_impl(
        self,
        session: Session,
        task_id: int
    ) -> bool:
        """删除任务实现"""
        task = session.query(Task).filter(Task.id == task_id).first()
        if task is None:
            return False

        # 不能删除正在运行的任务
        if task.status == DBTaskStatus.running:
            return False

        purge_task_runtime_rows(task_id, db=session)
        session.delete(task)
        session.flush()

        # 删除文件记录
        from pathlib import Path
        task_file = Path(settings.data_dir) / "tasks" / "records" / f"{task_id}.json"
        if task_file.exists():
            task_file.unlink()

        return True

    def update_task_progress(
        self,
        task_id: int,
        progress: int,
        total: int = None,
        db: Session = None
    ):
        """
        更新任务进度

        Args:
            task_id: 任务ID
            progress: 当前进度
            total: 总数
            db: 数据库会话
        """
        session = db or self._get_db()
        use_context = session is None

        def _update_impl(session: Session):
            task = session.query(Task).filter(Task.id == task_id).first()
            if task:
                task.progress = progress
                if total is not None:
                    task.total = total
                task.last_heartbeat = datetime.now()
                session.flush()

        if use_context:
            with get_db_session() as session:
                _update_impl(session)
        else:
            _update_impl(session)

        # 同时更新文件记录
        if total:
            pct = progress / total if total > 0 else 0
        else:
            pct = 0

        update_task_record(str(task_id), progress=pct)

    def get_pending_tasks(
        self,
        task_type: str = None,
        limit: int = 10,
        db: Session = None
    ) -> List[Task]:
        """
        获取待处理的任务

        Args:
            task_type: 任务类型筛选
            limit: 限制数量
            db: 数据库会话

        Returns:
            待处理的任务列表
        """
        session = db or self._get_db()
        use_context = session is None

        def _get_impl(session: Session) -> List[Task]:
            query = session.query(Task).filter(Task.status == DBTaskStatus.pending)

            if task_type:
                query = query.filter(Task.task_type == task_type)

            return query.order_by(
                desc(Task.created_at),
                desc(Task.id),
                desc(Task.priority)
            ).limit(limit).all()

        if use_context:
            with get_db_session() as session:
                return _get_impl(session)
        else:
            return _get_impl(session)

    def get_running_tasks(
        self,
        task_type: str = None,
        db: Session = None
    ) -> List[Task]:
        """
        获取正在运行的任务

        Args:
            task_type: 任务类型筛选
            db: 数据库会话

        Returns:
            正在运行的任务列表
        """
        session = db or self._get_db()
        use_context = session is None

        def _get_impl(session: Session) -> List[Task]:
            query = session.query(Task).filter(Task.status == DBTaskStatus.running)

            if task_type:
                query = query.filter(Task.task_type == task_type)

            return query.order_by(Task.started_at).all()

        if use_context:
            with get_db_session() as session:
                return _get_impl(session)
        else:
            return _get_impl(session)

    def find_zombie_tasks_in_db(
        self,
        heartbeat_timeout: int = HEARTBEAT_TIMEOUT,
        db: Session = None
    ) -> List[Task]:
        """
        在数据库中查找僵尸任务

        Args:
            heartbeat_timeout: 心跳超时时间(秒)
            db: 数据库会话

        Returns:
            僵尸任务列表
        """
        session = db or self._get_db()
        use_context = session is None

        def _find_impl(session: Session) -> List[Task]:
            now = datetime.now()
            cutoff_time = now - timedelta(seconds=heartbeat_timeout)

            # 查找RUNNING状态且心跳超时的任务
            return session.query(Task).filter(
                Task.status == DBTaskStatus.running,
                Task.last_heartbeat < cutoff_time
            ).all()

        if use_context:
            with get_db_session() as session:
                return _find_impl(session)
        else:
            return _find_impl(session)

    def cleanup_zombie_tasks_in_db(
        self,
        heartbeat_timeout: int = HEARTBEAT_TIMEOUT,
        db: Session = None
    ) -> List[int]:
        """
        清理数据库中的僵尸任务

        Args:
            heartbeat_timeout: 心跳超时时间(秒)
            db: 数据库会话

        Returns:
            被清理的任务ID列表
        """
        session = db or self._get_db()
        use_context = session is None

        def _cleanup_impl(session: Session) -> List[int]:
            zombies = self.find_zombie_tasks_in_db(heartbeat_timeout, session)
            cleaned_ids = []

            for task in zombies:
                now = datetime.now()

                # 检查是否超时
                is_timeout = False
                if task.started_at and task.timeout_seconds:
                    task_age = (now - task.started_at).total_seconds()
                    is_timeout = task_age > task.timeout_seconds

                if is_timeout:
                    task.status = DBTaskStatus.timeout
                    task.error_message = f"Task timed out after {task.timeout_seconds} seconds"
                else:
                    task.status = DBTaskStatus.failed
                    task.error_message = "Worker crashed or task became unresponsive (heartbeat timeout)"

                task.completed_at = now
                cleaned_ids.append(task.id)

            session.flush()
            return cleaned_ids

        if use_context:
            with get_db_session() as session:
                return _cleanup_impl(session)
        else:
            return _cleanup_impl(session)

    def cleanup_all_zombies(self) -> Dict[str, List]:
        """
        清理所有僵尸任务（文件记录+数据库）

        Returns:
            {"file_records": [...], "db_records": [...]}
        """
        # 清理文件记录中的僵尸任务
        file_cleaned = cleanup_zombie_tasks()

        # 清理数据库中的僵尸任务
        db_cleaned = self.cleanup_zombie_tasks_in_db()

        return {
            "file_records": file_cleaned,
            "db_records": db_cleaned
        }

    def get_task_health(self, task_id: int) -> Dict[str, Any]:
        """
        获取任务健康状态

        Args:
            task_id: 任务ID

        Returns:
            健康状态信息
        """
        record = get_task_record(str(task_id))
        if record is None:
            return {"status": "not_found"}

        now = datetime.now()
        health = {
            "task_id": task_id,
            "status": record.status.value,
            "is_running": record.status == TaskStatus.RUNNING,
            "is_healthy": True,
            "issues": []
        }

        if record.status == TaskStatus.RUNNING:
            # 检查心跳
            if record.last_heartbeat:
                heartbeat_age = (now - record.last_heartbeat).total_seconds()
                health["heartbeat_age_seconds"] = heartbeat_age
                if heartbeat_age > HEARTBEAT_TIMEOUT:
                    health["is_healthy"] = False
                    health["issues"].append(f"Heartbeat timeout ({heartbeat_age:.0f}s > {HEARTBEAT_TIMEOUT}s)")

            # 检查超时
            if record.started_at:
                task_age = (now - record.started_at).total_seconds()
                health["task_age_seconds"] = task_age
                if task_age > record.timeout_seconds:
                    health["is_healthy"] = False
                    health["issues"].append(f"Task timeout ({task_age:.0f}s > {record.timeout_seconds}s)")

        return health
