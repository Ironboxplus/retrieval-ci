"""上传 PDF → extract 产物 MD 全文定位（功能2 pdf 输入）。

upload 流水线把 extract 出的 markdown 写到 `upload_md_path(file_path)`（单一约定，
见 upload_context.py）。task 的 params 里存了 doc_id + file_path，所以
task_id → params.file_path → md_path → 读全文。

与 service.py 分离：保持纯 IO，service 只注入 callable。
"""
from __future__ import annotations

import logging

from app.services.pipeline.store import TaskStore
from app.services.pipeline.upload_context import upload_md_path

_logger = logging.getLogger(__name__)


def load_upload_md_for_task(task_id: str | int, *, store: TaskStore | None = None) -> str:
    """返回该上传任务 extract 出的 MD 全文；任务/产物缺失 → 抛 ValueError。"""
    store = store or TaskStore()
    try:
        tid = int(task_id)
    except (TypeError, ValueError):
        raise ValueError(f"非法 upload_task_id: {task_id!r}")

    task = store.get_task(tid)
    if task is None:
        raise ValueError(f"upload 任务不存在: {tid}")
    file_path = (task.get("params") or {}).get("file_path")
    if not file_path:
        raise ValueError(f"upload 任务 {tid} 无 file_path（无法定位 extract 产物）")

    md_path = upload_md_path(file_path)
    if not (md_path.is_file() and md_path.stat().st_size > 0):
        raise ValueError(
            f"upload 任务 {tid} 的 MD 尚未就绪（{md_path.name}）—— extract 未完成？")
    text = md_path.read_text(encoding="utf-8", errors="replace")
    _logger.info("match_doc.upload_md.loaded task=%s md=%s chars=%d",
                 tid, md_path.name, len(text))
    return text
