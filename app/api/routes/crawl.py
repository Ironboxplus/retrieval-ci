"""
爬虫路由
"""
import os
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel, Field
from datetime import datetime
from bs4 import BeautifulSoup
import re

from core.db.database import get_db
from core.db.models import Task, TaskStatus, CrawlRecord, CrawlStatus as DBCrawlStatus
from app.tasks.base import create_task_record
from app.tasks.crawl_tasks import enqueue_crawl_urls

router = APIRouter()


# ==================== Pydantic模型 ====================

class CrawlUrlsRequest(BaseModel):
    """URL爬取请求"""
    urls: List[str] = Field(..., description="要爬取的URL列表")
    use_proxy: bool = Field(False, description="是否使用代理")
    auto_index: bool = Field(True, description="爬取后是否自动索引")
    max_concurrent: int = Field(5, ge=1, le=20, description="最大并发数")


class CrawlTaskResponse(BaseModel):
    """爬取任务响应"""
    success: bool
    task_id: str
    total_urls: int
    message: str


class CrawlRecordResponse(BaseModel):
    """爬取记录响应"""
    id: int
    task_id: Optional[int] = None
    url: str
    status: str
    title: Optional[str] = None
    content_length: Optional[int] = None
    doc_id: Optional[int] = None
    error_message: Optional[str] = None
    # W41: diagnostic fields surfaced from CrawlRecord. Old pre-migration rows
    # have NULL here — clients should treat None as "not captured".
    status_code: Optional[int] = None
    retry_count: Optional[int] = None
    crawl_time: Optional[float] = None
    proxy_used: Optional[str] = None
    created_at: datetime


class CrawlRecordListResponse(BaseModel):
    """爬取记录列表响应"""
    total: int
    page: int
    page_size: int
    records: List[CrawlRecordResponse]


class HtmlUploadResponse(BaseModel):
    """HTML上传解析响应"""
    success: bool
    task_id: Optional[str] = None
    total_urls: int
    extracted_urls: List[str]
    message: str


# ==================== 路由 ====================

@router.post("/urls", response_model=CrawlTaskResponse)
async def crawl_urls(
    request: CrawlUrlsRequest,
    db: Session = Depends(get_db)
):
    """
    提交URL列表进行爬取

    返回task_id用于后台处理进度查询
    """
    if not request.urls:
        raise HTTPException(400, "URL list cannot be empty")

    # 验证URL格式
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )

    valid_urls = []
    invalid_urls = []
    for url in request.urls:
        url = url.strip()
        if url and url_pattern.match(url):
            valid_urls.append(url)
        elif url:
            invalid_urls.append(url)

    if not valid_urls:
        raise HTTPException(400, "No valid URLs provided")

    # 创建任务记录
    task = Task(
        task_type="crawl_urls",
        status=TaskStatus.pending,
        total=len(valid_urls),
        params_json={
            "urls": valid_urls,
            "use_proxy": request.use_proxy,
            "auto_index": request.auto_index,
            "max_concurrent": request.max_concurrent
        }
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # 为每个URL创建爬取记录
    for url in valid_urls:
        record = CrawlRecord(
            task_id=task.id,
            url=url,
            status=DBCrawlStatus.pending
        )
        db.add(record)
    db.commit()

    # 后台线程内联执行 huey crawl_urls_task（call_local 绕过队列），
    # 完整复用 EnhancedCrawler + subtask/event/workflow 写入逻辑
    import threading
    from core.logging import get_logger as _get_logger
    from app.tasks.crawl_tasks import crawl_urls_task

    _crawl_log = _get_logger("crawl_thread")

    def _run_crawl(task_id: int, urls: list, use_proxy: bool, auto_index: bool, max_concurrent: int):
        _crawl_log.info(f"Crawl thread started (inline): task={task_id}, urls={len(urls)}")
        try:
            crawl_urls_task.call_local(
                str(task_id), urls, use_proxy, auto_index, max_concurrent
            )
        except Exception as e:
            _crawl_log.error(f"Crawl thread failed: task={task_id}, error={e}", exc_info=True)

    threading.Thread(
        target=_run_crawl,
        args=(task.id, valid_urls, request.use_proxy, request.auto_index, request.max_concurrent),
        daemon=True,
    ).start()

    message = f"Crawl task created for {len(valid_urls)} URLs"
    if invalid_urls:
        message += f" ({len(invalid_urls)} invalid URLs skipped)"

    return CrawlTaskResponse(
        success=True,
        task_id=str(task.id),
        total_urls=len(valid_urls),
        message=message
    )


@router.post("/html", response_model=HtmlUploadResponse)
async def crawl_from_html(
    file: UploadFile = File(..., description="HTML文件"),
    use_proxy: bool = Form(False, description="是否使用代理"),
    auto_index: bool = Form(True, description="爬取后是否自动索引"),
    auto_crawl: bool = Form(True, description="是否自动开始爬取"),
    db: Session = Depends(get_db)
):
    """
    上传HTML文件，解析其中的链接并爬取

    适用于导出的书签页面或包含链接列表的HTML
    """
    # 检查文件类型
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in {".html", ".htm"}:
        raise HTTPException(400, f"Expected HTML file, got: {ext}")

    # 读取文件内容
    content = await file.read()
    try:
        html_content = content.decode("utf-8", errors="ignore")
    except Exception:
        html_content = content.decode("latin-1", errors="ignore")

    # 解析HTML提取链接
    soup = BeautifulSoup(html_content, "html.parser")
    extracted_urls = []

    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        # 只保留http/https链接
        if href.startswith(("http://", "https://")):
            extracted_urls.append(href)

    # 去重
    extracted_urls = list(dict.fromkeys(extracted_urls))

    if not extracted_urls:
        return HtmlUploadResponse(
            success=False,
            task_id=None,
            total_urls=0,
            extracted_urls=[],
            message="No valid URLs found in the HTML file"
        )

    task_id = None
    if auto_crawl:
        # 创建任务记录
        task = Task(
            task_type="crawl_urls",
            status=TaskStatus.pending,
            total=len(extracted_urls),
            params_json={
                "urls": extracted_urls,
                "use_proxy": use_proxy,
                "auto_index": auto_index,
                "source_file": filename
            }
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = str(task.id)

        # 为每个URL创建爬取记录
        for url in extracted_urls:
            record = CrawlRecord(
                task_id=task.id,
                url=url,
                status=DBCrawlStatus.pending
            )
            db.add(record)
        db.commit()

        # 入队后台任务. enqueue_crawl_urls spawns a daemon thread + call_local
        # — failures here mean the task row exists but no worker will run, so
        # log loudly instead of silently dropping (W44, was masked by bare pass).
        try:
            enqueue_crawl_urls(
                extracted_urls,
                use_proxy=use_proxy,
                auto_index=auto_index,
                task_id=str(task.id)  # 传递数据库任务ID
            )
        except Exception:
            from core.logging import get_logger
            get_logger("crawl_route").exception(
                "enqueue_crawl_urls failed for task=%s urls=%d — task will stay pending",
                task.id, len(extracted_urls),
            )

    return HtmlUploadResponse(
        success=True,
        task_id=task_id,
        total_urls=len(extracted_urls),
        extracted_urls=extracted_urls[:50],  # 只返回前50个作为预览
        message=f"Extracted {len(extracted_urls)} URLs from HTML" +
                (" and started crawling" if auto_crawl else "")
    )


@router.get("/records", response_model=CrawlRecordListResponse)
async def list_crawl_records(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    task_id: Optional[int] = Query(None, description="任务ID筛选"),
    status: Optional[str] = Query(None, description="状态筛选: pending|success|failed|skipped"),
    db: Session = Depends(get_db)
):
    """
    获取爬取记录列表
    """
    query = db.query(CrawlRecord)

    # 任务ID筛选
    if task_id is not None:
        query = query.filter(CrawlRecord.task_id == task_id)

    # 状态筛选
    if status:
        try:
            status_enum = DBCrawlStatus(status)
            query = query.filter(CrawlRecord.status == status_enum)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    # 获取总数
    total = query.count()

    # 分页并按创建时间倒序
    records = query.order_by(desc(CrawlRecord.created_at)).offset(
        (page - 1) * page_size
    ).limit(page_size).all()

    return CrawlRecordListResponse(
        total=total,
        page=page,
        page_size=page_size,
        records=[
            CrawlRecordResponse(
                id=record.id,
                task_id=record.task_id,
                url=record.url,
                status=record.status.value if record.status else "pending",
                title=record.title,
                content_length=record.content_length,
                doc_id=record.doc_id,
                error_message=record.error_message,
                status_code=record.status_code,
                retry_count=record.retry_count,
                crawl_time=record.crawl_time,
                proxy_used=record.proxy_used,
                created_at=record.created_at
            )
            for record in records
        ]
    )
