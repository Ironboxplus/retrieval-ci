"""
文档管理路由
"""
import os
import uuid
import mimetypes
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import desc
from pydantic import BaseModel
from datetime import datetime

from core.db.database import get_db
from core.db.models import Document, Chunk, Paper, SourceType, DataSource, DocumentStatus
from config import settings


# 支持下载的文件类型及其MIME类型
DOWNLOADABLE_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

SOURCE_FILE_METADATA_KEYS = (
    "source_file_path",
    "source_pdf_path",
    "pdf_path",
    "original_file_path",
    "downloaded_file_path",
)

router = APIRouter()


# ==================== Pydantic模型 ====================

class ChunkBrief(BaseModel):
    """分块简要信息"""
    id: int
    level: int
    chunk_index: int
    content_preview: str
    token_count: Optional[int] = None


class DocumentResponse(BaseModel):
    """文档响应模型"""
    id: int
    title: Optional[str] = None
    source: Optional[str] = None
    source_type: str
    data_source: str
    status: str
    url: Optional[str] = None
    file_path: Optional[str] = None
    chunk_count: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[dict] = None
    download_url: Optional[str] = None  # 文件下载链接
    source_file_path: Optional[str] = None
    source_download_url: Optional[str] = None  # 原始源文件下载链接

    class Config:
        from_attributes = True


class DocumentDetailResponse(DocumentResponse):
    """文档详情响应（包含chunks）"""
    chunks: List[ChunkBrief] = []


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    total: int
    page: int
    page_size: int
    documents: List[DocumentResponse]


class DocumentUpdateRequest(BaseModel):
    """文档更新请求"""
    title: Optional[str] = None
    metadata: Optional[dict] = None


class DocumentActionResponse(BaseModel):
    """文档操作响应"""
    success: bool
    message: str
    doc_id: int


class UploadDocumentResponse(BaseModel):
    """上传文档响应"""
    success: bool
    message: str
    task_id: Optional[str] = None
    doc_id: Optional[int] = None
    file_path: Optional[str] = None


# ==================== 辅助函数 ====================

def get_download_url(doc_id: int, file_path: Optional[str], request: Request = None) -> Optional[str]:
    """
    根据文档ID和文件路径生成下载URL

    Args:
        doc_id: 文档ID
        file_path: 文件路径
        request: FastAPI Request对象，用于获取基础URL

    Returns:
        下载URL或None（如果文件不可下载）
    """
    if not file_path:
        return None

    # 检查文件扩展名是否支持下载
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in DOWNLOADABLE_EXTENSIONS:
        return None

    # 检查文件是否存在
    if not os.path.exists(file_path):
        return None

    # 生成相对下载URL
    return f"/api/documents/{doc_id}/download"


def is_downloadable_file(file_path: Optional[str]) -> bool:
    """检查给定文件路径是否可下载。"""
    if not file_path:
        return False

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in DOWNLOADABLE_EXTENSIONS:
        return False

    return os.path.exists(file_path)


def resolve_document_source_file_path(doc: Document) -> Optional[str]:
    """
    解析文档对应的原始源文件路径。

    规则：
    - 论文文档优先返回 Paper 表里的 pdf_path
    - 其次尝试文档 metadata 中约定的源文件字段
    - 如果源文件与当前 doc.file_path 相同，则视为没有独立源文件
    """
    candidate_paths: list[str] = []

    if doc.paper and doc.paper.pdf_path:
        candidate_paths.append(doc.paper.pdf_path)

    metadata = doc.metadata_json or {}
    if isinstance(metadata, dict):
        for key in SOURCE_FILE_METADATA_KEYS:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                candidate_paths.append(value.strip())

    normalized_primary_path = os.path.normcase(os.path.abspath(doc.file_path)) if doc.file_path else None
    for candidate_path in candidate_paths:
        if not is_downloadable_file(candidate_path):
            continue

        normalized_candidate_path = os.path.normcase(os.path.abspath(candidate_path))
        if normalized_primary_path and normalized_candidate_path == normalized_primary_path:
            continue
        return candidate_path

    return None


def get_source_download_url(doc_id: int, doc: Document) -> Optional[str]:
    """根据文档信息生成原始源文件下载 URL。"""
    if not resolve_document_source_file_path(doc):
        return None
    return f"/api/documents/{doc_id}/source-download"


def build_document_response(doc: Document, chunks_brief: Optional[List[ChunkBrief]] = None) -> DocumentResponse | DocumentDetailResponse:
    """统一构建文档响应，避免列表/详情/更新接口重复拼装字段。"""
    source_file_path = resolve_document_source_file_path(doc)
    response_payload = dict(
        id=doc.id,
        title=doc.title,
        source=doc.source,
        source_type=doc.source_type.value if doc.source_type else "upload",
        data_source=doc.data_source.value if doc.data_source else "user",
        status=doc.status.value if doc.status else "pending",
        url=doc.url,
        file_path=doc.file_path,
        chunk_count=len(doc.chunks) if doc.chunks else 0,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        metadata=doc.metadata_json,
        download_url=get_download_url(doc.id, doc.file_path),
        source_file_path=source_file_path,
        source_download_url=get_source_download_url(doc.id, doc),
    )

    if chunks_brief is None:
        return DocumentResponse(**response_payload)
    return DocumentDetailResponse(**response_payload, chunks=chunks_brief)


# ==================== 路由 ====================

@router.get("", response_model=DocumentListResponse)
def list_documents(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    source_type: Optional[str] = Query(None, description="来源类型筛选: webpage|paper|upload"),
    data_source: Optional[str] = Query(None, description="数据来源筛选: system|user"),
    status: Optional[str] = Query(None, description="状态筛选: pending|processing|indexed|failed"),
    db: Session = Depends(get_db)
):
    """
    获取文档列表

    支持分页和多种筛选条件
    """
    query = db.query(Document)

    # 来源类型筛选
    if source_type:
        try:
            source_type_enum = SourceType(source_type)
            query = query.filter(Document.source_type == source_type_enum)
        except ValueError:
            raise HTTPException(400, f"Invalid source_type: {source_type}")

    # 数据来源筛选
    if data_source:
        try:
            data_source_enum = DataSource(data_source)
            query = query.filter(Document.data_source == data_source_enum)
        except ValueError:
            raise HTTPException(400, f"Invalid data_source: {data_source}")

    # 状态筛选
    if status:
        try:
            status_enum = DocumentStatus(status)
            query = query.filter(Document.status == status_enum)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    # 获取总数
    total = query.count()

    # 分页并按创建时间倒序
    documents = query.order_by(desc(Document.created_at)).offset(
        (page - 1) * page_size
    ).limit(page_size).all()

    return DocumentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        documents=[build_document_response(doc) for doc in documents]
    )


@router.get("/{doc_id}", response_model=DocumentDetailResponse)
def get_document(
    doc_id: int,
    db: Session = Depends(get_db)
):
    """
    获取文档详情

    包含文档的所有chunks概要信息
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    # 获取chunks并构建简要信息
    chunks_brief = [
        ChunkBrief(
            id=chunk.id,
            level=chunk.level or 0,
            chunk_index=chunk.chunk_index or 0,
            content_preview=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            token_count=chunk.token_count
        )
        for chunk in doc.chunks
    ]

    return build_document_response(doc, chunks_brief=chunks_brief)


@router.delete("/{doc_id}", response_model=DocumentActionResponse)
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db)
):
    """
    删除文档

    同时删除对应的chunks和索引（ChromaDB和Whoosh）
    SQLite中的chunks通过级联删除自动清理
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    # 获取文件路径（如果存在）
    file_path = doc.file_path

    # 获取文档的chunks数量用于日志
    chunk_count = len(doc.chunks) if doc.chunks else 0

    # 从向量索引中删除对应的embeddings（ChromaDB和Whoosh）
    # 注意：不删除数据库记录，因为后面db.delete(doc)会级联删除chunks
    try:
        from app.services.index_service import get_index_service
        # delete_db_records=False，因为文档删除会级联删除chunks
        get_index_service().delete_document_index(doc_id, db=db, delete_db_records=False)
    except Exception as e:
        # 记录错误但继续删除文档
        print(f"Warning: Failed to delete document index: {e}")

    # 删除文档（级联删除chunks）
    db.delete(doc)
    db.commit()

    # 尝试删除物理文件
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    return DocumentActionResponse(
        success=True,
        message=f"Document deleted successfully (removed {chunk_count} chunks from indexes)",
        doc_id=doc_id
    )

@router.put("/{doc_id}", response_model=DocumentResponse)
def update_document(
    doc_id: int,
    request: DocumentUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    更新文档元数据
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    # 更新字段
    if request.title is not None:
        doc.title = request.title

    if request.metadata is not None:
        # 合并元数据
        existing_metadata = dict(doc.metadata_json or {})
        existing_metadata.update(request.metadata)
        doc.metadata_json = dict(existing_metadata)
        flag_modified(doc, "metadata_json")

    db.commit()
    db.refresh(doc)

    return build_document_response(doc)


@router.post("/upload", response_model=UploadDocumentResponse)
def upload_document(
    file: UploadFile = File(..., description="上传的文件 (PDF/TXT/HTML)"),
    title: Optional[str] = Form(None, description="文档标题（可选）"),
    auto_index: bool = Form(True, description="是否自动索引"),
    db: Session = Depends(get_db)
):
    """
    上传文档

    支持PDF、TXT、HTML格式

    注意:
    - `auto_index=true` 时把文档交给上传流水线（extract marker→index）在后台 daemon
      线程跑，**不再在事件循环里同步解析+索引**（大 PDF 曾把整个 app 卡死）。接口立即
      返回 `task_id`，前端轮询 `/api/pipeline/tasks/{task_id}` 看进度。后端只 poll 状态，
      实际抽取/嵌入/入库由本地 worker 执行。
    - `auto_index=false` 时只落盘+建 Document 记录，不启动流水线。
    """
    # 检查文件类型
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    supported_extensions = {".pdf", ".txt", ".html", ".htm", ".md"}
    if ext not in supported_extensions:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext}. Supported: {', '.join(supported_extensions)}"
        )

    # 生成文件ID和保存路径
    file_id = str(uuid.uuid4())
    save_dir = os.path.join(settings.data_dir, "uploads")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file_id}{ext}")

    # 保存文件
    content = file.file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 创建文档记录
    doc = Document(
        title=title or filename,
        source=filename,
        source_type=SourceType.upload,
        data_source=DataSource.user,
        file_path=save_path,
        status=DocumentStatus.pending,
        metadata_json={"original_filename": filename, "file_size": len(content)}
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    if not auto_index:
        return UploadDocumentResponse(
            success=True,
            message="File uploaded successfully",
            task_id=None,
            doc_id=doc.id,
            file_path=save_path,
        )

    # Hand off to the upload pipeline (extract marker → index), which runs in a
    # daemon thread OFF the event loop. The previous path parsed (pypdf) + indexed
    # synchronously right here in the request handler — a large PDF blocked the
    # whole app (the freeze bug). Now the request returns immediately with a
    # task_id; the local worker does extract/embed/upsert and the frontend polls.
    doc.status = DocumentStatus.processing
    db.commit()
    db.refresh(doc)

    from app.services.pipeline.upload_service import get_upload_runner

    res = get_upload_runner().start_upload(doc.id, save_path)
    task_id = res.get("task_id")
    if res.get("status") == "conflict":
        message = res.get("error") or "an upload pipeline is already running for this document"
    else:
        message = "File uploaded; extracting (marker) + indexing in background"

    return UploadDocumentResponse(
        success=res.get("status") in ("started", "conflict"),
        message=message,
        task_id=str(task_id) if task_id is not None else None,
        doc_id=doc.id,
        file_path=save_path,
    )


def _resolve_document_markdown(doc: Document):
    """Path to an uploaded document's extracted Markdown, or raise 404.

    Papers have a crawler artifact resolved by core/storage/artifact_resolver;
    uploads have none — their extracted Markdown lives at the single path
    `upload_md_path()` dictates, next to the stored original. Reassembling the
    text from SQL chunks is NOT an alternative: L3 chunks overlap by ~25 chars
    (measured on doc 114854: 8,948 chunks summing to 4,562,377 chars over a
    4,712,964-char span), so a concatenation both duplicates and drops text.
    """
    from pathlib import Path
    from app.services.pipeline.upload_context import upload_md_path

    if not doc.file_path:
        raise HTTPException(404, "Document has no stored file, so no extracted Markdown")
    md = Path(upload_md_path(doc.file_path))
    if not (md.is_file() and md.stat().st_size > 0):
        raise HTTPException(404, "Markdown has not been extracted for this document")
    return md


@router.get("/{doc_id}/fulltext")
def get_document_fulltext(doc_id: int, db: Session = Depends(get_db)):
    """Full extracted Markdown for one document, as JSON.

    Mirrors /api/papers/{paper_id}/fulltext for documents that have no paper_id
    (uploads, webpages). Books are large — doc 114854 is 4.7 MB — so callers
    that only need to save the text should use /{doc_id}/markdown, which streams
    and honours Range instead of building the whole body in memory.
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    md = _resolve_document_markdown(doc)
    content = md.read_text(encoding="utf-8", errors="replace")
    return {
        "doc_id": doc_id,
        "title": doc.title,
        "content": content,
        "size": len(content),
    }


@router.get(
    "/{doc_id}/markdown",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Extracted Markdown file.",
            "content": {"text/markdown": {"schema": {"type": "string", "format": "binary"}}},
        },
        404: {"description": "Document or extracted Markdown not found."},
    },
)
def get_document_markdown_file(doc_id: int, db: Session = Depends(get_db)):
    """Stream a document's extracted Markdown with HTTP Range support."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    md = _resolve_document_markdown(doc)
    return FileResponse(
        path=md,
        media_type="text/markdown; charset=utf-8",
        filename=f"document-{doc_id}.md",
    )


@router.get("/{doc_id}/download")
def download_document(
    doc_id: int,
    db: Session = Depends(get_db)
):
    """
    下载文档文件

    支持PDF、TXT、MD格式的文档下载
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    if not doc.file_path:
        raise HTTPException(400, "This document has no downloadable file")

    if not os.path.exists(doc.file_path):
        raise HTTPException(404, "File not found on disk")

    # 检查文件扩展名是否支持下载
    ext = os.path.splitext(doc.file_path)[1].lower()
    if ext not in DOWNLOADABLE_EXTENSIONS:
        raise HTTPException(400, f"File type {ext} is not downloadable")

    # 获取MIME类型
    media_type = DOWNLOADABLE_EXTENSIONS.get(ext, "application/octet-stream")

    # 生成下载文件名
    filename = doc.title or os.path.basename(doc.file_path)
    if not filename.endswith(ext):
        filename = filename + ext

    return FileResponse(
        path=doc.file_path,
        media_type=media_type,
        filename=filename
    )


@router.get("/{doc_id}/source-download")
def download_document_source_file(
    doc_id: int,
    db: Session = Depends(get_db)
):
    """
    下载文档对应的原始源文件。

    对于论文文档，这个接口优先返回原始 PDF；
    对于其他文档，则尝试返回 metadata 中声明的源文件路径。
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    source_file_path = resolve_document_source_file_path(doc)
    if not source_file_path:
        raise HTTPException(400, "This document has no separate downloadable source file")

    if not os.path.exists(source_file_path):
        raise HTTPException(404, "Source file not found on disk")

    ext = os.path.splitext(source_file_path)[1].lower()
    if ext not in DOWNLOADABLE_EXTENSIONS:
        raise HTTPException(400, f"File type {ext} is not downloadable")

    media_type = DOWNLOADABLE_EXTENSIONS.get(ext, "application/octet-stream")
    filename = doc.title or os.path.basename(source_file_path)
    if not filename.endswith(ext):
        filename = filename + ext

    return FileResponse(
        path=source_file_path,
        media_type=media_type,
        filename=filename
    )


def _count_qdrant_points_for_document(
    corpus,
    *,
    doc_id: int,
    paper_id: str | None,
    source_type=None,
    venue_tier: str | None = None,
) -> int:
    """Count one document's points through a stable paper-corpus keyword.

    Historical paper imports contain both integer and string ``doc_id`` payloads,
    whereas their canonical ``paper_id`` is always a keyword string.  Prefer it
    whenever the SQLite document belongs to a paper.  Non-paper documents retain
    the normalized string ``doc_id`` lookup without an unindexed integer probe.
    Under the split layout the count reads the collection the document's points
    belong to (papers / arxiv / reviews).
    """
    from core.storage.corpus_store import role_for_document

    payload_key, payload_value = (
        ("paper_id", paper_id) if paper_id is not None else ("doc_id", str(doc_id))
    )
    role = role_for_document(source_type=source_type, venue_tier=venue_tier)
    return corpus.count_points(payload_key, payload_value, roles=[role])


def _count_persisted_document_points(
    *,
    doc_id: int,
    paper_id: str | None,
    source_type=None,
    venue_tier: str | None = None,
) -> int:
    """Create/use the shared Qdrant client off the async request event loop."""
    from core.storage.corpus_store import paper_store

    return _count_qdrant_points_for_document(
        paper_store(),
        doc_id=doc_id,
        paper_id=paper_id,
        source_type=source_type,
        venue_tier=venue_tier,
    )


@router.get("/{doc_id}/index-status")
def get_document_index_status(doc_id: int, db: Session = Depends(get_db)):
    """文档索引状态：SQL chunks 数 + Qdrant 向量数 + indexed bool。

    同 /api/papers/{paper_id}/index-status 模式，但按 documents.id (int PK) 查询，
    给前端文档库页面显示「已索引」徽章用。Qdrant 查询失败（如 indexing tick 阻塞）
    时返回 qdrant_points_count=0 + indexed=false，不抛 500。
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    sql_chunks_count = db.query(Chunk).filter(Chunk.doc_id == doc_id).count()
    paper_row = (
        db.query(Paper.paper_id, Paper.venue_tier).filter(Paper.doc_id == doc_id).first()
    )
    paper_id, venue_tier = paper_row if paper_row is not None else (None, None)

    qdrant_points_count = 0
    try:
        qdrant_points_count = _count_persisted_document_points(
            doc_id=doc_id,
            paper_id=paper_id,
            source_type=doc.source_type,
            venue_tier=venue_tier,
        )
    except Exception as e:  # noqa: BLE001
        # W8: Qdrant indexing tick may block count. Degrade gracefully.
        import logging
        logging.getLogger(__name__).warning(
            f"[doc-index-status] Qdrant count failed for doc_id={doc_id}: {e}"
        )

    return {
        "document_id": doc_id,
        "sql_chunks_count": sql_chunks_count,
        "qdrant_points_count": qdrant_points_count,
        "indexed": qdrant_points_count > 0,
    }
