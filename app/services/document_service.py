"""
文档服务 - 封装文档的增删改查业务逻辑
"""
import os
import hashlib
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session
from bs4 import BeautifulSoup

from core.db import get_db_session
from core.db.models import (
    Document, Chunk, SourceType, DataSource, DocumentStatus
)
from config import settings
from pipeline.parsers.pdf_parser import PDFParser
from pipeline.parsers.txt_parser import TXTParser
from pipeline.parsers.bookmark_parser import BookmarkParser


class DocumentService:
    """文档服务类"""

    def __init__(self, db: Session = None):
        """
        初始化文档服务

        Args:
            db: 数据库会话，如果不传则在每个方法中自动管理
        """
        self._db = db
        self._pdf_parser = PDFParser()
        self._txt_parser = TXTParser()
        self._bookmark_parser = BookmarkParser()

    def _get_db(self):
        """获取数据库会话"""
        if self._db is not None:
            return self._db
        return None

    def create_document(
        self,
        title: str,
        content: str = None,
        source: str = None,
        source_type: SourceType = SourceType.upload,
        data_source: DataSource = DataSource.user,
        file_path: str = None,
        url: str = None,
        metadata: Dict[str, Any] = None,
        db: Session = None
    ) -> Document:
        """
        创建文档记录

        Args:
            title: 文档标题
            content: 文档内容（用于计算哈希）
            source: 来源描述
            source_type: 来源类型 (webpage|paper|upload)
            data_source: 数据来源 (system|user)
            file_path: 本地文件路径
            url: 网页或论文URL
            metadata: 额外元数据
            db: 数据库会话

        Returns:
            创建的Document对象
        """
        session = db or self._get_db()
        use_context = session is None

        if use_context:
            with get_db_session() as session:
                return self._create_document_impl(
                    session, title, content, source, source_type,
                    data_source, file_path, url, metadata
                )
        else:
            return self._create_document_impl(
                session, title, content, source, source_type,
                data_source, file_path, url, metadata
            )

    def _create_document_impl(
        self,
        session: Session,
        title: str,
        content: str,
        source: str,
        source_type: SourceType,
        data_source: DataSource,
        file_path: str,
        url: str,
        metadata: Dict[str, Any]
    ) -> Document:
        """创建文档实现"""
        # 计算内容哈希
        content_hash = None
        if content:
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

        doc = Document(
            title=title,
            source=source,
            source_type=source_type,
            data_source=data_source,
            content_hash=content_hash,
            file_path=file_path,
            url=url,
            status=DocumentStatus.pending,
            metadata_json=metadata or {}
        )

        session.add(doc)
        session.flush()  # 获取ID

        return doc

    def get_document(
        self,
        doc_id: int,
        db: Session = None
    ) -> Optional[Document]:
        """
        获取文档

        Args:
            doc_id: 文档ID
            db: 数据库会话

        Returns:
            Document对象或None
        """
        session = db or self._get_db()
        use_context = session is None

        if use_context:
            with get_db_session() as session:
                return session.query(Document).filter(
                    Document.id == doc_id
                ).first()
        else:
            return session.query(Document).filter(
                Document.id == doc_id
            ).first()

    def list_documents(
        self,
        source_type: SourceType = None,
        data_source: DataSource = None,
        status: DocumentStatus = None,
        keyword: str = None,
        offset: int = 0,
        limit: int = 50,
        db: Session = None
    ) -> Tuple[List[Document], int]:
        """
        列出文档（支持筛选）

        Args:
            source_type: 按来源类型筛选
            data_source: 按数据来源筛选
            status: 按状态筛选
            keyword: 标题关键词搜索
            offset: 偏移量
            limit: 限制数量
            db: 数据库会话

        Returns:
            (文档列表, 总数)
        """
        session = db or self._get_db()
        use_context = session is None

        if use_context:
            with get_db_session() as session:
                return self._list_documents_impl(
                    session, source_type, data_source, status,
                    keyword, offset, limit
                )
        else:
            return self._list_documents_impl(
                session, source_type, data_source, status,
                keyword, offset, limit
            )

    def _list_documents_impl(
        self,
        session: Session,
        source_type: SourceType,
        data_source: DataSource,
        status: DocumentStatus,
        keyword: str,
        offset: int,
        limit: int
    ) -> Tuple[List[Document], int]:
        """列出文档实现"""
        query = session.query(Document)

        if source_type is not None:
            query = query.filter(Document.source_type == source_type)
        if data_source is not None:
            query = query.filter(Document.data_source == data_source)
        if status is not None:
            query = query.filter(Document.status == status)
        if keyword:
            query = query.filter(Document.title.ilike(f"%{keyword}%"))

        total = query.count()
        documents = query.order_by(Document.created_at.desc()).offset(offset).limit(limit).all()

        return documents, total

    def update_document(
        self,
        doc_id: int,
        title: str = None,
        source: str = None,
        status: DocumentStatus = None,
        metadata: Dict[str, Any] = None,
        db: Session = None
    ) -> Optional[Document]:
        """
        更新文档

        Args:
            doc_id: 文档ID
            title: 新标题
            source: 新来源描述
            status: 新状态
            metadata: 新元数据（会与现有合并）
            db: 数据库会话

        Returns:
            更新后的Document对象或None
        """
        session = db or self._get_db()
        use_context = session is None

        if use_context:
            with get_db_session() as session:
                return self._update_document_impl(
                    session, doc_id, title, source, status, metadata
                )
        else:
            return self._update_document_impl(
                session, doc_id, title, source, status, metadata
            )

    def _update_document_impl(
        self,
        session: Session,
        doc_id: int,
        title: str,
        source: str,
        status: DocumentStatus,
        metadata: Dict[str, Any]
    ) -> Optional[Document]:
        """更新文档实现"""
        doc = session.query(Document).filter(Document.id == doc_id).first()
        if doc is None:
            return None

        if title is not None:
            doc.title = title
        if source is not None:
            doc.source = source
        if status is not None:
            doc.status = status
        if metadata is not None:
            existing_meta = doc.metadata_json or {}
            existing_meta.update(metadata)
            doc.metadata_json = existing_meta

        session.flush()
        return doc

    def delete_document(
        self,
        doc_id: int,
        delete_index: bool = True,
        db: Session = None
    ) -> bool:
        """
        删除文档及其chunks、索引

        Args:
            doc_id: 文档ID
            delete_index: 是否同时删除索引
            db: 数据库会话

        Returns:
            是否删除成功
        """
        session = db or self._get_db()
        use_context = session is None

        if use_context:
            with get_db_session() as session:
                return self._delete_document_impl(session, doc_id, delete_index)
        else:
            return self._delete_document_impl(session, doc_id, delete_index)

    def _delete_document_impl(
        self,
        session: Session,
        doc_id: int,
        delete_index: bool
    ) -> bool:
        """删除文档实现"""
        doc = session.query(Document).filter(Document.id == doc_id).first()
        if doc is None:
            return False

        # 如果需要删除索引，调用索引服务（singleton，避免每次重新 load BGE-M3 模型）
        if delete_index:
            from app.services.index_service import get_index_service
            get_index_service().delete_document_index(doc_id)

        # 删除文档（chunks会级联删除）
        session.delete(doc)
        session.flush()

        return True

    def process_uploaded_file(
        self,
        file_path: str,
        filename: str = None,
        db: Session = None,
        existing_doc: Document = None,
        title: str = None,
        base_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        处理上传的文件，根据类型分发

        Args:
            file_path: 文件路径
            filename: 原始文件名（用于判断类型）
            db: 数据库会话

        Returns:
            处理结果字典，包含:
            - success: 是否成功
            - doc_id: 文档ID（如果创建了文档）
            - content: 提取的内容
            - metadata: 提取的元数据
            - urls: 提取的URL列表（书签文件）
            - error: 错误信息
        """
        if filename is None:
            filename = os.path.basename(file_path)

        ext = Path(filename).suffix.lower()
        base_metadata = dict(base_metadata or {})

        try:
            if ext == '.pdf':
                return self._process_pdf(
                    file_path, filename, db,
                    existing_doc=existing_doc,
                    title=title,
                    base_metadata=base_metadata
                )
            elif ext == '.txt' or ext == '.md':
                return self._process_txt(
                    file_path, filename, db,
                    existing_doc=existing_doc,
                    title=title,
                    base_metadata=base_metadata
                )
            elif ext in ('.html', '.htm'):
                return self._process_html_bookmark(
                    file_path, filename, db,
                    existing_doc=existing_doc,
                    title=title,
                    base_metadata=base_metadata
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {ext}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _process_pdf(
        self,
        file_path: str,
        filename: str,
        db: Session,
        existing_doc: Document = None,
        title: str = None,
        base_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """处理PDF文件"""
        content, extracted_metadata = self._pdf_parser.parse(file_path)
        metadata = dict(base_metadata or {})
        metadata.update(extracted_metadata)

        resolved_title = title or metadata.get("title") or filename

        doc = self._upsert_uploaded_document(
            filename=filename,
            file_path=file_path,
            content=content,
            title=resolved_title,
            metadata=metadata,
            db=db,
            existing_doc=existing_doc
        )

        return {
            "success": True,
            "doc_id": doc.id,
            "content": content,
            "metadata": metadata,
            "file_type": "pdf",
            "title": resolved_title,
            "source": filename
        }

    def _process_txt(
        self,
        file_path: str,
        filename: str,
        db: Session,
        existing_doc: Document = None,
        title: str = None,
        base_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """处理TXT/MD文件"""
        content, extracted_metadata = self._txt_parser.parse(file_path)
        metadata = dict(base_metadata or {})
        metadata.update(extracted_metadata)

        # 尝试从内容中提取标题（第一行非空行）
        lines = content.split('\n')
        resolved_title = title or filename
        for line in lines:
            line = line.strip()
            if line:
                # 移除Markdown标题符号
                if not title:
                    resolved_title = line.lstrip('#').strip()[:200]
                break

        doc = self._upsert_uploaded_document(
            filename=filename,
            file_path=file_path,
            content=content,
            title=resolved_title,
            metadata=metadata,
            db=db,
            existing_doc=existing_doc
        )

        return {
            "success": True,
            "doc_id": doc.id,
            "content": content,
            "metadata": metadata,
            "file_type": "txt",
            "title": resolved_title,
            "source": filename
        }

    def _process_html_bookmark(
        self,
        file_path: str,
        filename: str,
        db: Session,
        existing_doc: Document = None,
        title: str = None,
        base_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """处理HTML书签文件"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()

        bookmarks = self._bookmark_parser.parse(html_content)
        urls = self._bookmark_parser.get_urls()

        # 创建一个汇总文档
        content_parts = []
        if bookmarks:
            for bm in bookmarks:
                folder_info = f"[{bm.folder}] " if bm.folder else ""
                content_parts.append(f"{folder_info}{bm.title}\n{bm.url}")
        else:
            extracted_links = self.extract_links_from_html(html_content)
            urls = [item["url"] for item in extracted_links]
            for link in extracted_links:
                label = link.get("text") or link.get("title") or link["url"]
                content_parts.append(f"{label}\n{link['url']}")

        content = "\n\n".join(content_parts)

        metadata = dict(base_metadata or {})
        metadata.update({
            "bookmark_count": len(bookmarks),
            "urls_count": len(urls),
            "parsed_as": "bookmark_export" if bookmarks else "generic_html_links",
        })
        resolved_title = title or f"Bookmarks from {filename}"

        doc = self._upsert_uploaded_document(
            filename=filename,
            file_path=file_path,
            content=content,
            title=resolved_title,
            metadata=metadata,
            db=db,
            existing_doc=existing_doc
        )

        return {
            "success": True,
            "doc_id": doc.id,
            "content": content,
            "metadata": metadata,
            "urls": urls,
            "bookmarks": [
                {
                    "title": bm.title,
                    "url": bm.url,
                    "folder": bm.folder
                }
                for bm in bookmarks
            ],
            "file_type": "bookmark",
            "title": resolved_title,
            "source": filename
        }

    def _upsert_uploaded_document(
        self,
        filename: str,
        file_path: str,
        content: str,
        title: str,
        metadata: Dict[str, Any],
        db: Session,
        existing_doc: Document = None
    ) -> Document:
        """Create or update the upload document row before indexing."""
        if existing_doc is not None:
            existing_doc.title = title
            existing_doc.source = filename
            existing_doc.source_type = SourceType.upload
            existing_doc.data_source = DataSource.user
            existing_doc.file_path = file_path
            existing_doc.metadata_json = metadata
            existing_doc.content_hash = (
                hashlib.sha256(content.encode('utf-8')).hexdigest()
                if content else None
            )
            existing_doc.status = DocumentStatus.pending
            if db is not None:
                db.flush()
            return existing_doc

        return self.create_document(
            title=title,
            content=content,
            source=filename,
            source_type=SourceType.upload,
            file_path=file_path,
            metadata=metadata,
            db=db
        )

    def extract_links_from_html(
        self,
        html_content: str,
        base_url: str = None
    ) -> List[Dict[str, str]]:
        """
        从HTML中提取链接

        Args:
            html_content: HTML内容
            base_url: 基础URL（用于解析相对链接）

        Returns:
            链接列表，每个元素包含 url, text, title
        """
        soup = BeautifulSoup(html_content, 'lxml')
        links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '')

            # 跳过锚点和javascript链接
            if href.startswith('#') or href.startswith('javascript:'):
                continue

            # 处理相对URL
            if base_url and not href.startswith(('http://', 'https://')):
                from urllib.parse import urljoin
                href = urljoin(base_url, href)

            # 只保留http/https链接
            if not href.startswith(('http://', 'https://')):
                continue

            link_info = {
                "url": href,
                "text": a_tag.get_text(strip=True) or href,
                "title": a_tag.get('title', '')
            }
            links.append(link_info)

        return links

    def check_duplicate(
        self,
        content: str,
        db: Session = None
    ) -> Optional[Document]:
        """
        检查内容是否重复

        Args:
            content: 文档内容
            db: 数据库会话

        Returns:
            重复的Document对象或None
        """
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

        session = db or self._get_db()
        use_context = session is None

        if use_context:
            with get_db_session() as session:
                return session.query(Document).filter(
                    Document.content_hash == content_hash
                ).order_by(Document.id.desc()).first()
        else:
            return session.query(Document).filter(
                Document.content_hash == content_hash
            ).order_by(Document.id.desc()).first()
