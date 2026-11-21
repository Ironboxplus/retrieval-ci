from __future__ import annotations

"""
索引服务 - 封装文档索引的业务逻辑

支持内容去重和 upsert 语义，避免重复索引问题
"""
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime
import os
import threading

from sqlalchemy.orm import Session
from sqlalchemy.orm import selectinload

from core.db import get_db_session
from core.db.models import Document, Chunk, DocumentStatus
from config import settings
from app.models.schemas import Chunk as SchemaChunk

if TYPE_CHECKING:
    from core.indexing.chunker import Chunker
    from core.storage.qdrant_store import QdrantStore


_SOURCE_TYPE_TO_CONTENT_TYPE = {
    "webpage": "html",
}


class IndexService:
    """索引服务类"""

    def __init__(
        self,
        collection_name: str = "papers",
        embedding_device: Optional[str] = None,
        require_gpu: bool = False,
        chunker_type: Optional[str] = None,
    ):
        """
        初始化索引服务

        Args:
            collection_name: ChromaDB集合名称
            embedding_device: 显式 embedding 设备，例如 "cuda" 或 "cpu"
            require_gpu: 是否要求索引必须运行在 CUDA 上
            chunker_type: 指定分块策略。``None`` (默认) 走多粒度
                ``TextChunker``，保留 6.68M 历史 chunk 的语义；
                ``"markdown"``/``"html"``/``"text"`` 走
                ``SectionAwareChunker``（推荐用于 MinerU 抽取的新语料）。
        """
        self._collection_name = collection_name
        self._embedding_device = embedding_device
        self._require_gpu = require_gpu
        self._chunker_type = chunker_type
        self._chunker = None
        self._qdrant_store = None
        self._new_embedder = None

    @property
    def chunker(self) -> "Chunker":
        """懒加载分块器（按 ``chunker_type`` 走对应策略）。"""
        if self._chunker is None:
            from core.indexing.chunker import create_chunker, TextChunker

            if self._chunker_type is None:
                self._chunker = TextChunker()
            else:
                self._chunker = create_chunker(content_type=self._chunker_type)
        return self._chunker

    @property
    def new_embedder(self):
        """Process-wide shared BGE-M3 embedder.

        W46 (2026-05-20): routed through `get_embedder()` so all
        IndexService variants (one per chunker_type via W36 cache) share
        a single hot-in-GPU instance. Was per-instance lazy → N variants
        meant N model copies wasting GPU memory.
        """
        if self._new_embedder is None:
            from core.indexing.embedders import get_embedder
            self._new_embedder = get_embedder()
        return self._new_embedder

    def _ensure_index_device_ready(self) -> None:
        """在要求 GPU 时，拒绝静默回退到 CPU。"""
        if not self._require_gpu:
            return

        device = getattr(self.new_embedder, 'device', 'cpu')
        if device != "cuda":
            raise RuntimeError(
                "GPU indexing is required for this run, but the embedder is not using CUDA."
            )

    @property
    def qdrant_store(self) -> "QdrantStore":
        if self._qdrant_store is None:
            from core.storage.qdrant_store import QdrantStore
            self._qdrant_store = QdrantStore(collection_name=self._collection_name)
        return self._qdrant_store

    chroma_store = qdrant_store

    def index_document(
        self,
        doc_id: int,
        content: str,
        title: str = None,
        source: str = None,
        metadata: Dict[str, Any] = None,
        include_summary: bool = True,
        db: Session = None,
        source_type: str = None,
        data_source: str = None,
        force_reindex: bool = False,
        skip_whoosh: bool = False,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        对单个文档建立索引（支持 upsert 语义，自动去重）

        使用内容 hash 作为 chunk ID 的一部分，确保：
        1. 重新索引相同内容不会产生重复
        2. 内容变更时自动更新

        Args:
            doc_id: 文档ID
            content: 文档内容
            title: 文档标题
            source: 文档来源
            metadata: 额外元数据
            include_summary: 是否包含文档摘要
            db: 数据库会话
            source_type: 来源类型 (webpage|paper|upload)
            data_source: 数据来源 (system|user)
            force_reindex: 是否强制重建索引（先删除旧数据）

        Returns:
            索引结果，包含:
            - success: 是否成功
            - chunk_count: 分块数量
            - chunks: 分块ID列表
            - error: 错误信息
        """
        try:
            self._ensure_index_device_ready()

            if not (content or "").strip():
                return {
                    "success": False,
                    "chunk_count": 0,
                    "chunks": [],
                    "error": "Content is empty",
                }

            # 使用字符串doc_id作为分块的关联ID
            str_doc_id = str(doc_id)

            # 如果强制重建索引，先删除旧数据
            if force_reindex:
                deleted_count = self.chroma_store.delete_by_doc_id(str_doc_id)
                if deleted_count > 0:
                    print(f"Force reindex: deleted {deleted_count} old chunks from ChromaDB for doc_id={doc_id}")

            # 分块（HTML 源默认走 content_type='html' 让 chunker 剥标签）
            effective_content_type = content_type
            if effective_content_type is None and source_type:
                effective_content_type = _SOURCE_TYPE_TO_CONTENT_TYPE.get(source_type)
            chunk_kwargs = {
                "doc_id": str_doc_id,
                "text": content,
                "include_summary": include_summary,
            }
            if effective_content_type:
                chunk_kwargs["content_type"] = effective_content_type
            chunks = self.chunker.chunk_document(**chunk_kwargs)

            if not chunks:
                return {
                    "success": False,
                    "error": "No chunks generated from content"
                }

            # 生成嵌入向量（BGE-M3: dense + sparse）
            chunks, sparse_vectors = self.new_embedder.embed_chunks(chunks)

            # 为chunks添加元数据
            extra_metadata = {
                "title": title or "",
                "source": source or "",
                "doc_id": str_doc_id,
                **(metadata or {})
            }

            for chunk in chunks:
                chunk.metadata.update(extra_metadata)

            # 存储到向量数据库（默认使用 upsert 语义）
            self.chroma_store.add_chunks(
                chunks,
                source_type=source_type,
                data_source=data_source,
                use_upsert=True,
                sparse_vectors=sparse_vectors,
            )

            # Whoosh removed — sparse vectors in Qdrant replace BM25

            # 更新数据库中的文档状态和chunks
            self._save_chunks_to_db(doc_id, chunks, db)
            return {
                "success": True,
                "chunk_count": len(chunks),
                "chunks": [c.id for c in chunks],
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _save_chunks_to_db(
        self,
        doc_id: int,
        chunks: list,
        db: Session = None
    ):
        """
        将分块信息保存到数据库

        Args:
            doc_id: 文档ID
            chunks: 分块列表
            db: 数据库会话
        """
        session = db
        use_context = session is None

        def _save_impl(session: Session):
            # 更新文档状态
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = DocumentStatus.indexed

            # 删除旧的chunks
            session.query(Chunk).filter(Chunk.doc_id == doc_id).delete()

            # 添加新的chunks
            for idx, chunk in enumerate(chunks):
                db_chunk = Chunk(
                    doc_id=doc_id,
                    content=chunk.content,
                    level=chunk.level,
                    chunk_index=idx,
                    start_idx=chunk.start_idx,
                    end_idx=chunk.end_idx,
                    token_count=len(chunk.content),  # 简单估计
                    metadata_json=chunk.metadata
                )
                session.add(db_chunk)

            session.flush()

        if use_context:
            with get_db_session() as session:
                _save_impl(session)
        else:
            _save_impl(session)

    def delete_document_index(
        self,
        doc_id: int,
        db: Session = None,
        delete_db_records: bool = False
    ) -> bool:
        """
        删除文档的索引

        使用新的 delete_by_doc_id 方法确保完全删除所有 chunks

        Args:
            doc_id: 文档ID
            db: 数据库会话
            delete_db_records: 是否同时删除数据库中的chunks记录
                              (当文档本身也要被删除时设为False，依赖级联删除)

        Returns:
            是否删除成功
        """
        try:
            str_doc_id = str(doc_id)

            # 从ChromaDB删除 - 使用新的 delete_by_doc_id 方法
            try:
                deleted_count = self.chroma_store.delete_by_doc_id(str_doc_id)
                if deleted_count > 0:
                    print(f"Deleted {deleted_count} chunks from ChromaDB for doc_id={doc_id}")
            except Exception as e:
                print(f"Warning: Failed to delete from ChromaDB: {e}")

            # Whoosh removed — sparse vectors in Qdrant replace BM25

            # 仅在需要时删除数据库中的chunks
            if delete_db_records:
                session = db
                use_context = session is None

                if use_context:
                    with get_db_session() as session:
                        deleted_count = session.query(Chunk).filter(Chunk.doc_id == doc_id).delete()
                        print(f"Deleted {deleted_count} chunks from database for doc_id={doc_id}")
                        doc = session.query(Document).filter(Document.id == doc_id).first()
                        if doc:
                            doc.status = DocumentStatus.pending
                else:
                    deleted_count = session.query(Chunk).filter(Chunk.doc_id == doc_id).delete()
                    print(f"Deleted {deleted_count} chunks from database for doc_id={doc_id}")
                    doc = session.query(Document).filter(Document.id == doc_id).first()
                    if doc:
                        doc.status = DocumentStatus.pending

            return True

        except Exception as e:
            print(f"Error deleting document index: {e}")
            return False

    def rebuild_index(
        self,
        doc_ids: List[int] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        重建索引

        Args:
            doc_ids: 要重建的文档ID列表，如果为None则重建所有
            db: 数据库会话

        Returns:
            重建结果
        """
        session = db
        use_context = session is None

        def _rebuild_impl(session: Session) -> Dict[str, Any]:
            # 获取需要重建的文档
            query = session.query(Document)
            if doc_ids:
                query = query.filter(Document.id.in_(doc_ids))

            documents = query.all()

            success_count = 0
            failed_count = 0
            failed_docs = []

            for doc in documents:
                # 获取文档内容
                content = None

                # 尝试从文件读取
                if doc.file_path and os.path.exists(doc.file_path):
                    try:
                        from pipeline.parsers.pdf_parser import PDFParser
                        from pipeline.parsers.txt_parser import TXTParser

                        if doc.file_path.endswith('.pdf'):
                            content, _ = PDFParser().parse(doc.file_path)
                        else:
                            content, _ = TXTParser().parse(doc.file_path)
                    except Exception as e:
                        print(f"Error reading file {doc.file_path}: {e}")

                # 如果有现有的chunks，合并内容
                if not content and doc.chunks:
                    content = "\n\n".join([c.content for c in doc.chunks])

                if not content:
                    failed_count += 1
                    failed_docs.append({
                        "doc_id": doc.id,
                        "error": "No content available"
                    })
                    continue

                # 删除旧索引（重建时需要删除数据库记录）
                self.delete_document_index(doc.id, session, delete_db_records=True)

                # 建立新索引
                result = self.index_document(
                    doc_id=doc.id,
                    content=content,
                    title=doc.title,
                    source=doc.source,
                    metadata=doc.metadata_json,
                    db=session
                )

                if result["success"]:
                    success_count += 1
                else:
                    failed_count += 1
                    failed_docs.append({
                        "doc_id": doc.id,
                        "error": result.get("error", "Unknown error")
                    })

            return {
                "success": True,
                "total": len(documents),
                "success_count": success_count,
                "failed_count": failed_count,
                "failed_docs": failed_docs
            }

        if use_context:
            with get_db_session() as session:
                return _rebuild_impl(session)
        else:
            return _rebuild_impl(session)

    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息

        Returns:
            统计信息字典
        """
        try:
            vector_count = self.qdrant_store.count()

            return {
                "vector_index_size": vector_count,
                "collection_name": self._collection_name,
            }
        except Exception as e:
            return {
                "error": str(e)
            }

    def backfill_vector_index_from_db(
        self,
        db: Session = None,
        batch_size: int = 128
    ) -> Dict[str, Any]:
        """
        从 SQLite chunks 回填向量索引。

        仅补齐 ChromaDB，不修改 Whoosh 或现有 SQLite 行。
        """
        session = db
        use_context = session is None

        def _backfill_impl(session: Session) -> Dict[str, Any]:
            total_chunks = session.query(Chunk).count()
            vector_count = self.chroma_store.count()

            if total_chunks == 0:
                return {
                    "success": True,
                    "repaired": False,
                    "reason": "no_chunks",
                    "sqlite_chunks": 0,
                    "vector_chunks": vector_count
                }

            if vector_count >= total_chunks:
                return {
                    "success": True,
                    "repaired": False,
                    "reason": "vector_index_already_populated",
                    "sqlite_chunks": total_chunks,
                    "vector_chunks": vector_count
                }

            repaired_count = 0
            chunk_rows = (
                session.query(Chunk)
                .options(selectinload(Chunk.document))
                .order_by(Chunk.id.asc())
                .all()
            )

            for start in range(0, len(chunk_rows), batch_size):
                batch = chunk_rows[start:start + batch_size]
                schema_chunks = []
                source_rows = []
                for row in batch:
                    doc = row.document
                    if doc is None or not row.content:
                        continue

                    schema_chunks.append(
                        SchemaChunk(
                            id=f"db_chunk_{row.id}",
                            doc_id=str(row.doc_id),
                            content=row.content,
                            level=row.level or 0,
                            start_idx=row.start_idx or 0,
                            end_idx=row.end_idx or len(row.content),
                            metadata=row.metadata_json or {}
                        )
                    )
                    source_rows.append(row)

                if not schema_chunks:
                    continue

                embedded_chunks = self.embedder.embed_chunks(schema_chunks, batch_size=batch_size)
                for chunk, row in zip(embedded_chunks, source_rows):
                    doc = row.document
                    if doc is None:
                        continue
                    chunk.metadata.update({
                        "title": doc.title or "",
                        "source": doc.source or "",
                        "doc_id": str(row.doc_id),
                        "source_type": doc.source_type.value if doc.source_type else "",
                        "data_source": doc.data_source.value if doc.data_source else ""
                    })

                self.chroma_store.add_chunks(
                    embedded_chunks,
                    use_upsert=True
                )
                repaired_count += len(embedded_chunks)

            final_vector_count = self.chroma_store.count()
            return {
                "success": True,
                "repaired": repaired_count > 0,
                "sqlite_chunks": total_chunks,
                "vector_chunks": final_vector_count,
                "repaired_chunks": repaired_count
            }

        if use_context:
            with get_db_session() as session:
                return _backfill_impl(session)
        return _backfill_impl(session)


_INSTANCE_CACHE: Dict[str, "IndexService"] = {}
_CACHE_LOCK = threading.Lock()


def get_index_service(
    chunker_type: Optional[str] = None,
    *,
    collection_name: str = "papers",
    embedding_device: Optional[str] = None,
    require_gpu: bool = False,
) -> "IndexService":
    """Return a process-wide cached IndexService keyed by chunker_type.

    Each IndexService lazily loads BGE-M3 (~6.4GB on disk → CUDA memory) on
    first use; constructing one per crawl iteration both wastes time and races
    on the same shared model checkpoint file under concurrent workers. The
    cache makes "give me the service for this chunker type" a true singleton.

    Pass `chunker_type=None` for the legacy multigranularity TextChunker (paper
    preindex back-compat); pass `'html'`/`'section_l1'`/etc. for new pipelines.
    """
    key = chunker_type or "__default__"
    cached = _INSTANCE_CACHE.get(key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _INSTANCE_CACHE.get(key)
        if cached is not None:
            return cached
        cached = IndexService(
            collection_name=collection_name,
            embedding_device=embedding_device,
            require_gpu=require_gpu,
            chunker_type=chunker_type,
        )
        _INSTANCE_CACHE[key] = cached
        return cached
