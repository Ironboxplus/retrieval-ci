"""
BM25检索器 - 支持 SQLite FTS5（默认）和 Whoosh（兼容）
"""
from typing import List, Optional

from app.models.schemas import SearchResult
from config import settings


class BM25Retriever:
    """BM25全文检索器"""

    def __init__(self, store=None):
        if store is not None:
            self.store = store
        else:
            from core.storage.fts_store import FtsStore
            self.store = FtsStore()

    def retrieve(self, query: str,
                 top_k: int = None,
                 source_type: str = None,
                 data_source: str = None) -> List[SearchResult]:
        """
        BM25检索

        Args:
            query: 查询文本
            top_k: 返回数量
            source_type: 按来源类型筛选 (webpage|paper|upload)
            data_source: 按数据来源筛选 (system|user)

        Returns:
            检索结果列表
        """
        if top_k is None:
            top_k = settings.retrieval.initial_top_k

        results = self.store.search(
            query,
            top_k=top_k,
            source_type=source_type,
            data_source=data_source
        )

        return [
            SearchResult(
                doc_id=r["doc_id"],
                chunk_id=r["chunk_id"],
                content=r["content"],
                score=r["score"],
                doc_title=r["title"],
                source=r.get("source", ""),
                metadata={
                    "level": r.get("level"),
                    "method": "bm25",
                    "source_type": r.get("source_type", ""),
                    "data_source": r.get("data_source", "")
                }
            )
            for r in results
        ]

    def retrieve_with_highlight(self, query: str,
                                 top_k: int = 20,
                                 source_type: str = None,
                                 data_source: str = None) -> List[dict]:
        """
        检索并返回高亮结果

        Args:
            query: 查询文本
            top_k: 返回数量
            source_type: 按来源类型筛选
            data_source: 按数据来源筛选

        Returns:
            包含高亮的结果
        """
        return self.store.search_with_highlight(
            query,
            top_k,
            source_type=source_type,
            data_source=data_source
        )
