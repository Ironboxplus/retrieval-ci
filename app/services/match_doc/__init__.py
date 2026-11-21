"""match_doc — 独立模块：给定一篇查询文档（粘贴文本 / 上传 PDF / 已入库论文），
找语料库里内容最相近的论文（doc → paper 匹配，功能2）。

设计为自包含、对正常检索业务零侵入：
  - aggregator.py  纯函数：多查询块的候选 chunk → 论文级打分（coverage / maxsim）。DB/Qdrant-free。
  - adapters.py    把三种输入归一成 QueryChunk 列表（注入 chunker，可离线测）。
  - service.py     编排：adapt → 逐块检索（复用已暖的 SearchPipeline）→ 聚合 → 返回。
  - schemas.py     Pydantic 契约。

复用现有：SearchPipeline / BGE-M3 embedder / Qdrant store / bge-reranker / section_l1 chunker。
不新建模型实例、不改动现有检索路由。
"""
from app.services.match_doc.aggregator import aggregate_matches
from app.services.match_doc.adapters import build_query_chunks
from app.services.match_doc.schemas import (
    MatchDocRequest,
    MatchDocResponse,
    MatchedSpan,
    PaperMatch,
    QueryChunk,
)
from app.services.match_doc.service import MatchDocService, get_match_doc_service

__all__ = [
    "MatchDocRequest",
    "MatchDocResponse",
    "MatchedSpan",
    "PaperMatch",
    "QueryChunk",
    "aggregate_matches",
    "build_query_chunks",
    "MatchDocService",
    "get_match_doc_service",
]
