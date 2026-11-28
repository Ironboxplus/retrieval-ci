"""Pydantic 契约 for the match_doc module（功能2：doc → 最相近论文匹配）。"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class QueryChunk(BaseModel):
    """查询文档被切成的一段。idx 是它在文档里的顺序位置。"""
    idx: int
    text: str
    level: Optional[int] = None  # chunker 给的层级（1=L1摘要 / 2=L2段落），可选


class MatchedSpan(BaseModel):
    """一条命中明细：你的第 query_chunk_idx 段 ↔ 某论文的 paper_chunk，相似度 sim。"""
    query_chunk_idx: int
    paper_chunk: str
    sim: float


class PaperMatch(BaseModel):
    """一篇被匹配到的论文卡片。"""
    paper_id: str
    doc_id: str
    title: str = ""
    venue: str = ""
    year: Optional[int] = None
    score: float
    coverage: int = 0  # 你有多少个查询块在这篇论文里找到匹配
    metadata: Dict[str, Any] = {}
    matched: List[MatchedSpan] = []


class MatchDocRequest(BaseModel):
    """匹配请求。三种输入二选一填对应字段。"""
    input_type: Literal["text", "pdf", "paper_id"]
    text: Optional[str] = Field(default=None, description="input_type=text：直接粘贴的文本")
    upload_task_id: Optional[str] = Field(
        default=None, description="input_type=pdf：异步 extract 完成后的 task_id（产物 MD）"
    )
    paper_id: Optional[str] = Field(
        default=None, description="input_type=paper_id：已入库论文，找它的内容邻居"
    )
    top_k: int = Field(default=20, ge=1, le=200, description="返回论文数")
    per_chunk_k: int = Field(
        default=50, ge=1, le=200, description="每个查询块的召回深度（聚合前）"
    )
    agg: Literal["coverage", "maxsim"] = Field(
        default="coverage",
        description="coverage=整篇重合（每段都被覆盖才高分，默认）；maxsim=最强单段命中",
    )
    sim_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="低于此相似度的命中忽略"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="venue/year/venue_tier 等，复用现有 filter schema"
    )
    max_query_chunks: int = Field(
        default=40, ge=1, le=200, description="查询文档切块上限，防超长文档放大检索成本"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "文本 → 最相近论文（相关工作发现）",
                    "value": {
                        "input_type": "text",
                        "text": "We propose a retrieval-augmented method that reranks passages with a cross-encoder before generation.",
                        "top_k": 20,
                    },
                },
                {
                    "summary": "已入库论文 → 内容邻居（自动排除自身）",
                    "value": {"input_type": "paper_id", "paper_id": "neurips-2024-some-paper", "top_k": 20},
                },
            ]
        }
    )


class MatchDocResponse(BaseModel):
    input_type: str
    query_chunks: List[QueryChunk] = []
    papers: List[PaperMatch] = []
    metrics: Dict[str, Any] = {}


class MatchDocJobReceipt(BaseModel):
    status: Literal["pending"]
    task_id: int


class MatchDocJobStatus(BaseModel):
    """One durable match job as its submitter sees it.

    Served by owner-scoped routes rather than the generic `/api/tasks/{id}`:
    task ids are sequential integers, so on the public surface those routes let
    anyone read or cancel anyone else's job (including other task types).
    """

    task_id: int
    status: str
    progress: int = 0
    total: int = 0
    error: Optional[str] = None
    result: Optional[MatchDocResponse] = None
