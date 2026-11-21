"""输入归一层（功能2）：三种输入 → 统一 `QueryChunk` 列表。

设计为 DB/Qdrant-free —— 真正的数据访问（论文本体块、上传 extract 的 MD）通过
注入的 loader 回调进来，所以本模块离线可测、对正常业务零侵入：

  - load_paper_chunks(paper_id) -> Sequence[(content, level)]   已入库论文的本体块
  - load_upload_md(task_id)     -> str                          上传 PDF extract 出的 MD 全文

text / pdf 输入复用语料同款 chunker 切块；paper_id 输入直接拿已切好的库内块。
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence, Tuple

from app.services.match_doc.schemas import MatchDocRequest, QueryChunk

_logger = logging.getLogger(__name__)

LoadPaperChunks = Callable[[str], Sequence[Tuple[str, Optional[int]]]]
LoadUploadMd = Callable[[str], str]


def _reindex(texts_levels: Sequence[Tuple[str, Optional[int]]],
             max_chunks: int) -> list[QueryChunk]:
    """过滤空白、重新连续编号、截断到 max_chunks。"""
    out: list[QueryChunk] = []
    for content, level in texts_levels:
        text = (content or "").strip()
        if not text:
            continue
        out.append(QueryChunk(idx=len(out), text=text, level=level))
        if len(out) >= max_chunks:
            break
    return out


def chunks_from_text(text: str, *, chunker, max_chunks: int,
                     content_type: Optional[str] = None) -> list[QueryChunk]:
    """用语料同款 chunker 把任意文本切成查询块。"""
    chunks = chunker.chunk_document("query", text or "", content_type=content_type)
    qcs = _reindex([(c.content, getattr(c, "level", None)) for c in chunks], max_chunks)
    _logger.info("match_doc.adapt.text raw_chunks=%d query_chunks=%d", len(chunks), len(qcs))
    return qcs


def chunks_from_paper(paper_id: str, *, load_paper_chunks: LoadPaperChunks,
                      max_chunks: int) -> list[QueryChunk]:
    """已入库论文 → 它自己的本体块作为查询块（找内容邻居）。"""
    rows = load_paper_chunks(paper_id) or []
    qcs = _reindex([(c, lv) for (c, lv) in rows], max_chunks)
    _logger.info("match_doc.adapt.paper paper_id=%s rows=%d query_chunks=%d",
                 paper_id, len(rows), len(qcs))
    return qcs


def build_query_chunks(req: MatchDocRequest, *, chunker,
                       load_paper_chunks: Optional[LoadPaperChunks] = None,
                       load_upload_md: Optional[LoadUploadMd] = None) -> list[QueryChunk]:
    """按 input_type 分派到对应 adapter。字段缺失 / loader 未注入 → ValueError。"""
    max_chunks = req.max_query_chunks
    if req.input_type == "text":
        if not (req.text and req.text.strip()):
            raise ValueError("input_type=text 需要非空 text")
        return chunks_from_text(req.text, chunker=chunker, max_chunks=max_chunks)

    if req.input_type == "paper_id":
        if not req.paper_id:
            raise ValueError("input_type=paper_id 需要 paper_id")
        if load_paper_chunks is None:
            raise ValueError("paper_id 输入需要 load_paper_chunks loader")
        return chunks_from_paper(req.paper_id, load_paper_chunks=load_paper_chunks,
                                 max_chunks=max_chunks)

    if req.input_type == "pdf":
        if not req.upload_task_id:
            raise ValueError("input_type=pdf 需要 upload_task_id")
        if load_upload_md is None:
            raise ValueError("pdf 输入需要 load_upload_md loader")
        md = load_upload_md(req.upload_task_id) or ""
        return chunks_from_text(md, chunker=chunker, max_chunks=max_chunks,
                                content_type="markdown")

    raise ValueError(f"未知 input_type: {req.input_type}")
