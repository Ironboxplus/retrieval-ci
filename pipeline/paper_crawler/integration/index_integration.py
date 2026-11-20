"""
PaperIndexer - 论文索引器

将论文 API 对齐到项目当前真实运行拓扑：
- 写入使用 IndexService，确保 SQLite + Chroma + Whoosh 一致
- 向量检索使用独立的 papers collection
- 全文检索共享 Whoosh，但强制按 source_type=paper 过滤
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import func

# 添加项目根目录到路径以支持相对导入
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.cache import DISTRIBUTION_TTL_SECONDS, TOTAL_COUNT_TTL_SECONDS, stats_cache
from core.db.database import get_db_session
from core.db.models import (
    Chunk as DBChunk,
    DataSource,
    Document as DBDocument,
    DocumentStatus,
    Paper as DBPaper,
    SourceType,
)
from core.indexing.embedders import get_embedder as _get_shared_embedder
from core.indexing.embedding_profile import current_embed_fp
from app.models.schemas import AggregatedResult, SearchResult
from app.services.index_service import IndexService
from core.storage.qdrant_store import QdrantStore as ChromaStore

DEFAULT_PAPER_COLLECTION_NAME = "papers"
DEFAULT_PAPER_DATA_SOURCE = "system"


def existing_doc_id(db, paper_id: str) -> Optional[int]:
    """The SQL document currently holding `paper_id`'s text, or None.

    `papers.doc_id` (UNIQUE FK -> documents.id) is the one mapping from a business
    paper id to its document; it is written in the same transaction as the
    document, so it cannot disagree with it the way a side JSON file could.
    """
    return db.query(DBPaper.doc_id).filter(DBPaper.paper_id == paper_id).scalar()


def _load_module_from_path(module_name: str, module_path: Path):
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_paper_models():
    """延迟导入论文模型，避免循环依赖。"""
    try:
        from pipeline.paper_crawler.models.paper import Conference, Paper, PaperSource, PaperType
    except Exception:
        models_module = _load_module_from_path(
            "_retrival_paper_models_runtime",
            _project_root / "pipeline" / "paper_crawler" / "models" / "paper.py",
        )
        Conference = models_module.Conference
        Paper = models_module.Paper
        PaperSource = models_module.PaperSource
        PaperType = models_module.PaperType

    return Paper, PaperSource, PaperType, Conference


@dataclass
class PaperIndexConfig:
    """论文索引配置。"""

    collection_name: str = DEFAULT_PAPER_COLLECTION_NAME
    include_summary: bool = True
    index_metadata: bool = True
    generate_structured_summary: bool = True
    embedding_model: Optional[str] = None
    embedding_device: Optional[str] = None
    require_gpu: bool = False
    store_paper_metadata: bool = True
    max_concurrent_indexing: int = 5
    progress_callback: Optional[Callable[[int, int, str], None]] = None
    data_source: str = DEFAULT_PAPER_DATA_SOURCE
    # ``None`` keeps the legacy multi-granularity TextChunker (back-compat
    # with the 6.68M historical points). New MinerU-driven ingest should
    # pass ``chunker_type="markdown"`` to use SectionAwareChunker.
    chunker_type: Optional[str] = None


@dataclass
class PaperSearchResult:
    """论文聚合搜索结果。"""

    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    source: str
    conference: str
    year: int
    paper_type: str
    score: float
    hit_count: int
    matched_chunks: List[SearchResult] = field(default_factory=list)
    pdf_url: Optional[str] = None
    paper_page_url: Optional[str] = None
    arxiv_id: Optional[str] = None
    code_url: Optional[str] = None
    pdf_path: Optional[str] = None
    markdown_path: Optional[str] = None
    citations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "source": self.source,
            "conference": self.conference,
            "year": self.year,
            "paper_type": self.paper_type,
            "score": self.score,
            "hit_count": self.hit_count,
            "matched_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "score": chunk.score,
                }
                for chunk in self.matched_chunks
            ],
            "pdf_url": self.pdf_url,
            "paper_page_url": self.paper_page_url,
            "arxiv_id": self.arxiv_id,
            "code_url": self.code_url,
            "pdf_path": self.pdf_path,
            "markdown_path": self.markdown_path,
            "citations": self.citations,
            "metadata": self.metadata,
        }


class PaperIndexer:
    """论文索引器。"""

    def __init__(self, config: Optional[PaperIndexConfig] = None):
        self.config = config or PaperIndexConfig()
        self._index_service: Optional[IndexService] = None
        self._new_embedder = None
        self._chroma_store: Optional[ChromaStore] = None

    @property
    def index_service(self) -> IndexService:
        if self._index_service is None:
            self._index_service = IndexService(
                collection_name=self.config.collection_name,
                embedding_device=self.config.embedding_device,
                require_gpu=self.config.require_gpu,
                chunker_type=self.config.chunker_type,
            )
        return self._index_service

    @property
    def new_embedder(self):
        """BGE-M3 embedder with dense + sparse support."""
        if self._new_embedder is None:
            self._new_embedder = _get_shared_embedder()
        return self._new_embedder

    @property
    def chroma_store(self) -> ChromaStore:
        if self._chroma_store is None:
            # `papers` resolves to the routed papers/arxiv/reviews corpus.
            from core.storage.corpus_store import paper_store
            self._chroma_store = paper_store(self.config.collection_name)
        return self._chroma_store

    def _paper_source_value(self, paper) -> Optional[str]:
        source = getattr(paper, "source", None)
        return source.value if hasattr(source, "value") else source

    def _paper_type_value(self, paper) -> str:
        paper_type = getattr(paper, "paper_type", None)
        if hasattr(paper_type, "value"):
            return paper_type.value
        if paper_type:
            return str(paper_type)
        return "unknown"

    def _conference_value(self, paper) -> str:
        conference = getattr(paper, "conference", None)
        if hasattr(conference, "value"):
            return conference.value
        if conference:
            return str(conference)
        metadata = getattr(paper, "metadata", None)
        if metadata and getattr(metadata, "venue", None):
            return str(metadata.venue)
        return "Other"

    def _paper_year(self, paper) -> int:
        year = getattr(paper, "year", None)
        if year:
            return int(year)
        metadata = getattr(paper, "metadata", None)
        if metadata and getattr(metadata, "year", None):
            return int(metadata.year)
        raise ValueError("Paper year is required")

    def _paper_authors(self, paper) -> List[str]:
        authors = []
        for author in getattr(paper, "authors", []) or []:
            authors.append(author.name if hasattr(author, "name") else str(author))
        return authors

    def _paper_keywords(self, paper) -> List[str]:
        keywords = list(getattr(paper, "keywords", []) or [])
        metadata = getattr(paper, "metadata", None)
        if metadata and getattr(metadata, "keywords", None):
            for keyword in metadata.keywords:
                if keyword not in keywords:
                    keywords.append(keyword)
        return keywords

    def _build_paper_metadata(self, paper, markdown_path: Optional[Path]) -> Dict[str, Any]:
        metadata = {
            "paper_id": paper.id,
            "title": paper.title,
            "abstract": getattr(paper, "abstract", "") or "",
            "authors": self._paper_authors(paper),
            "source": self._paper_source_value(paper) or "",
            "paper_type": self._paper_type_value(paper),
            "conference": self._conference_value(paper),
            "year": self._paper_year(paper),
            "keywords": self._paper_keywords(paper),
            "doi": getattr(paper, "doi", None),
            "arxiv_id": getattr(paper, "arxiv_id", None),
            "pdf_url": str(getattr(paper, "pdf_url", "")) if getattr(paper, "pdf_url", None) else None,
            "paper_page_url": str(getattr(paper, "paper_page_url", "")) if getattr(paper, "paper_page_url", None) else None,
            "code_url": str(getattr(paper, "code_url", "")) if getattr(paper, "code_url", None) else None,
            "citations": getattr(paper, "citations", 0) or 0,
            "venue_track": getattr(paper, "venue_track", None),
            "full_text_source": getattr(paper, "full_text_source", None),
            "markdown_path": str(markdown_path) if markdown_path else None,
        }
        return {key: value for key, value in metadata.items() if value is not None}

    def _build_vector_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        vector_metadata: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                vector_metadata[key] = value
                continue
            if isinstance(value, list):
                scalar_items = [self._coerce_metadata_scalar(item) for item in value if item is not None]
                if key == "authors":
                    vector_metadata["author_names"] = " | ".join(str(item) for item in scalar_items)
                elif key == "keywords":
                    vector_metadata["keyword_terms"] = " | ".join(str(item) for item in scalar_items)
                else:
                    vector_metadata[f"{key}_json"] = json.dumps(scalar_items, ensure_ascii=False)
                continue
            if isinstance(value, dict):
                vector_metadata[f"{key}_json"] = json.dumps(
                    {sub_key: self._coerce_metadata_scalar(sub_value) for sub_key, sub_value in value.items()},
                    ensure_ascii=False,
                )
                continue
            vector_metadata[key] = self._coerce_metadata_scalar(value)
        return vector_metadata

    def _coerce_metadata_scalar(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {key: self._coerce_metadata_scalar(sub_value) for key, sub_value in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._coerce_metadata_scalar(item) for item in value]
        return str(getattr(value, "name", value))

    def _build_structured_summary(self, paper) -> str:
        parts = [f"Title: {paper.title}"]
        authors = self._paper_authors(paper)
        if authors:
            parts.append(f"Authors: {', '.join(authors[:5])}")
        parts.append(f"Published: {self._conference_value(paper)} {self._paper_year(paper)}")
        paper_type = self._paper_type_value(paper)
        if paper_type != "unknown":
            parts.append(f"Type: {paper_type}")
        keywords = self._paper_keywords(paper)
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords[:10])}")
        abstract = getattr(paper, "abstract", None)
        if abstract:
            parts.append(f"\nAbstract: {abstract}")
        return "\n".join(parts)

    def _read_markdown_content(self, markdown_path: Path) -> str:
        if not markdown_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
        return markdown_path.read_text(encoding="utf-8")

    def _paper_url(self, paper) -> str:
        page_url = getattr(paper, "paper_page_url", None)
        if page_url:
            return str(page_url)
        arxiv_url = getattr(paper, "arxiv_url", None)
        if arxiv_url:
            return str(arxiv_url)
        source_url = getattr(paper, "url", None)
        if source_url:
            return str(source_url)
        return paper.id

    def _upsert_paper_row(
        self,
        db,
        doc_id: int,
        paper,
        metadata: Dict[str, Any],
        markdown_path: Optional[Path],
    ) -> DBPaper:
        # 方案 X 后：paper_id 是 PK NOT NULL，必须显式 set。
        # dataclass Paper.id 即业务 paper_id 字符串 (e.g. "aaai-2022-19873")。
        biz_paper_id = getattr(paper, "id", None) or getattr(paper, "paper_id", None)
        if not biz_paper_id:
            raise ValueError(f"paper without business id; cannot insert: doc_id={doc_id}")
        # One row per paper: a re-index writes a NEW document and re-points the
        # existing row at it (papers.doc_id is UNIQUE, so the old document simply
        # stops being referenced; the caller deletes it once its vectors are gone).
        paper_row = db.get(DBPaper, str(biz_paper_id))
        if paper_row is None:
            paper_row = DBPaper(paper_id=str(biz_paper_id), doc_id=doc_id)
            db.add(paper_row)
        else:
            paper_row.doc_id = doc_id

        paper_row.arxiv_id = getattr(paper, "arxiv_id", None)
        paper_row.doi = getattr(paper, "doi", None)
        if getattr(paper, "full_text_source", None):
            paper_row.full_text_source = paper.full_text_source
        paper_row.title = paper.title
        paper_row.abstract = getattr(paper, "abstract", None)
        paper_row.authors_json = self._paper_authors(paper)
        paper_row.year = metadata.get("year")
        paper_row.venue = metadata.get("conference")
        paper_row.paper_source = metadata.get("source")
        paper_row.pdf_path = str(getattr(paper, "pdf_path", "") or "") or None
        paper_row.txt_path = str(markdown_path) if markdown_path else metadata.get("markdown_path")
        paper_row.citation_count = metadata.get("citations", 0)
        if not paper_row.venue_tier:
            # The registry owns tiers; a blanket CCF-A default mislabelled IJCAI
            # 2026 (CCF-B). A venue outside the registry stays untiered.
            from pipeline.paper_crawler.ccf_a_main_corpus import registry_venue_tier

            paper_row.venue_tier = (
                getattr(paper, "venue_tier", None) or registry_venue_tier(paper_row.venue)
            )
        if not paper_row.venue_source:
            paper_row.venue_source = paper_row.venue
        if not paper_row.track_type:
            paper_row.track_type = getattr(paper, "track_type", None) or "main"
        return paper_row

    def _prepare_paper(self, paper, markdown_path: Optional[Path], content: Optional[str] = None) -> Dict[str, Any]:
        """Prepare paper content, metadata, and chunks (CPU only, no embedding)."""
        if content is None:
            if markdown_path is None:
                raise ValueError(f"No content source for paper {paper.id}")
            content = self._read_markdown_content(markdown_path)

        metadata = self._build_paper_metadata(paper, markdown_path)
        vector_metadata = self._build_vector_metadata(metadata)
        if self.config.generate_structured_summary:
            summary = self._build_structured_summary(paper)
            if summary and not content.startswith(summary[:100]):
                content = summary + "\n\n---\n\n" + content

        return {"paper": paper, "markdown_path": markdown_path, "content": content,
                "metadata": metadata, "vector_metadata": vector_metadata}

    def _write_paper_to_stores(
        self, prepared: Dict[str, Any], chunks: List, db, *,
        skip_whoosh: bool = False, wait_qdrant: bool = True,
        sparse_vectors: Optional[List[Dict]] = None,
    ) -> str:
        """Create DB records and write pre-embedded chunks to stores. Returns doc_id."""
        paper = prepared["paper"]
        doc = DBDocument(
            title=paper.title, source=self._paper_url(paper),
            source_type=SourceType.paper, data_source=DataSource(self.config.data_source),
            file_path=str(prepared["markdown_path"]) if prepared["markdown_path"] else None,
            url=self._paper_url(paper), status=DocumentStatus.processing,
            metadata_json=prepared["metadata"],
        )
        db.add(doc)
        db.flush()

        str_doc_id = str(doc.id)
        paper_url = self._paper_url(paper)
        extra_metadata = {**(prepared["vector_metadata"] or {}),
                          "title": paper.title, "source": paper_url,
                          "doc_id": str_doc_id, "embed_fp": current_embed_fp()}
        for chunk in chunks:
            chunk.doc_id = str_doc_id
            chunk.metadata.update(extra_metadata)

        # No data_source kwarg here — D3 (papers_payload_contract_plan.md):
        # `data_source` is not part of the paper/review Qdrant payload
        # profile (JOIN documents.data_source via doc_id when needed). The
        # SQL DBDocument row above still records it; only the Qdrant payload
        # excludes it. Passing it anyway would work (the projector drops it)
        # but would fire an out-of-profile-key warning on every single chunk.
        self.index_service.chroma_store.add_chunks(
            chunks, source_type=SourceType.paper.value,
            use_upsert=True, wait=wait_qdrant, sparse_vectors=sparse_vectors,
        )
        self.index_service._save_chunks_to_db(doc.id, chunks, db)
        doc.status = DocumentStatus.indexed
        self._upsert_paper_row(db=db, doc_id=doc.id, paper=paper,
                               metadata=prepared["metadata"], markdown_path=prepared["markdown_path"])
        return str_doc_id

    async def index_paper(
        self,
        paper,
        markdown_path: Optional[Path] = None,
        content: Optional[str] = None,
        force_reindex: bool = True,
        skip_whoosh: bool = False,
    ) -> str:
        prepared = self._prepare_paper(paper, markdown_path, content)

        with get_db_session() as db:
            old_doc_id = existing_doc_id(db, paper.id)
            if force_reindex and old_doc_id is not None:
                self.index_service.delete_document_index(old_doc_id, db=db, delete_db_records=True)
                db.query(DBPaper).filter(DBPaper.doc_id == old_doc_id).delete()
                db.query(DBDocument).filter(DBDocument.id == old_doc_id).delete()

            self.index_service._ensure_index_device_ready()
            chunks = self.index_service.chunker.chunk_document(
                doc_id="_tmp", text=prepared["content"], include_summary=self.config.include_summary,
            )
            chunks, sparse_vectors = self.new_embedder.embed_chunks(chunks)
            doc_id = self._write_paper_to_stores(
                prepared, chunks, db, skip_whoosh=skip_whoosh,
                sparse_vectors=sparse_vectors,
            )

        if self.config.progress_callback:
            self.config.progress_callback(1, 1, paper.title)
        return doc_id

    async def batch_index_papers(
        self,
        papers: List,
        skip_whoosh: bool = True,
        per_batch_timeout_sec: int = 300,
    ) -> Dict[str, Any]:
        """Batch-optimized: chunk all(CPU) → embed all(GPU saturated) → bulk write(IO).

        Args:
            per_batch_timeout_sec: Max seconds for the entire batch. 0 = no timeout.
        """
        async def _do_batch() -> Dict[str, Any]:
            indexed = 0
            failed = 0
            results: List[Dict[str, Any]] = []
            prepared_list: List[Dict[str, Any]] = []
            all_chunks: List = []
            chunk_offsets: List[tuple[int, int]] = []

            for record in papers:
                paper = record.to_runtime_paper() if hasattr(record, "to_runtime_paper") else record
                md_path = Path(record.markdown_path) if hasattr(record, "markdown_path") else None
                try:
                    prep = self._prepare_paper(paper, md_path)
                    chunks = self.index_service.chunker.chunk_document(
                        doc_id=f"_batch_{len(prepared_list)}", text=prep["content"],
                        include_summary=self.config.include_summary,
                    )
                    if not chunks:
                        raise RuntimeError("No chunks generated")
                    start = len(all_chunks)
                    all_chunks.extend(chunks)
                    chunk_offsets.append((start, len(all_chunks)))
                    prepared_list.append(prep)
                except Exception as exc:
                    failed += 1
                    pid = getattr(paper, "id", getattr(record, "paper_id", "?"))
                    results.append({"paper_id": pid, "status": "failed", "error": str(exc)})

            all_sparse = None
            if all_chunks:
                all_chunks, all_sparse = self.new_embedder.embed_chunks(all_chunks)

            for i, prep in enumerate(prepared_list):
                start, end = chunk_offsets[i]
                chunks = all_chunks[start:end]
                chunk_sparse = all_sparse[start:end] if all_sparse else None
                try:
                    with get_db_session() as db:
                        doc_id = self._write_paper_to_stores(
                            prep, chunks, db, skip_whoosh=skip_whoosh, wait_qdrant=False,
                            sparse_vectors=chunk_sparse,
                        )
                    indexed += 1
                    results.append({"paper_id": prep["paper"].id, "status": "indexed", "doc_id": doc_id})
                except Exception as exc:
                    failed += 1
                    results.append({"paper_id": prep["paper"].id, "status": "failed", "error": str(exc)})

            return {"indexed": indexed, "failed": failed, "results": results}

        if per_batch_timeout_sec > 0:
            return await asyncio.wait_for(_do_batch(), timeout=per_batch_timeout_sec)
        return await _do_batch()

    async def index_batch(
        self,
        papers: List,
        markdown_dir: Optional[Path] = None,
        contents: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        results: Dict[str, str] = {}
        total = len(papers)
        semaphore = asyncio.Semaphore(self.config.max_concurrent_indexing)

        async def index_single(idx: int, paper):
            async with semaphore:
                try:
                    content = contents.get(paper.id) if contents else None
                    markdown_path = None
                    if content is None and markdown_dir:
                        candidate = markdown_dir / f"{paper.id}.md"
                        if candidate.exists():
                            markdown_path = candidate
                    doc_id = await self.index_paper(paper=paper, markdown_path=markdown_path, content=content)
                    if self.config.progress_callback:
                        self.config.progress_callback(idx + 1, total, paper.title)
                    return paper.id, doc_id
                except Exception as exc:
                    print(f"Error indexing paper {paper.id}: {exc}")
                    return paper.id, None

        completed = await asyncio.gather(*(index_single(i, paper) for i, paper in enumerate(papers)))
        for paper_id, doc_id in completed:
            if doc_id:
                results[paper_id] = doc_id
        return results

    def _load_context_by_doc_ids(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not doc_ids:
            return {}
        normalized_ids = []
        for doc_id in doc_ids:
            try:
                normalized_ids.append(int(doc_id))
            except (TypeError, ValueError):
                continue
        if not normalized_ids:
            return {}

        context: Dict[str, Dict[str, Any]] = {}
        with get_db_session() as db:
            rows = (
                db.query(DBDocument, DBPaper)
                .outerjoin(DBPaper, DBPaper.doc_id == DBDocument.id)
                .filter(DBDocument.id.in_(normalized_ids))
                .all()
            )
            for document, paper in rows:
                metadata = dict(document.metadata_json or {})
                metadata.setdefault("paper_id", paper.paper_id if paper else str(document.id))
                metadata.setdefault("title", document.title or "")
                metadata.setdefault("source", paper.paper_source if paper else "")
                metadata.setdefault("conference", paper.venue if paper else metadata.get("conference", "Other"))
                metadata.setdefault("year", paper.year if paper else metadata.get("year", 0))
                metadata.setdefault("abstract", paper.abstract if paper else metadata.get("abstract", ""))
                metadata.setdefault("authors", paper.authors_json if paper else metadata.get("authors", []))
                metadata.setdefault("paper_type", metadata.get("paper_type", "unknown"))
                metadata.setdefault("pdf_path", paper.pdf_path if paper else metadata.get("pdf_path"))
                metadata.setdefault("markdown_path", paper.txt_path if paper else metadata.get("markdown_path"))
                metadata.setdefault("citations", paper.citation_count if paper else metadata.get("citations", 0))
                metadata.setdefault("arxiv_id", paper.arxiv_id if paper else metadata.get("arxiv_id"))
                metadata.setdefault("doi", paper.doi if paper else metadata.get("doi"))
                context[str(document.id)] = {
                    "doc_id": str(document.id),
                    "title": document.title or metadata.get("title", ""),
                    "source": document.source or metadata.get("paper_page_url") or metadata.get("source", ""),
                    "metadata": metadata,
                }
        return context

    def _points_to_search_results(self, points) -> List[SearchResult]:
        """Convert QdrantStore formatted dicts to SearchResult schema objects."""
        results = []
        for point in points:
            meta = point.get("metadata", {}) if isinstance(point, dict) else (point.payload or {})
            content = point.get("content", "") if isinstance(point, dict) else meta.get("content", "")
            score = point.get("score", 0.0) if isinstance(point, dict) else point.score
            pid = point.get("id", "") if isinstance(point, dict) else str(point.id)
            results.append(SearchResult(
                # Historical Qdrant payloads store the SQL document key as an int,
                # while the shared SearchResult contract intentionally uses strings.
                # Normalize at the storage adapter boundary so one legacy point cannot
                # turn an otherwise valid paper search into an HTTP 500.
                doc_id=str(meta.get("doc_id", "")),
                chunk_id=str(pid),
                content=content,
                score=score,
                doc_title=meta.get("title", ""),
                source=meta.get("source", ""),
                metadata=meta,
            ))
        return results

    def _enrich_results(self, results: List[SearchResult]) -> List[SearchResult]:
        context = self._load_context_by_doc_ids([result.doc_id for result in results])
        enriched: List[SearchResult] = []
        for result in results:
            doc_context = context.get(str(result.doc_id))
            if doc_context is None:
                continue
            enriched_metadata = {**doc_context["metadata"], **result.metadata}
            enriched.append(
                SearchResult(
                    doc_id=str(result.doc_id),
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=result.score,
                    doc_title=doc_context["title"] or result.doc_title,
                    source=doc_context["source"] or result.source,
                    metadata=enriched_metadata,
                )
            )
        return enriched

    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        filtered: List[SearchResult] = []
        for result in results:
            metadata = result.metadata
            matched = True
            for key, expected in filters.items():
                actual = metadata.get(key)
                if isinstance(expected, list):
                    if actual not in expected:
                        matched = False
                        break
                elif actual != expected:
                    matched = False
                    break
            if matched:
                filtered.append(result)
        return filtered

    def _aggregate_by_paper(self, results: List[SearchResult]) -> List[PaperSearchResult]:
        grouped: Dict[str, List[SearchResult]] = {}
        metadata_by_paper: Dict[str, Dict[str, Any]] = {}

        for result in results:
            paper_id = result.metadata.get("paper_id")
            if not paper_id:
                continue
            grouped.setdefault(paper_id, []).append(result)
            metadata_by_paper.setdefault(paper_id, result.metadata)

        aggregated: List[PaperSearchResult] = []
        for paper_id, chunks in grouped.items():
            meta = metadata_by_paper[paper_id]
            total_score = sum(chunk.score for chunk in chunks)
            avg_score = total_score / len(chunks)
            combined_score = avg_score * (1 + 0.1 * min(len(chunks), 10))
            aggregated.append(
                PaperSearchResult(
                    paper_id=paper_id,
                    title=chunks[0].doc_title if chunks else meta.get("title", ""),
                    authors=meta.get("authors", []),
                    abstract=meta.get("abstract", ""),
                    source=meta.get("source", ""),
                    conference=meta.get("conference", "Other"),
                    year=meta.get("year", 0) or 0,
                    paper_type=meta.get("paper_type", "unknown"),
                    score=combined_score,
                    hit_count=len(chunks),
                    matched_chunks=chunks[:5],
                    pdf_url=meta.get("pdf_url"),
                    paper_page_url=meta.get("paper_page_url"),
                    arxiv_id=meta.get("arxiv_id"),
                    code_url=meta.get("code_url"),
                    pdf_path=meta.get("pdf_path"),
                    markdown_path=meta.get("markdown_path"),
                    citations=meta.get("citations", 0),
                    metadata=meta,
                )
            )
        aggregated.sort(key=lambda item: item.score, reverse=True)
        return aggregated

    def _convert_to_paper_results(self, results: List[SearchResult]) -> List[PaperSearchResult]:
        converted: List[PaperSearchResult] = []
        for result in results:
            meta = result.metadata
            converted.append(
                PaperSearchResult(
                    paper_id=meta.get("paper_id", result.doc_id),
                    title=result.doc_title,
                    authors=meta.get("authors", []),
                    abstract=meta.get("abstract", ""),
                    source=meta.get("source", ""),
                    conference=meta.get("conference", "Other"),
                    year=meta.get("year", 0) or 0,
                    paper_type=meta.get("paper_type", "unknown"),
                    score=result.score,
                    hit_count=1,
                    matched_chunks=[result],
                    pdf_url=meta.get("pdf_url"),
                    paper_page_url=meta.get("paper_page_url"),
                    arxiv_id=meta.get("arxiv_id"),
                    code_url=meta.get("code_url"),
                    pdf_path=meta.get("pdf_path"),
                    markdown_path=meta.get("markdown_path"),
                    citations=meta.get("citations", 0),
                    metadata=meta,
                )
            )
        return converted

    def search_papers(
        self,
        query: str,
        top_k: int = 20,
        method: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        aggregate_by_paper: bool = True,
    ) -> List[PaperSearchResult]:
        method = (method or "hybrid").lower()
        embed_result = self.new_embedder.embed_texts([query])
        dense_vector = embed_result.dense[0].tolist()
        search_top_k = top_k * 3

        if method == "vector":
            points = self.chroma_store.search(
                query_embedding=dense_vector,
                top_k=search_top_k,
            )
        else:
            from core.storage.qdrant_store import sparse_dict_to_qdrant
            sparse = sparse_dict_to_qdrant(embed_result.sparse[0]) if embed_result.sparse else None
            points = self.chroma_store.hybrid_search(
                dense_vector=dense_vector,
                sparse_vector=sparse,
                top_k=search_top_k,
            )

        raw_results = self._points_to_search_results(points)
        paper_results = self._enrich_results(raw_results)
        if filters:
            paper_results = self._apply_filters(paper_results, filters)

        if not aggregate_by_paper:
            return self._convert_to_paper_results(paper_results[:top_k])

        return self._aggregate_by_paper(paper_results)[:top_k]

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        doc_id = self.get_indexed_doc_id(paper_id)
        if doc_id is None:
            return None
        doc_id_int = int(doc_id)

        with get_db_session() as db:
            row = (
                db.query(DBDocument, DBPaper)
                .outerjoin(DBPaper, DBPaper.doc_id == DBDocument.id)
                .filter(DBDocument.id == doc_id_int)
                .first()
            )
            if row is None:
                return None
            document, paper = row
            chunks = (
                db.query(DBChunk)
                .filter(DBChunk.doc_id == doc_id_int)
                .order_by(DBChunk.chunk_index.asc())
                .all()
            )
            return {
                "id": paper_id,
                "paper_id": paper_id,
                "doc_id": document.id,
                "title": document.title,
                "source": document.source,
                "content_preview": chunks[0].content[:1000] if chunks else "",
                "metadata": document.metadata_json or {},
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "level": chunk.level,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata_json or {},
                    }
                    for chunk in chunks
                ],
            }

    def get_indexed_doc_id(self, paper_id: str) -> Optional[str]:
        with get_db_session() as db:
            document = (
                db.query(DBDocument.id)
                .join(DBPaper, DBPaper.doc_id == DBDocument.id)
                .filter(
                    DBPaper.paper_id == paper_id,
                    DBDocument.source_type == SourceType.paper,
                    DBDocument.status == DocumentStatus.indexed,
                )
                .first()
            )
        return str(document[0]) if document is not None else None

    def delete_paper(self, paper_id: str) -> bool:
        with get_db_session() as db:
            doc_id = existing_doc_id(db, paper_id)
            if doc_id is None:
                return False
            self.index_service.delete_document_index(doc_id, db=db, delete_db_records=True)
            db.query(DBPaper).filter(DBPaper.doc_id == doc_id).delete()
            return bool(db.query(DBDocument).filter(DBDocument.id == doc_id).delete())

    def get_indexed_paper_ids(self) -> List[str]:
        with get_db_session() as db:
            return sorted(row[0] for row in db.query(DBPaper.paper_id).all())

    def _cached_vector_count(self) -> int:
        """Vector count from the shared stale-while-revalidate cache; never probes on the request path.

        Qdrant's `get_collection()` blocks up to 60 s during in-flight HNSW indexing (W8), so a
        cold call returns 0 at once and the first probe fills the cache on a daemon thread.
        """
        return stats_cache.get(
            ("papers.vector_count", self.config.collection_name),
            lambda: int(self.chroma_store.count()),
            ttl=TOTAL_COUNT_TTL_SECONDS,
            default=0,
        )

    def _compute_chunk_levels(self) -> Dict[int, int]:
        with get_db_session() as db:
            rows = (
                db.query(DBChunk.level, func.count(DBChunk.id))
                .join(DBDocument, DBChunk.doc_id == DBDocument.id)
                .filter(DBDocument.source_type == SourceType.paper)
                .group_by(DBChunk.level)
                .all()
            )
        return {int(level if level is not None else 0): int(count) for level, count in rows}

    def _cached_chunk_levels(self) -> Dict[int, int]:
        """Chunk-level histogram from the shared stale-while-revalidate cache.

        The underlying GROUP BY scans 23M rows (19 s on the production host). A cold call returns
        {} at once and the first scan fills the cache on a daemon thread; afterwards a stale
        histogram is served while one background refresh runs every 10 minutes.
        """
        return stats_cache.get(
            "papers.chunk_levels",
            self._compute_chunk_levels,
            ttl=DISTRIBUTION_TTL_SECONDS,
            default={},
        )

    @staticmethod
    def _count_paper_totals() -> Dict[str, int]:
        with get_db_session() as db:
            total_papers = db.query(func.count(DBPaper.paper_id)).scalar() or 0
            total_documents = (
                db.query(func.count(DBDocument.id))
                .filter(DBDocument.source_type == SourceType.paper)
                .scalar()
                or 0
            )
            total_chunks = (
                db.query(func.count(DBChunk.id))
                .join(DBDocument, DBChunk.doc_id == DBDocument.id)
                .filter(DBDocument.source_type == SourceType.paper)
                .scalar()
                or 0
            )
        return {
            "total_papers": total_papers,
            "total_documents": total_documents,
            "total_chunks": total_chunks,
        }

    def get_stats(self) -> Dict[str, Any]:
        totals = stats_cache.get(
            "papers.totals", self._count_paper_totals, ttl=TOTAL_COUNT_TTL_SECONDS
        )
        return {
            **totals,
            "chunk_levels": self._cached_chunk_levels(),
            "vector_index_size": self._cached_vector_count(),
            "bm25_index_size": 0,
            "collection_name": self.config.collection_name,
        }

    def to_search_response(self, query: str, results: List[PaperSearchResult]) -> Dict[str, Any]:
        aggregated = []
        chunks: List[SearchResult] = []
        with get_db_session() as db:
            doc_ids = {
                pid: str(doc_id)
                for pid, doc_id in db.query(DBPaper.paper_id, DBPaper.doc_id)
                .filter(DBPaper.paper_id.in_([r.paper_id for r in results])).all()
            } if results else {}
        for result in results:
            aggregated.append(
                AggregatedResult(
                    doc_id=doc_ids.get(result.paper_id, result.paper_id),
                    doc_title=result.title,
                    source=result.paper_page_url or result.source or result.paper_id,
                    hit_count=result.hit_count,
                    total_score=result.score * result.hit_count,
                    avg_score=result.score,
                    chunks=result.matched_chunks,
                )
            )
            chunks.extend(result.matched_chunks)

        return {
            "query": query,
            "method": "hybrid",
            "total_results": len(chunks),
            "results": [chunk.model_dump() for chunk in chunks],
            "aggregated": [item.model_dump() for item in aggregated],
            "paper_results": [result.to_dict() for result in results],
        }


async def index_papers_from_pipeline(
    papers: List,
    markdown_dir: Path,
    config: Optional[PaperIndexConfig] = None,
) -> Dict[str, str]:
    indexer = PaperIndexer(config=config)
    return await indexer.index_batch(papers=papers, markdown_dir=markdown_dir)


def create_paper_indexer(collection_name: str = "papers") -> PaperIndexer:
    return PaperIndexer(config=PaperIndexConfig(collection_name=collection_name))


def create_runtime_paper_indexer(
    collection_name: str = DEFAULT_PAPER_COLLECTION_NAME,
    data_source: str = DEFAULT_PAPER_DATA_SOURCE,
) -> PaperIndexer:
    return PaperIndexer(
        config=PaperIndexConfig(
            collection_name=collection_name,
            data_source=data_source,
        )
    )
