"""Text chunking strategies.

This module exposes three building blocks, intentionally kept in one file
so the public surface (`from core.indexing.chunker import …`) stays stable:

- ``Chunker``               -- the common interface (Strategy pattern)
- ``SectionAwareChunker``   -- markdown / html / plain text chunker that
                              splits along headings when present and falls
                              back to a bounded size-based split otherwise
- ``TextChunker``           -- legacy multi-granularity chunker
                              (summary + paragraph + sentence levels) kept
                              for the IndexService.index_document path that
                              still expects three-level fan-out

``create_chunker(content_type, **kwargs)`` picks the right strategy at
runtime so callers don't hardcode which class to use.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import (
    HTMLNodeParser,
    MarkdownNodeParser,
)
from llama_index.core.schema import Document as _LlamaDocument

from app.models.schemas import Chunk
from config import settings
from core.indexing.embedding_profile import get_profile, load_tokenizer, tokens_over_limit
from core.indexing.token_budget import SEGMENT_CHARS, split_to_budget

_logger = logging.getLogger(__name__)

_YAML_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)

# Markdown ATX heading at the start of a line. MarkdownNodeParser splits on
# any depth, but we still detect "is there any heading at all?" cheaply
# before running the full parser on plain-text input.
_MD_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S")

# Extract the leading heading line from a node's text. MarkdownNodeParser
# preserves the heading line at the top of each split, so the first
# ``# .* / ## .* / ### .*`` line is the section's own title.
_LEADING_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Marker writes page anchor spans inline with headings:
#   `# <span id="page-0-2"></span><span id="page-0-0"></span>Title`
# These leak into section_name + L1 title without a strip pass.
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Match an `Abstract` heading at any depth, case-insensitive. The body
# (everything until the next heading at any depth) becomes the L1 summary.
_ABSTRACT_HEADING_RE = re.compile(
    r"(?im)^\s{0,3}#{1,6}\s+abstract\s*$",
)
_NEXT_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S")

# Content-type aliases collapsed onto canonical kinds. The factory and the
# SectionAwareChunker both consult this table so adding a new alias only
# touches one place.
_CONTENT_TYPE_ALIASES = {
    None: "auto",
    "": "auto",
    "auto": "auto",
    "markdown": "markdown",
    "md": "markdown",
    "text/markdown": "markdown",
    "html": "html",
    "htm": "html",
    "text/html": "html",
    "text": "text",
    "txt": "text",
    "plain": "text",
    "text/plain": "text",
}


def _normalize_content_type(content_type: Optional[str]) -> str:
    """Return one of {'auto', 'markdown', 'html', 'text'} — never raises."""
    if content_type is None:
        return "auto"
    key = content_type.strip().lower()
    return _CONTENT_TYPE_ALIASES.get(key, "auto")


def _strip_html(text: str) -> str:
    """Strip tags + script/style bodies. Returns visible text only.

    Uses BeautifulSoup (already a project dependency) rather than a regex
    so nested/malformed markup still degrades gracefully.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    # ``separator='\n'`` keeps paragraph boundaries so downstream splitters
    # still see structural cues. Multiple blank lines collapse later.
    visible = soup.get_text(separator="\n")
    visible = re.sub(r"\n{3,}", "\n\n", visible)
    return visible.strip()


def _has_markdown_headings(text: str) -> bool:
    return bool(_MD_HEADING_RE.search(text))


def enforce_budget(chunks: List[Chunk]) -> List[Chunk]:
    """The final guarantee behind every chunker: no chunk is longer than the embedding
    profile's ``chunk_max_tokens`` real tokens.

    Chunkers aim at character or token targets and none of them promises a bound (a
    chunker's own token estimate is approximate; an abstract or an unbroken CJK or base64
    run has no natural break). A chunk over the budget is cut into pieces at whitespace (or
    at token offsets when it has none). A piece keeps the chunk's level and metadata, gets a
    new id, and ``start_idx``/``end_idx`` of its own text: the original offset plus the
    piece's position in the chunk, so ``text[start:end] == content`` still holds wherever it
    held for the chunk. Chunks within the budget are returned as they are.
    """
    if not chunks:
        return chunks
    profile = get_profile()
    limit = profile.chunk_max_tokens
    oversized = {i for i, _ in tokens_over_limit([c.content for c in chunks], limit, profile)}
    if not oversized:
        return chunks
    out: List[Chunk] = []
    for index, chunk in enumerate(chunks):
        if index not in oversized:
            out.append(chunk)
            continue
        for start, end in split_to_budget(chunk.content, max_tokens=limit, profile=profile, prefer="space"):
            piece = chunk.content[start:end]
            lead = len(piece) - len(piece.lstrip())
            piece = piece.strip()
            if not piece:
                continue
            out.append(chunk.model_copy(update={
                "id": str(uuid.uuid4()),
                "content": piece,
                "start_idx": chunk.start_idx + start + lead,
                "end_idx": chunk.start_idx + start + lead + len(piece),
                "embedding": None,
                "metadata": dict(chunk.metadata),
            }))
    _logger.info("chunker.budget split %d of %d chunks over %d tokens into %d pieces",
                 len(oversized), len(chunks), limit, len(out) - (len(chunks) - len(oversized)))
    return out


class Chunker(ABC):
    """Common interface for text chunking strategies.

    Implementations must be deterministic given the same input so that
    point_id hashing downstream (W23) stays stable across re-runs.
    """

    @abstractmethod
    def chunk_document(self, doc_id: str, text: str, **kwargs: Any) -> List[Chunk]:
        """Return chunks for ``text`` belonging to ``doc_id``.

        Implementations may accept extra kwargs (e.g. ``content_type``,
        ``include_summary``) but must tolerate the absence of all of them.
        """


class SectionAwareChunker(Chunker):
    """Markdown-aware chunker with fallbacks for headerless / html / plain text.

    Strategy:
      1. If ``content_type='html'``, strip tags down to visible text first.
      2. Strip a YAML frontmatter block if present.
      3. If headings are present (and we weren't told this is plain text),
         split along the heading hierarchy, then bound each section by
         ``max_chunk_chars`` via the recursive splitter.
      4. Otherwise fall back to a single ``body`` section split purely by
         size; this is the path that handles uploaded txt files, headerless
         pdf extracts, and html pages whose markup didn't include headings.

    All produced chunks carry ``metadata['section_name']`` so consumers can
    surface where the snippet came from.
    """

    DEFAULT_MAX_CHUNK_CHARS = 1500
    DEFAULT_OVERLAP = 100
    FALLBACK_SECTION_NAME = "body"

    def __init__(
        self,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        chunk_overlap: int = DEFAULT_OVERLAP,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap = chunk_overlap
        # llama-index NodeParsers — single source of truth for chunking
        # primitives. Markdown for header-aware split, HTML for tag-aware
        # strip+flatten, Sentence for size-bounded fallback.
        self._md_parser = MarkdownNodeParser()
        self._html_parser = HTMLNodeParser()
        # char-bound size split: llama-index's SentenceSplitter is
        # token-based (~4 chars/token), which makes "max_chunk_chars" a
        # leaky abstraction. langchain's RecursiveCharacterTextSplitter
        # is the right tool for the char-bound job — use it here and let
        # llama-index handle the structural (markdown/html/semantic)
        # parts where it shines.
        self._size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_chars,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        return _YAML_FRONTMATTER_RE.sub("", text)

    @staticmethod
    def _section_name_from_node_text(text: str) -> Optional[str]:
        """MarkdownNodeParser preserves the heading line at the top of
        each section node. Pull the actual heading text out of the first
        ``#/##/###`` line so callers see something like ``"1 INTRODUCTION"``
        instead of llama-index's slash-path representation.

        Strip embedded HTML (marker emits `<span id="page-X-Y"></span>` page
        anchors inline with headings) so section_name stays clean for
        Qdrant payload + frontend UI tag display.
        """
        m = _LEADING_HEADING_RE.search(text)
        if not m:
            return None
        raw = m.group(2)
        cleaned = _HTML_TAG_RE.sub("", raw)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or None

    def _flatten_html(self, html_text: str) -> str:
        """HTMLNodeParser strips ``<script>/<style>`` and yields one node
        per tag (h1/p/etc.). For our chunker we want a single clean text
        stream to feed back through the markdown-or-text pipeline."""
        try:
            nodes = self._html_parser.get_nodes_from_documents(
                [_LlamaDocument(text=html_text)]
            )
        except Exception as exc:  # pragma: no cover -- defensive
            _logger.warning("html_parser.fail err=%s — falling through raw text", exc)
            return html_text
        return "\n\n".join(n.text for n in nodes if n.text and n.text.strip())

    def _preprocess(self, text: str, content_type: str) -> str:
        if content_type == "html":
            text = self._flatten_html(text)
        return self._strip_frontmatter(text).strip()

    def _should_use_heading_split(self, text: str, content_type: str) -> bool:
        if content_type == "text":
            return False
        return _has_markdown_headings(text)

    def _split_section_into_chunks(
        self,
        section_text: str,
        section_name: str,
        doc_id: str,
        seen: set[str],
    ) -> List[Chunk]:
        section_text = section_text.strip()
        if not section_text:
            return []

        if len(section_text) <= self.max_chunk_chars:
            pieces = [section_text]
        else:
            pieces = [
                doc.page_content
                for doc in self._size_splitter.create_documents([section_text])
            ]

        chunks: List[Chunk] = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            normalized = re.sub(r"\s+", " ", piece)
            if normalized in seen:
                continue
            seen.add(normalized)
            chunks.append(self._make_chunk(doc_id, piece, section_name))
        return chunks

    @staticmethod
    def _make_chunk(doc_id: str, content: str, section_name: str) -> Chunk:
        return Chunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            content=content,
            level=2,
            start_idx=0,
            end_idx=len(content),
            metadata={"section_name": section_name},
        )

    def chunk_document(
        self,
        doc_id: str,
        text: str,
        content_type: Optional[str] = None,
        **_: Any,
    ) -> List[Chunk]:
        t0 = time.monotonic()
        kind = _normalize_content_type(content_type)
        raw_len = len(text or "")
        clean_text = self._preprocess(text or "", kind)
        if not clean_text:
            _logger.info(
                "chunker.skip doc_id=%s reason=empty content_type=%s raw_len=%d",
                doc_id, content_type, raw_len,
            )
            return []

        seen: set[str] = set()
        if not self._should_use_heading_split(clean_text, kind):
            chunks = self._split_section_into_chunks(
                clean_text, self.FALLBACK_SECTION_NAME, doc_id, seen,
            )
            _logger.info(
                "chunker.done doc_id=%s path=flat-fallback kind=%s chunks=%d "
                "clean_len=%d elapsed_ms=%.1f",
                doc_id, kind, len(chunks), len(clean_text),
                (time.monotonic() - t0) * 1000,
            )
            return enforce_budget(chunks)

        chunks: List[Chunk] = []
        section_count = 0
        nodes = self._md_parser.get_nodes_from_documents(
            [_LlamaDocument(text=clean_text)]
        )
        for node in nodes:
            section_name = (
                self._section_name_from_node_text(node.text)
                or self.FALLBACK_SECTION_NAME
            )
            section_count += 1
            chunks.extend(self._split_section_into_chunks(
                node.text, section_name, doc_id, seen,
            ))
        _logger.info(
            "chunker.done doc_id=%s path=heading-split kind=%s sections=%d "
            "chunks=%d clean_len=%d elapsed_ms=%.1f",
            doc_id, kind, section_count, len(chunks), len(clean_text),
            (time.monotonic() - t0) * 1000,
        )
        return enforce_budget(chunks)


class TextChunker(Chunker):
    """Multi-granularity chunker (level 1 summary + level 2 paragraph + level 3 sentence).

    Kept for the legacy IndexService.index_document fan-out path. New
    paper-corpus ingest uses :class:`SectionAwareChunker` instead — the two
    coexist while the index migration is in flight.
    """

    def __init__(self):
        self.chunk_sizes = settings.chunk.chunk_sizes
        self.overlap = settings.chunk.chunk_overlap

    def chunk_document(
        self,
        doc_id: str,
        text: str,
        include_summary: bool = True,
        **_: Any,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []

        if include_summary:
            summary = self._create_summary(text)
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                content=summary,
                level=1,
                start_idx=0,
                end_idx=len(text),
                metadata={"type": "summary"},
            ))

        chunks.extend(self._chunk_by_size(
            doc_id=doc_id, text=text,
            chunk_size=self.chunk_sizes[0], overlap=self.overlap, level=2,
        ))
        chunks.extend(self._chunk_by_size(
            doc_id=doc_id, text=text,
            chunk_size=self.chunk_sizes[-1], overlap=self.overlap // 2, level=3,
        ))
        return enforce_budget(self._dedupe_chunks(chunks))

    def chunk_document_iter(
        self,
        doc_id: str,
        text: str,
        include_summary: bool = True,
    ) -> Iterator[Chunk]:
        # Deduplication needs every chunk of the document, so nothing is streamed
        # before chunk_document has finished.
        yield from self.chunk_document(doc_id, text, include_summary)

    def _create_summary(self, text: str, max_length: Optional[int] = None) -> str:
        if max_length is None:
            max_length = settings.chunk.summary_max_length

        paragraphs = text.split("\n\n")
        summary = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(summary) + len(para) > max_length:
                break
            summary += para + " "
        return summary.strip()[:max_length]

    def _chunk_by_size(
        self,
        doc_id: str,
        text: str,
        chunk_size: int,
        overlap: int,
        level: int,
    ) -> List[Chunk]:
        return list(self._chunk_by_size_iter(doc_id, text, chunk_size, overlap, level))

    def _chunk_by_size_iter(
        self,
        doc_id: str,
        text: str,
        chunk_size: int,
        overlap: int,
        level: int,
    ) -> Iterator[Chunk]:
        text_len = len(text)
        start = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                boundary = self._find_sentence_boundary(text, end)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                yield Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=chunk_text,
                    level=level,
                    start_idx=start,
                    end_idx=end,
                    metadata={"chunk_size": chunk_size},
                )

            if end >= text_len:
                break
            start = max(1, end - overlap)

    @staticmethod
    def _find_sentence_boundary(text: str, pos: int, search_range: int = 100) -> int:
        enders = ".。!！?？\n"
        for i in range(pos, min(pos + search_range, len(text))):
            if text[i] in enders:
                return i + 1
        for i in range(pos, max(pos - search_range, 0), -1):
            if text[i] in enders:
                return i + 1
        return pos

    @staticmethod
    def _normalize_chunk_content(content: str) -> str:
        return re.sub(r"\s+", " ", (content or "").strip())

    def _chunk_priority(self, chunk: Chunk) -> Tuple[int, int]:
        """Prefer real paragraphs over synthetic summaries when content collides."""
        priority_map = {2: 3, 3: 2, 1: 1}
        return (
            priority_map.get(chunk.level, 0),
            len(self._normalize_chunk_content(chunk.content)),
        )

    def _dedupe_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        chosen: dict[str, Chunk] = {}
        order: List[str] = []
        for chunk in chunks:
            normalized = self._normalize_chunk_content(chunk.content)
            if not normalized:
                continue
            existing = chosen.get(normalized)
            if existing is None:
                chosen[normalized] = chunk
                order.append(normalized)
                continue
            if self._chunk_priority(chunk) > self._chunk_priority(existing):
                chosen[normalized] = chunk
        deduped = [chosen[key] for key in order]
        deduped.sort(key=lambda chunk: (
            chunk.start_idx if chunk.start_idx is not None else 0,
            chunk.level,
        ))
        return deduped

    def chunk_by_semantic(self, doc_id: str, text: str) -> List[Chunk]:
        """Paragraph-grouping fallback (legacy, kept for callers that still use it)."""
        chunks: List[Chunk] = []
        paragraphs = re.split(r"\n\s*\n", text)
        current = ""
        start_idx = 0
        chunk_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                start_idx += 2
                continue
            if len(current) + len(para) < self.chunk_sizes[0]:
                current += para + "\n\n"
            else:
                if current.strip():
                    chunks.append(Chunk(
                        id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        content=current.strip(),
                        level=2,
                        start_idx=chunk_start,
                        end_idx=start_idx,
                        metadata={"type": "semantic"},
                    ))
                chunk_start = start_idx
                current = para + "\n\n"
            start_idx += len(para) + 2

        if current.strip():
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                content=current.strip(),
                level=2,
                start_idx=chunk_start,
                end_idx=start_idx,
                metadata={"type": "semantic"},
            ))
        return enforce_budget(chunks)


class _BgeM3LlamaAdapter:
    """Adapt our :class:`BgeM3Embedder` to llama-index ``BaseEmbedding``.

    Semantic chunking only consumes dense vectors — sparse + ColBERT are
    skipped to keep batch encode lean. We construct the BaseEmbedding
    subclass lazily inside ``build()`` to defer the llama-index import
    until SemanticSectionChunker is actually used (the import pulls in
    pydantic-v2 + a handful of heavy deps).
    """

    @staticmethod
    def build(embedder: Any = None):
        from typing import List as _List
        from llama_index.core.embeddings import BaseEmbedding as _LlamaBase
        from pydantic import PrivateAttr as _PrivateAttr

        if embedder is None:
            # W46: use the process-wide shared BGE-M3 so the semantic
            # boundary detector doesn't load its own copy when callers
            # also init IndexService / SearchPipeline.
            from core.indexing.embedders import get_embedder
            embedder = get_embedder()

        class _Adapter(_LlamaBase):
            _embedder: Any = _PrivateAttr()

            def __init__(self, **kw):
                super().__init__(**kw)

            @classmethod
            def class_name(cls) -> str:
                return "BgeM3LlamaAdapter"

            def _get_text_embedding(self, text: str) -> _List[float]:
                return self._embedder.embed_texts(
                    [text], priority="background").dense[0].tolist()

            def _get_text_embeddings(self, texts: _List[str]) -> _List[_List[float]]:
                from core.indexing.embedders import embed_in_pieces
                result = embed_in_pieces(self._embedder, list(texts), "background")
                return [vector.tolist() for vector in result.dense]

            def _get_query_embedding(self, query: str) -> _List[float]:
                return self._get_text_embedding(query)

            async def _aget_query_embedding(self, query: str) -> _List[float]:
                return self._get_query_embedding(query)

            async def _aget_text_embedding(self, text: str) -> _List[float]:
                return self._get_text_embedding(text)

            async def _aget_text_embeddings(self, texts: _List[str]) -> _List[_List[float]]:
                return self._get_text_embeddings(texts)

        adapter = _Adapter()
        adapter._embedder = embedder
        return adapter


def _simple_sentence_splitter(text: str) -> List[str]:
    """Regex sentence splitter — avoids pulling in NLTK / spaCy.

    Splits after ``. ! ?`` followed by whitespace. Good enough for
    English academic prose, which is what we run semantic chunking on.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if s and s.strip()]


class SemanticSectionChunker(SectionAwareChunker):
    """Hybrid chunker: markdown headers OUTER, semantic split INNER.

    Why not pure semantic? Marker recovers real section structure (`# 1
    INTRODUCTION`, `## 4.3 EXPERIMENTAL SETTING`) — that structure is
    high-signal metadata (``section_name``) we'd lose if we semantically
    split the whole doc as one stream. Keeping headers as the outer
    boundary and applying semantic split only INSIDE long sections lets
    us preserve marker's structural quality while still getting tighter
    semantic boundaries within long sections.

    The embedder defaults to :class:`BgeM3Embedder` so the boundary
    detector lives in the same vector space as the retrieval index —
    boundaries reflect the same "similarity" the retriever later sees.

    SemanticSplitterNodeParser is created lazily and cached at class
    scope; embedding model load is ~5-10s and we don't want to pay it
    per-document.
    """

    DEFAULT_BREAKPOINT_PERCENTILE = 95  # llama-index default
    DEFAULT_BUFFER_SIZE = 1             # llama-index default
    # Skip semantic split for sections shorter than this — embedding
    # overhead isn't worth it and a short section is already coherent.
    DEFAULT_MIN_SECTION_CHARS = 800
    # 2026-05-19 user-mandated schema: emit one L1 chunk per doc =
    # title + first N bge-m3 tokens. Replaces legacy L1=title-only +
    # L3=512-char paragraphs. Aligns retrieval signal with paper-level
    # summary (title alone was too sparse for hybrid scoring).
    L1_SUMMARY_TOKENS = 500
    # An abstract longer than this is not an abstract: ``## Abstract`` with no heading after it
    # makes the whole body "the abstract" (one measured L1 of 19,424 tokens). It is capped to the
    # first L1_SUMMARY_TOKENS tokens, like a paper with no abstract heading. Real abstracts are a
    # few hundred tokens, so none of them reaches this.
    L1_ABSTRACT_MAX_TOKENS = 1024
    # A title longer than this is not a title. A paper whose title is not an H1 can carry a later
    # whole-line H1 (measured: ICML 2026 doc 105830, an appendix table of contents of 3,600
    # characters) that the first-H1 rule took as the title, pushing the L1 into two pieces.
    L1_TITLE_MAX_TOKENS = 128
    L1_SECTION_NAME = "title_summary"
    # Filter L2 chunks below this size — semantic split occasionally emits
    # 1-10 char tail nodes (sentence boundary artifacts) that embed to noise.
    MIN_CHUNK_CHARS = 40

    _semantic_parser_cache: Any = None

    def __init__(
        self,
        max_chunk_chars: int = SectionAwareChunker.DEFAULT_MAX_CHUNK_CHARS,
        chunk_overlap: int = SectionAwareChunker.DEFAULT_OVERLAP,
        breakpoint_percentile: int = DEFAULT_BREAKPOINT_PERCENTILE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        min_section_chars: int = DEFAULT_MIN_SECTION_CHARS,
        embedder: Any = None,
        semantic_inner_split: bool = True,
    ):
        super().__init__(max_chunk_chars=max_chunk_chars, chunk_overlap=chunk_overlap)
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.min_section_chars = min_section_chars
        self._embedder = embedder
        # When False, skip bge-m3 boundary detection inside long sections
        # and fall back to parent SectionAwareChunker's size-splitter (with
        # configured chunk_overlap). L1 title+abstract emission unchanged.
        # 2026-05-19 default ICLR run: False — 11.5h → <30min, 5-15%
        # retrieval quality tradeoff judged acceptable by user.
        self.semantic_inner_split = semantic_inner_split

    @classmethod
    def reset_cache(cls) -> None:
        """Drop the cached parser — used by tests."""
        cls._semantic_parser_cache = None

    def _get_semantic_parser(self):
        if SemanticSectionChunker._semantic_parser_cache is not None:
            return SemanticSectionChunker._semantic_parser_cache
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        adapter = _BgeM3LlamaAdapter.build(self._embedder)
        parser = SemanticSplitterNodeParser.from_defaults(
            embed_model=adapter,
            breakpoint_percentile_threshold=self.breakpoint_percentile,
            buffer_size=self.buffer_size,
            sentence_splitter=_simple_sentence_splitter,
        )
        SemanticSectionChunker._semantic_parser_cache = parser
        _logger.info(
            "semantic_chunker.init.ok percentile=%d buffer=%d",
            self.breakpoint_percentile, self.buffer_size,
        )
        return parser

    def _split_section_into_chunks(  # type: ignore[override]
        self,
        section_text: str,
        section_name: str,
        doc_id: str,
        seen: set[str],
    ) -> List[Chunk]:
        section_text = section_text.strip()
        if not section_text:
            return []

        # Drop sections entirely below MIN_CHUNK_CHARS — embed would be noise.
        if len(section_text) < self.MIN_CHUNK_CHARS:
            return []

        # Fast path: skip bge-m3 semantic detection entirely. Defers to
        # parent SectionAwareChunker (size-splitter with chunk_overlap).
        # L1 emission is unaffected — chunk_document override still runs.
        if not self.semantic_inner_split:
            return super()._split_section_into_chunks(
                section_text, section_name, doc_id, seen
            )

        # Short sections: keep as one chunk — semantic split has overhead
        # and a single short section is already coherent.
        if len(section_text) <= self.min_section_chars:
            normalized = re.sub(r"\s+", " ", section_text)
            if normalized in seen:
                return []
            seen.add(normalized)
            return [self._make_chunk(doc_id, section_text, section_name)]

        try:
            from llama_index.core.schema import Document
            parser = self._get_semantic_parser()
            nodes = parser.get_nodes_from_documents([Document(text=section_text)])
            pieces = [n.text for n in nodes if n.text and n.text.strip()]
        except Exception as exc:
            _logger.warning(
                "semantic_chunker.fallback section=%s len=%d err=%s — using recursive splitter",
                section_name, len(section_text), exc,
            )
            pieces = [
                doc.page_content
                for doc in self._size_splitter.create_documents([section_text])
            ]

        # SemanticSplitter doesn't enforce max char bound — for very long
        # sections it can return single nodes well over max_chunk_chars. Run
        # the size splitter as a post-pass so embed-time tokens stay sane.
        bounded_pieces: List[str] = []
        for piece in pieces:
            if len(piece) > self.max_chunk_chars:
                bounded_pieces.extend(
                    doc.page_content
                    for doc in self._size_splitter.create_documents([piece])
                )
            else:
                bounded_pieces.append(piece)

        chunks: List[Chunk] = []
        for piece in bounded_pieces:
            piece = piece.strip()
            # Drop noise: semantic split sometimes emits 1-10 char tail
            # fragments. MIN_CHUNK_CHARS keeps signal:noise high.
            if len(piece) < self.MIN_CHUNK_CHARS:
                continue
            normalized = re.sub(r"\s+", " ", piece)
            if normalized in seen:
                continue
            seen.add(normalized)
            chunks.append(self._make_chunk(doc_id, piece, section_name))
        return chunks

    @staticmethod
    def _get_tokenizer():
        # The embedding profile's own tokenizer (BGE_M3_PATH aware, loaded once per process).
        return load_tokenizer(get_profile())

    @staticmethod
    def _extract_abstract(text: str) -> Optional[str]:
        """Find ``## Abstract`` (any depth, case-insensitive) and return
        the body text until the next heading. Strips embedded HTML noise
        (marker page anchors). Returns ``None`` if no abstract heading."""
        m = _ABSTRACT_HEADING_RE.search(text)
        if not m:
            return None
        body_start = m.end()
        rest = text[body_start:]
        next_m = _NEXT_HEADING_RE.search(rest)
        body = rest[: next_m.start()] if next_m else rest
        body = _HTML_TAG_RE.sub("", body)
        body = re.sub(r"\s+", " ", body).strip()
        return body or None

    @classmethod
    def _summary_head(cls, text: str, max_tokens: Optional[int] = None) -> str:
        """The first ``max_tokens`` (default ``L1_SUMMARY_TOKENS``) bge-m3 tokens of ``text``, decoded.

        ``truncation`` cuts after the whole text is tokenized, so only a bounded prefix is handed
        over: the first L1_SUMMARY_TOKENS tokens come from far fewer characters."""
        tokenizer = cls._get_tokenizer()
        token_ids = tokenizer.encode(
            text[:SEGMENT_CHARS],
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens or cls.L1_SUMMARY_TOKENS,
        )
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _clean_title(raw: Optional[str]) -> Optional[str]:
        title = _HTML_TAG_RE.sub("", raw).strip() if raw else None
        return re.sub(r"\s+", " ", title) if title else None

    def _is_title_length(self, title: Optional[str]) -> bool:
        return bool(title) and not tokens_over_limit([title], self.L1_TITLE_MAX_TOKENS, get_profile())

    def _build_l1_chunk(
        self,
        doc_id: str,
        text: str,
        *,
        title_hint: Optional[str] = None,
    ) -> Optional[Chunk]:
        """Emit one L1 chunk per document. Preferred summary source is the
        paper's own ``## Abstract`` section (richer signal than a token-
        truncated prefix). Falls back to first ``L1_SUMMARY_TOKENS`` bge-m3
        tokens of body when no abstract heading is present.

        Title detection: first ATX H1 (``# X``), HTML tags stripped. Body
        excludes the title line so summary doesn't redundantly repeat it.
        Token truncation uses bge-m3's own tokenizer so token count matches
        what the embedder sees at index-time.
        """
        if not text or not text.strip():
            return None

        title_m = re.search(r"^\s*#\s+(.+?)\s*$", text, re.MULTILINE)
        title = self._clean_title(title_m.group(1)) if title_m else None
        if title_m and not self._is_title_length(title):
            title_m, title = None, None          # a whole-line H1 of something else, not the title
        if title is None:
            title = self._clean_title(title_hint)
            if title and not self._is_title_length(title):
                title = self._summary_head(title, self.L1_TITLE_MAX_TOKENS - 2) or None   # 2 = specials

        # Prefer abstract path
        abstract = self._extract_abstract(text)
        if abstract and not tokens_over_limit([abstract], self.L1_ABSTRACT_MAX_TOKENS, get_profile()):
            summary = abstract
            path = "abstract"
        elif abstract:
            summary = self._summary_head(abstract)
            path = "abstract_capped"
        else:
            body = text[title_m.end():].lstrip() if title_m else text.lstrip()
            if not body:
                body = title or ""
            summary = self._summary_head(body)
            path = "n_tokens_fallback"

        content = f"{title}\n\n{summary}" if title else summary
        if not content.strip():
            return None

        _logger.info(
            "semantic_chunker.l1.built doc_id=%s path=%s title_len=%d summary_len=%d",
            doc_id, path, len(title or ""), len(summary),
        )

        return Chunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            content=content,
            level=1,
            start_idx=0,
            end_idx=len(summary),
            metadata={
                "section_name": self.L1_SECTION_NAME,
                "title": title or "",
                "l1_source": path,
            },
        )

    def chunk_document(  # type: ignore[override]
        self,
        doc_id: str,
        text: str,
        content_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Chunk]:
        """Emit one L1 (title + 500 bge-m3 token summary) + L2 section chunks."""
        title_hint = kwargs.pop("title_hint", None)
        l2_chunks = super().chunk_document(
            doc_id, text, content_type=content_type, **kwargs
        )
        if not l2_chunks:
            return []
        l1 = self._build_l1_chunk(
            doc_id,
            text or "",
            title_hint=title_hint,
        )
        if l1 is None:
            return l2_chunks
        # The L2 chunks passed the budget in the parent; the L1 summary is at most
        # L1_ABSTRACT_MAX_TOKENS (or L1_SUMMARY_TOKENS when capped) plus the title, so only a
        # title of absurd length could still need the budget.
        return [*enforce_budget([l1]), *l2_chunks]


def create_chunker(
    content_type: Optional[str] = None,
    *,
    max_chunk_chars: int = SectionAwareChunker.DEFAULT_MAX_CHUNK_CHARS,
    chunk_overlap: int = SectionAwareChunker.DEFAULT_OVERLAP,
) -> Chunker:
    """Factory: pick the chunker for a given ``content_type``.

    ``content_type`` accepts canonical names (``markdown``, ``html``,
    ``text``), MIME equivalents (``text/html`` etc.), the special token
    ``multigranularity`` for the legacy three-level fan-out, the token
    ``semantic`` for :class:`SemanticSectionChunker` (markdown headers
    + bge-m3-driven semantic boundary detection within sections), or
    ``None`` (defaults to SectionAware which itself falls back
    gracefully). Unknown types fall through to SectionAware rather than
    raising — the caller may not always know the source format upstream.
    """
    kind = (content_type or "").strip().lower()
    if kind == "multigranularity":
        return TextChunker()
    if kind == "semantic":
        return SemanticSectionChunker(
            max_chunk_chars=max_chunk_chars,
            chunk_overlap=chunk_overlap,
        )
    # "section_l1" / "fast" / "semantic_fast": same as semantic (L1 + L2 +
    # html-clean + size cap + abstract-first) BUT skips bge-m3 boundary
    # detection inside long sections — 20-30× faster on local CPU/GPU at
    # ~5-15% retrieval quality cost. Use this for bulk reindex when wall
    # time dominates.
    if kind in ("section_l1", "fast", "semantic_fast"):
        return SemanticSectionChunker(
            max_chunk_chars=max_chunk_chars,
            chunk_overlap=chunk_overlap,
            semantic_inner_split=False,
        )
    return SectionAwareChunker(
        max_chunk_chars=max_chunk_chars,
        chunk_overlap=chunk_overlap,
    )


__all__ = [
    "Chunker",
    "SectionAwareChunker",
    "SemanticSectionChunker",
    "TextChunker",
    "create_chunker",
]
