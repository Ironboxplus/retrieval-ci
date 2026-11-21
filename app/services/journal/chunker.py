"""Chunking for ACE journal entries."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.services.journal.repository import new_chunk_id
from core.indexing.embedding_profile import ascii_within_limit, count_tokens, get_profile
from core.indexing.token_budget import SPECIAL_TOKENS, segment_text, split_to_budget, token_offsets


@dataclass
class JournalTextChunk:
    chunk_id: str
    entry_id: str
    chunk_index: int
    level: int
    content: str
    heading: str | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)


# How far a window edge may move to land on a word boundary instead of inside a word.
_SNAP_TOKENS = 16


def _clip(text: str, limit: int = 3000) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\n...[truncated]"


class JournalChunker:
    """Markdown-aware chunker that preserves short journal notes.

    Sizes are model tokens (the embedding profile's tokenizer): ``target_tokens`` is the
    window size, ``overlap_tokens`` the overlap, and no chunk leaves here longer than the
    profile's ``chunk_max_tokens``.
    """

    def __init__(self, target_tokens: int = 520, overlap_tokens: int = 80):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_entry(
        self,
        *,
        entry_id: str,
        title: str | None,
        summary: str | None,
        body: str,
        tags: list[str],
        source_kind: str,
        client: str,
    ) -> list[JournalTextChunk]:
        chunks: list[JournalTextChunk] = []
        summary_content = "\n".join(p for p in [
            title or "(untitled journal entry)",
            summary or "",
            f"source_kind: {source_kind}",
            f"client: {client}",
            "tags: " + ", ".join(tags) if tags else "",
        ] if p.strip()).strip()
        items = [{"level": 1, "content": summary_content, "heading": "summary"}, *self._chunk_body(body)]
        for item in items:
            # The final guarantee: whatever a window rule above produced, a piece never
            # exceeds the embedding profile's chunk budget.
            for piece in self._fit_budget(item["content"]):
                chunks.append(self._make(
                    entry_id,
                    len(chunks),
                    item["level"],
                    piece,
                    heading=item.get("heading"),
                    metadata=item.get("metadata") or {},
                ))
        return chunks

    @staticmethod
    def _fit_budget(content: str) -> list[str]:
        clean = content.strip()
        spans = split_to_budget(clean, max_tokens=get_profile().chunk_max_tokens, prefer="line")
        return [piece for piece in (clean[start:end] for start, end in spans) if piece.strip()]

    def _make(
        self,
        entry_id: str,
        chunk_index: int,
        level: int,
        content: str,
        *,
        heading: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JournalTextChunk:
        clean = content.strip()
        return JournalTextChunk(
            chunk_id=new_chunk_id(entry_id, chunk_index, level, clean),
            entry_id=entry_id,
            chunk_index=chunk_index,
            level=level,
            heading=heading,
            content=clean,
            token_count=count_tokens([clean], get_profile())[0],
            metadata=metadata or {},
        )

    def _chunk_body(self, body: str) -> list[dict[str, Any]]:
        body = (body or "").strip()
        if not body:
            return []
        chunks: list[dict[str, Any]] = []
        consumed: list[tuple[int, int]] = []
        for match in _FENCE_RE.finditer(body):
            fenced = match.group(0).strip()
            if len(fenced) >= 1200:
                chunks.append({
                    "level": 3,
                    "heading": "detail",
                    "content": _clip(fenced),
                    "metadata": {"kind": "fenced_block"},
                })
                consumed.append(match.span())
        text = self._remove_spans(body, consumed)
        for heading, section_text in self._split_markdown_sections(text):
            for content in self._window_text(section_text):
                if content.strip():
                    chunks.append({
                        "level": 2,
                        "heading": heading,
                        "content": content,
                        "metadata": {},
                    })
        if not chunks:
            chunks.append({"level": 2, "heading": None, "content": body, "metadata": {}})
        return chunks

    @staticmethod
    def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
        if not spans:
            return text
        out = []
        last = 0
        for start, end in spans:
            out.append(text[last:start])
            out.append("\n\n[long fenced block indexed as detail chunk]\n\n")
            last = end
        out.append(text[last:])
        return "".join(out)

    @staticmethod
    def _split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
        sections: list[tuple[str | None, str]] = []
        heading_stack: list[str] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        def flush() -> None:
            nonlocal current_lines
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_heading, content))
            current_lines = []

        for line in text.splitlines():
            match = _HEADING_RE.match(line)
            if match:
                flush()
                level = len(match.group(1))
                heading_stack[:] = heading_stack[: level - 1]
                heading_stack.append(match.group(2).strip())
                current_heading = " / ".join(heading_stack)
                continue
            current_lines.append(line)
        flush()
        return sections or [(None, text.strip())]

    def _window_text(self, text: str) -> list[str]:
        """Windows of ``target_tokens`` model tokens overlapping by ``overlap_tokens``.

        Each window is a slice of the original text. Its edges move up to ``_SNAP_TOKENS``
        tokens to land between words when the text has any, and fall on a token boundary
        otherwise (CJK, base64).

        A section longer than ``SEGMENT_CHARS`` (a conversation log or a dump with no
        headings is one section) is windowed one segment at a time, so no tokenizer call
        sees an unbounded text. Windows do not overlap across a segment edge; a section
        within the bound is one segment and is windowed exactly as a whole.
        """
        text = text.strip()
        if not text:
            return []
        windows: list[str] = []
        for start, end in segment_text(text):
            windows.extend(self._window_segment(text[start:end].strip()))
        return windows

    def _window_segment(self, text: str) -> list[str]:
        """``_window_text`` of one segment: the text is already stripped and bounded."""
        if not text:
            return []
        # A short ASCII text needs no tokenizer (the target counts content tokens only).
        if ascii_within_limit(text, self.target_tokens + SPECIAL_TOKENS):
            return [text]
        tokens = token_offsets(text, get_profile())
        if len(tokens) <= self.target_tokens:
            return [text]

        def starts_word(index: int) -> bool:
            return index == 0 or text[tokens[index][0] - 1].isspace()

        windows: list[str] = []
        start = 0
        while True:
            end = min(start + self.target_tokens, len(tokens))
            if end < len(tokens):
                end = next((i for i in range(end, max(start, end - _SNAP_TOKENS), -1) if starts_word(i)), end)
            windows.append(text[tokens[start][0]:tokens[end - 1][1]])
            if end >= len(tokens):
                return windows
            next_start = max(start + 1, end - self.overlap_tokens)
            start = next((i for i in range(next_start, max(start, next_start - _SNAP_TOKENS), -1) if starts_word(i)),
                         next_start)
