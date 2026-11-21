"""Chunking for ACE journal entries."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.services.journal.repository import new_chunk_id


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


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t]


def _clip(text: str, limit: int = 3000) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\n...[truncated]"


class JournalChunker:
    """Markdown-aware chunker that preserves short journal notes."""

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
        chunks.append(self._make(entry_id, 0, 1, summary_content, heading="summary"))
        for idx, item in enumerate(self._chunk_body(body), start=1):
            chunks.append(self._make(
                entry_id,
                idx,
                item["level"],
                item["content"],
                heading=item.get("heading"),
                metadata=item.get("metadata") or {},
            ))
        return chunks

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
            token_count=len(_tokens(clean)),
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
        words = _tokens(text)
        if not words:
            return []
        if len(words) <= self.target_tokens:
            return [text.strip()]
        windows: list[str] = []
        step = max(1, self.target_tokens - self.overlap_tokens)
        start = 0
        while start < len(words):
            windows.append(" ".join(words[start:start + self.target_tokens]))
            if start + self.target_tokens >= len(words):
                break
            start += step
        return windows
