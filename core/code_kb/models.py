"""Pure-Python data contracts shared across code indexing boundaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    content: str
    code: str
    label: str = ""


__all__ = ["CodeChunk"]
