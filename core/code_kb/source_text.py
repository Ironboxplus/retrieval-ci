"""Deterministic decoding for source files from local paths and Git objects."""

from __future__ import annotations


class BinarySourceTextError(ValueError):
    """Raised when an index candidate contains a binary NUL byte."""


def decode_source_bytes(content: bytes) -> str:
    """Decode source text without dropping legacy Western-encoded repositories."""

    if b"\x00" in content:
        raise BinarySourceTextError("source content contains a NUL byte")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return content.decode("cp1252")
        except UnicodeDecodeError:
            return content.decode("latin-1")
