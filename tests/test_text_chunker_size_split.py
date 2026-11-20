"""``TextChunker`` has one fixed-size splitter; the list and iterator entry points agree.

Chunk ids are random uuids, so outputs are compared field by field. Inputs are the
repository's own Markdown documents (real text with headings, tables, code fences and
CJK), at both chunk sizes the production path uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.indexing.chunker import TextChunker

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [name for name in ("README.md", "overview.md", "AGENTS.md", "INDEX.md", "env.md", "plan.md")
             if (ROOT / name).exists()]


def _fields(chunks):
    return [(c.doc_id, c.content, c.level, c.start_idx, c.end_idx, c.metadata) for c in chunks]


@pytest.mark.parametrize("name", DOCUMENTS)
def test_list_and_iterator_splitters_return_the_same_chunks(name: str) -> None:
    chunker = TextChunker()
    text = (ROOT / name).read_text(encoding="utf-8", errors="replace")
    for size, overlap, level in (
        (chunker.chunk_sizes[0], chunker.overlap, 2),
        (chunker.chunk_sizes[-1], chunker.overlap // 2, 3),
    ):
        kwargs = dict(doc_id="doc", text=text, chunk_size=size, overlap=overlap, level=level)
        from_list = chunker._chunk_by_size(**kwargs)
        from_iterator = list(chunker._chunk_by_size_iter(**kwargs))
        assert from_list, f"{name} produced no chunks at size {size}"
        assert _fields(from_list) == _fields(from_iterator)
        assert len({c.id for c in from_list}) == len(from_list)


def test_chunk_document_and_iterator_agree_on_a_real_document() -> None:
    chunker = TextChunker()
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8", errors="replace")
    eager = chunker.chunk_document("doc", text)
    streamed = list(chunker.chunk_document_iter("doc", text))
    assert _fields(eager) == _fields(streamed)
