"""TDD red tests for SectionAwareChunker fallback paths + create_chunker factory.

Covers the cases the original SectionAwareChunker silently mishandled:
- markdown without any headers (returned 1 dummy chunk with "Untitled")
- HTML input (tags leaked into the chunk text)
- empty / whitespace-only docs
- oversized sections (no headings)
- factory selection by content_type
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import Chunk

PLAIN_TEXT_DOC = "First paragraph of an uploaded plain text document.\n\n" + (
    "A long body that needs to be split into multiple bounded chunks. " * 200
) + "\n\nClosing remarks."

SHORT_NO_HEADER_DOC = "A short note. Just a sentence or two."

ONE_PARAGRAPH_DOC = "A single research paper about something. " * 50

HTML_DOC = """<!DOCTYPE html>
<html>
  <head><title>Sample</title><script>alert('xss');</script></head>
  <body>
    <h1>Real Title</h1>
    <p>First paragraph of the page.</p>
    <p>Second paragraph with <a href="#">a link</a> inside.</p>
    <style>body { color: red; }</style>
    <script>console.log('inline');</script>
  </body>
</html>"""

MARKDOWN_WITH_HEADERS = """# Title

## Section A

Some content for section A.

## Section B

Some content for section B that is a bit longer to verify chunking inside.
"""

OVERSIZED_HEADERLESS_DOC = "\n\n".join(
    f"Paragraph number {i} discussing aspect {i % 7} of the topic in detail."
    for i in range(400)
)


class TestSectionChunkerHandlesNoHeaders(unittest.TestCase):
    """Headerless markdown / plain text must still produce useful chunks."""

    def setUp(self):
        from core.indexing.chunker import SectionAwareChunker

        self.chunker = SectionAwareChunker(max_chunk_chars=500, chunk_overlap=50)

    def test_plain_text_produces_multiple_bounded_chunks(self):
        chunks = self.chunker.chunk_document("doc1", PLAIN_TEXT_DOC)
        self.assertGreater(len(chunks), 1, "long body must split")
        for c in chunks:
            self.assertLessEqual(len(c.content), 600, "chunks bounded by max_chunk_chars + overlap")
            self.assertIsInstance(c, Chunk)

    def test_short_headerless_returns_single_chunk_preserving_text(self):
        chunks = self.chunker.chunk_document("doc2", SHORT_NO_HEADER_DOC)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content.strip(), SHORT_NO_HEADER_DOC.strip())

    def test_section_name_is_body_when_no_headers(self):
        chunks = self.chunker.chunk_document("doc3", ONE_PARAGRAPH_DOC)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertEqual(
                c.metadata.get("section_name"), "body",
                "headerless text must use 'body' as section_name, not 'Untitled'",
            )

    def test_oversized_headerless_doc_splits_into_many_chunks(self):
        chunks = self.chunker.chunk_document("doc4", OVERSIZED_HEADERLESS_DOC)
        self.assertGreater(len(chunks), 5, "very long headerless text must split into many bounded chunks")


class TestSectionChunkerHandlesHtml(unittest.TestCase):
    """HTML input must be stripped before chunking."""

    def setUp(self):
        from core.indexing.chunker import SectionAwareChunker

        self.chunker = SectionAwareChunker(max_chunk_chars=500)

    def test_html_tags_are_stripped_from_chunks(self):
        chunks = self.chunker.chunk_document("doc1", HTML_DOC, content_type="html")
        for c in chunks:
            self.assertNotIn("<p>", c.content)
            self.assertNotIn("</p>", c.content)
            self.assertNotIn("<a ", c.content)
            self.assertNotIn("<script", c.content)
            self.assertNotIn("<style", c.content)

    def test_html_script_and_style_content_removed(self):
        chunks = self.chunker.chunk_document("doc1", HTML_DOC, content_type="html")
        joined = " ".join(c.content for c in chunks)
        self.assertNotIn("alert", joined, "script body must be stripped")
        self.assertNotIn("console.log", joined, "inline script body must be stripped")
        self.assertNotIn("color: red", joined, "style body must be stripped")

    def test_html_visible_text_preserved(self):
        chunks = self.chunker.chunk_document("doc1", HTML_DOC, content_type="html")
        joined = " ".join(c.content for c in chunks)
        self.assertIn("Real Title", joined)
        self.assertIn("First paragraph", joined)
        self.assertIn("Second paragraph", joined)


class TestSectionChunkerEmptyDocs(unittest.TestCase):
    def setUp(self):
        from core.indexing.chunker import SectionAwareChunker

        self.chunker = SectionAwareChunker()

    def test_empty_string_returns_no_chunks(self):
        self.assertEqual(self.chunker.chunk_document("d", ""), [])

    def test_whitespace_only_returns_no_chunks(self):
        self.assertEqual(self.chunker.chunk_document("d", "\n\n   \t  \n"), [])

    def test_html_with_no_visible_text_returns_no_chunks(self):
        empty_html = "<html><head><script>x();</script></head><body>   </body></html>"
        self.assertEqual(self.chunker.chunk_document("d", empty_html, content_type="html"), [])


class TestSectionChunkerOversizedSection(unittest.TestCase):
    """A long section under a single heading must split into multiple bounded chunks."""

    def test_long_section_splits_keeping_section_name(self):
        from core.indexing.chunker import SectionAwareChunker

        chunker = SectionAwareChunker(max_chunk_chars=400, chunk_overlap=40)
        text = "## Method\n\n" + ("Detailed methodology paragraph. " * 300)
        chunks = chunker.chunk_document("doc1", text)
        method_chunks = [c for c in chunks if c.metadata.get("section_name") == "Method"]
        self.assertGreater(len(method_chunks), 1)
        for c in method_chunks:
            self.assertLessEqual(len(c.content), 500)


class TestSectionChunkerContentTypeRouting(unittest.TestCase):
    """content_type='markdown' must still produce per-heading sections."""

    def test_markdown_content_type_produces_sections(self):
        from core.indexing.chunker import SectionAwareChunker

        chunker = SectionAwareChunker()
        chunks = chunker.chunk_document("doc1", MARKDOWN_WITH_HEADERS, content_type="markdown")
        names = {c.metadata.get("section_name") for c in chunks}
        self.assertTrue(names & {"Title", "Section A", "Section B"})

    def test_text_content_type_does_not_treat_pound_as_heading(self):
        from core.indexing.chunker import SectionAwareChunker

        chunker = SectionAwareChunker()
        text = "# This is a plain text file that starts with a hash\n\nbody text."
        chunks = chunker.chunk_document("doc1", text, content_type="text")
        for c in chunks:
            self.assertEqual(c.metadata.get("section_name"), "body")


class TestCreateChunkerFactory(unittest.TestCase):
    """Factory returns the appropriate Chunker instance per content_type."""

    def test_markdown_returns_section_aware(self):
        from core.indexing.chunker import SectionAwareChunker, create_chunker

        c = create_chunker(content_type="markdown")
        self.assertIsInstance(c, SectionAwareChunker)

    def test_html_returns_section_aware(self):
        from core.indexing.chunker import SectionAwareChunker, create_chunker

        c = create_chunker(content_type="html")
        self.assertIsInstance(c, SectionAwareChunker)

    def test_text_returns_section_aware(self):
        from core.indexing.chunker import SectionAwareChunker, create_chunker

        c = create_chunker(content_type="text")
        self.assertIsInstance(c, SectionAwareChunker)

    def test_default_returns_section_aware(self):
        from core.indexing.chunker import SectionAwareChunker, create_chunker

        c = create_chunker()
        self.assertIsInstance(c, SectionAwareChunker)

    def test_multigranularity_returns_text_chunker(self):
        from core.indexing.chunker import TextChunker, create_chunker

        c = create_chunker(content_type="multigranularity")
        self.assertIsInstance(c, TextChunker)

    def test_unknown_falls_back_to_section_aware(self):
        from core.indexing.chunker import SectionAwareChunker, create_chunker

        c = create_chunker(content_type="application/x-mystery")
        self.assertIsInstance(c, SectionAwareChunker)

    def test_factory_passes_through_size_params(self):
        from core.indexing.chunker import create_chunker

        c = create_chunker(content_type="text", max_chunk_chars=777, chunk_overlap=33)
        self.assertEqual(c.max_chunk_chars, 777)
        self.assertEqual(c.chunk_overlap, 33)


class TestChunkerBaseInterface(unittest.TestCase):
    """Both chunkers must implement the Chunker ABC."""

    def test_section_aware_is_chunker(self):
        from core.indexing.chunker import Chunker, SectionAwareChunker

        self.assertTrue(issubclass(SectionAwareChunker, Chunker))

    def test_text_chunker_is_chunker(self):
        from core.indexing.chunker import Chunker, TextChunker

        self.assertTrue(issubclass(TextChunker, Chunker))


if __name__ == "__main__":
    unittest.main()
