"""TDD: `/api/documents/{doc_id}/fulltext` and `/{doc_id}/markdown`.

Uploaded documents (books) have no `paper_id`, so `/api/papers/{id}/fulltext`
cannot reach them — and that route reads a crawler artifact uploads never have.
These two routes are the document-keyed equivalents.

Real SQLite + a real file on disk under the actual upload root, so the test
exercises `upload_md_path()`'s real convention rather than a stand-in. Both the
rows and the files are removed in teardown.
"""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.documents import router as documents_router
from app.services.pipeline.upload_context import upload_md_path
from core.db import get_db_session
from core.db.models import DataSource, Document, DocumentStatus, SourceType


MARKDOWN = "# Seeded Book\n\nChapter one body text.\n\n## 2 Latent variables\n\nBody.\n"


class DocumentFulltextRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suffix = uuid.uuid4().hex[:12]
        # Mirror a real upload: an original file plus extracted markdown at the
        # path upload_md_path() dictates.
        cls.file_path = str(Path(upload_md_path("x.pdf")).parent.parent / f"{cls.suffix}.pdf")
        cls.md_path = Path(upload_md_path(cls.file_path))
        cls.md_path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" on purpose: text-mode writes translate to CRLF on Windows,
        # which /markdown would then stream back verbatim while /fulltext decoded
        # it away — the two routes would disagree for a reason that has nothing
        # to do with either. The extraction pipeline writes LF.
        cls.md_path.write_text(MARKDOWN, encoding="utf-8", newline="\n")

        with get_db_session() as db:
            doc = Document(
                title=f"Seeded Book {cls.suffix}",
                source=f"{cls.suffix}.pdf",
                source_type=SourceType.upload,
                data_source=DataSource.user,
                status=DocumentStatus.indexed,
                file_path=cls.file_path,
            )
            db.add(doc)
            db.flush()
            cls.doc_id = doc.id

            # A second document with no extracted markdown, to pin the 404.
            bare = Document(
                title=f"No Markdown {cls.suffix}",
                source=f"{cls.suffix}-bare.pdf",
                source_type=SourceType.upload,
                data_source=DataSource.user,
                status=DocumentStatus.indexed,
                file_path=str(Path(cls.file_path).with_name(f"{cls.suffix}-bare.pdf")),
            )
            db.add(bare)
            db.flush()
            cls.bare_doc_id = bare.id

        app = FastAPI()
        app.include_router(documents_router, prefix="/api/documents")
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.md_path.unlink(missing_ok=True)
        with get_db_session() as db:
            db.query(Document).filter(
                Document.id.in_([cls.doc_id, cls.bare_doc_id])
            ).delete(synchronize_session=False)

    def test_fulltext_returns_complete_markdown(self) -> None:
        response = self.client.get(f"/api/documents/{self.doc_id}/fulltext")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["content"], MARKDOWN)
        self.assertEqual(body["size"], len(MARKDOWN))
        self.assertEqual(body["doc_id"], self.doc_id)

    def test_fulltext_is_not_reassembled_from_chunks(self) -> None:
        """The body must be the file verbatim — no collapsing, no chunk joins.

        L3 chunks overlap, so a concatenation would both duplicate and drop
        text; this asserts the blank lines and heading structure survive.
        """
        content = self.client.get(f"/api/documents/{self.doc_id}/fulltext").json()["content"]
        self.assertIn("\n\n## 2 Latent variables\n\n", content)

    def test_markdown_route_streams_the_file(self) -> None:
        response = self.client.get(f"/api/documents/{self.doc_id}/markdown")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.headers["content-type"].startswith("text/markdown"))
        self.assertEqual(response.text, MARKDOWN)

    def test_markdown_route_supports_range(self) -> None:
        """save_path downloads resume through Range; FileResponse must honour it."""
        response = self.client.get(
            f"/api/documents/{self.doc_id}/markdown", headers={"Range": "bytes=0-5"})
        self.assertEqual(response.status_code, 206, response.text)
        self.assertEqual(response.text, MARKDOWN[:6])

    def test_missing_markdown_is_404_not_500(self) -> None:
        response = self.client.get(f"/api/documents/{self.bare_doc_id}/fulltext")
        self.assertEqual(response.status_code, 404)
        self.assertIn("Markdown", response.json()["detail"])

    def test_unknown_document_is_404(self) -> None:
        response = self.client.get("/api/documents/999999999/fulltext")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Document not found")


if __name__ == "__main__":
    unittest.main()
