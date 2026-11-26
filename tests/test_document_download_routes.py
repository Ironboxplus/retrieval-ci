from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.documents import router as documents_router
from core.db import get_db_session
from core.db.models import DataSource, Document, DocumentStatus, Paper, SourceType


class DocumentDownloadRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix="retrival_document_download_test_",
            ignore_cleanup_errors=True,
        )
        self.markdown_path = Path(self.temp_dir.name) / "paper.md"
        self.pdf_path = Path(self.temp_dir.name) / "paper.pdf"
        self.markdown_path.write_text("# Sample Paper\n\nConverted markdown body.", encoding="utf-8")
        self.pdf_path.write_bytes(b"%PDF-1.4\n% source pdf\n")

        self.document_id = self._seed_paper_document()

        app = FastAPI()
        app.include_router(documents_router, prefix="/api/documents")
        self.client = TestClient(app)

    def tearDown(self) -> None:
        if self.document_id is not None:
            with get_db_session() as db:
                db.query(Paper).filter(Paper.doc_id == self.document_id).delete()
                db.query(Document).filter(Document.id == self.document_id).delete()
                db.commit()
        self.temp_dir.cleanup()

    def _seed_paper_document(self) -> int:
        with get_db_session() as db:
            document = Document(
                title="Readable Retrieval Paper",
                source="https://example.org/papers/readable-retrieval",
                source_type=SourceType.paper,
                data_source=DataSource.system,
                status=DocumentStatus.indexed,
                file_path=str(self.markdown_path),
                url="https://example.org/papers/readable-retrieval",
                metadata_json={
                    "paper_id": "neurips-2024-readable-retrieval",
                    "conference": "NeurIPS",
                    "year": 2024,
                    "paper_type": "main",
                    "pdf_path": str(self.pdf_path),
                },
            )
            db.add(document)
            db.flush()

            db.add(
                Paper(
                    paper_id="neurips-2024-readable-retrieval",
                    doc_id=document.id,
                    title=document.title,
                    abstract="A readable test paper for document download routes.",
                    authors_json=["Alice", "Bob"],
                    year=2024,
                    venue="NeurIPS",
                    paper_source="neurips",
                    pdf_path=str(self.pdf_path),
                    txt_path=str(self.markdown_path),
                    citation_count=1,
                )
            )
            db.commit()
            return int(document.id)

    def test_document_list_and_detail_expose_separate_source_download(self) -> None:
        list_response = self.client.get("/api/documents", params={"source_type": "paper", "page_size": 100})
        self.assertEqual(list_response.status_code, 200)
        documents = list_response.json()["documents"]
        payload = next(item for item in documents if item["id"] == self.document_id)
        self.assertEqual(payload["download_url"], f"/api/documents/{self.document_id}/download")
        self.assertEqual(payload["source_download_url"], f"/api/documents/{self.document_id}/source-download")
        self.assertEqual(payload["source_file_path"], str(self.pdf_path))

        detail_response = self.client.get(f"/api/documents/{self.document_id}")
        self.assertEqual(detail_response.status_code, 200)
        detail_payload = detail_response.json()
        self.assertEqual(detail_payload["download_url"], f"/api/documents/{self.document_id}/download")
        self.assertEqual(detail_payload["source_download_url"], f"/api/documents/{self.document_id}/source-download")
        self.assertEqual(detail_payload["file_path"], str(self.markdown_path))
        self.assertEqual(detail_payload["source_file_path"], str(self.pdf_path))

    def test_download_routes_return_markdown_and_original_pdf(self) -> None:
        document_download_response = self.client.get(f"/api/documents/{self.document_id}/download")
        self.assertEqual(document_download_response.status_code, 200)
        self.assertIn("text/markdown", document_download_response.headers["content-type"])
        self.assertIn("Converted markdown body.", document_download_response.text)

        source_download_response = self.client.get(f"/api/documents/{self.document_id}/source-download")
        self.assertEqual(source_download_response.status_code, 200)
        self.assertIn("application/pdf", source_download_response.headers["content-type"])
        self.assertEqual(source_download_response.content[:8], b"%PDF-1.4")


if __name__ == "__main__":
    unittest.main()
