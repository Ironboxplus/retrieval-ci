from __future__ import annotations

import asyncio
import io
import json
import tempfile
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from core.db import get_db_session
from core.db.models import DataSource, Document, DocumentStatus, Paper, SourceType
from pipeline.paper_crawler.integration import api_routes as paper_api_routes
from pipeline.paper_crawler.integration.api_routes import create_paper_router


def test_pdf_resolution_uses_paper_id_when_database_path_is_missing(tmp_path):
    pdf_root = tmp_path / "pdf-root"
    pdf_path = pdf_root / "aaai" / "2026" / "aaai-2026-37659.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.7\n4DSTR\n")
    paper = SimpleNamespace(
        paper_id="aaai-2026-37659",
        pdf_path=None,
        txt_path=None,
        venue="AAAI",
        year=2026,
    )
    settings = SimpleNamespace(
        paper_pdf_roots=[str(pdf_root)],
        paper_markdown_roots=[],
    )

    _, resolved = paper_api_routes._resolve_paper_artifact_path(
        paper, "pdf", settings)

    assert resolved == pdf_path


class TestPaperBatchDownload:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix="retrival_paper_batch_download_test_",
            ignore_cleanup_errors=True,
        )
        self.root = Path(self.temp_dir.name)
        self.document_ids: list[int] = []
        for index in range(2):
            paper_id = f"batch-paper-{index + 1}"
            markdown = self.root / f"{paper_id}.md"
            pdf = self.root / f"{paper_id}.pdf"
            markdown.write_bytes(f"# Paper {index + 1}\n".encode("utf-8"))
            pdf.write_bytes(f"%PDF-1.7\npaper-{index + 1}\n".encode())
            self.document_ids.append(self._seed_paper(paper_id, markdown, pdf))

        app = FastAPI()
        app.include_router(create_paper_router(indexer=object()))
        self.client = TestClient(app)

    def teardown_method(self):
        with get_db_session() as db:
            db.query(Paper).filter(Paper.doc_id.in_(self.document_ids)).delete(
                synchronize_session=False)
            db.query(Document).filter(Document.id.in_(self.document_ids)).delete(
                synchronize_session=False)
            db.commit()
        self.temp_dir.cleanup()

    def _seed_paper(self, paper_id: str, markdown: Path, pdf: Path) -> int:
        with get_db_session() as db:
            document = Document(
                title=paper_id,
                source=f"https://example.org/{paper_id}",
                source_type=SourceType.paper,
                data_source=DataSource.system,
                status=DocumentStatus.indexed,
                file_path=str(markdown),
                url=f"https://example.org/{paper_id}",
                metadata_json={"paper_id": paper_id, "conference": "ICML", "year": 2026},
            )
            db.add(document)
            db.flush()
            db.add(Paper(
                paper_id=paper_id,
                doc_id=document.id,
                title=paper_id,
                abstract="Batch download test paper.",
                authors_json=["Test Author"],
                year=2026,
                venue="ICML",
                paper_source="icml",
                pdf_path=str(pdf),
                txt_path=str(markdown),
                citation_count=0,
            ))
            db.commit()
            return int(document.id)

    def test_batch_download_returns_requested_pdf_and_markdown_files(self):
        response = self.client.post("/api/papers/download-batch", json={
            "paper_ids": ["batch-paper-1", "batch-paper-2"],
            "artifacts": "both",
        })

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/zip")
        assert response.headers["accept-ranges"] == "none"
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert set(archive.namelist()) == {
                "batch-paper-1.pdf",
                "batch-paper-1.md",
                "batch-paper-2.pdf",
                "batch-paper-2.md",
                "manifest.json",
            }
            assert archive.read("batch-paper-1.pdf").startswith(b"%PDF-")
            assert archive.read("batch-paper-2.md") == b"# Paper 2\n"
            manifest = json.loads(archive.read("manifest.json"))
        assert len(manifest["files"]) == 4
        assert manifest["errors"] == []

    def test_markdown_file_endpoint_streams_original_bytes_with_ranges(self):
        response = self.client.get(
            "/api/papers/batch-paper-1/markdown",
            headers={"Range": "bytes=0-4"},
        )

        assert response.status_code == 206
        assert response.headers["content-range"].startswith("bytes 0-4/")
        assert response.content == b"# Pap"

    def test_fulltext_response_never_exposes_the_resolved_filesystem_path(self):
        response = self.client.get("/api/papers/batch-paper-1/fulltext")

        assert response.status_code == 200
        assert response.json()["content"] == "# Paper 1\n"
        assert "path" not in response.json()
        assert str(self.root) not in response.text

    def test_batch_download_rejects_more_than_ten_papers(self):
        response = self.client.post("/api/papers/download-batch", json={
            "paper_ids": [f"paper-{index}" for index in range(11)],
            "artifacts": "pdf",
        })

        assert response.status_code == 422

    def test_batch_download_rejects_http_ranges_for_generated_archives(self):
        response = self.client.post(
            "/api/papers/download-batch",
            headers={"Range": "bytes=0-31"},
            json={
                "paper_ids": ["batch-paper-1"],
                "artifacts": "pdf",
            },
        )

        assert response.status_code == 416
        assert response.headers["accept-ranges"] == "none"

    def test_batch_download_rejects_oversized_source_sets(self, monkeypatch):
        monkeypatch.setattr(paper_api_routes, "_PAPER_BATCH_MAX_SOURCE_BYTES", 1)

        response = self.client.post("/api/papers/download-batch", json={
            "paper_ids": ["batch-paper-1"],
            "artifacts": "pdf",
        })

        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"]

    def test_batch_download_rejects_when_build_slots_are_busy(self):
        assert paper_api_routes._PAPER_BATCH_BUILD_SLOTS.acquire(blocking=False)
        assert paper_api_routes._PAPER_BATCH_BUILD_SLOTS.acquire(blocking=False)
        try:
            response = self.client.post("/api/papers/download-batch", json={
                "paper_ids": ["batch-paper-1"],
                "artifacts": "pdf",
            })
        finally:
            paper_api_routes._PAPER_BATCH_BUILD_SLOTS.release()
            paper_api_routes._PAPER_BATCH_BUILD_SLOTS.release()

        assert response.status_code == 429

    def test_batch_slot_is_held_until_response_cleanup(self, monkeypatch):
        slot = threading.BoundedSemaphore(value=1)
        archive_path = self.root / "held-until-cleanup.zip"
        archive_path.write_bytes(b"zip")
        archive = paper_api_routes.PaperBatchArchive(
            path=archive_path,
            filename="papers.zip",
            file_count=1,
            error_count=0,
        )
        monkeypatch.setattr(paper_api_routes, "_PAPER_BATCH_BUILD_SLOTS", slot)
        monkeypatch.setattr(
            paper_api_routes,
            "_create_paper_batch_archive",
            lambda _request: archive,
        )
        route = next(
            route
            for route in self.client.app.routes
            if getattr(route, "path", None) == "/api/papers/download-batch"
            and "POST" in getattr(route, "methods", set())
        )
        request = paper_api_routes.PaperBatchDownloadRequest(
            paper_ids=["batch-paper-1"],
            artifacts="pdf",
        )
        http_request = Request({
            "type": "http",
            "method": "POST",
            "path": "/api/papers/download-batch",
            "headers": [],
        })

        response = asyncio.run(route.endpoint(request, http_request))

        assert not slot.acquire(blocking=False)
        assert archive_path.exists()
        asyncio.run(response.background())
        assert not archive_path.exists()
        assert slot.acquire(blocking=False)
        slot.release()
