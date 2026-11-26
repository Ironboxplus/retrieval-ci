"""TDD for medical corpus preparation (no bulk pull in unit tests)."""

from __future__ import annotations

import json
import time
from pathlib import Path
import pytest

from pipeline.medical_corpus.cf_bypass import turnstile_available
from pipeline.medical_corpus.europe_pmc import record_from_europe_pmc_result
from pipeline.medical_corpus.models import (
    MedicalSourceKind,
    canonical_medical_id,
    europe_pmc_pdf_url,
)
from pipeline.medical_corpus.pdf_fetch import download_pdf, plan_pdf_fetch
from pipeline.medical_corpus.proxy import MedicalProxyConfig
from pipeline.medical_corpus.shield import ShieldKind, classify_http_shield


def test_canonical_id_prefers_pmcid():
    assert canonical_medical_id(pmcid="PMC12900525") == "med-pmc12900525"
    assert canonical_medical_id(pmcid="12900525") == "med-pmc12900525"
    assert canonical_medical_id(pmid="123") == "med-pmid-123"
    assert canonical_medical_id(doi="10.1/foo").startswith("med-doi-")


def test_europe_pmc_pdf_url():
    assert europe_pmc_pdf_url("PMC1").endswith("PMC1?pdf=render")


def test_record_from_europe_pmc_result_extracts_pdf():
    row = {
        "pmcid": "PMC12900525",
        "pmid": "41694193",
        "doi": "10.22037/iej.v21i1.45971",
        "title": "Bond Strength Study",
        "pubYear": "2026",
        "isOpenAccess": "Y",
        "hasPDF": "Y",
        "license": "cc by-nc-sa",
        "fullTextUrlList": {
            "fullTextUrl": [
                {
                    "availabilityCode": "OA",
                    "documentStyle": "pdf",
                    "url": "https://europepmc.org/articles/PMC12900525?pdf=render",
                }
            ]
        },
        "authorString": "A, B",
    }
    rec = record_from_europe_pmc_result(row)
    assert rec.paper_id == "med-pmc12900525"
    assert rec.has_pdf and rec.is_open_access
    assert "pdf=render" in (rec.pdf_url or "")
    assert rec.source_kind == MedicalSourceKind.EUROPE_PMC


def test_classify_cloudflare_challenge_html():
    html = "<html>Just a moment... Checking your browser challenges.cloudflare.com Ray ID: abc</html>"
    d = classify_http_shield(status_code=403, body=html, expect_pdf=True)
    assert d.needs_cf_bypass
    assert d.kind in {ShieldKind.CLOUDFLARE, ShieldKind.CAPTCHA, ShieldKind.FORBIDDEN}


def test_classify_pdf_magic_ok():
    d = classify_http_shield(
        status_code=200,
        content_type="application/pdf",
        body=b"%PDF-1.6\n%",
        expect_pdf=True,
    )
    assert d.kind == ShieldKind.NONE


def test_classify_html_instead_of_pdf():
    d = classify_http_shield(
        status_code=200,
        content_type="text/html",
        body=b"<!DOCTYPE html><html>nope</html>",
        expect_pdf=True,
    )
    assert d.kind == ShieldKind.HTML_INSTEAD_OF_PDF
    assert d.needs_cf_bypass


def test_plan_pdf_fetch_dry(tmp_path: Path):
    from pipeline.medical_corpus.models import MedicalPaperRecord

    rec = MedicalPaperRecord(
        paper_id="med-pmc1",
        title="t",
        pmcid="PMC1",
        year=2024,
        journal="Test J",
        pdf_url="https://europepmc.org/articles/PMC1?pdf=render",
        has_pdf=True,
        is_open_access=True,
    )
    plan = plan_pdf_fetch(rec, pdf_root=tmp_path / "pdf")
    assert plan.action == "would_download"
    assert plan.dest_path.name.endswith(".pdf")


def test_download_pdf_dry_run_does_not_hit_network(tmp_path: Path):
    from pipeline.medical_corpus.models import MedicalPaperRecord

    rec = MedicalPaperRecord(
        paper_id="med-pmc1",
        title="t",
        pdf_url="https://europepmc.org/articles/PMC1?pdf=render",
        has_pdf=True,
    )
    out = download_pdf(
        rec,
        pdf_root=tmp_path / "pdf",
        proxy=MedicalProxyConfig(),
        execute=False,
    )
    assert out.download_status == "planned"


def test_proxy_config_refuses_empty():
    with pytest.raises(RuntimeError):
        MedicalProxyConfig(http_proxy="", https_proxy="", require_proxy=True).as_requests_dict()


def test_turnstile_vendor_path_resolves():
    # Package should at least be cloneable path; import may work without Chrome.
    assert isinstance(turnstile_available(), bool)


class TestManifestReplaceSurvivesAConcurrentReader:
    """The manifest write must not fail because a reader had the file open.

    `_write_jsonl` is atomic by design (tmp + fsync + os.replace) and the medical
    download stage relies on that: the disk guard writes the manifest and only
    then returns the summary the pipeline gate reads. The pipeline's own poller
    thread reads that same manifest every `poll_interval_s` through
    `_manifest_progress` -> `_load_jsonl` -> `Path.read_text()`, and the poller's
    first probe fires the instant the stage starts, so reader and writer really
    do overlap.

    On POSIX `os.replace` over an open destination is fine. On Windows it raises
    PermissionError [WinError 5], which propagates out of `run_download_pass`
    before the summary is stored — the stage fails for the wrong reason and any
    caller reading `state["summary"]` gets a KeyError. That is what made
    test_durable_low_disk_pipeline_fails_instead_of_promoting_limited_batch fail
    once in four full-suite runs on 2026-09-02 while passing in isolation, since
    load widens the overlap window.

    Production runs this pipeline on Linux, so this is a Windows-only hazard —
    but a flaky suite costs the one signal the release gate depends on.
    """

    @staticmethod
    def _record(paper_id: str):
        from pipeline.medical_corpus.models import MedicalPaperRecord
        return MedicalPaperRecord(
            paper_id=paper_id, title="t", year=2026, journal="J",
            is_open_access=True, has_pdf=True,
            pdf_url="http://127.0.0.1:9/p.pdf", download_status="pending",
        )

    def test_replace_survives_a_reader_that_releases_promptly(self, tmp_path: Path):
        """The poller opens and closes; the write must ride out that overlap."""
        import threading

        from pipeline.medical_corpus.prepare import _load_jsonl, _write_jsonl

        manifest = tmp_path / "medical_manifest.jsonl"
        _write_jsonl(manifest, [self._record("med-a")])

        opened = threading.Event()
        handle = manifest.open("r", encoding="utf-8")
        handle.read(1)

        def release_soon() -> None:
            opened.set()
            time.sleep(0.2)
            handle.close()

        closer = threading.Thread(target=release_soon, daemon=True)
        closer.start()
        opened.wait(5)
        try:
            _write_jsonl(manifest, [self._record("med-a"), self._record("med-b")])
        finally:
            closer.join(timeout=5)
            handle.close()

        assert [r.paper_id for r in _load_jsonl(manifest)] == ["med-a", "med-b"]

    def test_a_destination_held_open_forever_still_raises(self, tmp_path: Path):
        """Retrying must not decay into swallowing a checkpoint that never landed.

        A reader that never lets go is not the poller; it is a real fault, and a
        silently skipped manifest write would make restart recovery re-download
        work it already has. The bound exists so the failure stays visible.
        """
        import platform

        from pipeline.medical_corpus.prepare import _load_jsonl, _write_jsonl

        manifest = tmp_path / "medical_manifest.jsonl"
        _write_jsonl(manifest, [self._record("med-a")])

        with manifest.open("r", encoding="utf-8") as reader:
            reader.read(1)
            if platform.system() == "Windows":
                with pytest.raises(PermissionError):
                    _write_jsonl(manifest, [self._record("med-b")])
            else:
                # POSIX replaces an open destination; there is nothing to retry.
                _write_jsonl(manifest, [self._record("med-b")])
                assert [r.paper_id for r in _load_jsonl(manifest)] == ["med-b"]

    def test_a_reader_never_observes_a_partial_manifest(self, tmp_path: Path):
        """Retrying the replace must not degrade it into a non-atomic write.

        The race is symmetric on Windows: a reader opening the destination while
        the swap is in flight can itself get PermissionError. That direction is
        already tolerated where it happens for real — the pipeline's poller wraps
        every probe in `except Exception` precisely so a probe can never take the
        stage down (orchestrator.py:67) — so a transient open failure is allowed
        here. What is NOT allowed is reading a truncated or malformed manifest,
        because restart recovery would then treat downloaded PDFs as unknown.
        """
        import threading

        from pipeline.medical_corpus.prepare import _load_jsonl, _write_jsonl

        manifest = tmp_path / "medical_manifest.jsonl"
        big = [self._record(f"med-{i}") for i in range(200)]
        _write_jsonl(manifest, big)

        seen: list[int] = []
        corrupt: list[BaseException] = []
        transient = 0
        stop = threading.Event()

        def reader() -> None:
            nonlocal transient
            while not stop.is_set():
                try:
                    seen.append(len(_load_jsonl(manifest)))
                except PermissionError:
                    transient += 1  # the swap was in flight; the poller ignores this
                except BaseException as exc:  # noqa: BLE001 - a real corruption
                    corrupt.append(exc)
                    return

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            for _ in range(20):
                _write_jsonl(manifest, big)
        finally:
            stop.set()
            thread.join(timeout=10)

        assert not corrupt, f"reader parsed a broken manifest: {corrupt[:2]}"
        assert seen, "the reader never managed a single successful read"
        assert set(seen) == {200}, f"reader observed partial manifests: {sorted(set(seen))}"
