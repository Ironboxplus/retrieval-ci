"""Full-text quality rules for non-publisher copies (operator, 2026-11-10).

The official listing fixes venue, year, track, DOI and title; an arXiv or
open-repository PDF can only be the full text of that record, never relabel it.
A copy is admitted by DOI identity (the Semantic Scholar bridge), its latest
arXiv version is pinned, an early draft is refused, and before it is stored the
PDF itself must carry the official title on page 1 with a sane page count.

Fixtures are real: arXiv API entries and official TPAMI records captured on
2026-11-10 (tests/fixtures/tier1_sources), and the first two pages of the arXiv
PDF of tpami-2024-3112205 downloaded through the proxy the same day.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from pypdf import PdfReader

from core.db.models import Chunk as DBChunk, Document as DBDocument, Paper as DBPaper, SourceType
from pipeline.paper_crawler.ccf_a_main_corpus import (
    CorpusPaper,
    FULL_TEXT_ARXIV,
    FULL_TEXT_CHECK_RULE,
    FULL_TEXT_OA_REPO,
    FULL_TEXT_PUBLISHER,
    MainConferenceCorpusBuilder,
    PvldbVolumeSource,
    apply_open_access_records,
    ascii_pdf_url,
    candidate_url_problem,
    doi_in_url,
    figshare_file_url,
    full_text_content_problem,
    full_text_text_problem,
    identity_rejection_path,
    parse_arxiv_feed,
    provenance_path,
    read_full_text_provenance,
    verify_full_text_identity,
    vet_arxiv_full_text,
)
from pipeline.paper_crawler.integration.index_integration import PaperIndexConfig, PaperIndexer
from scripts.sync.sync_local_conf_md import sync_slice
from tests.sqlite_task_db import make_session_factory
from tests.test_host_turn_download import RecordingProxy, tiny_pdf

FIXTURES = Path(__file__).parent / "fixtures" / "tier1_sources"
REPO = Path(__file__).resolve().parents[1]
ARXIV_PDF = FIXTURES / "tpami-2024-3112205-arxiv-p1-2.pdf"
ARXIV_PDF_TITLE = "Recognizing Predictive Substructures With Subgraph Information Bottleneck"
# Page 1 of the ACM publisher PDF of SOSP 2025 "Tai Chi", whose text layer
# prints "High-E!ciency": the ffi ligature glyph extracts as "!".
ACM_LIGATURE_PDF = FIXTURES / "sosp-2025-3764851-acm-p1.pdf"
ACM_LIGATURE_TITLE = (
    "Tai Chi: A General High-Efficiency Scheduling Framework for SmartNICs in Hyperscale Clouds"
)
TPAMI_ALIASES = ("TPAMI", "Pattern Analysis and Machine Intelligence", "PAMI")


def arxiv_records():
    return parse_arxiv_feed((FIXTURES / "arxiv_api_tpami.xml").read_text(encoding="utf-8"))


def official(arxiv_id: str) -> CorpusPaper:
    """The official TPAMI record whose S2 DOI bridge pointed at ``arxiv_id``."""
    record = json.loads((FIXTURES / "arxiv_official_records.json").read_text(encoding="utf-8"))[arxiv_id]
    return CorpusPaper(
        paper_id=record["paper_id"], title=record["title"], authors=record["authors"],
        conference=record["conference"], year=record["year"], pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        paper_page_url=f"https://doi.org/{record['doi']}", source_name="crossref", doi=record["doi"],
        arxiv_id=arxiv_id, full_text_source=FULL_TEXT_ARXIV, full_text_match="s2_doi",
        pdf_resolution_status="resolved_open_access",
    )


def vet(paper: CorpusPaper) -> dict[str, int]:
    return vet_arxiv_full_text([paper], arxiv_records(), venue_aliases=TPAMI_ALIASES, acceptance_lead_years=2)


# ------------------------------------------------------------ identity
def test_a_doi_match_is_accepted_and_pins_the_latest_arxiv_version():
    paper = official("1809.02108")
    record = arxiv_records()["1809.02108"]
    assert record.doi == paper.doi.lower()  # the arXiv record names the same DOI

    stats = vet(paper)

    assert stats["arxiv_accepted"] == 1
    assert paper.pdf_url == f"https://arxiv.org/pdf/1809.02108v{record.version}"
    assert paper.source_version == f"v{record.version}"
    assert paper.full_text_source == FULL_TEXT_ARXIV
    assert paper.full_text_match.startswith("s2_doi+arxiv_doi")


def test_a_fuzzy_or_title_only_match_is_rejected():
    data = json.loads((FIXTURES / "s2_batch.json").read_text(encoding="utf-8"))
    record = data["response"][0]  # real S2 record for 10.1109/tpami.2023.3296600
    same_title_other_doi = CorpusPaper(
        paper_id="tpami-x", title=record["title"], authors=[], conference="TPAMI", year=2024, pdf_url="",
        paper_page_url="", source_name="crossref", doi="10.1109/tpami.2023.9999999")
    near_title_same_doi = CorpusPaper(
        paper_id="tpami-y", title=record["title"] + " Revisited", authors=[], conference="TPAMI", year=2024,
        pdf_url="", paper_page_url="", source_name="crossref", doi=data["dois"][0])
    stats = {key: 0 for key in ("matched", "resolved", "arxiv", "publisher_oa", "other_oa",
                                "not_found", "identity_mismatch")}
    apply_open_access_records(
        [(same_title_other_doi.doi, same_title_other_doi), (near_title_same_doi.doi, near_title_same_doi)],
        [record, record], stats, preferred_hosts=frozenset(), excluded_hosts=frozenset({"doi.org"}),
    )
    assert stats["identity_mismatch"] == 2 and stats["resolved"] == 0
    for paper in (same_title_other_doi, near_title_same_doi):
        assert paper.pdf_url == "" and paper.full_text_source is None and paper.arxiv_id is None


def test_a_publisher_open_pdf_wins_and_keeps_no_preprint_fallback():
    data = json.loads((FIXTURES / "s2_batch.json").read_text(encoding="utf-8"))
    record, doi = data["response"][3], data["dois"][3]  # Springer OA + an arXiv id
    paper = CorpusPaper(paper_id="ijcv-x", title=record["title"], authors=[], conference="IJCV", year=2025,
                        pdf_url="", paper_page_url="", source_name="crossref", doi=doi)
    stats = {key: 0 for key in ("matched", "resolved", "arxiv", "publisher_oa", "other_oa",
                                "not_found", "identity_mismatch")}
    apply_open_access_records([(doi, paper)], [record], stats,
                              preferred_hosts=frozenset({"link.springer.com"}), excluded_hosts=frozenset())
    assert paper.pdf_url.startswith("https://link.springer.com/content/pdf/")
    assert paper.alternate_pdf_urls == [] and paper.full_text_source == FULL_TEXT_PUBLISHER
    assert paper.full_text_match == "official"


def test_the_official_record_keeps_its_venue_when_the_pdf_comes_from_arxiv(tmp_path):
    paper = official("1809.02108")
    official_fields = (paper.paper_id, paper.title, paper.conference, paper.year, paper.doi, list(paper.authors))
    vet(paper)
    assert (paper.paper_id, paper.title, paper.conference, paper.year, paper.doi, list(paper.authors)) == official_fields

    # ... through the runtime paper into SQL and the chunk payload, untouched.
    paper.pdf_path = str(tmp_path / "x.pdf")
    runtime = paper.to_runtime_paper()
    indexer = PaperIndexer(PaperIndexConfig(collection_name="unused_test"))
    factory = make_session_factory(tmp_path / "quality.db")
    with factory() as db:
        doc = DBDocument(title=runtime.title, source="s", source_type=SourceType.paper,
                         metadata_json={"paper_id": paper.paper_id})
        db.add(doc)
        db.flush()
        indexer._upsert_paper_row(db, doc.id, runtime, indexer._build_paper_metadata(runtime, None), None)
        db.add(DBChunk(doc_id=doc.id, chunk_index=0, content="c", level=1, start_idx=0, end_idx=1,
                       metadata_json={"paper_id": paper.paper_id}))
    with factory() as db:
        row = db.get(DBPaper, paper.paper_id)
        assert (row.venue, row.year, row.title, row.doi) == ("TPAMI", 2022, paper.title, paper.doi)
        assert row.venue_tier == "CCF-A" and row.arxiv_id == "1809.02108"
        assert row.full_text_source == FULL_TEXT_ARXIV
    sys.path.insert(0, str(REPO / "scripts" / "remote_embed"))
    try:
        import dump_chunks
    finally:
        sys.path.pop(0)
    conn = sqlite3.connect(tmp_path / "quality.db")
    try:
        payload = next(map(dump_chunks.build_payload, dump_chunks.iter_chunks(conn)))
    finally:
        conn.close()
    assert payload["venue"] == "TPAMI" and payload["year"] == 2022
    assert payload["preprint_full_text"] is True


def test_an_early_draft_is_rejected_and_the_paper_has_no_full_text():
    paper = official("1908.07141")  # TPAMI 2023; only arXiv v1 of 2019-08, no journal_ref or comment
    record = arxiv_records()["1908.07141"]
    assert record.updated.year == 2019 and not record.journal_ref and not record.comment

    stats = vet(paper)

    assert stats["arxiv_early_draft"] == 1
    assert paper.pdf_url == "" and paper.arxiv_id is None and paper.full_text_source is None
    assert paper.pdf_resolution_status == "needs_open_access"
    assert "arxiv_early_draft" in paper.pdf_resolution_note


def test_an_early_draft_falls_back_only_to_an_open_repository_copy():
    paper = official("1908.07141")
    paper.alternate_pdf_urls = ["https://ieeexplore.ieee.org/stamp/x.pdf", "https://eprints.example.ac.uk/1/paper.pdf"]
    vet(paper)
    assert paper.pdf_url == "https://eprints.example.ac.uk/1/paper.pdf"
    assert paper.full_text_source == FULL_TEXT_OA_REPO and paper.source_version is None


def test_arxiv_doi_typos_are_tolerated_but_a_different_doi_is_a_conflict():
    trailing_dot = official("2606.21289")  # arXiv doi "10.1109/tpami.2023.3347617." (typed period)
    truncated = official("2307.06797")  # arXiv doi "10.1109/tpami.2024.34959" (not this DOI)
    assert vet(trailing_dot)["arxiv_accepted"] == 1
    assert vet(truncated)["arxiv_doi_conflict"] == 1 and truncated.pdf_url == ""


# ------------------------------------------------------------ content check
def test_page_one_title_mismatch_is_rejected():
    right = CorpusPaper(paper_id="tpami-2024-3112205", title=ARXIV_PDF_TITLE, authors=[], conference="TPAMI",
                        year=2024, pdf_url="", paper_page_url="", source_name="crossref", official_pages=2)
    assert full_text_content_problem(ARXIV_PDF, right) is None
    wrong = CorpusPaper(paper_id="tpami-2022-2889052", title="Deep Audio-Visual Speech Recognition", authors=[],
                        conference="TPAMI", year=2022, pdf_url="", paper_page_url="", source_name="crossref")
    assert full_text_content_problem(ARXIV_PDF, wrong) == "page-1 title does not match the official title"


def test_a_broken_ligature_glyph_on_page_one_still_matches_the_title():
    def paper(title):
        return CorpusPaper(paper_id="sosp-2025-3764851", title=title, authors=[], conference="SOSP", year=2025,
                           pdf_url="", paper_page_url="", source_name="crossref")

    assert full_text_content_problem(ACM_LIGATURE_PDF, paper(ACM_LIGATURE_TITLE)) is None
    # Only the ligature letters are forgiven: another word still fails.
    other = ACM_LIGATURE_TITLE.replace("Hyperscale", "Hybrid")
    assert full_text_content_problem(ACM_LIGATURE_PDF, paper(other)) == "page-1 title does not match the official title"


def test_a_page_count_far_from_the_official_one_is_rejected():
    paper = CorpusPaper(paper_id="tpami-2024-3112205", title=ARXIV_PDF_TITLE, authors=[], conference="TPAMI",
                        year=2024, pdf_url="", paper_page_url="", source_name="crossref", official_pages=13)
    assert full_text_content_problem(ARXIV_PDF, paper) == "2 pages, outside 6-49 for the official 13"


@pytest.fixture()
def pdf_proxy():
    server = RecordingProxy(refuse_once=set())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_download_checks_non_publisher_copies_and_records_their_provenance(tmp_path, pdf_proxy):
    pdf_proxy.pdf = ARXIV_PDF.read_bytes()

    def paper(paper_id, title, url, alternates=()):
        return CorpusPaper(paper_id=paper_id, title=title, authors=["A"], conference="TPAMI", year=2024,
                           pdf_url=url, paper_page_url="", source_name="crossref", alternate_pdf_urls=list(alternates),
                           full_text_source=FULL_TEXT_ARXIV, full_text_match="s2_doi", source_version="v3",
                           official_pages=2)

    good = paper("tpami-good", ARXIV_PDF_TITLE, "http://arxiv.org/pdf/2103.11155v3")
    # A different paper's preprint, then an open repository copy of the same wrong bytes.
    bad = paper("tpami-bad", "Deep Audio-Visual Speech Recognition", "http://arxiv.org/pdf/1809.02108v2",
                alternates=["http://eprints.example.ac.uk/bad.pdf"])
    builder = MainConferenceCorpusBuilder(corpus_root=tmp_path / "corpus",
                                          proxy_url=f"http://127.0.0.1:{pdf_proxy.server_address[1]}")
    result = {p.paper_id: p for p in asyncio.run(builder.download_papers(
        [good, bad], max_bytes_per_second=0, host_turns=True, host_cooldown_seconds=0.5))}

    assert result["tpami-good"].pdf_path and result["tpami-good"].full_text_source == FULL_TEXT_ARXIV
    assert result["tpami-good"].source_version == "v3"
    assert result["tpami-good"].downloaded_from_url == "http://arxiv.org/pdf/2103.11155v3"
    assert result["tpami-bad"].pdf_path is None
    assert "page-1 title does not match" in result["tpami-bad"].download_error
    assert not list((tmp_path / "corpus" / "pdf" / "tpami" / "2024").glob("tpami-bad*"))
    assert sum("/bad.pdf" in url or "1809.02108" in url for url in pdf_proxy.requests) == 2  # both tried once
    # The rejected copies sit outside the PDF tree, with the reason per URL.
    reason = identity_rejection_path(tmp_path / "corpus", "TPAMI", 2024, "tpami-bad")
    assert reason.with_suffix(".pdf").exists()
    rejections = json.loads(reason.read_text(encoding="utf-8"))["rejections"]
    assert [(r["full_text_source"], r["url"]) for r in rejections] == [
        (FULL_TEXT_ARXIV, "http://arxiv.org/pdf/1809.02108v2"),
        (FULL_TEXT_OA_REPO, "http://eprints.example.ac.uk/bad.pdf"),
    ]

    # A later run does not fetch a known-bad copy again, and the slice sync marks the record.
    again = paper("tpami-bad", "Deep Audio-Visual Speech Recognition", "http://arxiv.org/pdf/1809.02108v2",
                  alternates=["http://eprints.example.ac.uk/bad.pdf"])
    requests_before = len(pdf_proxy.requests)
    [retried] = asyncio.run(builder.download_papers([again], max_bytes_per_second=0, host_turns=True))
    assert retried.pdf_path is None and len(pdf_proxy.requests) == requests_before
    records = [again.to_manifest_record()]
    assert sync_slice(records, "TPAMI", 2024, tmp_path / "corpus")["identity_rejected"] == 1
    assert "page-1 title does not match" in records[0]["pdf_rejected_identity"]
    assert records[0]["conference"] == "TPAMI" and not records[0].get("pdf_path")


def test_publisher_pdfs_are_checked_but_never_rejected(tmp_path, pdf_proxy):
    pdf_proxy.pdf = tiny_pdf()  # a blank page: would fail the title check
    paper = CorpusPaper(paper_id="vldb-2025-p1", title="Some VLDB Paper", authors=["A"], conference="VLDB",
                        year=2025, pdf_url="http://www.vldb.org/pvldb/vol18/p1.pdf", paper_page_url="",
                        source_name="pvldb", full_text_source=FULL_TEXT_PUBLISHER, full_text_match="official")
    builder = MainConferenceCorpusBuilder(corpus_root=tmp_path / "corpus",
                                          proxy_url=f"http://127.0.0.1:{pdf_proxy.server_address[1]}")
    [result] = asyncio.run(builder.download_papers([paper], max_bytes_per_second=0, host_turns=True))
    assert result.pdf_path and result.full_text_source == FULL_TEXT_PUBLISHER
    assert result.full_text_check == "mismatch: page-1 title does not match the official title"
    assert read_full_text_provenance(Path(result.pdf_path))["full_text_check"] == result.full_text_check


def test_a_rejected_arxiv_copy_falls_back_to_the_repository_copy_and_the_provenance_follows(tmp_path, pdf_proxy):
    pdf_proxy.pdf_by_host = {"arxiv.org": tiny_pdf(), "eprints.example.ac.uk": ARXIV_PDF.read_bytes()}
    paper = CorpusPaper(
        paper_id="tpami-2024-3112205", title=ARXIV_PDF_TITLE, authors=["A"], conference="TPAMI", year=2024,
        pdf_url="http://arxiv.org/pdf/2103.11155v3", paper_page_url="", source_name="crossref",
        alternate_pdf_urls=["http://eprints.example.ac.uk/sib.pdf"], arxiv_id="2103.11155",
        full_text_source=FULL_TEXT_ARXIV, full_text_match="s2_doi", source_version="v3", official_pages=2)
    planned = paper.to_manifest_record()
    corpus_root = tmp_path / "corpus"
    builder = MainConferenceCorpusBuilder(corpus_root=corpus_root,
                                          proxy_url=f"http://127.0.0.1:{pdf_proxy.server_address[1]}")
    builder.save_manifest = lambda *a, **k: None  # as download_subset.py does
    [result] = asyncio.run(builder.download_papers([paper], max_bytes_per_second=0, host_turns=True))

    assert result.downloaded_from_url == "http://eprints.example.ac.uk/sib.pdf"
    assert (result.full_text_source, result.arxiv_id, result.source_version) == (FULL_TEXT_OA_REPO, None, None)
    # The manifest only learns it through the sidecar, folded in by the slice sync.
    records = [planned]
    counts = sync_slice(records, "TPAMI", 2024, corpus_root)
    assert counts["provenance_applied"] == 1
    assert records[0]["full_text_source"] == FULL_TEXT_OA_REPO
    assert records[0]["downloaded_from_url"] == "http://eprints.example.ac.uk/sib.pdf"
    assert "arxiv_id" not in records[0] and "source_version" not in records[0]
    assert (records[0]["conference"], records[0]["title"]) == ("TPAMI", ARXIV_PDF_TITLE)


# ------------------------------------------------------------ stored PDFs
def stored_corpus(tmp_path: Path) -> tuple[Path, list[dict]]:
    """Three TPAMI 2024 PDFs already on disk, as a download batch leaves them:
    a right arXiv copy, a wrong arXiv copy, and a publisher PDF downloaded
    before provenance existed (no sidecar, no ``full_text_source``)."""
    corpus_root = tmp_path / "corpus"
    pdf_dir = corpus_root / "pdf" / "tpami" / "2024"
    pdf_dir.mkdir(parents=True)

    def arxiv_copy(paper_id, title):
        paper = CorpusPaper(paper_id=paper_id, title=title, authors=["A"], conference="TPAMI", year=2024,
                            pdf_url="https://arxiv.org/pdf/2103.11155v3", paper_page_url="", source_name="crossref",
                            arxiv_id="2103.11155", source_version="v3", full_text_source=FULL_TEXT_ARXIV,
                            full_text_match="s2_doi")
        shutil.copy(ARXIV_PDF, pdf_dir / f"{paper_id}.pdf")
        paper.downloaded_from_url = paper.pdf_url
        provenance_path(pdf_dir / f"{paper_id}.pdf").write_text(json.dumps({
            "paper_id": paper_id, "downloaded_from_url": paper.pdf_url, "full_text_source": FULL_TEXT_ARXIV,
            "full_text_match": "s2_doi", "source_version": "v3", "arxiv_id": "2103.11155",
        }), encoding="utf-8")
        return paper.to_manifest_record()

    publisher = CorpusPaper(paper_id="tpami-2024-ieee", title="A Title The Blank Page Lacks", authors=["A"],
                            conference="TPAMI", year=2024, pdf_url="https://www.computer.org/csdl/x.pdf",
                            paper_page_url="", source_name="crossref", doi="10.1109/tpami.2024.1")
    (pdf_dir / "tpami-2024-ieee.pdf").write_bytes(tiny_pdf())
    records = [
        arxiv_copy("tpami-2024-right", ARXIV_PDF_TITLE),
        arxiv_copy("tpami-2024-wrong", "Deep Audio-Visual Speech Recognition"),
        publisher.to_manifest_record(),
    ]
    return corpus_root, records


def test_the_stored_pdf_pass_rejects_wrong_preprints_and_keeps_publisher_misses(tmp_path):
    corpus_root, records = stored_corpus(tmp_path)
    pdf_dir = corpus_root / "pdf" / "tpami" / "2024"
    papers = [CorpusPaper.from_manifest_record(r) for r in records]

    report = verify_full_text_identity(papers, corpus_root=corpus_root, pdf_root=corpus_root / "pdf")

    assert (report["ok"], report["kept_mismatch"], report["rejected"]) == (1, 1, 1)
    assert not (pdf_dir / "tpami-2024-wrong.pdf").exists()
    assert not provenance_path(pdf_dir / "tpami-2024-wrong.pdf").exists()
    assert identity_rejection_path(corpus_root, "TPAMI", 2024, "tpami-2024-wrong").with_suffix(".pdf").exists()
    assert (pdf_dir / "tpami-2024-ieee.pdf").exists()  # a publisher PDF is never moved
    assert [m["paper_id"] for m in report["kept_mismatches"]] == ["tpami-2024-ieee"]
    assert read_full_text_provenance(pdf_dir / "tpami-2024-ieee.pdf") == {
        "paper_id": "tpami-2024-ieee", "full_text_check": "mismatch: page-1 title does not match the official title",
        "full_text_source": FULL_TEXT_PUBLISHER, "full_text_check_rule": FULL_TEXT_CHECK_RULE,
    }
    # The verdict is cached: a second pass reads no PDF.
    again = verify_full_text_identity(papers, corpus_root=corpus_root, pdf_root=corpus_root / "pdf")
    assert (again["ok"], again["kept_mismatch"], again["rejected"], again["already_checked"]) == (0, 0, 0, 2)

    counts = sync_slice(records, "TPAMI", 2024, corpus_root)
    by_id = {r["paper_id"]: r for r in records}
    assert counts["identity_rejected"] == 1 and counts["provenance_applied"] == 2
    assert "page-1 title does not match" in by_id["tpami-2024-wrong"]["pdf_rejected_identity"]
    assert by_id["tpami-2024-right"]["full_text_check"] == "ok"
    assert by_id["tpami-2024-right"]["arxiv_id"] == "2103.11155"  # untouched by the check
    assert by_id["tpami-2024-ieee"]["full_text_source"] == FULL_TEXT_PUBLISHER
    assert by_id["tpami-2024-ieee"]["full_text_check"].startswith("mismatch:")


def test_download_subset_verify_only_checks_stored_pdfs_without_a_proxy(tmp_path):
    corpus_root, records = stored_corpus(tmp_path)
    manifest = corpus_root / "manifests" / "paper_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(records), encoding="utf-8")
    env = {**os.environ, "PROXY__HTTP_PROXY": "", "HTTP_PROXY": "", "HTTPS_PROXY": ""}

    run = subprocess.run(
        [sys.executable, "scripts/download_subset.py", "--verify-only", "--target", "TPAMI:2024",
         "--corpus-root", str(corpus_root)],
        cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )

    assert run.returncode == 0, run.stderr
    assert "identity check: ok=1 kept_mismatch=1 rejected=1" in run.stdout
    assert "kept (publisher): tpami-2024-ieee" in run.stdout
    assert not (corpus_root / "pdf" / "tpami" / "2024" / "tpami-2024-wrong.pdf").exists()
    assert json.loads(manifest.read_text(encoding="utf-8")) == records  # the manifest is not written


# ------------------------------------------------------------ holes found 2026-11-12
# Real data: page-1 text of PDFs stored on .88 that the old title check rejected
# although they are the right paper, the TPAMI record whose Springer URL was a 2018
# book chapter, and the live figshare record API answer (all captured 2026-11-12).
QUALITY = Path(__file__).parent / "fixtures" / "full_text_quality"
TITLE_MISMATCH = "page-1 title does not match the official title"


def quality_json(name: str):
    return json.loads((QUALITY / name).read_text(encoding="utf-8"))


FALSE_NEGATIVES = quality_json("page1_false_negatives.json")
TPAMI_RECORD = quality_json("tpami_2022_3054304_record.json")
SPRINGER_CHAPTER = "https://link.springer.com/content/pdf/10.1007/978-3-319-99978-4_16.pdf"


def listed(row: dict, **overrides) -> CorpusPaper:
    fields = dict(paper_id=row["paper_id"], title=row["title"], authors=row["authors"], conference=row["conference"],
                  year=row["year"], pdf_url="", paper_page_url="", source_name="official")
    fields.update(overrides)
    return CorpusPaper(**fields)


def tpami_paper(**overrides) -> CorpusPaper:
    fields = {key: TPAMI_RECORD[key] for key in (
        "paper_id", "title", "authors", "conference", "year", "paper_page_url", "source_name", "doi")}
    fields.update(pdf_url="")
    fields.update(overrides)
    return CorpusPaper(**fields)


def open_access_stats() -> dict[str, int]:
    return {key: 0 for key in ("matched", "resolved", "arxiv", "publisher_oa", "other_oa", "not_found",
                               "identity_mismatch", "doi_url_mismatch")}


# ---- hole 1: a full-text URL that names another DOI
@pytest.mark.parametrize("url, expected", [
    (SPRINGER_CHAPTER, "10.1007/978-3-319-99978-4_16"),
    ("https://link.springer.com/content/pdf/10.1007%2Fs11263-025-02489-1.pdf?error=cookies_not_supported",
     "10.1007/s11263-025-02489-1"),
    ("https://dl.acm.org/doi/pdf/10.1145/3611643.3616364?download=true", "10.1145/3611643.3616364"),
    ("https://doi.org/10.1109/TPAMI.2021.3054304", "10.1109/tpami.2021.3054304"),
    ("https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/abc.123/pdf", "10.1002/abc.123"),
    ("https://arxiv.org/pdf/2103.11155v3", None),
    ("https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9340247", None),
    ("https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00004.pdf", None),
])
def test_the_doi_a_full_text_url_names_is_read_from_the_url(url, expected):
    assert doi_in_url(url) == expected


def test_a_url_naming_a_different_doi_is_refused():
    paper = tpami_paper()
    assert paper.doi == "10.1109/tpami.2021.3054304"
    problem = candidate_url_problem(SPRINGER_CHAPTER, paper)
    assert problem and "10.1007/978-3-319-99978-4_16" in problem and paper.doi in problem
    same = "https://link.springer.com/content/pdf/10.1109/TPAMI.2021.3054304.pdf"
    assert candidate_url_problem(same, paper) is None  # case-insensitive equality
    assert candidate_url_problem("https://arxiv.org/pdf/2103.11155v3", paper) is None
    assert candidate_url_problem("https://eprints.example.ac.uk/1/paper.pdf", paper) is None
    # A paper without an official DOI cannot vouch for a URL that names one.
    nameless = tpami_paper(doi=None, paper_page_url="")
    assert candidate_url_problem(SPRINGER_CHAPTER, nameless)


def test_a_publisher_url_without_a_doi_must_belong_to_the_papers_own_publisher():
    tpami = tpami_paper()  # 10.1109 -> IEEE
    ijcv = CorpusPaper(paper_id="ijcv-2025-x", title="T", authors=[], conference="IJCV", year=2025, pdf_url="",
                       paper_page_url="https://doi.org/10.1007/s11263-025-02489-1", source_name="crossref",
                       doi="10.1007/s11263-025-02489-1")
    ieee_stamp = "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9340247"
    assert candidate_url_problem(ieee_stamp, tpami) is None
    problem = candidate_url_problem(ieee_stamp, ijcv)
    assert problem and "ieeexplore.ieee.org" in problem
    assert candidate_url_problem("https://www.sciencedirect.com/science/article/pii/S0000", tpami)
    assert candidate_url_problem("https://dl.acm.org/doi/epdf", ijcv)
    # Venue-native hosts and repositories are not aggregators: nothing to prove.
    assert candidate_url_problem("https://www.vldb.org/pvldb/vol18/p1-arch.pdf", ijcv) is None
    assert candidate_url_problem("https://www.repository.cam.ac.uk/bitstream/1810/1/x.pdf", ijcv) is None


def test_resolution_skips_a_url_naming_another_doi_but_keeps_the_other_candidates():
    paper = tpami_paper()
    # Semantic Scholar's record for this DOI: its open PDF is the 2018 chapter, no arXiv copy.
    record = {"title": paper.title, "externalIds": {"DOI": paper.doi},
              "openAccessPdf": {"url": SPRINGER_CHAPTER, "status": "HYBRID"}}
    stats = open_access_stats()
    apply_open_access_records([(paper.doi, paper)], [record], stats,
                              preferred_hosts=frozenset(), excluded_hosts=frozenset({"doi.org"}))
    assert paper.pdf_url == "" and paper.full_text_source is None
    assert stats["doi_url_mismatch"] == 1 and stats["resolved"] == 0 and stats["matched"] == 1

    # With an arXiv copy the paper still resolves, to arXiv.
    with_arxiv = tpami_paper()
    record["externalIds"] = {"DOI": with_arxiv.doi, "ArXiv": "2103.11155"}
    stats = open_access_stats()
    apply_open_access_records([(with_arxiv.doi, with_arxiv)], [record], stats,
                              preferred_hosts=frozenset(), excluded_hosts=frozenset({"doi.org"}))
    assert with_arxiv.pdf_url == "https://arxiv.org/pdf/2103.11155" and with_arxiv.alternate_pdf_urls == []
    assert stats["doi_url_mismatch"] == 1 and stats["arxiv"] == 1

    # A preferred publisher host does not exempt it either.
    preferred = tpami_paper()
    stats = open_access_stats()
    apply_open_access_records([(preferred.doi, preferred)], [record], stats,
                              preferred_hosts=frozenset({"link.springer.com"}), excluded_hosts=frozenset())
    assert preferred.pdf_url == "https://arxiv.org/pdf/2103.11155" and stats["publisher_oa"] == 0


def test_download_refuses_a_stored_url_naming_another_doi_and_never_requests_it(tmp_path, pdf_proxy):
    pdf_proxy.pdf = ARXIV_PDF.read_bytes()
    # Manifest rows collected before the fix carry the chapter as the planned URL.
    paper = tpami_paper(pdf_url=SPRINGER_CHAPTER, full_text_source=FULL_TEXT_OA_REPO, full_text_match="s2_doi",
                        official_pages=2)
    paper.title = ARXIV_PDF_TITLE  # the bytes the proxy serves would pass the content check
    paper.authors = ["Junchi Yu"]
    builder = MainConferenceCorpusBuilder(corpus_root=tmp_path / "corpus",
                                          proxy_url=f"http://127.0.0.1:{pdf_proxy.server_address[1]}")
    [result] = asyncio.run(builder.download_papers([paper], max_bytes_per_second=0, host_turns=True))
    assert result.pdf_path is None and not pdf_proxy.requests
    assert "10.1007/978-3-319-99978-4_16" in result.download_error
    rejections = json.loads(identity_rejection_path(tmp_path / "corpus", "TPAMI", 2022, paper.paper_id)
                            .with_suffix(".json").read_text(encoding="utf-8"))["rejections"]
    assert [r["url"] for r in rejections] == [SPRINGER_CHAPTER]

    # Another candidate for the same paper is still tried and used.
    paper.alternate_pdf_urls = ["http://arxiv.org/pdf/2103.11155v3"]
    [again] = asyncio.run(builder.download_papers([paper], max_bytes_per_second=0, host_turns=True))
    assert again.pdf_path and again.downloaded_from_url == "http://arxiv.org/pdf/2103.11155v3"
    assert [urlsplit(url).hostname for url in pdf_proxy.requests] == ["arxiv.org"]


# ---- hole 2: the page-1 title check
@pytest.mark.parametrize("row", FALSE_NEGATIVES, ids=[row["paper_id"] for row in FALSE_NEGATIVES])
def test_the_right_paper_passes_although_pypdf_mangled_or_the_listing_differs(row):
    # These are the venue's own PDFs (ecva, vldb): the near-match rule is for them.
    assert full_text_text_problem(row["page1_text"], row["page_count"], listed(row), fuzzy=True) is None


def test_a_page_of_another_paper_never_passes_whoever_its_authors_are():
    checked = 0
    for page in FALSE_NEGATIVES:
        for other in FALSE_NEGATIVES:
            if other is page:
                continue
            problem = full_text_text_problem(page["page1_text"], page["page_count"], listed(other), fuzzy=True)
            assert problem == TITLE_MISMATCH, (page["paper_id"], other["paper_id"])
            checked += 1
    assert checked == 72


def test_a_fuzzy_title_needs_an_official_author_on_the_page():
    row = next(r for r in FALSE_NEGATIVES if r["paper_id"] == "eccv-2026-2836")  # camera-ready added "Human"
    assert full_text_text_problem(row["page1_text"], row["page_count"], listed(row), fuzzy=True) is None
    for authors in (["Zzyzx Qwertyuiop"], []):
        paper = listed(row, authors=authors)
        assert full_text_text_problem(row["page1_text"], row["page_count"], paper, fuzzy=True) == TITLE_MISMATCH
    # An exact title needs no corroboration.
    exact = next(r for r in FALSE_NEGATIVES if r["paper_id"] == "vldb-2025-p1677-tang")
    assert full_text_text_problem(exact["page1_text"], exact["page_count"], listed(exact, authors=[])) is None


def test_the_near_match_rule_is_off_for_non_publisher_copies():
    # Without ``fuzzy`` only the exact title passes, forgiving ff/fi/fl ligature letters and the
    # listing's "(Revision)": 5 of the 9 right PDFs. The other 4 are a camera-ready retitle
    # (a dropped word, a typo in the listing, a new name) and math in the title.
    exact = {"eccv-2024-00354", "eccv-2026-6676", "vldb-2024-p4683-yuan", "vldb-2025-p1677-tang", "vldb-2025-p53-yuan"}
    for row in FALSE_NEGATIVES:
        strict = full_text_text_problem(row["page1_text"], row["page_count"], listed(row))
        assert (strict is None) == (row["paper_id"] in exact), row["paper_id"]
        assert full_text_text_problem(row["page1_text"], row["page_count"], listed(row), fuzzy=True) is None


ARXIV_RETITLED = quality_json("arxiv_retitled_versions.json")


@pytest.mark.parametrize("row", ARXIV_RETITLED, ids=[row["paper_id"] for row in ARXIV_RETITLED])
def test_a_retitled_arxiv_version_is_another_version_and_stays_rejected(tmp_path, pdf_proxy, row):
    # Real arXiv copies rejected by task 974 on 2026-11-11. "I2DFormer" is the conference
    # predecessor of "I2DFormer+"; arXiv v4 prints "Slow vs. Fast", the journal "Slow versus Fast".
    pdf = QUALITY / row["pdf"]
    paper = CorpusPaper(paper_id=row["paper_id"], title=row["title"], authors=row["authors"], conference=row["conference"],
                        year=row["year"], pdf_url="", paper_page_url="", source_name="crossref")
    assert full_text_content_problem(pdf, paper) == TITLE_MISMATCH
    assert full_text_content_problem(pdf, paper, fuzzy=True) is None  # what a fuzzy rule alone would let through

    pdf_proxy.pdf = pdf.read_bytes()
    paper.pdf_url = f"http://arxiv.org/pdf/{row['arxiv_id']}v2"
    paper.full_text_source, paper.full_text_match, paper.doi = FULL_TEXT_ARXIV, "s2_doi", row["doi"]
    builder = MainConferenceCorpusBuilder(corpus_root=tmp_path / "corpus",
                                          proxy_url=f"http://127.0.0.1:{pdf_proxy.server_address[1]}")
    [result] = asyncio.run(builder.download_papers([paper], max_bytes_per_second=0, host_turns=True))
    assert result.pdf_path is None and TITLE_MISMATCH in result.download_error


def test_a_same_topic_paper_by_the_same_authors_is_rejected():
    page = PdfReader(str(ARXIV_PDF)).pages[0].extract_text()
    other = CorpusPaper(paper_id="tpami-x", title="Graph Information Bottleneck for Subgraph Recognition",
                        authors=["Junchi Yu", "Ran He"], conference="TPAMI", year=2024, pdf_url="",
                        paper_page_url="", source_name="crossref")
    assert full_text_text_problem(page, 2, other) == TITLE_MISMATCH
    assert full_text_content_problem(ARXIV_PDF, other) == TITLE_MISMATCH


def test_the_venues_own_pdf_gets_the_near_match_rule(tmp_path, pdf_proxy):
    row = ARXIV_RETITLED[0]
    pdf_proxy.pdf = (QUALITY / row["pdf"]).read_bytes()
    paper = CorpusPaper(paper_id="vldb-2025-near", title=row["title"], authors=row["authors"], conference="VLDB",
                        year=2025, pdf_url="http://www.vldb.org/pvldb/vol18/near.pdf", paper_page_url="",
                        source_name="pvldb", full_text_source=FULL_TEXT_PUBLISHER, full_text_match="official")
    builder = MainConferenceCorpusBuilder(corpus_root=tmp_path / "corpus",
                                          proxy_url=f"http://127.0.0.1:{pdf_proxy.server_address[1]}")
    [result] = asyncio.run(builder.download_papers([paper], max_bytes_per_second=0, host_turns=True))
    assert result.pdf_path and result.full_text_check == "ok"


def test_slow_versus_fast_is_still_a_different_version_and_a_revision_suffix_is_ignored():
    # journal.md 2026-11-11: arXiv v4 prints "Slow vs. Fast", the official title says "versus".
    page = "Slow vs. Fast\nHanbin Zhao, Xi Li\nAbstract. We study something else entirely."
    short = CorpusPaper(paper_id="tpami-x", title="Slow versus Fast", authors=["Hanbin Zhao"], conference="TPAMI",
                        year=2024, pdf_url="", paper_page_url="", source_name="crossref")
    assert full_text_text_problem(page, 12, short, fuzzy=True) == TITLE_MISMATCH
    row = next(r for r in FALSE_NEGATIVES if r["paper_id"] == "vldb-2025-p1677-tang")
    assert row["title"].endswith("(Revision)")
    assert full_text_text_problem(row["page1_text"], row["page_count"], listed(row)) is None


def test_the_page_count_range_still_applies_to_a_fuzzy_match():
    row = next(r for r in FALSE_NEGATIVES if r["paper_id"] == "eccv-2026-6676")
    paper = listed(row, official_pages=40)
    assert full_text_text_problem(row["page1_text"], row["page_count"], paper, fuzzy=True) == (
        f"{row['page_count']} pages, outside 20-130 for the official 40")


# ---- hole 4: repository landing pages
def test_a_figshare_record_resolves_to_its_single_pdf_file():
    record = quality_json("figshare_article_22895105.json")
    title = "Learning Geometric Transformation for Point Cloud Completion"
    assert figshare_file_url(record, title) == "https://ndownloader.figshare.com/files/40628105"
    pdf = record["files"][0]
    two = {**record, "files": [
        {**pdf, "name": "supplementary.pdf", "download_url": "https://ndownloader.figshare.com/files/1"},
        {**pdf, "name": "Learning_Geometric_Transformation_for_Point_Cloud_Completion.pdf",
         "download_url": "https://ndownloader.figshare.com/files/2"},
        {**pdf, "name": "data.zip", "mimetype": "application/zip", "download_url": "https://ndownloader.figshare.com/files/3"},
    ]}
    assert figshare_file_url(two, title).endswith("/files/2")
    ambiguous = {**two, "files": two["files"][:1] + [{**pdf, "name": "other.pdf"}]}
    with pytest.raises(ValueError, match="no single PDF"):
        figshare_file_url(ambiguous, title)
    with pytest.raises(ValueError, match="no PDF"):
        figshare_file_url({**record, "files": []}, title)


# ---- hole 3: the VLDB listing's non-ASCII file names
def test_pvldb_asks_for_the_ascii_file_name_the_server_has():
    # vol15's listing (and the page's own <a href>) say p2559-könig.pdf; the browser sends
    # k%C3%B6nig and the server answers an HTML 404. The file is p2559-konig.pdf (probed 2026-11-12).
    html = (QUALITY / "pvldb_vol15.html").read_text(encoding="utf-8")
    papers = PvldbVolumeSource().papers_from_volume_page(2022, html, "https://www.vldb.org/pvldb/volumes/15/")
    urls = {p.paper_id: p.pdf_url for p in papers}
    assert urls == {
        "vldb-2022-p1779-wenig": "https://www.vldb.org/pvldb/vol15/p1779-wenig.pdf",
        "vldb-2022-p2559-könig": "https://www.vldb.org/pvldb/vol15/p2559-konig.pdf",  # the id keeps the listing's stem
    }


def test_only_raw_non_ascii_pdf_urls_are_folded():
    assert ascii_pdf_url("https://www.vldb.org/pvldb/vol15/p3586-hättasch.pdf").endswith("p3586-hattasch.pdf")
    assert ascii_pdf_url("https://vldb.org/pvldb/vol12/p85-bleifuß.pdf").endswith("p85-bleifuss.pdf")
    # Already percent-encoded or ASCII with an apostrophe: the server serves these as listed.
    for url in ("https://vldb.org/pvldb/vol12/p85-bleifu%C3%9F.pdf", "https://www.vldb.org/pvldb/vol18/p4737-d'ascenzo.pdf",
                "https://www.vldb.org/pvldb/vol18/p1-arch.pdf"):
        assert ascii_pdf_url(url) == url


# ---- hole 4 (download): a landing page is resolved or fails once, cleanly
def figshare_routes(proxy, record):
    body = json.dumps(record).replace("https://ndownloader", "http://ndownloader").encode("utf-8")
    proxy.routes["http://api.figshare.com/v2/articles/22895105"] = (200, "application/json", body, {})
    proxy.routes["http://figshare.com/articles/journal_contribution/Learning_Geometric_Transformation/22895105"] = (
        202, "text/html; charset=UTF-8", b"", {"x-amzn-waf-action": "challenge"})


def test_a_figshare_landing_page_is_resolved_through_its_api_to_the_pdf_file(tmp_path, pdf_proxy):
    pdf_proxy.pdf = ARXIV_PDF.read_bytes()
    figshare_routes(pdf_proxy, quality_json("figshare_article_22895105.json"))
    landing = "http://figshare.com/articles/journal_contribution/Learning_Geometric_Transformation/22895105"
    paper = CorpusPaper(paper_id="ijcv-2023-x", title=ARXIV_PDF_TITLE, authors=["Junchi Yu"], conference="IJCV",
                        year=2023, pdf_url=landing, paper_page_url="", source_name="crossref", official_pages=2,
                        full_text_source=FULL_TEXT_OA_REPO, full_text_match="s2_doi")
    builder = MainConferenceCorpusBuilder(corpus_root=tmp_path / "corpus",
                                          proxy_url=f"http://127.0.0.1:{pdf_proxy.server_address[1]}")
    [result] = asyncio.run(builder.download_papers([paper], max_bytes_per_second=0, host_turns=True))
    assert result.pdf_path and result.download_error is None
    assert result.downloaded_from_url == "http://ndownloader.figshare.com/files/40628105"
    assert pdf_proxy.requests == ["http://api.figshare.com/v2/articles/22895105",
                                  "http://ndownloader.figshare.com/files/40628105"]  # the challenged page is never asked
    assert read_full_text_provenance(Path(result.pdf_path))["full_text_source"] == FULL_TEXT_OA_REPO


def test_the_file_a_figshare_record_points_to_still_goes_through_the_content_check(tmp_path, pdf_proxy):
    pdf_proxy.pdf = ARXIV_PDF.read_bytes()
    figshare_routes(pdf_proxy, quality_json("figshare_article_22895105.json"))
    landing = "http://figshare.com/articles/journal_contribution/Learning_Geometric_Transformation/22895105"
    paper = CorpusPaper(paper_id="ijcv-2023-y", title="Deep Audio-Visual Speech Recognition", authors=["A"],
                        conference="IJCV", year=2023, pdf_url=landing, paper_page_url="", source_name="crossref",
                        full_text_source=FULL_TEXT_OA_REPO, full_text_match="s2_doi")
    builder = MainConferenceCorpusBuilder(corpus_root=tmp_path / "corpus",
                                          proxy_url=f"http://127.0.0.1:{pdf_proxy.server_address[1]}")
    [result] = asyncio.run(builder.download_papers([paper], max_bytes_per_second=0, host_turns=True))
    assert result.pdf_path is None and TITLE_MISMATCH in result.download_error
    assert identity_rejection_path(tmp_path / "corpus", "IJCV", 2023, "ijcv-2023-y").with_suffix(".pdf").exists()


@pytest.mark.parametrize("host_turns", [True, False])
def test_a_repository_landing_page_that_is_not_a_pdf_fails_once_without_retries(tmp_path, pdf_proxy, host_turns):
    landing = "http://repository.example.ac.uk/handle/1234/5678"
    pdf_proxy.routes[landing] = (200, "text/html; charset=utf-8", b"<!DOCTYPE html><html><body>Item page</body></html>", {})
    paper = CorpusPaper(paper_id="tpami-2024-landing", title="T", authors=["A"], conference="TPAMI", year=2024,
                        pdf_url=landing, paper_page_url="", source_name="crossref",
                        full_text_source=FULL_TEXT_OA_REPO, full_text_match="s2_doi")
    builder = MainConferenceCorpusBuilder(corpus_root=tmp_path / "corpus",
                                          proxy_url=f"http://127.0.0.1:{pdf_proxy.server_address[1]}")
    [result] = asyncio.run(builder.download_papers([paper], max_bytes_per_second=0, host_turns=host_turns))
    assert pdf_proxy.requests == [landing]
    assert result.pdf_path is None and result.pdf_resolution_status == "download_failed"
    assert "not a PDF" in result.download_error and "text/html" in result.download_error


# ---- hole 5: an upstream file that is empty
@pytest.mark.parametrize("host_turns", [True, False])
def test_an_empty_upstream_pdf_is_a_clear_download_failure_without_retries(tmp_path, pdf_proxy, host_turns):
    # jmlr 22-1047 is served as a 0-byte application/pdf.
    url = "http://jmlr.org/papers/volume23/22-1047/22-1047.pdf"
    pdf_proxy.routes[url] = (200, "application/pdf", b"", {})
    paper = CorpusPaper(paper_id="jmlr-2022-22-1047", title="T", authors=["A"], conference="JMLR", year=2022,
                        pdf_url=url, paper_page_url="", source_name="jmlr_volume_index")
    builder = MainConferenceCorpusBuilder(corpus_root=tmp_path / "corpus",
                                          proxy_url=f"http://127.0.0.1:{pdf_proxy.server_address[1]}")
    [result] = asyncio.run(builder.download_papers([paper], max_bytes_per_second=0, host_turns=host_turns))
    assert pdf_proxy.requests == [url]
    assert result.pdf_path is None and result.pdf_resolution_status == "download_failed"
    assert "empty file (0 bytes)" in result.download_error


# ---- hole 6: the stored PDFs get the same rules
def chapter_corpus(tmp_path: Path):
    """Two stored copies of the same bytes: one fetched from the 2018 Springer chapter URL
    (provenance says publisher), one from arXiv."""
    corpus_root = tmp_path / "corpus"
    pdf_dir = corpus_root / "pdf" / "tpami" / "2024"
    pdf_dir.mkdir(parents=True)
    papers = []
    for paper_id, url, source in (("tpami-2024-chapter", SPRINGER_CHAPTER, FULL_TEXT_PUBLISHER),
                                  ("tpami-2024-arxiv", "https://arxiv.org/pdf/2103.11155v3", FULL_TEXT_ARXIV)):
        paper = tpami_paper(paper_id=paper_id, title=ARXIV_PDF_TITLE, authors=["Junchi Yu"], year=2024,
                            pdf_url=url, full_text_source=source)
        shutil.copy(ARXIV_PDF, pdf_dir / f"{paper_id}.pdf")
        provenance_path(pdf_dir / f"{paper_id}.pdf").write_text(json.dumps({
            "paper_id": paper_id, "downloaded_from_url": url, "full_text_source": source, "full_text_match": "s2_doi",
        }), encoding="utf-8")
        papers.append(paper)
    return corpus_root, pdf_dir, papers


def test_the_stored_pdf_pass_moves_out_a_copy_whose_url_names_another_doi(tmp_path):
    corpus_root, pdf_dir, papers = chapter_corpus(tmp_path)

    report = verify_full_text_identity(papers, corpus_root=corpus_root, pdf_root=corpus_root / "pdf")

    # The chapter's host is a publisher, but its DOI proves it is another paper: not a 2% title miss.
    assert (report["ok"], report["kept_mismatch"], report["rejected"]) == (1, 0, 1)
    assert not (pdf_dir / "tpami-2024-chapter.pdf").exists() and (pdf_dir / "tpami-2024-arxiv.pdf").exists()
    [rejected] = report["rejections"]
    assert rejected["paper_id"] == "tpami-2024-chapter" and "10.1007/978-3-319-99978-4_16" in rejected["problem"]
    assert identity_rejection_path(corpus_root, "TPAMI", 2024, "tpami-2024-chapter").with_suffix(".pdf").exists()


def test_a_cached_ok_verdict_is_still_checked_against_the_url_rule(tmp_path):
    corpus_root, pdf_dir, papers = chapter_corpus(tmp_path)
    sidecar = provenance_path(pdf_dir / "tpami-2024-chapter.pdf")
    sidecar.write_text(json.dumps({**json.loads(sidecar.read_text(encoding="utf-8")), "full_text_check": "ok"}),
                       encoding="utf-8")
    report = verify_full_text_identity(papers, corpus_root=corpus_root, pdf_root=corpus_root / "pdf")
    assert report["rejected"] == 1 and not (pdf_dir / "tpami-2024-chapter.pdf").exists()


def test_a_dry_run_reports_what_would_be_rejected_and_touches_nothing(tmp_path):
    corpus_root, pdf_dir, papers = chapter_corpus(tmp_path)
    before = {path.name: path.read_bytes() for path in pdf_dir.iterdir()}

    report = verify_full_text_identity(papers, corpus_root=corpus_root, pdf_root=corpus_root / "pdf", dry_run=True)

    assert (report["ok"], report["rejected"]) == (1, 1)
    assert [r["paper_id"] for r in report["rejections"]] == ["tpami-2024-chapter"]
    assert {path.name: path.read_bytes() for path in pdf_dir.iterdir()} == before
    assert not (corpus_root / "rejected_identity").exists()


def test_a_cached_title_mismatch_from_an_older_rule_is_checked_again(tmp_path):
    corpus_root, pdf_dir, papers = chapter_corpus(tmp_path)
    arxiv_pdf = pdf_dir / "tpami-2024-arxiv.pdf"
    arxiv_sidecar = provenance_path(arxiv_pdf)
    arxiv_sidecar.write_text(json.dumps({
        **json.loads(arxiv_sidecar.read_text(encoding="utf-8")), "full_text_source": FULL_TEXT_PUBLISHER,
        "full_text_check": f"mismatch: {TITLE_MISMATCH}",  # written by the 2026-11-11 rule, no rule marker
    }), encoding="utf-8")
    papers[1].pdf_url = "https://www.computer.org/csdl/x.pdf"
    report = verify_full_text_identity(papers[1:], corpus_root=corpus_root, pdf_root=corpus_root / "pdf")
    assert report["ok"] == 1 and report["already_checked"] == 0
    assert read_full_text_provenance(arxiv_pdf)["full_text_check"] == "ok"
    again = verify_full_text_identity(papers[1:], corpus_root=corpus_root, pdf_root=corpus_root / "pdf")
    assert again["already_checked"] == 1


def test_download_subset_verify_only_dry_run_lists_without_moving(tmp_path):
    corpus_root, pdf_dir, papers = chapter_corpus(tmp_path)
    manifest = corpus_root / "manifests" / "paper_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps([p.to_manifest_record() for p in papers]), encoding="utf-8")
    env = {**os.environ, "PROXY__HTTP_PROXY": "", "HTTP_PROXY": "", "HTTPS_PROXY": ""}

    run = subprocess.run(
        [sys.executable, "scripts/download_subset.py", "--verify-only", "--dry-run", "--target", "TPAMI:2024",
         "--corpus-root", str(corpus_root)],
        cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )

    assert run.returncode == 0, run.stderr
    assert "would reject: tpami-2024-chapter" in run.stdout and "rejected=1" in run.stdout
    assert (pdf_dir / "tpami-2024-chapter.pdf").exists()
