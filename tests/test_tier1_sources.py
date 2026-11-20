"""Tier-1 venue sources parse real captured pages (2026-11-10, tests/fixtures/tier1_sources).

The fixtures are trimmed copies of the live responses (ECVA archive, ECCV 2026
virtual JSON, JMLR v25, PVLDB vol 18, IEEE CSDL GraphQL, Crossref, Semantic
Scholar), so these tests pin the parsers to the publishers' real markup. The
`network` tests repeat one live request per source through the proxy.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from pipeline.paper_crawler import ccf_a_main_corpus as corpus
from pipeline.paper_crawler.ccf_a_main_corpus import (
    CorpusPaper,
    CrossrefJournalSource,
    DblpAcmProceedingsSource,
    EcvaProceedingsSource,
    IeeeCsdlProceedingsSource,
    JmlrVolumeSource,
    OfficialProceedingsClient,
    PvldbVolumeSource,
    apply_open_access_records,
    build_sources,
)

FIXTURES = Path(__file__).parent / "fixtures" / "tier1_sources"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture_json(name: str):
    return json.loads(fixture_text(name))


def test_ecva_archive_reads_only_the_requested_edition():
    soup = BeautifulSoup(fixture_text("ecva_papers.html"), "html.parser")
    source = EcvaProceedingsSource()
    papers_2024 = source.papers_from_archive(2024, soup)
    assert [p.paper_id for p in papers_2024] == [
        "eccv-2024-00004", "eccv-2024-00006", "eccv-2024-00010", "eccv-2024-00019", "eccv-2024-12638",
    ]
    first = papers_2024[0]
    assert first.title.startswith("Is Retain Set All You Need in Machine Unlearning?")
    assert first.authors == ["Jacopo Bonato", "Marco Cotogni", "Luigi Sabetta"]
    assert first.pdf_url == "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00004.pdf"
    assert first.doi == "10.1007/978-3-031-73232-4_1"
    assert papers_2024[-1].title.startswith("DεpS: Delayed ε-Shrinking")
    papers_2022 = source.papers_from_archive(2022, soup)
    assert len(papers_2022) == 2 and all("eccv_2022" in p.pdf_url for p in papers_2022)
    with pytest.raises(RuntimeError):
        source.papers_from_archive(2018, soup)


def test_eccv_2026_virtual_json_keeps_one_row_per_paper_with_its_track():
    papers = EcvaProceedingsSource().papers_from_virtual_json(2026, fixture_json("eccv2026_orals_posters.json"))
    by_id = {p.paper_id: p for p in papers}
    assert len(papers) == 5  # two oral/spotlight events repeat posters under negative ids
    assert by_id["eccv-2026-13401"].venue_track == "oral"
    assert by_id["eccv-2026-7293"].venue_track == "spotlight"
    assert by_id["eccv-2026-7443"].venue_track == "poster"
    assert by_id["eccv-2026-7443"].pdf_url == "https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/7443.pdf"
    assert by_id["eccv-2026-7443"].paper_page_url == "https://eccv.ecva.net/virtual/2026/poster/4784"
    assert all(p.authors for p in papers)


def test_jmlr_volume_page_marks_mloss_papers():
    soup = BeautifulSoup(fixture_text("jmlr_v25.html"), "html.parser")
    papers = JmlrVolumeSource().papers_from_volume_page(2024, soup, "https://jmlr.org/papers/v25/")
    assert [p.paper_id for p in papers] == ["jmlr-2024-19-301", "jmlr-2024-21-0264", "jmlr-2024-23-0572"]
    assert papers[0].pdf_url == "https://jmlr.org/papers/volume25/19-301/19-301.pdf"
    assert papers[0].authors == ["Jonathan K. Su"]
    assert [p.venue_track for p in papers] == ["research", "research", "mloss"]


def test_pvldb_volume_drops_demos_and_front_matter():
    papers = PvldbVolumeSource().papers_from_volume_page(
        2025, fixture_text("pvldb_vol18.html"), "https://www.vldb.org/pvldb/volumes/18/")
    assert [p.paper_id for p in papers] == ["vldb-2025-p1-arch", "vldb-2025-p14-hajidehi", "vldb-2025-p4763-he"]  # the 5-page "A Demonstration of Q2O" is a demo
    assert papers[0].pdf_url == "https://www.vldb.org/pvldb/vol18/p1-arch.pdf"
    assert papers[0].abstract.startswith("Although user-defined functions")
    assert all(p.conference == "VLDB" and p.year == 2025 for p in papers)


def test_csdl_listing_skips_front_matter_and_leaves_closed_papers_for_resolution():
    source = IeeeCsdlProceedingsSource("IEEE S&P", "sp", {2024: "1RjE8VKKk1y", 2025: "21B7ONGXzZ6"})
    papers_2024 = source.papers_from_graphql(2024, fixture_json("csdl_1RjE8VKKk1y.json"))
    assert [p.title for p in papers_2024] == [
        "Side-Channel-Assisted Reverse-Engineering of Encrypted DNN Hardware Accelerator IP and Attack Surface Exploration"
    ]
    paper = papers_2024[0]
    assert paper.paper_id == "sp-2024-00001"
    assert paper.pdf_url.startswith("https://www.computer.org/csdl/pds/api/csdl/proceedings/download-article/")
    papers_2025 = source.papers_from_graphql(2025, fixture_json("csdl_21B7ONGXzZ6.json"))
    assert len(papers_2025) == 2  # "Program Committee" (6 pp) and "Author Index" (17 pp) are front matter
    assert all(p.pdf_url == "" and p.pdf_resolution_status == "needs_open_access" for p in papers_2025)
    assert all(p.doi.lower().startswith("10.1109/sp61157.2025.") for p in papers_2025)


def test_crossref_proceedings_drop_posters_by_page_count():
    source = DblpAcmProceedingsSource("CCS", "ccs", crossref_container_titles={2024: "x"})
    papers = source.papers_from_crossref(2024, fixture_json("crossref_ccs2024.json")["message"]["items"])
    assert [p.paper_id for p in papers] == [
        "ccs-2024-3658644.3670337", "ccs-2024-3658644.3690254", "ccs-2024-3658644.3690261",
    ]
    assert papers[0].pdf_url == "https://dl.acm.org/doi/pdf/10.1145/3658644.3670337"
    assert papers[0].doi == "10.1145/3658644.3670337" and papers[0].authors


def test_crossref_journal_keeps_its_volume_and_drops_editorials_and_online_first():
    source = CrossrefJournalSource("TPAMI", "0162-8828", "tpami", 1978)
    papers = source.papers_from_crossref(2024, fixture_json("crossref_tpami.json")["message"]["items"])
    assert [p.paper_id for p in papers] == ["tpami-2024-3296600", "tpami-2024-3329725", "tpami-2024-3328397"]
    assert all(p.pdf_url == "" and p.pdf_resolution_status == "needs_open_access" for p in papers)
    assert papers[0].title.startswith("OPAL: Occlusion Pattern Aware Loss")
    ijcv = CrossrefJournalSource("IJCV", "0920-5691", "ijcv", 1892, doi_id="suffix")
    assert ijcv._id_of("10.1007/s11263-024-02023-9") == "s11263-024-02023-9"


def _doi_paper(paper_id: str, title: str, doi: str) -> CorpusPaper:
    return CorpusPaper(paper_id=paper_id, title=title, authors=[], conference="TPAMI", year=2024,
                       pdf_url="", paper_page_url=f"https://doi.org/{doi}", source_name="crossref", doi=doi)


def test_open_access_records_prefer_publisher_host_then_arxiv_and_check_identity():
    data = fixture_json("s2_batch.json")
    titles = [record["title"] if record else "" for record in data["response"]]
    papers = [_doi_paper(f"p{i}", title, doi) for i, (title, doi) in enumerate(zip(titles, data["dois"]))]
    papers[2].title = "A Different Paper Entirely"  # same DOI, other title -> refused
    stats = {key: 0 for key in ("matched", "resolved", "arxiv", "publisher_oa", "other_oa",
                                "not_found", "identity_mismatch")}
    apply_open_access_records(
        [(doi, paper) for doi, paper in zip(data["dois"], papers)], data["response"], stats,
        preferred_hosts=frozenset({"link.springer.com"}), excluded_hosts=frozenset({"doi.org"}),
    )
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2203.02231" and papers[0].arxiv_id == "2203.02231"
    assert papers[0].pdf_resolution_status == "resolved_open_access"
    assert papers[2].pdf_url == ""
    assert papers[3].pdf_url == "https://link.springer.com/content/pdf/10.1007/s11263-025-02489-1.pdf"
    assert papers[3].alternate_pdf_urls == []  # the publisher's PDF gets no preprint fallback
    assert papers[0].alternate_pdf_urls == []  # S2's OA url is the same arXiv copy
    assert papers[4].pdf_url == ""  # closed article without an open copy
    assert stats["identity_mismatch"] == 1 and stats["publisher_oa"] == 1 and stats["arxiv"] == 2


def test_registry_builds_every_tier1_source():
    sources = build_sources()
    assert isinstance(sources["ECCV"], EcvaProceedingsSource)
    assert isinstance(sources["TPAMI"], CrossrefJournalSource) and sources["TPAMI"].issn == "0162-8828"
    assert sources["IJCV"].open_access_hosts == frozenset({"link.springer.com"})
    assert isinstance(sources["IEEE S&P"], IeeeCsdlProceedingsSource)
    assert sources["IEEE S&P"].proceeding_ids[2026] == "2bojuokAJK8"
    assert isinstance(sources["VLDB"], PvldbVolumeSource)
    assert sources["CCS"].crossref_container_titles[2024].startswith("Proceedings of the 2024 on ACM SIGSAC")
    assert sources["SOSP"].supports_year(2023) and sorted(sources["SOSP"].crossref_container_titles) == [2023, 2024, 2025, 2026]
    assert sources["OSDI"].url_slug == "osdi"


# ------------------------------------------------------------------ live network
@pytest.fixture(scope="module")
def client():
    c = OfficialProceedingsClient()
    yield c
    c.close()


@pytest.mark.network
@pytest.mark.parametrize("venue,year,expected_min", [
    ("JMLR", 2024, 400), ("VLDB", 2025, 380), ("OSDI", 2026, 120), ("SOSP", 2024, 40), ("IEEE S&P", 2024, 255),
])
def test_live_listing(client, venue, year, expected_min):
    papers = build_sources()[venue].collect_papers(year, client)
    assert len(papers) >= expected_min
    assert len({p.paper_id for p in papers}) == len(papers)
    assert all(p.title and p.conference == venue and p.year == year for p in papers)


def test_collect_targets_and_slice_merge():
    from scripts.collect.collect_2026_to_manifest import merge_slices, parse_targets

    assert parse_targets(["IEEE S&P:2024", "JMLR:2024", "IEEE S&P:2024"], "ECCV", 2026) == [
        ("ECCV", 2026), ("IEEE S&P", 2024), ("JMLR", 2024)]
    for bad in ("IEEE S&P", "JMLR:24x", ":2024"):
        with pytest.raises(SystemExit):
            parse_targets([bad], None, None)

    def paper(paper_id, venue="JMLR", year=2024):
        return CorpusPaper(paper_id=paper_id, title=paper_id, authors=["A"], conference=venue, year=year,
                           pdf_url=f"https://example.org/{paper_id}.pdf", paper_page_url="", source_name="t")

    existing = [paper("jmlr-2024-stale").to_manifest_record(), paper("jmlr-2024-kept").to_manifest_record(),
                paper("vldb-2024-x", "VLDB").to_manifest_record()]
    collected = {("JMLR", 2024): [paper("jmlr-2024-kept"), paper("jmlr-2024-new")]}
    merged, counts = merge_slices(existing, collected, replace_slices=True)
    assert sorted(r["paper_id"] for r in merged) == ["jmlr-2024-kept", "jmlr-2024-new", "vldb-2024-x"]
    assert counts == {"removed_stale": 2, "added": 2, "updated": 0}
    merged, counts = merge_slices(existing, collected, replace_slices=False)  # a --limit smoke keeps rows
    assert len(merged) == 4 and counts == {"removed_stale": 0, "added": 1, "updated": 1}


@pytest.mark.network
def test_live_open_access_only_slice_keeps_only_open_full_text(client):
    papers, report = corpus.collect_slice(build_sources()["IEEE S&P"], 2025, client)
    assert report["collected"] >= 250 and report["without_open_full_text"] >= 150
    assert report["kept"] == len(papers) == report["collected"] - report["without_open_full_text"]
    assert all(p.pdf_url and p.venue_tier == "CCF-A" for p in papers)


@pytest.mark.network
def test_live_batch_that_semantic_scholar_knows_none_of_is_not_found(client):
    # S2 answers HTTP 400 "No valid paper ids given" for such a batch (IEEE S&P 2023 hit it).
    paper = _doi_paper("sp-2023-00007", "Toss a Fault to Your Witcher", "10.1109/SP46215.2023.00007")
    paper.pdf_url = ""
    stats = corpus.resolve_open_access_pdf_urls([paper], client)
    assert stats["requested"] == 1 and stats["not_found"] == 1 and paper.pdf_url == ""
