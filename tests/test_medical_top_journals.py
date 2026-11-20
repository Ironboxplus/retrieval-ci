"""TDD: medical top-journal whitelist gate."""

from __future__ import annotations

from pipeline.medical_corpus.models import MedicalPaperRecord, MedicalSourceKind
from pipeline.medical_corpus.top_journals import (
    TOP_MEDICAL_JOURNALS,
    default_top_journal_query,
    filter_top_journal_records,
    is_top_medical_journal,
    match_top_journal,
)
from pipeline.medical_corpus.prepare import PrepareConfig


def test_whitelist_includes_flagships():
    names = {n.casefold() for n in TOP_MEDICAL_JOURNALS}
    assert any("new england" in n for n in names)
    assert any("lancet" in n for n in names)
    assert any(n == "jama" for n in names) or any("jama" in n for n in names)
    assert any("nature medicine" in n for n in names)


def test_match_aliases():
    assert match_top_journal("N Engl J Med") == "The New England Journal of Medicine"
    assert match_top_journal("The Lancet") == "The Lancet"
    assert match_top_journal("Lancet") == "The Lancet"
    assert match_top_journal("JAMA") == "JAMA"
    assert match_top_journal("BMJ (Clinical research ed.)") == "BMJ"
    assert match_top_journal("Nature Medicine") == "Nature Medicine"
    assert match_top_journal("Some Random Mid-Tier Journal of Nursing") is None
    assert not is_top_medical_journal("Iranian Endodontic Journal")


def test_filter_drops_non_top():
    rows = [
        MedicalPaperRecord(
            paper_id="med-pmc1", title="a", journal="The Lancet",
            source_kind=MedicalSourceKind.EUROPE_PMC,
        ),
        MedicalPaperRecord(
            paper_id="med-pmc2", title="b", journal="Iranian Endodontic Journal",
            source_kind=MedicalSourceKind.EUROPE_PMC,
        ),
        MedicalPaperRecord(
            paper_id="med-pmc3", title="c", journal=None,
            source_kind=MedicalSourceKind.EUROPE_PMC,
        ),
        MedicalPaperRecord(
            paper_id="med-pmc4", title="d", journal="N Engl J Med",
            source_kind=MedicalSourceKind.EUROPE_PMC,
        ),
    ]
    kept, stats = filter_top_journal_records(rows)
    assert {r.paper_id for r in kept} == {"med-pmc1", "med-pmc4"}
    assert stats["dropped_not_top"] == 1
    assert stats["dropped_no_journal"] == 1
    assert kept[0].journal == "The Lancet"
    assert kept[1].journal == "The New England Journal of Medicine"


def test_default_query_includes_journal_clause_and_oa():
    q = default_top_journal_query()
    assert "OPEN_ACCESS:Y" in q
    assert "HAS_PDF:Y" in q
    assert "SRC:MED" in q
    assert 'JOURNAL:"' in q
    assert "PUB_YEAR:" in q


def test_prepare_config_defaults_top_journals_only():
    cfg = PrepareConfig()
    assert cfg.top_journals_only is True
    assert "JOURNAL:" in cfg.query or "Lancet" in cfg.query or "OPEN_ACCESS" in cfg.query
