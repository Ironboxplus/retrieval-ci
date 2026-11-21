"""Test CorpusPaper.to_runtime_paper conference mapping is registry-driven.

Regression for KeyError 'WWW' (and other new venues) when conference_map was
hardcoded and missed entries that the registry/scrapers had already added.
"""
from __future__ import annotations

import pytest

from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper
from pipeline.paper_crawler.models.paper import Conference


def _make_record(conf: str, paper_id: str = "test-2022-1.1") -> CorpusPaper:
    return CorpusPaper(
        paper_id=paper_id,
        conference=conf,
        year=2022,
        title="t",
        authors=["a"],
        abstract=None,
        pdf_url="https://x",
        paper_page_url=None,
        downloaded_from_url=None,
        pdf_path=None,
        markdown_path=None,
        source_name="test",
    )


@pytest.mark.parametrize(
    "conf,expected_enum",
    [
        ("AAAI", Conference.AAAI),
        ("ACL", Conference.ACL),
        ("CVPR", Conference.CVPR),
        ("EMNLP", Conference.EMNLP),
        ("ICCV", Conference.ICCV),
        ("ICLR", Conference.ICLR),
        ("ICML", Conference.ICML),
        ("IJCAI", Conference.IJCAI),
        ("KDD", Conference.KDD),
        ("NeurIPS", Conference.NEURIPS),
        ("SIGIR", Conference.SIGIR),
        ("SIGMOD", Conference.SIGMOD),
        ("WWW", Conference.WWW),
    ],
)
def test_known_conferences_map_to_enum(conf, expected_enum):
    """Each conference present in the registry must map to its enum member."""
    r = _make_record(conf)
    p = r.to_runtime_paper()
    assert p.conference == expected_enum


@pytest.mark.parametrize(
    "new_venue",
    ["CSUR", "USENIX Security", "CCS", "ACM MM", "VLDB"],
)
def test_new_venues_do_not_raise(new_venue):
    """Venues newly added to the registry must not KeyError; OTHER is acceptable."""
    r = _make_record(new_venue)
    p = r.to_runtime_paper()
    assert p.conference is not None


def test_unknown_conference_falls_back_to_other():
    """Truly unknown conference names should fall back, not raise."""
    r = _make_record("MadeUpVenue9999")
    p = r.to_runtime_paper()
    assert p.conference == Conference.OTHER
