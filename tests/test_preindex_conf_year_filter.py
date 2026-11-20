"""TDD: --conf/--year slice filter for preindex_md_to_sql_only.py.

The filter must be ADDITIVE: None/None preserves the existing no-filter path.
conf match is case-insensitive on the manifest 'conference' field. Generic for
ANY conference.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load the script module directly (it is a script, not a package member).
_SPEC = importlib.util.spec_from_file_location(
    "preindex_md_to_sql_only", REPO_ROOT / "scripts" / "preindex_md_to_sql_only.py"
)
preindex = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(preindex)  # type: ignore

from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper
from app.models.schemas import Chunk


def _rec(conf, year, pid):
    return CorpusPaper(
        paper_id=pid, title="t", authors=[], conference=conf, year=year,
        pdf_url="u", paper_page_url="p", source_name="s",
    )


@pytest.fixture
def records():
    return [
        _rec("CVPR", 2026, "cvpr-2026-a"),
        _rec("CVPR", 2026, "cvpr-2026-b"),
        _rec("CVPR", 2025, "cvpr-2025-x"),
        _rec("AAAI", 2026, "aaai-2026-y"),
    ]


def test_no_filter_preserves_all(records):
    assert preindex._filter_by_conf_year(records, None, None) == records


def test_conf_filter_case_insensitive(records):
    out = preindex._filter_by_conf_year(records, "cvpr", None)
    assert {r.paper_id for r in out} == {"cvpr-2026-a", "cvpr-2026-b", "cvpr-2025-x"}


def test_year_filter(records):
    out = preindex._filter_by_conf_year(records, None, 2026)
    assert {r.paper_id for r in out} == {"cvpr-2026-a", "cvpr-2026-b", "aaai-2026-y"}


def test_conf_and_year_narrows_to_slice(records):
    out = preindex._filter_by_conf_year(records, "CVPR", 2026)
    assert {r.paper_id for r in out} == {"cvpr-2026-a", "cvpr-2026-b"}


def test_argparser_accepts_conf_year():
    args = preindex.build_argparser().parse_args(
        ["--conf", "AAAI", "--year", "2026", "--chunker-type", "section_l1"]
    )
    assert args.conf == "AAAI"
    assert args.year == 2026


def test_argparser_defaults_none():
    args = preindex.build_argparser().parse_args([])
    assert args.conf is None
    assert args.year is None


def test_argparser_defaults_to_l1_l2_paper_schema():
    args = preindex.build_argparser().parse_args([])
    assert args.chunker_type == "section_l1"


def test_argparser_rejects_l2_only_markdown_chunker():
    with pytest.raises(SystemExit):
        preindex.build_argparser().parse_args(["--chunker-type", "markdown"])


def test_paper_chunk_schema_requires_exactly_one_l1():
    l2 = Chunk(
        id="l2",
        doc_id="1",
        content="body",
        level=2,
        start_idx=0,
        end_idx=4,
        metadata={},
    )
    with pytest.raises(RuntimeError, match="exactly one level-1"):
        preindex._validate_paper_chunk_schema([l2])
