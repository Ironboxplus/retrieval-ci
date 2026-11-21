"""TDD: `papers` Qdrant payload projector — the single executable schema for
the `papers` collection (docs/papers_payload_contract_plan.md 3.1/3.4).

`papers` is a heterogeneous container of four document profiles measured in
production 2026-09-01 (plan.md 1.1): paper/review share an 8-field whitelist;
upload/webpage have no paper_id and carry real business fields the paper
whitelist would incorrectly drop (D1). This module is the ONLY place that
decides what payload keys reach Qdrant for this collection; every write path
must route through `project_papers_payload`.
"""
from __future__ import annotations

import logging

import pytest

from core.storage.papers_payload import (
    PAPER_PROFILE_KEYS,
    UPLOAD_PROFILE_KEYS,
    WEBPAGE_PROFILE_KEYS,
    PROFILE_BY_SOURCE_TYPE,
    UnknownSourceTypeError,
    project_papers_payload,
)


class TestProfileKeySets:
    """Key sets are measured production facts (plan.md 1.1), not guesses."""

    def test_paper_profile_is_the_8_field_whitelist(self):
        assert PAPER_PROFILE_KEYS == frozenset({
            "doc_id", "paper_id", "source_type", "level",
            "venue", "year", "venue_tier", "content",
        })

    def test_upload_profile_matches_measured_production_keys(self):
        assert UPLOAD_PROFILE_KEYS == frozenset({
            "chunk_id", "chunk_size", "content", "data_source", "doc_id",
            "end_idx", "file_size", "level", "original_filename", "source",
            "source_type", "start_idx", "title",
        })

    def test_webpage_profile_matches_measured_production_keys(self):
        assert WEBPAGE_PROFILE_KEYS == frozenset({
            "chunk_id", "content", "crawl_time", "crawled_at", "data_source",
            "doc_id", "end_idx", "level", "proxy_used", "retry_count",
            "section_name", "source", "source_type", "start_idx",
            "status_code", "title",
        })

    def test_upload_and_webpage_have_no_paper_id(self):
        """D1: they aren't papers — a paper_id would be meaningless identity."""
        assert "paper_id" not in UPLOAD_PROFILE_KEYS
        assert "paper_id" not in WEBPAGE_PROFILE_KEYS

    def test_review_shares_the_paper_profile_object(self):
        assert PROFILE_BY_SOURCE_TYPE["review"] is PROFILE_BY_SOURCE_TYPE["paper"]


class TestPaperProfile:
    def test_projects_to_whitelist_and_drops_extras(self):
        raw = {
            "source_type": "paper", "doc_id": 5, "paper_id": "x", "level": 2,
            "venue": "CVPR", "year": 2026, "venue_tier": "CCF-A",
            "content": "hi", "data_source": "system", "chunk_id": "c1",
        }
        out = project_papers_payload(raw)
        assert out == {
            "source_type": "paper", "doc_id": "5", "paper_id": "x", "level": 2,
            "venue": "CVPR", "year": 2026, "venue_tier": "CCF-A", "content": "hi",
        }

    def test_review_shares_paper_whitelist(self):
        raw = {"source_type": "review", "doc_id": 9, "paper_id": "y", "level": 1,
               "content": "review text", "data_source": "system"}
        out = project_papers_payload(raw)
        assert "data_source" not in out
        assert out["doc_id"] == "9"


class TestDocIdNormalization:
    """D2: Qdrant KEYWORD index silently drops int-typed doc_id (plan.md 1.2)."""

    def test_int_doc_id_becomes_str(self):
        out = project_papers_payload({"source_type": "paper", "doc_id": 20286})
        assert out["doc_id"] == "20286"
        assert isinstance(out["doc_id"], str)

    def test_str_doc_id_stays_str(self):
        out = project_papers_payload({"source_type": "paper", "doc_id": "20286"})
        assert out["doc_id"] == "20286"

    def test_none_doc_id_stays_absent_not_stringified(self):
        out = project_papers_payload({"source_type": "paper", "doc_id": None})
        assert "doc_id" not in out

    def test_missing_doc_id_stays_absent(self):
        out = project_papers_payload({"source_type": "paper", "level": 1})
        assert "doc_id" not in out


class TestNoneValuesDropped:
    def test_none_valued_profile_keys_are_dropped(self):
        """Missing SQL data must never overwrite real data with a null (matches
        the pre-projector callers' `if v is not None` pattern this replaces)."""
        out = project_papers_payload({"source_type": "paper", "venue": None, "year": 2026})
        assert "venue" not in out
        assert out["year"] == 2026


class TestUnknownSourceTypeFailsClosed:
    def test_unknown_source_type_raises(self):
        with pytest.raises(UnknownSourceTypeError):
            project_papers_payload({"source_type": "citation_graph", "doc_id": 1})

    def test_missing_source_type_raises(self):
        with pytest.raises(UnknownSourceTypeError):
            project_papers_payload({"doc_id": 1})

    def test_does_not_pass_through_arbitrary_payload(self):
        """Even a papers-shaped payload must fail closed without a registered
        profile — no accidental 5th document profile (D1: 'this period we do
        not add a 5th profile')."""
        with pytest.raises(UnknownSourceTypeError):
            project_papers_payload({"source_type": "citation", "content": "x", "doc_id": 1})


class TestUploadProfile:
    def test_keeps_business_fields_paper_whitelist_would_drop(self):
        raw = {
            "source_type": "upload", "doc_id": "3", "content": "c",
            "chunk_id": "u1", "chunk_size": 100, "data_source": "user",
            "end_idx": 10, "file_size": 999, "level": 2,
            "original_filename": "report.pdf", "source": "upload",
            "start_idx": 0, "title": "Report",
        }
        out = project_papers_payload(raw)
        assert out == raw

    def test_paper_id_dropped_if_present(self):
        out = project_papers_payload({"source_type": "upload", "doc_id": "1",
                                       "paper_id": "should-not-exist"})
        assert "paper_id" not in out


class TestWebpageProfile:
    def test_keeps_webpage_only_fields(self):
        raw = {
            "source_type": "webpage", "doc_id": "9", "content": "c",
            "chunk_id": "w1", "crawl_time": "t", "crawled_at": "t2",
            "data_source": "user", "end_idx": 5, "level": 1,
            "proxy_used": True, "retry_count": 0, "section_name": "intro",
            "source": "http://x", "start_idx": 0, "status_code": 200,
            "title": "Page",
        }
        out = project_papers_payload(raw)
        assert out == raw


class TestDropWarning:
    """Dropping out-of-profile keys must be observable, so a new leak is
    caught instead of silently matching old behavior (plan.md 3.1)."""

    def test_out_of_profile_keys_log_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="core.storage.papers_payload"):
            project_papers_payload({"source_type": "paper", "doc_id": 1, "author_names": "A|B"})
        assert any("author_names" in r.message for r in caplog.records)

    def test_clean_payload_logs_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger="core.storage.papers_payload"):
            project_papers_payload({"source_type": "paper", "doc_id": 1})
        assert caplog.records == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
