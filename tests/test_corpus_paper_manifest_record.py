"""Regression: a manifest written by a NEWER build must not crash an older reader.

2026-09-14 production failure (Tasks task #325, SIGIR 2026 `ingest_pipeline`):

    TypeError: CorpusPaper.__init__() got an unexpected keyword argument 'venue_track'
      ccf_a_main_corpus.py from_manifest_record  ->  return cls(**filtered_record)
      scripts/preindex_md_to_sql_only.py:184

The KDD 2026 collection added `venue_track` to every manifest record. `.88` still ran
a build whose `CorpusPaper` had no such field, and `from_manifest_record` expanded the
record wholesale — it only ever popped the single known extra key `selection_rule`.
The whole preindex stage died 53 seconds in.

The manifest is shared state between hosts that upgrade at different times, so a
record carrying a field this build does not know is a NORMAL condition, not a bug.
Drop unknown keys and say so in the log; never take the process down. A typo'd field
name is the cost, which is why the dropped keys must be named in the warning.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper


def _minimal_record(**extra) -> dict:
    record = {
        "paper_id": "kdd-2026-3770854.3780155",
        "title": "A Paper",
        "authors": ["A. Author"],
        "conference": "KDD",
        "year": 2026,
        "pdf_url": "https://dl.acm.org/doi/pdf/10.1145/3770854.3780155",
        "paper_page_url": "https://doi.org/10.1145/3770854.3780155",
        "source_name": "official",
    }
    record.update(extra)
    return record


def test_known_fields_still_round_trip():
    paper = CorpusPaper.from_manifest_record(
        _minimal_record(venue_track="research", markdown_path="data/papers/markdown/kdd/2026/x.md")
    )
    assert paper.paper_id == "kdd-2026-3770854.3780155"
    assert paper.venue_track == "research"
    assert paper.markdown_path == "data/papers/markdown/kdd/2026/x.md"
    assert paper.to_manifest_record()["venue_track"] == "research"


def test_unknown_manifest_field_is_dropped_instead_of_raising(caplog):
    """The exact shape of the task #325 failure, one version earlier."""
    record = _minimal_record(
        venue_track="research",
        some_field_a_newer_build_added="value",
    )

    with caplog.at_level(logging.WARNING):
        paper = CorpusPaper.from_manifest_record(record)

    assert paper.paper_id == "kdd-2026-3770854.3780155"
    assert not hasattr(paper, "some_field_a_newer_build_added")
    # Named in the log: a silently swallowed typo would be worse than the crash.
    assert "some_field_a_newer_build_added" in caplog.text


def test_selection_rule_stays_silent():
    """A long-known extra key must not start emitting warnings on every record."""
    import logging as _logging

    records = [_minimal_record(selection_rule="main-conference") for _ in range(3)]
    logger = _logging.getLogger("pipeline.paper_crawler.ccf_a_main_corpus")
    seen: list[_logging.LogRecord] = []
    handler = _logging.Handler()
    handler.emit = seen.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        for record in records:
            CorpusPaper.from_manifest_record(record)
    finally:
        logger.removeHandler(handler)

    assert [r for r in seen if r.levelno >= _logging.WARNING] == []
