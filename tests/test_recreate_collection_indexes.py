"""TDD: `recreate_qdrant_collection.py`'s PAYLOAD_INDEXES must be DERIVED
from the papers_payload projector, not hand-typed (plan.md 3.2) — a second,
independently-maintained key list is exactly how the script grew a stray
9th `data_source` index in production (docs/papers_payload_contract_plan.md
1.3/1.4): it duplicated local_ingest.PAYLOAD_KEEP_KEYS instead of importing
it, and the two drifted.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from qdrant_client.models import PayloadSchemaType

from core.storage.papers_payload import (
    COMMON_OPTIONAL_STRING_KEYS,
    OPTIONAL_PAPER_KEYS,
    PAPER_PROFILE_KEYS,
)
from recreate_qdrant_collection import PAYLOAD_INDEXES


def test_index_fields_are_exactly_the_paper_profile_minus_content_plus_optional_keys():
    fields = {field for field, _ in PAYLOAD_INDEXES}
    assert fields == ((PAPER_PROFILE_KEYS - {"content"}) | OPTIONAL_PAPER_KEYS
                      | COMMON_OPTIONAL_STRING_KEYS)


def test_no_duplicate_fields():
    fields = [field for field, _ in PAYLOAD_INDEXES]
    assert len(fields) == len(set(fields))


def test_data_source_has_no_index():
    """D3 conclusion: no Qdrant search path filters by data_source (grep of
    core/retrieval + main.py found zero call sites passing a real value into
    QdrantStore.search/hybrid_search/sparse_search's data_source kwarg). An
    index over 13.5M points for a predicate nothing uses is dead weight —
    exactly the stray index the 2026-09-01 audit found covering only 0.04%
    of the collection. `data_source` stays a legitimate upload/webpage
    payload KEY (papers_payload.UPLOAD_PROFILE_KEYS /
    WEBPAGE_PROFILE_KEYS); it is just not indexed."""
    fields = {field for field, _ in PAYLOAD_INDEXES}
    assert "data_source" not in fields


def test_content_is_never_indexed():
    fields = {field for field, _ in PAYLOAD_INDEXES}
    assert "content" not in fields


def test_year_and_level_are_integer_schema():
    by_field = dict(PAYLOAD_INDEXES)
    assert by_field["year"] == PayloadSchemaType.INTEGER
    assert by_field["level"] == PayloadSchemaType.INTEGER


def test_keyword_fields_are_keyword_schema():
    by_field = dict(PAYLOAD_INDEXES)
    for field in ("doc_id", "paper_id", "source_type", "venue", "venue_tier"):
        assert by_field[field] == PayloadSchemaType.KEYWORD


def test_withdrawn_is_a_bool_index():
    """The optional `withdrawn` key is written as a JSON true, which only a `bool`
    index covers (a keyword index would never see it)."""
    assert dict(PAYLOAD_INDEXES)["withdrawn"] == PayloadSchemaType.BOOL


def test_embed_fp_is_a_keyword_index():
    assert dict(PAYLOAD_INDEXES)["embed_fp"] == PayloadSchemaType.KEYWORD


def test_ten_indexes_total():
    """AGENTS.md 2.1: 8-field paper whitelist, 7 of them indexed (content isn't),
    plus the optional `withdrawn` and `preprint_full_text` flags and the
    `embed_fp` profile fingerprint."""
    assert len(PAYLOAD_INDEXES) == 10
