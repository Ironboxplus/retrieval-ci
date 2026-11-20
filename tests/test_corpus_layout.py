"""Write routing and the pre-write guard for the papers / arxiv / reviews split."""
from __future__ import annotations

import pytest

from core.storage.corpus_layout import (
    CorpusLayout,
    MisroutedPaperPoint,
    check_paper_point,
    collection_for_payload,
)

FORMAL = {"paper_id": "iclr-2025-1", "venue_tier": "CCF-A", "source_type": "paper"}
PREPRINT = {"paper_id": "arxiv-2609.08019", "venue_tier": "preprint", "source_type": "paper"}
REVIEW = {"paper_id": "iclr-2025-1", "venue_tier": "CCF-A", "source_type": "review"}
UPLOAD = {"doc_id": "77", "source_type": "upload"}
MIXED, SPLIT = CorpusLayout(split=False), CorpusLayout(split=True)


def test_mixed_layout_writes_everything_to_papers():
    for payload in (FORMAL, PREPRINT, REVIEW, UPLOAD):
        assert collection_for_payload(payload, MIXED) == "papers"


def test_split_layout_routes_by_payload():
    assert collection_for_payload(FORMAL, SPLIT) == "papers"
    assert collection_for_payload(UPLOAD, SPLIT) == "papers"
    assert collection_for_payload(PREPRINT, SPLIT) == "arxiv"
    assert collection_for_payload(REVIEW, SPLIT) == "reviews"


def test_guard_accepts_correct_targets():
    check_paper_point("papers", PREPRINT, MIXED)  # the legacy layout still writes here
    check_paper_point("papers", FORMAL, SPLIT)
    check_paper_point("papers", UPLOAD, SPLIT)
    check_paper_point("arxiv", PREPRINT, MIXED)  # migration copies before the switch
    check_paper_point("reviews", REVIEW, SPLIT)
    check_paper_point("code_chunks", PREPRINT, SPLIT)  # not a paper collection


@pytest.mark.parametrize("collection,payload,layout", [
    ("papers", PREPRINT, SPLIT),
    ("papers", REVIEW, SPLIT),
    ("arxiv", FORMAL, MIXED),
    ("arxiv", REVIEW, SPLIT),
    ("reviews", FORMAL, SPLIT),
    ("reviews", PREPRINT, MIXED),
])
def test_guard_refuses_misrouted_points_before_the_write(collection, payload, layout):
    with pytest.raises(MisroutedPaperPoint):
        check_paper_point(collection, payload, layout)
