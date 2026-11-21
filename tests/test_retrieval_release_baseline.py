"""Release API baseline comparison never persists candidate text."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def baseline_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "capture_retrieval_release_baseline.py"
    spec = importlib.util.spec_from_file_location("retrieval_release_baseline_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_baseline_target_is_limited_to_private_88_compute(baseline_module):
    assert baseline_module.validate_release_target("http://10.100.100.88:18080") == (
        "http://10.100.100.88:18080"
    )
    with pytest.raises(ValueError):
        baseline_module.validate_release_target("http://example.invalid:18080")


def test_canonical_result_keeps_identity_score_and_content_hash_not_content(baseline_module):
    case = baseline_module.ApiCase("case", "/api/example", {}, "hits")
    result = baseline_module.canonical_case_result(case, {
        "hits": [{
            "doc_id": "doc-1", "chunk_id": "chunk-1", "score": 0.75,
            "code": "private source text must not be written to the baseline",
        }],
    })

    stored = result["hits"][0]
    assert stored["identity"]["doc_id"] == "doc-1"
    assert stored["score"] == 0.75
    assert "content_sha256" in stored["identity"]
    assert "private source text" not in repr(result)


def test_compare_requires_exact_order_identity_and_score_equivalence(baseline_module):
    expected = {"schema_version": 1, "cases": [
        {"name": "case", "result_count": 1, "hits": [{"identity": {"id": "a"}, "score": 1.0}]},
    ]}
    assert baseline_module.compare_baseline(expected, expected) == []
    actual = {"schema_version": 1, "cases": [
        {"name": "case", "result_count": 1, "hits": [{"identity": {"id": "a"}, "score": 0.9}]},
    ]}
    assert baseline_module.compare_baseline(expected, actual) == [
        "case: final result identities, order, or scores changed",
    ]
