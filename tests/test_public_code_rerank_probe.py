"""Smoke the local full-pool Public Code rerank performance probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "probe_public_code_rerank_candidates.py"
    spec = importlib.util.spec_from_file_location("public_code_rerank_probe_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_pool_probe_reports_the_top_k_five_retrieval_pool_without_a_model_load():
    probe = _probe_module()

    result = probe.run_probe(candidate_count=20, content_characters=2_048)

    assert result["retrieval_candidate_count"] == 20
    assert result["rerank_candidate_count"] == 20
    assert result["scored_candidate_count"] == 20
    assert result["returned_hit_count"] == 5
    assert result["scored_content_characters"] >= 20 * 2_048
    assert result["elapsed_ms"] >= 0
