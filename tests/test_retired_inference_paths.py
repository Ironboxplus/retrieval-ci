"""Every in-process model path is retired: constructing or calling it raises RetiredInferencePath.

Models run as vLLM services now; the old torch / FlagEmbedding / sentence-transformers classes and
the scripts that load a model into their own process stay in the tree but fail the moment they are
used, naming what to use instead. Nothing here loads a model: each call must raise before it would.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_the_exception_names_what_is_retired_and_what_replaces_it():
    from core.inference_retired import EMBEDDER_INSTEAD, RERANKER_INSTEAD, RetiredInferencePath

    error = RetiredInferencePath("Thing", EMBEDDER_INSTEAD)
    assert isinstance(error, RuntimeError)
    assert "Thing is retired" in str(error) and "get_embedder()" in str(error)
    assert error.what == "Thing" and error.use_instead == EMBEDDER_INSTEAD
    assert "get_shared_reranker()" in RERANKER_INSTEAD


def test_bge_m3_embedder_raises_when_constructed():
    from core.indexing.embedders.bge_m3 import BgeM3Embedder
    from core.inference_retired import RetiredInferencePath

    with pytest.raises(RetiredInferencePath, match="BgeM3Embedder"):
        BgeM3Embedder(device="cpu")
    with pytest.raises(RetiredInferencePath):
        BgeM3Embedder()


def test_sentence_transformer_embedder_raises_when_constructed():
    from core.indexing.embedders.sentence_transformer import SentenceTransformerEmbedder
    from core.inference_retired import RetiredInferencePath

    with pytest.raises(RetiredInferencePath, match="SentenceTransformerEmbedder"):
        SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def test_bge_m3_core_load_model_raises_before_importing_flagembedding(monkeypatch):
    from core.indexing.embedders import bge_m3_core
    from core.indexing.embedding_profile import get_profile
    from core.inference_retired import RetiredInferencePath

    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)       # an import would raise ImportError instead
    with pytest.raises(RetiredInferencePath, match="load_model"):
        bge_m3_core.load_model(get_profile(), "BAAI/bge-m3", "cpu")


def test_torch_bge_reranker_raises_when_constructed():
    from core.inference_retired import RetiredInferencePath
    from core.retrieval.reranker import BgeReranker

    with pytest.raises(RetiredInferencePath, match="BgeReranker"):
        BgeReranker(device="cpu")
    with pytest.raises(RetiredInferencePath):
        BgeReranker()


@pytest.mark.parametrize("module, needle", [
    ("scripts.verify_reranker_cuda_canary", "verify_reranker_cuda_canary"),
    ("scripts.smoke.rerank_cancel_equivalence_real_gate", "rerank_cancel_equivalence_real_gate"),
    ("scripts.eval.code_embed_eval", "code_embed_eval"),
])
def test_scripts_raise_in_main_before_loading_anything(module, needle, monkeypatch):
    from core.inference_retired import RetiredInferencePath

    for heavy in ("torch", "FlagEmbedding", "sentence_transformers"):
        monkeypatch.setitem(sys.modules, heavy, None)               # loading a model would fail differently
    main = importlib.import_module(module).main

    with pytest.raises(RetiredInferencePath, match=needle):
        main()


@pytest.mark.parametrize("script", [
    "scripts/verify_reranker_cuda_canary.py",
    "scripts/smoke/rerank_cancel_equivalence_real_gate.py",
    "scripts/eval/code_embed_eval.py",
])
def test_running_a_retired_script_exits_with_the_exception(script):
    done = subprocess.run([sys.executable, str(REPO / script)], cwd=REPO, capture_output=True, text=True,
                          timeout=180, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert done.returncode != 0
    assert "RetiredInferencePath" in done.stderr and "is retired" in done.stderr


def test_the_default_factories_no_longer_build_a_model_in_process():
    """create_embedder / create_reranker return the service clients, built without contacting a service."""
    from core.indexing.embedders import create_embedder
    from core.retrieval.reranker import VllmReranker, create_reranker

    assert isinstance(create_reranker(), VllmReranker)
    assert create_embedder().__class__.__name__ == "VllmBgeM3Embedder"
