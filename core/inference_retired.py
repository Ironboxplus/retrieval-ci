"""One exception for every in-process model path that has been retired.

Every model runs as a vLLM service now (``EMBEDDING__SERVICE_URL``, ``RERANKER__SERVICE_URL``).
The old torch / FlagEmbedding / sentence-transformers classes and scripts stay in the tree, but
constructing or calling them raises this, naming what to use instead, so nothing quietly loads a
second copy of a model into the backend again.

Kept dependency-free: the standalone embed worker and the retired scripts import it without the
project's settings layer, torch or FlagEmbedding.
"""
from __future__ import annotations


class RetiredInferencePath(RuntimeError):
    """An in-process inference path that was retired in favour of a vLLM service."""

    def __init__(self, what: str, use_instead: str):
        super().__init__(
            f"{what} is retired: models run as vLLM services now. Use {use_instead}.")
        self.what, self.use_instead = what, use_instead


EMBEDDER_INSTEAD = (
    "core.indexing.embedders.get_embedder() (a client of the vLLM embedding service at "
    "EMBEDDING__SERVICE_URL)")
RERANKER_INSTEAD = (
    "core.retrieval.reranker.get_shared_reranker() (a client of the vLLM reranking service at "
    "RERANKER__SERVICE_URL)")
EXTRACTOR_INSTEAD = (
    "a registered extractor worker: set PIPELINE_LOCAL_EXTRACTOR to a registry block such as "
    "deploy/compute88/local_extractor.json (the PaddleOCR-VL worker, scripts/remote_pvl/pvl_worker.py)")
