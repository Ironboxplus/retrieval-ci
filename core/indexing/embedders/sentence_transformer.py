"""SentenceTransformer embedder — dense-only, backward compatible."""
from __future__ import annotations

from core.indexing.embedders import (
    BaseEmbedder,
)
from core.inference_retired import EMBEDDER_INSTEAD, RetiredInferencePath


class SentenceTransformerEmbedder(BaseEmbedder):
    """RETIRED: a sentence-transformers model loaded into this process.

    Kept for reference only; constructing it raises. The only embedding model the backend
    serves is BGE-M3, through the vLLM service (``core.indexing.embedders.get_embedder()``).
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        raise RetiredInferencePath("SentenceTransformerEmbedder (in-process sentence-transformers)",
                                   EMBEDDER_INSTEAD)

    @property
    def dimension(self) -> int:
        raise RetiredInferencePath("SentenceTransformerEmbedder (in-process sentence-transformers)",
                                   EMBEDDER_INSTEAD)

    def embed_texts(self, texts, *, priority="interactive"):
        raise RetiredInferencePath("SentenceTransformerEmbedder (in-process sentence-transformers)",
                                   EMBEDDER_INSTEAD)
