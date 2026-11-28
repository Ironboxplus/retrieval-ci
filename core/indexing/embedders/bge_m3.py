"""BGE-M3 embedder — dense + sparse dual output."""
from __future__ import annotations

from core.indexing.embedders import (
    BaseEmbedder,
)
from core.inference_retired import EMBEDDER_INSTEAD, RetiredInferencePath


class BgeM3Embedder(BaseEmbedder):
    """RETIRED: BGE-M3 loaded into this process with FlagEmbedding.

    Kept for reference only; constructing it raises. The backend embeds through the vLLM
    service (``core.indexing.embedders.get_embedder()``).
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        raise RetiredInferencePath("BgeM3Embedder (in-process FlagEmbedding)", EMBEDDER_INSTEAD)

    @property
    def dimension(self) -> int:
        raise RetiredInferencePath("BgeM3Embedder (in-process FlagEmbedding)", EMBEDDER_INSTEAD)

    def embed_texts(self, texts, *, priority="interactive"):
        raise RetiredInferencePath("BgeM3Embedder (in-process FlagEmbedding)", EMBEDDER_INSTEAD)
