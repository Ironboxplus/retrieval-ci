"""SentenceTransformer embedder — dense-only, backward compatible."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings, EMBEDDING_MODELS
from core.indexing.embedders import (
    BaseEmbedder,
    EmbeddingPriority,
    EmbeddingResult,
    embedding_slot,
)
from core.gpu_memory import (
    gpu_model_activity,
    is_cuda_device,
    request_cuda_cache_trim_if_idle,
    trim_cuda_cache_if_idle,
)


class SentenceTransformerEmbedder(BaseEmbedder):

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self._model_name = model_name or settings.embedding.model_name
        self._device = device or settings.embedding.device
        info = EMBEDDING_MODELS.get(self._model_name, {})
        self._dim = info.get("dimension", 384)
        with gpu_model_activity(self._device):
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
        if is_cuda_device(self._device):
            trim_cuda_cache_if_idle(self._device)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def supports_sparse(self) -> bool:
        return False

    def embed_texts(
        self, texts: list[str], *, priority: EmbeddingPriority = "interactive",
    ) -> EmbeddingResult:
        try:
            with embedding_slot(priority):
                with gpu_model_activity(self._device):
                    vectors = self._model.encode(
                        texts,
                        batch_size=settings.embedding.batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
        finally:
            self._release_idle_cuda_cache()
        return EmbeddingResult(dense=np.asarray(vectors, dtype=np.float32))

    def _release_idle_cuda_cache(self) -> None:
        """Return temporary encode workspaces after the shared GPU becomes idle."""
        if not is_cuda_device(self._device):
            return
        request_cuda_cache_trim_if_idle(self._device)
