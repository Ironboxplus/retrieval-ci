"""BGE-M3 embedder — dense + sparse dual output."""
from __future__ import annotations

import numpy as np
from FlagEmbedding import BGEM3FlagModel

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


class BgeM3Embedder(BaseEmbedder):

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self._model_name = model_name
        self._device = device or settings.embedding.device
        info = EMBEDDING_MODELS.get(self._model_name, {})
        self._dim = info.get("dimension", 1024)
        use_bf16 = is_cuda_device(self._device)
        # Deployment override: load weights from a local dir (e.g. modelscope
        # cache on a HF-blocked host) while keeping ``_model_name`` as the
        # repo-id so factory routing + dim lookup stay correct. No-op when unset.
        import os
        model_path = os.environ.get("BGE_M3_PATH") or self._model_name
        with gpu_model_activity(self._device):
            self._model = BGEM3FlagModel(
                model_path,
                use_bf16=use_bf16,
                use_fp16=False,
                device=self._device,
            )
            if is_cuda_device(self._device):
                self._warmup()
        if is_cuda_device(self._device):
            trim_cuda_cache_if_idle(self._device)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def supports_sparse(self) -> bool:
        return True

    def _warmup(self):
        """Eager GPU weight transfer — avoids lazy copy on first real request."""
        import gc
        self._model.encode(["warmup"], return_dense=True, return_sparse=True)
        gc.collect()

    def embed_texts(
        self, texts: list[str], *, priority: EmbeddingPriority = "interactive",
    ) -> EmbeddingResult:
        try:
            with embedding_slot(priority):
                with gpu_model_activity(self._device):
                    output = self._model.encode(
                        texts,
                        batch_size=settings.embedding.batch_size,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=False,
                    )
        finally:
            self._release_idle_cuda_cache()
        dense = np.asarray(output["dense_vecs"], dtype=np.float32)
        sparse = [
            {str(token_id): float(weight) for token_id, weight in weights.items()}
            for weights in output["lexical_weights"]
        ]
        return EmbeddingResult(dense=dense, sparse=sparse)

    def _release_idle_cuda_cache(self) -> None:
        """Return temporary encode workspaces after the shared GPU becomes idle."""
        if not is_cuda_device(self._device):
            return
        request_cuda_cache_trim_if_idle(self._device)
