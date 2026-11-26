"""BGE-M3 embedder — dense + sparse dual output."""
from __future__ import annotations

import os

from config import settings
from core.indexing.embedders import (
    BaseEmbedder,
    EmbeddingPriority,
    EmbeddingResult,
    embedding_slot,
)
from core.indexing.embedders.bge_m3_core import encode, load_model
from core.indexing.embedding_profile import MODEL_PATH_ENV, get_profile
from core.inference_retired import EMBEDDER_INSTEAD, RetiredInferencePath
from core.gpu_memory import (
    gpu_model_activity,
    is_cuda_device,
    request_cuda_cache_trim_if_idle,
    trim_cuda_cache_if_idle,
)


class BgeM3Embedder(BaseEmbedder):
    """RETIRED: BGE-M3 loaded into this process with FlagEmbedding.

    Kept for reference only; constructing it raises. The backend embeds through the vLLM
    service (``core.indexing.embedders.get_embedder()``).
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        raise RetiredInferencePath("BgeM3Embedder (in-process FlagEmbedding)", EMBEDDER_INSTEAD)
        self._model_name = model_name
        self._device = device or settings.embedding.device
        self._profile = get_profile(self._model_name)
        self._dim = self._profile.dimension
        # Deployment override: load weights from a local dir (e.g. modelscope
        # cache on a HF-blocked host) while keeping ``_model_name`` as the
        # repo-id so factory routing + dim lookup stay correct. No-op when unset.
        model_path = os.environ.get(MODEL_PATH_ENV) or self._model_name
        with gpu_model_activity(self._device):
            self._model = load_model(self._profile, model_path, self._device)
            if is_cuda_device(self._device):
                self._warmup()
        if is_cuda_device(self._device):
            trim_cuda_cache_if_idle(self._device)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def supports_sparse(self) -> bool:
        return self._profile.sparse

    def _warmup(self):
        """Eager GPU weight transfer — avoids lazy copy on first real request."""
        import gc
        encode(self._model, ["warmup"], self._profile)
        gc.collect()

    def embed_texts(
        self, texts: list[str], *, priority: EmbeddingPriority = "interactive",
    ) -> EmbeddingResult:
        try:
            with embedding_slot(priority):
                with gpu_model_activity(self._device):
                    dense, sparse = encode(self._model, texts, self._profile)
        finally:
            self._release_idle_cuda_cache()
        return EmbeddingResult(
            dense=dense,
            sparse=[
                {str(token_id): weight for token_id, weight in weights.items()}
                for weights in sparse
            ],
        )

    def _release_idle_cuda_cache(self) -> None:
        """Return temporary encode workspaces after the shared GPU becomes idle."""
        if not is_cuda_device(self._device):
            return
        request_cuda_cache_trim_if_idle(self._device)
