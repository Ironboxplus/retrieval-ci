"""The one BGE-M3 loader and the one encode call, shared by the backend and the embed worker.

Importing this module pulls in neither ``config`` nor FlagEmbedding/torch (they are imported
when a model is loaded), so the standalone worker can use it without the project's settings
layer. Every limit and dtype comes from the :class:`EmbeddingProfile`; callers never pass a
``max_length`` and an over-long input raises instead of being cut.
"""
from __future__ import annotations

from typing import Any, Sequence

from core.indexing.embedding_profile import EmbeddingProfile, tokens_over_limit
from core.inference_retired import EMBEDDER_INSTEAD, RetiredInferencePath

# A corruption guard, not a precision check: fp16 unit vectors measure a few 1e-4 of norm
# error on real text (3.4e-4 over 500 mixed texts) and the bf16 vectors this replaces were off
# by 2e-3. Anything 1e-2 from unit length is a broken vector, and a false positive here
# would fail indexing.
NORM_TOLERANCE = 1e-2


class InputTooLong(ValueError):
    """A text is longer than the model's sequence limit; the index and token count say which."""

    def __init__(self, index: int, tokens: int, limit: int):
        super().__init__(
            f"input {index} has {tokens} tokens, over the model limit of {limit}; "
            "split it before embedding instead of letting the model cut it")
        self.index, self.tokens, self.limit = index, tokens, limit


def _is_cuda(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def load_model(profile: EmbeddingProfile, model_path: str, device: str) -> Any:
    """Load BGE-M3 under ``profile`` onto ``device`` (a ``BGEM3FlagModel``).

    The limit is bound at construction: ``passage_max_length`` and ``query_max_length``
    are the profile's sequence limit, not FlagEmbedding's silent 512. Computation is fp16
    on CUDA; FlagEmbedding cannot run fp16 on CPU, so a CPU model computes in float32 and
    its vectors are not the CUDA deployment's (the profile fingerprint describes CUDA).

    RETIRED: raises. Nothing loads BGE-M3 into a process any more; the models run as vLLM
    services and ``vllm_client.encode`` is the client (same limits, same guards).
    """
    raise RetiredInferencePath("bge_m3_core.load_model (in-process FlagEmbedding)", EMBEDDER_INSTEAD)
    from FlagEmbedding import BGEM3FlagModel

    if profile.dtype not in ("float16", "float32"):
        raise ValueError(f"unsupported embedding dtype {profile.dtype!r}")
    use_fp16 = profile.dtype == "float16" and _is_cuda(device)
    return BGEM3FlagModel(
        model_path,
        normalize_embeddings=profile.normalize,
        use_fp16=use_fp16,
        use_bf16=False,
        devices=device,
        pooling_method=profile.pooling,
        batch_size=profile.encode_batch_size,
        query_max_length=profile.max_seq_len,
        passage_max_length=profile.max_seq_len,
    )


def encode(model: Any, texts: Sequence[str], profile: EmbeddingProfile) -> tuple[Any, list[dict[int, float]]]:
    """``(dense float32 [n, dim], sparse [n] of token id -> weight)`` for ``texts``.

    Counts real tokens (specials included) with the model's own tokenizer first and raises
    :class:`InputTooLong` before the GPU is touched. The dense vectors are checked finite and
    unit length before they are returned: a value corrupted here would be written to Qdrant
    and survive any read-back comparison.
    """
    import numpy as np

    texts = list(texts)
    if not texts:
        return np.zeros((0, profile.dimension), dtype=np.float32), []
    for index, tokens in tokens_over_limit(
            texts, profile.max_seq_len, profile, tokenizer=getattr(model, "tokenizer", None))[:1]:
        raise InputTooLong(index, tokens, profile.max_seq_len)

    output = model.encode(
        texts,
        batch_size=profile.encode_batch_size,
        max_length=profile.max_seq_len,
        return_dense=True,
        return_sparse=profile.sparse,
        return_colbert_vecs=False,
    )
    dense = np.asarray(output["dense_vecs"], dtype=np.float32)
    if dense.shape != (len(texts), profile.dimension):
        raise ValueError(f"dense vectors have shape {dense.shape}, expected {(len(texts), profile.dimension)}")
    if not np.isfinite(dense).all():
        raise ValueError("dense vectors are not finite")
    if profile.normalize:
        deviation = float(np.abs(np.linalg.norm(dense, axis=1) - 1.0).max())
        if deviation >= NORM_TOLERANCE:
            raise ValueError(f"dense vectors are not unit length (largest norm error {deviation:.2e})")
    sparse = [
        {int(token_id): float(weight) for token_id, weight in weights.items()}
        for weights in (output["lexical_weights"] if profile.sparse else [{}] * len(texts))
    ]
    return dense, sparse
