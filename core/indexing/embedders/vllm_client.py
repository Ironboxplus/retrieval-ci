"""The client of the BGE-M3 vLLM service: the one encode call once the model runs in its own process.

Text is tokenized here, with the profile's tokenizer, and only token ids are sent. vLLM's own
tokenizer (transformers 5.x) drops the trailing "▁" the production tokenizer emits before ``</s>``
on text that ends in whitespace, which moved code-chunk vectors down to cos 0.978; with ids the
service's vectors match FlagEmbedding's to fp16 noise (RESEARCH-2026-11-14-vllm-serving.md §3).

The contract is ``bge_m3_core.encode``'s: ``(dense float32 [n, dim], sparse [n] of int token id ->
weight)``, an over-long text raises :class:`InputTooLong` before any request leaves, and the dense
rows are checked finite and unit length before they are returned. The sparse plugin also returns
weights <= 0 and ``<unk>`` entries; they are dropped here the way FlagEmbedding's
``lexical_weights`` drop them (weight > 0, no ``<s>``/``<pad>``/``</s>``/``<unk>``).

A call is cut into requests of ``rows_per_request`` texts with up to ``max_in_flight`` of them
outstanding, and the next request is tokenized while the earlier ones run on the GPU, so one
large call keeps the service busy from its first request to its last.

Standard library, numpy, httpx and the tokenizer only: no torch, no vLLM, no ``config``, so the
standalone embed worker and the backend share it.
"""
from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from functools import lru_cache
from typing import Any, Optional, Sequence

from core.indexing.embedders.bge_m3_core import NORM_TOLERANCE, InputTooLong
from core.indexing.embedding_profile import EmbeddingProfile, load_tokenizer, tokens_over_limit

# Same variable the settings layer will read (``Settings`` nested delimiter, ``embedding.service_url``).
SERVICE_URL_ENV = "EMBEDDING__SERVICE_URL"
DEFAULT_SERVICE_URL = "http://127.0.0.1:18200"
# ``served-model-name`` in deploy/compute88/vllm_bge_m3.yaml.
SERVED_MODEL_NAME = "bge-m3"
# Measured on the 3090 (PLAN-2026-11-14-vllm-embed-host.md §2): 32 texts per request with 8
# outstanding saturate the engine; one text per request leaves the API process as the limit.
ROWS_PER_REQUEST = 32
MAX_IN_FLIGHT = 8


class ServiceError(RuntimeError):
    """The service is unreachable, serves something other than the profile, or failed a request."""


def service_url(base_url: Optional[str] = None) -> str:
    return (base_url or os.environ.get(SERVICE_URL_ENV) or DEFAULT_SERVICE_URL).rstrip("/")


@lru_cache(maxsize=4)
def _http(base_url: str) -> Any:
    import httpx

    # trust_env=False: the host's HTTP proxy (127.0.0.1:10808) must never carry loopback traffic.
    return httpx.Client(base_url=base_url, trust_env=False, timeout=httpx.Timeout(900.0, connect=10.0),
                        limits=httpx.Limits(max_connections=64, max_keepalive_connections=64))


def check_service(profile: EmbeddingProfile, base_url: Optional[str] = None) -> dict:
    """The service's model entry, after checking it serves this profile.

    The served directory must name the profile's revision and the length limit must be the
    profile's: weights of another snapshot would change every vector without changing the
    fingerprint stamped on them.
    """
    url = service_url(base_url)
    try:
        response = _http(url).get("/v1/models", timeout=10.0)
        response.raise_for_status()
        models = response.json()["data"]
    except Exception as exc:
        raise ServiceError(f"BGE-M3 service at {url} is not answering ({exc}); start it with "
                           "scripts/remote_embed/bge_m3_service.sh start") from exc
    entry = next((m for m in models if m.get("id") == SERVED_MODEL_NAME), None)
    if entry is None:
        raise ServiceError(f"service at {url} serves {[m.get('id') for m in models]}, not {SERVED_MODEL_NAME!r}")
    if profile.revision not in str(entry.get("root", "")):
        raise ServiceError(f"service at {url} serves {entry.get('root')!r}, not revision {profile.revision}")
    if int(entry.get("max_model_len") or 0) != profile.max_seq_len:
        raise ServiceError(f"service at {url} has max_model_len {entry.get('max_model_len')}, "
                           f"the profile {profile.max_seq_len}")
    return entry


def _unused_token_ids(tokenizer: Any) -> frozenset:
    """The ids FlagEmbedding leaves out of ``lexical_weights``: cls, eos, pad and unk."""
    return frozenset(i for i in (tokenizer.cls_token_id, tokenizer.eos_token_id,
                                 tokenizer.pad_token_id, tokenizer.unk_token_id) if i is not None)


def encode(texts: Sequence[str], profile: EmbeddingProfile, *, base_url: Optional[str] = None,
           priority: Optional[int] = None, tokenizer: Any = None,
           rows_per_request: int = ROWS_PER_REQUEST,
           max_in_flight: int = MAX_IN_FLIGHT) -> tuple[Any, list[dict[int, float]]]:
    """``(dense float32 [n, dim], sparse [n] of token id -> weight)`` for ``texts`` from the service.

    ``priority`` goes in the request body's top level (vLLM ignores it inside ``data``); lower
    runs first when the service schedules by priority.
    """
    import numpy as np

    texts = list(texts)
    if not texts:
        return np.zeros((0, profile.dimension), dtype=np.float32), []
    tokenizer = tokenizer or load_tokenizer(profile)
    for index, tokens in tokens_over_limit(texts, profile.max_seq_len, profile, tokenizer=tokenizer)[:1]:
        raise InputTooLong(index, tokens, profile.max_seq_len)

    url = service_url(base_url)
    client = _http(url)
    unused = _unused_token_ids(tokenizer)
    embed_task = "dense&sparse" if profile.sparse else "dense"
    dense = np.empty((len(texts), profile.dimension), dtype=np.float32)
    sparse: list = [None] * len(texts)

    def post(start: int, ids: list) -> tuple[int, int, list]:
        body: dict = {"model": SERVED_MODEL_NAME, "task": "plugin",
                      "data": {"input": ids, "embed_task": embed_task, "return_tokens": False}}
        if priority is not None:
            body["priority"] = priority
        try:
            response = client.post("/pooling", json=body)
        except Exception as exc:          # connection refused, reset, timeout
            raise ServiceError(f"BGE-M3 service at {url} failed texts {start}..{start + len(ids) - 1}: "
                               f"{exc!r}") from exc
        if response.status_code != 200:
            raise ServiceError(f"/pooling returned {response.status_code} for texts "
                               f"{start}..{start + len(ids) - 1}: {response.text[:300]}")
        return start, len(ids), response.json()["data"]["data"]

    def store(future: Future) -> None:
        start, count, rows = future.result()
        if len(rows) != count:
            raise ServiceError(f"/pooling returned {len(rows)} rows for {count} texts")
        for row in rows:
            i = start + int(row["index"])
            dense[i] = row["dense_embedding"]
            sparse[i] = {int(e["token_id"]): float(e["weight"]) for e in (row.get("sparse_embedding") or ())
                         if e["weight"] > 0 and e["token_id"] not in unused}

    with ThreadPoolExecutor(max_workers=max_in_flight, thread_name_prefix="bge-m3-client") as pool:
        pending: set = set()
        try:
            for start in range(0, len(texts), rows_per_request):
                ids = tokenizer(texts[start:start + rows_per_request], add_special_tokens=True, truncation=False,
                                return_attention_mask=False, return_token_type_ids=False,
                                verbose=False)["input_ids"]
                for offset, row in enumerate(ids):
                    if len(row) > profile.max_seq_len:      # the guard above vouched for it; never sent
                        raise InputTooLong(start + offset, len(row), profile.max_seq_len)
                while len(pending) >= max_in_flight:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        store(future)
                pending.add(pool.submit(post, start, ids))
            for future in wait(pending).done:
                store(future)
        except BaseException:
            for future in pending:
                future.cancel()
            raise

    if not np.isfinite(dense).all():
        raise ValueError("dense vectors are not finite")
    if profile.normalize:
        deviation = float(np.abs(np.linalg.norm(dense, axis=1) - 1.0).max())
        if deviation >= NORM_TOLERANCE:
            raise ValueError(f"dense vectors are not unit length (largest norm error {deviation:.2e})")
    if not profile.sparse:
        sparse = [{} for _ in texts]
    return dense, sparse
