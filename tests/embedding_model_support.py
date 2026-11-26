"""Locate the real BGE-M3 snapshot and a CUDA device for the embedding tests.

The tests use the real tokenizer and the real model; nothing here stands in for either.
They are skipped (never faked) when the weights or the GPU are not on the machine.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

SNAPSHOT_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_HF_SNAPSHOT = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"
    / "snapshots" / SNAPSHOT_REVISION
)
GPU_DEVICE = os.environ.get("EMBEDDING_TEST_DEVICE", "cuda:0")


def bge_m3_snapshot() -> str:
    """Directory of the BGE-M3 snapshot: ``BGE_M3_PATH`` if set, else the local HF cache."""
    configured = os.environ.get("BGE_M3_PATH")
    if configured and Path(configured).is_dir():
        return configured
    if _HF_SNAPSHOT.is_dir():
        return str(_HF_SNAPSHOT)
    pytest.skip("BGE-M3 snapshot is not available locally")


def require_gpu() -> str:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return GPU_DEVICE


def repo_text() -> str:
    """The fixed concatenation of tracked, non-personal repository files that real text is cut
    from (README.md, core/retrieval/pipeline.py, core/indexing/chunker.py — always present,
    also in the public CI snapshot, which leaves the notes out)."""
    repo = Path(__file__).resolve().parents[1]
    return "\n\n".join(
        (repo / name).read_text(encoding="utf-8")
        for name in ("README.md", "core/retrieval/pipeline.py", "core/indexing/chunker.py"))


def real_text(tokenizer, tokens: int) -> str:
    """The first ``tokens`` real tokens of ``repo_text()``, cut at a token end."""
    source = repo_text()
    encoded = tokenizer(source[: tokens * 12], add_special_tokens=False,
                        return_offsets_mapping=True, truncation=False)
    assert len(encoded["input_ids"]) >= tokens, "the repo files are shorter than the requested text"
    return source[: encoded["offset_mapping"][tokens - 1][1]]


def text_of_tokens(tokenizer, tokens: int) -> str:
    """Natural-looking English text whose real token count (specials included) is ``tokens``."""
    vocabulary = ("retrieval index chunk vector embedding model sparse dense token budget "
                  "journal paper code scope qdrant payload fingerprint reuse").split()
    words = [vocabulary[i % len(vocabulary)] + str(i % 97) for i in range(tokens + 16)]

    def count(n: int) -> int:
        return len(tokenizer(" ".join(words[:n]))["input_ids"])

    low, high = 1, len(words)          # the largest prefix that does not exceed the target
    while low < high:
        middle = (low + high + 1) // 2
        if count(middle) <= tokens:
            low = middle
        else:
            high = middle - 1
    text = " ".join(words[:low])
    text += " x" * (tokens - len(tokenizer(text)["input_ids"]))   # one token each
    assert len(tokenizer(text)["input_ids"]) == tokens
    return text
