"""The embedding profile: one frozen record, one fingerprint of what changes a vector."""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from core.indexing import embedding_profile as profile_module
from core.indexing.embedding_profile import EmbeddingProfile, get_profile

REPO_ROOT = Path(__file__).resolve().parent.parent

# field -> a different value, for every field that changes the vector a text encodes to
VECTOR_FIELDS = {
    "model_id": "BAAI/other",
    "revision": "0" * 40,
    "dimension": 768,
    "max_seq_len": 512,
    "dtype": "bfloat16",
    "pooling": "mean",
    "normalize": False,
    "sparse": False,
}


def test_fingerprint_is_stable_across_processes():
    here = get_profile().fingerprint
    code = ("from core.indexing.embedding_profile import get_profile;"
            "print(get_profile().fingerprint)")
    other = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert other == here
    assert len(here) == 16 and int(here, 16) >= 0


@pytest.mark.parametrize("field,value", sorted(VECTOR_FIELDS.items()))
def test_fingerprint_changes_with_every_vector_field(field, value):
    base = get_profile()
    assert dataclasses.replace(base, **{field: value}).fingerprint != base.fingerprint


@pytest.mark.parametrize("field,value", [("encode_batch_size", 7), ("chunk_max_tokens", 512)])
def test_fingerprint_ignores_batching_and_chunk_budget(field, value):
    base = get_profile()
    assert dataclasses.replace(base, **{field: value}).fingerprint == base.fingerprint


def test_every_profile_field_is_classified():
    """A new field must be a deliberate choice: in the fingerprint, or listed as excluded."""
    excluded = {"encode_batch_size", "chunk_max_tokens"}
    fields = {f.name for f in dataclasses.fields(EmbeddingProfile)}
    assert fields == set(VECTOR_FIELDS) | excluded


def test_default_profile_matches_the_deployed_model():
    profile = get_profile()
    assert profile.model_id == "BAAI/bge-m3"
    assert profile.revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert (profile.dimension, profile.max_seq_len, profile.dtype) == (1024, 8192, "float16")
    assert (profile.pooling, profile.normalize, profile.sparse) == ("cls", True, True)
    assert profile.chunk_max_tokens <= profile.max_seq_len


def test_profile_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        get_profile().max_seq_len = 512  # type: ignore[misc]


def test_model_id_comes_from_the_settings_environment_variable(monkeypatch):
    monkeypatch.setenv(profile_module.MODEL_ID_ENV, "BAAI/bge-m3")
    assert get_profile() is profile_module.PROFILES["BAAI/bge-m3"]
    monkeypatch.setenv(profile_module.MODEL_ID_ENV, "no/such-model")
    with pytest.raises(KeyError, match="no/such-model"):
        get_profile()
    assert get_profile("BAAI/bge-m3").model_id == "BAAI/bge-m3"


def test_config_numbers_for_bge_m3_are_the_profile_numbers():
    from config import EMBEDDING_MODELS, settings

    row, profile = EMBEDDING_MODELS["BAAI/bge-m3"], get_profile("BAAI/bge-m3")
    assert row["dimension"] == profile.dimension
    assert row["max_tokens"] == profile.max_seq_len
    assert row["supports_sparse"] is profile.sparse
    assert not hasattr(settings.embedding, "use_fp16")
    assert not hasattr(settings.embedding, "batch_size")


def test_the_profile_module_needs_nothing_from_the_project_config():
    code = ("import sys; import core.indexing.embedding_profile;"
            "assert 'config' not in sys.modules and 'pydantic_settings' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


def test_worker_modules_import_without_the_project_config():
    code = ("import sys; import core.indexing.embedders.bge_m3_core;"
            "import core.indexing.token_budget;"
            "assert 'config' not in sys.modules and 'pydantic_settings' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)
