"""Shared Qdrant payload-index helpers must remain domain-neutral."""

from __future__ import annotations

from types import SimpleNamespace

from core.storage.qdrant_operations import ensure_keyword_payload_indexes


def test_creates_only_missing_keyword_indexes_with_wait_true_and_injected_retry():
    class Client:
        def __init__(self):
            self.created = []

        def get_collection(self, _name):
            return SimpleNamespace(payload_schema={"source_id": {}})

        def create_payload_index(self, **kwargs):
            self.created.append(kwargs)

    client = Client()
    retried = []

    failed = ensure_keyword_payload_indexes(
        client=client,
        collection_name="public_code_chunks",
        field_names=("source_id", "revision", "path"),
        retry=lambda operation, run: retried.append(operation) or run(),
    )

    assert failed == set()
    assert retried == [
        "create_payload_index(collection=public_code_chunks field=revision)",
        "create_payload_index(collection=public_code_chunks field=path)",
    ]
    assert [item["field_name"] for item in client.created] == ["revision", "path"]
    assert all(item["wait"] is True for item in client.created)
