from __future__ import annotations

from types import SimpleNamespace

from scripts.configure_qdrant_low_memory import configure_collection


def _info(*, compliant: bool):
    on_disk = True if compliant else None
    return SimpleNamespace(
        model_dump=lambda **_kwargs: {
            "status": "green",
            "optimizer_status": "ok",
            "config": {
                "params": {
                    "vectors": {
                        "dense": {
                            "on_disk": on_disk,
                            "hnsw_config": {"on_disk": on_disk},
                        }
                    },
                    "sparse_vectors": {
                        "sparse": {"index": {"on_disk": on_disk}}
                    },
                    "on_disk_payload": compliant,
                },
                "hnsw_config": {"on_disk": compliant},
                "optimizer_config": {
                    "memmap_threshold": 5000 if compliant else None,
                    "max_optimization_threads": 1 if compliant else None,
                },
            },
        }
    )


class _Client:
    def __init__(self, *, compliant: bool):
        self.info = _info(compliant=compliant)
        self.updates = []

    def get_collection(self, collection_name):
        return self.info

    def update_collection(self, **kwargs):
        self.updates.append(kwargs)
        return True


def test_dry_run_reports_needed_changes_without_updating():
    client = _Client(compliant=False)

    result = configure_collection(client, "code_chunks", execute=False)

    assert result == {
        "collection": "code_chunks",
        "action": "would_update",
        "before": {
            "dense_on_disk": False,
            "dense_hnsw_on_disk": False,
            "sparse_on_disk": False,
            "payload_on_disk": False,
            "memmap_threshold": None,
            "max_optimization_threads": None,
        },
    }
    assert client.updates == []


def test_execute_uses_qdrant_client_models_for_complete_low_memory_update():
    client = _Client(compliant=False)

    result = configure_collection(client, "public_code_chunks", execute=True)

    assert result["action"] == "update_submitted"
    assert len(client.updates) == 1
    update = client.updates[0]
    assert update["collection_name"] == "public_code_chunks"
    dense = update["vectors_config"]["dense"]
    sparse = update["sparse_vectors_config"]["sparse"]
    assert dense.on_disk is True
    assert dense.hnsw_config.on_disk is True
    assert sparse.index.on_disk is True
    assert update["collection_params"].on_disk_payload is True
    assert update["hnsw_config"].on_disk is True
    assert update["optimizers_config"].memmap_threshold == 5000
    assert update["optimizers_config"].max_optimization_threads == 1


def test_already_compliant_collection_is_never_rebuilt_again():
    client = _Client(compliant=True)

    result = configure_collection(client, "journal_chunks", execute=True)

    assert result["action"] == "already_compliant"
    assert client.updates == []
