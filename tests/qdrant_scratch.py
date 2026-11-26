"""Scratch collections for real-Qdrant tests.

On the Windows test Qdrant a delete right after heavy use can fail with "access
denied" while the storage directory is renamed, and the half-deleted directory then
blocks re-creating the same name. Each test therefore gets fresh names, and drops
retry with backoff and only warn if the collection still lingers.
"""
from __future__ import annotations

import time
import uuid
import warnings


def scratch_names(prefix: str, roles=("papers", "arxiv", "reviews")) -> dict[str, str]:
    suffix = uuid.uuid4().hex[:8]
    return {role: f"{prefix}_{role}_{suffix}_test" for role in roles}


def drop(client, names, attempts: int = 4) -> None:
    for name in names:
        last = None
        for attempt in range(attempts):
            try:
                if client.collection_exists(name):
                    client.delete_collection(name)
                last = None
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.5 * (attempt + 1))
        if last is not None:
            warnings.warn(f"could not drop scratch collection {name} ({last})", stacklevel=2)
