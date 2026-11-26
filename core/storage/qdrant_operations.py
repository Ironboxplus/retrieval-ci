"""Domain-neutral durable Qdrant write and payload-index operations."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import Any

from qdrant_client import models


logger = logging.getLogger(__name__)


def retry_qdrant_operation(
    operation: str,
    run: Callable[[], Any],
    *,
    max_attempts: int = 5,
    initial_delay_seconds: float = 1.0,
) -> Any:
    """Execute an idempotent Qdrant operation with bounded exponential backoff."""

    delay_seconds = initial_delay_seconds
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return run()
        except Exception as exc:  # Qdrant transport and segment-merge failures are transient.
            last_error = exc
            logger.warning("%s failed (attempt %d/%d): %s", operation, attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 30.0)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{operation} did not execute")


def ensure_keyword_payload_indexes(
    *,
    client: Any,
    collection_name: str,
    field_names: Iterable[str],
    retry: Callable[[str, Callable[[], Any]], Any],
    existing_schema: dict[str, Any] | None = None,
) -> set[str]:
    """Create missing keyword payload indexes and return fields that still failed.

    ``retry`` is injected so callers can keep their existing logging/metrics and
    tests while centralizing the Qdrant schema operation itself.
    """

    schema = existing_schema if existing_schema is not None else (
        client.get_collection(collection_name).payload_schema or {}
    )
    failed: set[str] = set()
    for field_name in field_names:
        if field_name in schema:
            continue
        operation = f"create_payload_index(collection={collection_name} field={field_name})"
        try:
            retry(
                operation,
                lambda field_name=field_name: client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                ),
            )
        except Exception:
            failed.add(field_name)
            logger.exception("could not create Qdrant payload index field=%s", field_name)
    return failed
