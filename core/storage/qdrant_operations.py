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
    max_delay_seconds: float = 30.0,
    sleep: Callable[[float], None] | None = None,
) -> Any:
    """Execute an idempotent Qdrant operation with bounded exponential backoff.

    ``sleep`` defaults to ``time.sleep`` looked up per call, so tests that patch
    the module's clock keep working; scripts inject their own to stay testable.
    """

    delay_seconds = initial_delay_seconds
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return run()
        except Exception as exc:  # Qdrant transport and segment-merge failures are transient.
            last_error = exc
            logger.warning("%s failed (attempt %d/%d): %s", operation, attempt, max_attempts, exc)
            if attempt < max_attempts:
                (sleep or time.sleep)(delay_seconds)
                delay_seconds = min(delay_seconds * 2, max_delay_seconds)
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
    on_disk: bool = False,
) -> set[str]:
    """Create missing keyword payload indexes and return fields that still failed.

    ``retry`` is injected so callers can keep their existing logging/metrics and
    tests while centralizing the Qdrant schema operation itself. ``on_disk`` must
    follow the deployment's storage mode: without Qdrant's LOW_MEMORY_MODE an
    in-RAM index is loaded into memory at every restart. Existing indexes are
    left as they are.
    """

    schema = existing_schema if existing_schema is not None else (
        client.get_collection(collection_name).payload_schema or {}
    )
    field_schema: Any = (
        models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, on_disk=True)
        if on_disk else models.PayloadSchemaType.KEYWORD
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
                    field_schema=field_schema,
                    wait=True,
                ),
            )
        except Exception:
            failed.add(field_name)
            logger.exception("could not create Qdrant payload index field=%s", field_name)
    return failed


# ids per request: a `paper_id` MatchAny list of this size keeps one request small
# and each filtered write bounded (paper_id is keyword-indexed in every paper collection).
_PAPER_ID_BATCH = 500


def set_paper_withdrawn(
    client: Any,
    collection_name: str,
    paper_ids: Iterable[str],
    withdrawn: bool,
    *,
    retry: Callable[[str, Callable[[], Any]], Any] = retry_qdrant_operation,
) -> None:
    """Make every point of `paper_ids` carry the optional `withdrawn` payload key
    (True) or lack it (False), the shape core/storage/papers_payload.py writes.

    Both calls select points by filter and use wait=True, so they return once the
    change is applied. `set_payload` merges its keys into each selected point's
    payload ("If payload value with specified key already exists - it will be
    overwritten", qdrant_client.QdrantClient.set_payload) and `delete_payload`
    removes the listed keys (QdrantClient.delete_payload); a point without the key
    is left as it is, so both are idempotent and safe to repeat after a crash.
    """
    ids = sorted({pid for pid in paper_ids})
    for start in range(0, len(ids), _PAPER_ID_BATCH):
        selector = models.Filter(must=[models.FieldCondition(
            key="paper_id", match=models.MatchAny(any=ids[start:start + _PAPER_ID_BATCH]))])
        if withdrawn:
            operation = f"set_payload(collection={collection_name} withdrawn=true)"
            retry(operation, lambda selector=selector: client.set_payload(
                collection_name=collection_name, payload={"withdrawn": True},
                points=selector, wait=True))
        else:
            operation = f"delete_payload(collection={collection_name} withdrawn)"
            retry(operation, lambda selector=selector: client.delete_payload(
                collection_name=collection_name, keys=["withdrawn"],
                points=selector, wait=True))


def delete_points_by_doc_id(
    client: Any,
    collection_name: str,
    doc_id: str,
    *,
    retry: Callable[[str, Callable[[], Any]], Any] = retry_qdrant_operation,
) -> None:
    """Delete every point of one document, then prove none is left.

    Unlike QdrantStore.delete_by_doc_id (which logs and returns 0 on any error),
    a failure raises: the caller deletes the SQL document only after this
    returns, so a swallowed error would orphan the vectors for good. `doc_id`
    is the string the papers payload contract writes (papers_payload D2); an
    int-typed legacy payload would survive the filter and fail the count check.
    """
    selector = models.Filter(must=[models.FieldCondition(
        key="doc_id", match=models.MatchValue(value=doc_id))])
    retry(f"delete(collection={collection_name} doc_id={doc_id})",
          lambda: client.delete(collection_name=collection_name,
                                points_selector=models.FilterSelector(filter=selector), wait=True))
    left = client.count(collection_name=collection_name, count_filter=selector, exact=True).count
    if left:
        raise RuntimeError(
            f"{left} point(s) of doc_id={doc_id} remain in {collection_name} after delete")
