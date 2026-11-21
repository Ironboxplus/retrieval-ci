"""One bounded-backoff policy for every Qdrant write path.

``retry_qdrant_operation`` is the policy; ``QdrantStore`` and the maintenance
scripts used to carry their own copies (60 s cap here, no cap there, print
instead of logging). The callables raise the client's real transport exception;
the only stand-in is a thin wrapper that delegates every call to a real
in-memory ``QdrantClient`` and fails ``upsert`` on demand.
"""

from __future__ import annotations

import logging
import time

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient, models  # noqa: E402
from qdrant_client.http.exceptions import ResponseHandlingException  # noqa: E402

from core.storage.qdrant_operations import retry_qdrant_operation  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402
from scripts import compact_papers_index, split_papers_corpus  # noqa: E402
from scripts.remote_embed import local_ingest  # noqa: E402

TRANSPORT_ERROR = ResponseHandlingException(TimeoutError("read timed out"))


class FlakyUpserts:
    """The real client, failing the first ``failures`` upserts with a transport error."""

    def __init__(self, real: QdrantClient, failures: int):
        self._real = real
        self.failures = failures
        self.upsert_calls = 0

    def upsert(self, *args, **kwargs):
        self.upsert_calls += 1
        if self.upsert_calls <= self.failures:
            raise TRANSPORT_ERROR
        return self._real.upsert(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture()
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


@pytest.fixture()
def real_client():
    client = QdrantClient(":memory:")
    yield client
    client.close()


def _failing(times: int):
    state = {"calls": 0}

    def run():
        state["calls"] += 1
        if state["calls"] <= times:
            raise TRANSPORT_ERROR
        return "done"

    return run


def _points(count: int, *, named: str | None = None) -> list[models.PointStruct]:
    return [
        models.PointStruct(id=index + 1, vector={named: [0.1, 0.2, 0.3, 0.4]} if named else [0.1, 0.2, 0.3, 0.4])
        for index in range(count)
    ]


def test_delays_double_and_stop_at_the_default_cap() -> None:
    delays: list[float] = []
    result = retry_qdrant_operation("op", _failing(7), max_attempts=9, sleep=delays.append)
    assert result == "done"
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_cap_and_first_delay_are_parameters() -> None:
    delays: list[float] = []
    retry_qdrant_operation(
        "op", _failing(5), max_attempts=6, initial_delay_seconds=0.5, max_delay_seconds=3.0, sleep=delays.append
    )
    assert delays == [0.5, 1.0, 2.0, 3.0, 3.0]


def test_the_last_error_is_raised_after_the_attempts_run_out() -> None:
    delays: list[float] = []
    with pytest.raises(ResponseHandlingException):
        retry_qdrant_operation("op", _failing(99), max_attempts=3, sleep=delays.append)
    assert delays == [1.0, 2.0]


def test_store_upsert_batch_retries_with_the_shared_policy(real_client, sleeps, caplog) -> None:
    real_client.create_collection("retry_policy_store", vectors_config=models.VectorParams(
        size=4, distance=models.Distance.COSINE))
    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = "retry_policy_store"
    store.client = FlakyUpserts(real_client, failures=2)

    with caplog.at_level(logging.WARNING):
        store._upsert_batch(_points(3), wait=True)

    assert real_client.count("retry_policy_store", exact=True).count == 3
    assert sleeps == [1.0, 2.0]
    assert "upsert(collection=retry_policy_store points=3) failed (attempt 1/5)" in caplog.text


def test_local_ingest_backoff_stops_at_thirty_seconds(real_client, sleeps) -> None:
    real_client.create_collection("retry_policy_ingest", vectors_config=models.VectorParams(
        size=4, distance=models.Distance.COSINE))
    flaky = FlakyUpserts(real_client, failures=7)

    local_ingest._upsert_with_retry(flaky, "retry_policy_ingest", _points(2), max_retries=8)

    assert real_client.count("retry_policy_ingest", exact=True).count == 2
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_split_corpus_backoff_stops_at_thirty_seconds(real_client, sleeps) -> None:
    real_client.create_collection("retry_policy_split", vectors_config=models.VectorParams(
        size=4, distance=models.Distance.COSINE))
    flaky = FlakyUpserts(real_client, failures=7)

    split_papers_corpus.upsert_with_retry(flaky, "retry_policy_split", _points(2), max_retries=8)

    assert real_client.count("retry_policy_split", exact=True).count == 2
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_compact_upload_failures_are_logged_by_the_shared_policy(real_client, sleeps, caplog) -> None:
    real_client.create_collection("retry_policy_compact", vectors_config={
        "dense": models.VectorParams(size=4, distance=models.Distance.COSINE)})
    flaky = FlakyUpserts(real_client, failures=2)

    with caplog.at_level(logging.WARNING):
        compact_papers_index.upload(flaky, "retry_policy_compact", [(1, [0.1, 0.2, 0.3, 0.4])])

    assert real_client.count("retry_policy_compact", exact=True).count == 1
    assert sleeps == [1.0, 2.0]
    assert "upsert retry_policy_compact failed (attempt 1/6)" in caplog.text
