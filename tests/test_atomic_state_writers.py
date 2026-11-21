"""State/report writers in the maintenance scripts share one atomic-write helper.

A Windows reader (antivirus, a status poller) holds a file without
FILE_SHARE_DELETE for a few hundred milliseconds; the helper retries the
rename over that window, so a hand-written ``tmp.replace(path)`` copy fails
where the shared one succeeds.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts.compact_papers_index import _save_state as compact_save_state
from scripts.reconcile_paper_chunk_schema import _write_report
from scripts.reviews.harvest_all import _save_cursor
from scripts.split_papers_corpus import _save_state as split_save_state

pytestmark = pytest.mark.skipif(os.name != "nt", reason="needs Windows delete-sharing semantics")

HOLD_SECONDS = 0.4


@contextmanager
def held_open_briefly(path: Path):
    """Keep ``path`` open (no delete sharing) for HOLD_SECONDS, then release it."""
    handle = open(path, "rb")

    def release() -> None:
        time.sleep(HOLD_SECONDS)
        handle.close()

    releaser = threading.Thread(target=release)
    releaser.start()
    try:
        yield
    finally:
        releaser.join()


def _write_state_through(writer, path: Path, payload: dict) -> None:
    path.write_text(json.dumps({"version": 0}), encoding="utf-8")
    with held_open_briefly(path):
        writer(path, payload)


@pytest.mark.parametrize(
    "writer",
    [compact_save_state, split_save_state, _write_report, _save_cursor],
    ids=["compact_papers_index", "split_papers_corpus", "reconcile_report", "harvest_cursor"],
)
def test_writer_survives_a_briefly_held_destination(tmp_path: Path, writer) -> None:
    target = tmp_path / "state.json"
    _write_state_through(writer, target, {"version": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}
    assert [item.name for item in tmp_path.iterdir()] == ["state.json"]
