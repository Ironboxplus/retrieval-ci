"""The resplit stage reads select's oversized list one row at a time.

Production 2026-11-14: code_chunks' list was 898 MB (519 rows with whole payloads) and
public_code_chunks' 575 MB; loading a list at once swapped .88 until the backend stopped answering.
"""
from __future__ import annotations

import json
import tracemalloc
from types import SimpleNamespace

from app.services.pipeline.reembed_select import iter_oversized

ROW_CHARS = 1_000_000


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def row(point_id: int, fill: str) -> dict:
    return {"point_id": point_id, "tokens": 9999, "payload": {"content": fill * ROW_CHARS}}


def test_rows_come_one_at_a_time_and_memory_stays_near_one_row(tmp_path):
    ctx = SimpleNamespace(select_dir=tmp_path / "select")
    write_rows(ctx.select_dir / "oversized.jsonl", [row(i, "é") for i in range(30)])     # ~60 MB on disk

    tracemalloc.start()
    seen = [(point_id, len(r["payload"]["content"])) for point_id, r in iter_oversized(ctx)]
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert seen == [(i, ROW_CHARS) for i in range(30)]
    assert peak < 12 * ROW_CHARS          # a few copies of one row, not 30 rows


def test_a_point_listed_twice_comes_once_as_its_last_row(tmp_path):
    ctx = SimpleNamespace(select_dir=tmp_path / "select")
    rows = [{"point_id": 1, "payload": {"content": "old"}}, {"point_id": 2, "payload": {"content": "b"}},
            {"point_id": 1, "payload": {"content": "new"}}]
    write_rows(ctx.select_dir / "oversized.jsonl", rows)

    assert [(i, r["payload"]["content"]) for i, r in iter_oversized(ctx)] == [(2, "b"), (1, "new")]


def test_no_list_means_no_rows(tmp_path):
    assert list(iter_oversized(SimpleNamespace(select_dir=tmp_path / "select"))) == []
