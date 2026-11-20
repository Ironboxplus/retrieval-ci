"""Real-file contract between Code's incremental collector and shared snapshots."""
from __future__ import annotations

from core.indexing.file_snapshot import FileCandidate, SnapshotDiff
from mcp_server.server import _collect_incremental


def test_code_incremental_and_shared_snapshot_agree_on_real_file_change_sets(tmp_path):
    (tmp_path / "same.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "changed.py").write_text("value = 2\n", encoding="utf-8")
    (tmp_path / "added.py").write_text("value = 3\n", encoding="utf-8")
    candidates = [
        FileCandidate.from_path(tmp_path / name, tmp_path)
        for name in ("same.py", "changed.py", "added.py")
    ]
    known = {
        "same.py": candidates[0].content_hash,
        "changed.py": "0" * 64,
        "removed.py": "1" * 64,
    }

    snapshot = SnapshotDiff.from_candidates(candidates, known)
    payloads, manifest, current = _collect_incremental(
        str(tmp_path), known, respect_gitignore=False)

    assert snapshot.added == ("added.py",)
    assert snapshot.changed == ("changed.py",)
    assert snapshot.unchanged == ("same.py",)
    assert set(item["rel_path"] for item in payloads) == set(snapshot.changed_paths)
    assert set(manifest) == set(snapshot.live_paths)
    assert current == {candidate.rel_path: candidate.content_hash for candidate in candidates}
