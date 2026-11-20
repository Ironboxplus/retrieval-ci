"""Real-filesystem tests for the shared incremental file snapshot contract."""
from __future__ import annotations

from pathlib import Path

from core.indexing.file_snapshot import FileCandidate, ScanSpec, SnapshotDiff


def _candidate(root: Path, relative: str) -> FileCandidate:
    return FileCandidate.from_path(root / relative, root)


def test_snapshot_diff_classifies_add_change_unchanged_and_removed_from_real_files(tmp_path):
    (tmp_path / "same.md").write_text("same", encoding="utf-8")
    (tmp_path / "changed.md").write_text("new content", encoding="utf-8")
    (tmp_path / "added.md").write_text("added", encoding="utf-8")

    current = [_candidate(tmp_path, name) for name in ("same.md", "changed.md", "added.md")]
    known = {
        "same.md": {"content_hash": current[0].content_hash},
        "changed.md": {"content_hash": "0" * 64},
        "removed.md": {"content_hash": "1" * 64},
    }

    diff = SnapshotDiff.from_candidates(current, known)

    assert diff.added == ("added.md",)
    assert diff.changed == ("changed.md",)
    assert diff.unchanged == ("same.md",)
    assert diff.removed == ("removed.md",)
    assert diff.live_paths == ("added.md", "changed.md", "same.md")
    assert diff.replace_safe is True


def test_snapshot_diff_keeps_previously_indexed_unreadable_file_and_refuses_sweep(tmp_path):
    path = tmp_path / "locked.md"
    path.write_text("old live document", encoding="utf-8")
    candidate = _candidate(tmp_path, "locked.md")
    unreadable = FileCandidate(
        abs_path=str(path), rel_path="locked.md", content_hash=None,
        size_bytes=None, read_error="permission denied",
    )

    diff = SnapshotDiff.from_candidates(
        [unreadable], {"locked.md": {"content_hash": candidate.content_hash}},
    )

    assert diff.unreadable_keep == ("locked.md",)
    assert diff.removed == ()
    assert diff.replace_safe is False
    assert diff.replace_block_reason == "unreadable files"


def test_snapshot_diff_refuses_empty_or_disjoint_replacement_but_allows_initial_index(tmp_path):
    (tmp_path / "fresh.md").write_text("fresh", encoding="utf-8")
    fresh = _candidate(tmp_path, "fresh.md")
    known = {"old.md": {"content_hash": "a" * 64}}

    empty = SnapshotDiff.from_candidates([], known)
    disjoint = SnapshotDiff.from_candidates([fresh], known)
    initial = SnapshotDiff.from_candidates([fresh], {})

    assert empty.replace_safe is False
    assert empty.replace_block_reason == "empty snapshot"
    assert disjoint.replace_safe is False
    assert disjoint.replace_block_reason == "disjoint snapshot"
    assert initial.replace_safe is True


def test_windows_case_only_collision_is_fail_safe_not_added_removed(tmp_path):
    (tmp_path / "Readme.md").write_text("one", encoding="utf-8")
    (tmp_path / "README.md").write_text("two", encoding="utf-8")
    candidates = [_candidate(tmp_path, "Readme.md"), _candidate(tmp_path, "README.md")]
    diff = SnapshotDiff.from_candidates(candidates, {}, case_sensitive=False)

    assert diff.replace_safe is False
    assert diff.replace_block_reason == "case-insensitive path identity collision"


def test_scan_spec_normalizes_windows_paths_case_insensitively_and_enforces_envelope():
    spec = ScanSpec(root_dir=r"C:\Docs\Journal", patterns=("*.md",), max_depth=1)

    assert spec.normalize_relative_path(r"Sub\NOTE.MD") == "Sub/NOTE.MD"
    assert spec.matches(r"sub\note.md") is True
    assert spec.matches("sub/deep/note.md") is False
    assert spec.matches("other.txt") is False


def test_standalone_mcp_snapshot_mirror_uses_real_bytes_and_preserves_catalog_fields(tmp_path):
    """The published one-file MCP cannot import core, so verify its real-file mirror."""
    import importlib.util
    import sys

    plugin = Path(__file__).resolve().parents[1] / "mcp_plugin" / "retrieval_mcp.py"
    spec = importlib.util.spec_from_file_location("snapshot_plugin_real_file", plugin)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    path = tmp_path / "notes.md"
    path.write_text("stable journal body", encoding="utf-8")
    candidate = module._journal_file_candidate(str(path), str(tmp_path))
    unchanged, replace_safe, reason = module._journal_snapshot_diff(
        [candidate], {"notes.md": {
            "content_hash": candidate.content_hash,
            "size_bytes": candidate.size_bytes,
            "chunk_count": 3,
        }},
    )

    assert candidate.readable is True
    assert candidate.size_bytes == path.stat().st_size
    assert unchanged == {"notes.md"}
    assert replace_safe is True
    assert reason is None


def test_standalone_mcp_snapshot_envelope_does_not_cross_delete_other_patterns(tmp_path):
    import importlib.util
    import sys

    plugin = Path(__file__).resolve().parents[1] / "mcp_plugin" / "retrieval_mcp.py"
    spec = importlib.util.spec_from_file_location("snapshot_plugin_envelope", plugin)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    notes = tmp_path / "notes.md"
    notes.write_text("markdown current", encoding="utf-8")
    candidate = module._journal_file_candidate(str(notes), str(tmp_path))
    all_known = {
        "notes.md": {"content_hash": candidate.content_hash, "chunk_count": 2},
        "old.txt": {"content_hash": "a" * 64, "chunk_count": 1},
    }

    envelope = module._journal_known_in_envelope(all_known, ["*.md"], 0)
    unchanged, replace_safe, reason = module._journal_snapshot_diff([candidate], envelope)

    assert envelope == {"notes.md": all_known["notes.md"]}
    assert unchanged == {"notes.md"}
    assert replace_safe is True
    assert reason is None


def test_journal_known_in_envelope_excludes_dot_directory_rows():
    import importlib.util
    import sys
    from pathlib import Path

    plugin = Path(__file__).resolve().parents[1] / "mcp_plugin" / "retrieval_mcp.py"
    spec = importlib.util.spec_from_file_location("snapshot_plugin_dot_dirs", plugin)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    known = {
        ".conversations/claude/2026-11-04_x/001.md": {"content_hash": "a" * 64},
        "docs/.hidden/x.md": {"content_hash": "b" * 64},
        ".conversations\\codex\\y\\001.md": {"content_hash": "c" * 64},
        "notes/a.md": {"content_hash": "d" * 64},
        "notes/.dotfile.md": {"content_hash": "e" * 64},  # dot FILE, normal directory
    }

    for max_depth in (None, 8):
        envelope = module._journal_known_in_envelope(known, ["*.md"], max_depth)
        assert set(envelope) == {"notes/a.md", "notes/.dotfile.md"}
