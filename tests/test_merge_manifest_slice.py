"""TDD: merge_manifest_slice.py — fold a shipped slice into a host's working manifest.

The invariant that matters on `.88`: the working manifest holds OTHER venues
(ACL 2026, and historically a 136 MB full corpus). Merging a KDD/SIGIR payload
must replace only the paper_ids in the payload and leave every other record
byte-identical — including records whose paths point at hosts this box cannot
read, which a full re-sync would otherwise NULL.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sync" / "merge_manifest_slice.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("merge_manifest_slice", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["merge_manifest_slice"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def _rec(pid, conf, year, **extra):
    r = {"paper_id": pid, "conference": conf, "year": year, "markdown_path": None}
    r.update(extra)
    return r


def test_replaces_by_paper_id_and_appends_the_rest():
    base = [
        _rec("acl-2026-long-1", "ACL", 2026, markdown_path="/srv/acl/1.md"),
        _rec("sigir-2026-canary", "SIGIR", 2026, markdown_path="/old/canary.md"),
    ]
    payload = [
        _rec("sigir-2026-canary", "SIGIR", 2026, markdown_path="/new/canary.md", venue_track="research"),
        _rec("kdd-2026-a", "KDD", 2026, markdown_path="/new/kdd-a.md", venue_track="research"),
    ]

    counts = mod.merge_slice(base, payload)

    assert counts["replaced"] == 1
    assert counts["appended"] == 1
    assert counts["total_after"] == 3

    by_id = {r["paper_id"]: r for r in base}
    assert by_id["sigir-2026-canary"]["markdown_path"] == "/new/canary.md"
    assert by_id["sigir-2026-canary"]["venue_track"] == "research"
    assert by_id["kdd-2026-a"]["markdown_path"] == "/new/kdd-a.md"


def test_records_outside_the_payload_are_untouched():
    """The ACL 2026 slice and any unreachable-host paths must survive verbatim."""
    untouched = _rec(
        "acl-2026-long-1", "ACL", 2026,
        markdown_path="//10.100.100.9/lsb/retrival/papers/markdown/acl/2026/x.md",
    )
    base = [untouched, _rec("kdd-2026-a", "KDD", 2026)]
    before = json.dumps(untouched, sort_keys=True)

    mod.merge_slice(base, [_rec("kdd-2026-a", "KDD", 2026, markdown_path="/new/a.md")])

    assert json.dumps(base[0], sort_keys=True) == before


def test_merge_is_idempotent():
    base = [_rec("kdd-2026-a", "KDD", 2026)]
    payload = [_rec("kdd-2026-a", "KDD", 2026, markdown_path="/new/a.md")]

    first = mod.merge_slice(base, payload)
    second = mod.merge_slice(base, payload)

    assert first["appended"] == 0 and first["replaced"] == 1
    assert second["replaced"] == 1 and second["appended"] == 0
    assert len(base) == 1


def test_record_without_paper_id_is_skipped_not_appended():
    """An unkeyed record cannot be merged idempotently — appending it would
    duplicate on every re-run."""
    base = [_rec("kdd-2026-a", "KDD", 2026)]
    counts = mod.merge_slice(base, [{"conference": "KDD", "year": 2026}])
    assert counts["skipped_no_paper_id"] == 1
    assert counts["appended"] == 0
    assert len(base) == 1


def test_cli_backs_up_before_writing(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "paper_manifest.json"
    original = [_rec("acl-2026-1", "ACL", 2026, markdown_path="/keep/me.md")]
    manifest.write_text(json.dumps(original), encoding="utf-8")

    payload = tmp_path / "slice.json"
    payload.write_text(json.dumps([_rec("kdd-2026-a", "KDD", 2026, markdown_path="/new/a.md")]), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "merge_manifest_slice.py", "--payload", str(payload), "--manifest-path", str(manifest),
    ])
    assert mod.main() == 0

    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(written) == 2
    backups = list(tmp_path.glob("*.bak.merge.*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == original


def test_cli_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "paper_manifest.json"
    original = json.dumps([_rec("acl-2026-1", "ACL", 2026)])
    manifest.write_text(original, encoding="utf-8")

    payload = tmp_path / "slice.json"
    payload.write_text(json.dumps([_rec("kdd-2026-a", "KDD", 2026)]), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "merge_manifest_slice.py", "--payload", str(payload),
        "--manifest-path", str(manifest), "--dry-run",
    ])
    assert mod.main() == 0

    assert manifest.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak*"))
    assert "appended=1" in capsys.readouterr().out
