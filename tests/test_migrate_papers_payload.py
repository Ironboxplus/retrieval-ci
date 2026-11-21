"""TDD: scripts/migrate_papers_payload.py — the isolated-collection migration
tool for the D2 (doc_id -> str) / D3 (drop data_source from paper/review)
fixes (docs/papers_payload_contract_plan.md 3.3).

SAFETY under test:
  - dry-run (the default) must never call set_payload/delete_payload.
  - --collection has no default and must reject "papers" outright — the
    production migration is a separate, operator-approved step.
  - --execute batches writes with wait=True + bounded retry, and validates
    each batch against a server round-trip (a write that "succeeds" but
    silently no-ops must be caught, not trusted).
  - checkpoint file supports resuming a crashed scan.

Uses a FAKE Qdrant client only. Never connects to a real Qdrant instance, so
it covers control flow — scan bookkeeping, retry, checkpoint resume, batch
accounting — and nothing about what Qdrant actually does with the writes.
The claim the migration exists to make (an int doc_id rewritten to a str
enters the KEYWORD index) is unfalsifiable here and is measured against a real
Qdrant in tests/test_migrate_papers_payload_real_qdrant.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from migrate_papers_payload import (
    PRODUCTION_ACK,
    Checkpoint,
    build_arg_parser,
    migrate,
    validate_args,
)


class _Point:
    def __init__(self, id_, payload):
        self.id = id_
        self.payload = payload


class _FakeClient:
    """In-memory Qdrant stand-in: scroll/batch_update_points/delete_payload/retrieve."""

    def __init__(self, points):
        self._points = {p.id: p for p in points}
        self._order = [p.id for p in points]
        self.batch_update_calls = []
        self.delete_payload_calls = []
        self.retrieve_calls = []
        self.fail_next_batch_update = 0
        self.fail_next_delete_payload = 0

    def scroll(self, collection, limit, offset=None, with_payload=True, with_vectors=False):
        start = 0 if offset is None else self._order.index(offset)
        ids = self._order[start:start + limit]
        page = [self._points[i] for i in ids]
        next_offset = self._order[start + limit] if start + limit < len(self._order) else None
        return page, next_offset

    def batch_update_points(self, collection, update_operations, wait=True):
        """Apply real qdrant SetPayloadOperation objects, as the script builds them."""
        if self.fail_next_batch_update > 0:
            self.fail_next_batch_update -= 1
            raise RuntimeError("transient failure")
        self.batch_update_calls.append(
            [(dict(op.set_payload.payload), list(op.set_payload.points))
             for op in update_operations])
        for op in update_operations:
            for pid in op.set_payload.points:
                self._points[pid].payload.update(op.set_payload.payload)

    def delete_payload(self, collection, keys, points, wait=True):
        if self.fail_next_delete_payload > 0:
            self.fail_next_delete_payload -= 1
            raise RuntimeError("transient failure")
        self.delete_payload_calls.append((list(keys), list(points)))
        for pid in points:
            for k in keys:
                self._points[pid].payload.pop(k, None)

    def retrieve(self, collection, ids, with_payload=True):
        self.retrieve_calls.append((list(ids), with_payload))
        if isinstance(with_payload, (list, tuple)):
            keys = set(with_payload)
            return [_Point(i, {k: v for k, v in self._points[i].payload.items() if k in keys})
                    for i in ids]
        return [self._points[i] for i in ids]


def _sample_points():
    return [
        _Point(1, {"source_type": "paper", "doc_id": 100, "paper_id": "p1",
                   "level": 1, "venue": "AAAI", "year": 2026, "venue_tier": "CCF-A",
                   "content": "c1"}),
        _Point(2, {"source_type": "paper", "doc_id": "101", "paper_id": "p2",
                   "level": 2, "content": "c2"}),
        _Point(3, {"source_type": "review", "doc_id": 100, "paper_id": "p1",
                   "level": 1, "content": "review", "data_source": "system"}),
        _Point(4, {"source_type": "upload", "doc_id": "9", "content": "c4",
                   "data_source": "user", "original_filename": "x.pdf"}),
        _Point(5, {"source_type": "paper", "level": 1, "content": "no doc id"}),
    ]


class TestScanReport:
    def test_dry_run_does_not_write(self):
        client = _FakeClient(_sample_points())
        result = migrate(client, "isolated_test_collection", execute=False)
        assert client.batch_update_calls == []
        assert client.delete_payload_calls == []
        assert result["applied"]["doc_id_points_updated"] == 2  # points 1 and 3 (int doc_id)
        assert result["applied"]["batches_applied"] == 0

    def test_report_counts_by_source_type(self):
        client = _FakeClient(_sample_points())
        result = migrate(client, "isolated_test_collection", execute=False)
        assert result["report"]["by_source_type"] == {"paper": 3, "review": 1, "upload": 1}

    def test_report_doc_id_missing_and_needs_str(self):
        client = _FakeClient(_sample_points())
        result = migrate(client, "isolated_test_collection", execute=False)
        assert result["report"]["doc_id_missing"] == 1  # point 5
        assert result["report"]["doc_id_needs_str"] == 2  # points 1, 3 (int)

    def test_report_data_source_to_remove_only_counts_paper_review(self):
        client = _FakeClient(_sample_points())
        result = migrate(client, "isolated_test_collection", execute=False)
        # point 3 (review) illegitimately carries data_source; point 4
        # (upload) legitimately carries it and must NOT be counted.
        assert result["report"]["data_source_to_remove"] == 1

    def test_plan_matches_report(self):
        client = _FakeClient(_sample_points())
        result = migrate(client, "isolated_test_collection", execute=False)
        assert result["plan"] == {"doc_id_str_conversions": 2, "data_source_removals": 1}


class TestExecute:
    def test_execute_converts_int_doc_id_to_str(self):
        client = _FakeClient(_sample_points())
        migrate(client, "isolated_test_collection", execute=True, apply_batch=1)
        assert client._points[1].payload["doc_id"] == "100"
        assert client._points[3].payload["doc_id"] == "100"
        assert client._points[2].payload["doc_id"] == "101"  # untouched, already str

    def test_execute_removes_data_source_from_review_not_upload(self):
        client = _FakeClient(_sample_points())
        migrate(client, "isolated_test_collection", execute=True, apply_batch=1)
        assert "data_source" not in client._points[3].payload
        assert client._points[4].payload["data_source"] == "user"

    def test_dry_run_leaves_collection_untouched(self):
        client = _FakeClient(_sample_points())
        before = {pid: dict(p.payload) for pid, p in client._points.items()}
        migrate(client, "isolated_test_collection", execute=False)
        after = {pid: dict(p.payload) for pid, p in client._points.items()}
        assert before == after


class TestRetryAndBatching:
    def test_retries_transient_batch_update_failure(self):
        client = _FakeClient(_sample_points())
        client.fail_next_batch_update = 1
        migrate(client, "isolated_test_collection", execute=True, apply_batch=1,
                sleep=lambda *_a: None)
        assert client._points[1].payload["doc_id"] == "100"

    def test_gives_up_after_max_retries(self):
        client = _FakeClient(_sample_points())
        client.fail_next_batch_update = 99
        with pytest.raises(RuntimeError):
            migrate(client, "isolated_test_collection", execute=True, apply_batch=1,
                    max_retries=2, sleep=lambda *_a: None)

    def test_one_request_carries_many_distinct_doc_id_values(self):
        """The volume defect: a flush threshold counted per doc_id VALUE only
        fires once one value has accumulated apply_batch points, which on
        production's spread never happens, so every distinct value cost its own
        round trip. Ten distinct values, one apply_batch of 500 -> one request
        holding ten operations, not ten requests."""
        client = _FakeClient([_Point(i, {"source_type": "paper", "doc_id": 200 + i})
                              for i in range(10)])

        migrate(client, "isolated_test_collection", execute=True, apply_batch=500)

        assert len(client.batch_update_calls) == 1
        assert len(client.batch_update_calls[0]) == 10, "ten SetPayload ops in one call"
        assert all(p.payload["doc_id"] == str(200 + i)
                   for i, p in enumerate(client._points.values()))


class TestValidation:
    """A write call that raises no exception but silently no-ops must still
    be caught — this is the exact W4 failure mode (wait=False silently
    dropping writes) that AGENTS.md's wait=True rule exists to prevent;
    validation is the belt to wait=True's suspenders."""

    def test_raises_when_batch_update_silently_no_ops(self):
        client = _FakeClient(_sample_points())
        client.batch_update_points = lambda collection, update_operations, wait=True: None

        with pytest.raises(RuntimeError, match="validation failed"):
            migrate(client, "isolated_test_collection", execute=True, apply_batch=1)

    def test_raises_when_delete_payload_silently_no_ops(self):
        client = _FakeClient(_sample_points())
        client.delete_payload = lambda collection, keys, points, wait=True: (
            client.delete_payload_calls.append((list(keys), list(points)))
        )

        with pytest.raises(RuntimeError, match="validation failed"):
            migrate(client, "isolated_test_collection", execute=True, apply_batch=1)


class TestCheckpointResume:
    def test_full_pass_clears_checkpoint(self, tmp_path):
        ckpt_path = tmp_path / "ckpt.json"
        client = _FakeClient(_sample_points())
        checkpoint = Checkpoint(ckpt_path)
        migrate(client, "isolated_test_collection", execute=False,
                scan_batch=2, checkpoint=checkpoint)
        assert not ckpt_path.exists()

    def test_checkpoint_persists_mid_scan_offset_on_crash(self, tmp_path):
        ckpt_path = tmp_path / "ckpt.json"
        client = _FakeClient(_sample_points())
        checkpoint = Checkpoint(ckpt_path)

        calls = {"n": 0}
        real_scroll = client.scroll

        def flaky_scroll(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated crash")
            return real_scroll(*a, **k)

        client.scroll = flaky_scroll
        with pytest.raises(RuntimeError, match="simulated crash"):
            migrate(client, "isolated_test_collection", execute=False,
                    scan_batch=2, checkpoint=checkpoint)

        assert ckpt_path.exists()
        assert checkpoint.load() == 3  # first page = ids [1, 2], next offset = 3

    def test_resuming_from_checkpoint_skips_already_scanned_pages(self, tmp_path):
        ckpt_path = tmp_path / "ckpt.json"
        checkpoint = Checkpoint(ckpt_path)
        checkpoint.save(3)  # pretend page [1, 2] was already scanned

        client = _FakeClient(_sample_points())
        seen_ids = []
        real_scroll = client.scroll

        def recording_scroll(collection, limit, offset=None, **kw):
            page, next_offset = real_scroll(collection, limit, offset=offset, **kw)
            seen_ids.extend(p.id for p in page)
            return page, next_offset

        client.scroll = recording_scroll
        migrate(client, "isolated_test_collection", execute=False,
                scan_batch=2, checkpoint=checkpoint)

        assert 1 not in seen_ids and 2 not in seen_ids
        assert seen_ids == [3, 4, 5]


class TestOperationSelection:
    """D2 (doc_id -> str) and D3 (drop data_source) are independent, and only
    D3 removes information. Bundling an irreversible field deletion into a
    doc_id repair run is scope the operator did not ask for, so each has to be
    selectable on its own."""

    def test_doc_id_only_leaves_data_source_alone(self):
        client = _FakeClient(_sample_points())

        result = migrate(client, "isolated_test_collection", execute=True,
                         operations=("doc_id",))

        assert client._points[1].payload["doc_id"] == "100"
        assert client.delete_payload_calls == []
        assert client._points[3].payload["data_source"] == "system"
        assert result["applied"]["data_source_points_updated"] == 0
        # The scan still REPORTS what D3 would do, so one dry run sizes both.
        assert result["plan"]["data_source_removals"] == 1

    def test_data_source_only_leaves_doc_id_alone(self):
        client = _FakeClient(_sample_points())

        result = migrate(client, "isolated_test_collection", execute=True,
                         operations=("data_source",))

        assert client.batch_update_calls == []
        assert client._points[1].payload["doc_id"] == 100, "int doc_id untouched"
        assert "data_source" not in client._points[3].payload
        assert result["applied"]["doc_id_points_updated"] == 0
        assert result["plan"]["doc_id_str_conversions"] == 2

    def test_default_runs_both(self):
        client = _FakeClient(_sample_points())
        migrate(client, "isolated_test_collection", execute=True)
        assert client._points[1].payload["doc_id"] == "100"
        assert "data_source" not in client._points[3].payload

    def test_unknown_operation_is_rejected(self):
        client = _FakeClient(_sample_points())
        with pytest.raises(ValueError, match="unknown operation"):
            migrate(client, "isolated_test_collection", operations=("doc_ids",))

    def test_cli_defaults_to_both_operations(self):
        args = build_arg_parser().parse_args(["--collection", "iso"])
        assert sorted(args.operations) == ["data_source", "doc_id"]

    def test_cli_can_select_one_operation(self):
        args = build_arg_parser().parse_args(
            ["--collection", "iso", "--operations", "doc_id"])
        assert args.operations == ["doc_id"]


class TestVerificationReadsOnlyTheKeyItChecks:
    """Every batch is read back to prove the write stuck. A `papers` point
    carries 25 payload keys including `content` and `abstract`, so pulling the
    whole payload back would move tens of GB over a 6.9M-point run to compare
    one field."""

    def test_doc_id_verification_requests_only_doc_id(self):
        client = _FakeClient(_sample_points())

        migrate(client, "isolated_test_collection", execute=True,
                operations=("doc_id",), apply_batch=500)

        assert client.retrieve_calls, "the write must be verified at all"
        for ids, with_payload in client.retrieve_calls:
            assert with_payload == ["doc_id"], (
                f"verification pulled {with_payload!r} instead of just doc_id")

    def test_data_source_verification_requests_only_data_source(self):
        client = _FakeClient(_sample_points())

        migrate(client, "isolated_test_collection", execute=True,
                operations=("data_source",), apply_batch=500)

        assert client.retrieve_calls
        for ids, with_payload in client.retrieve_calls:
            assert with_payload == ["data_source"]

    def test_a_narrow_read_still_catches_a_write_that_did_not_stick(self):
        """The narrowing must not weaken the check it exists to perform."""
        client = _FakeClient(_sample_points())
        client.batch_update_points = lambda collection, update_operations, wait=True: None

        with pytest.raises(RuntimeError, match="validation failed"):
            migrate(client, "isolated_test_collection", execute=True,
                    operations=("doc_id",))


class TestProgressReporting:
    """A production --execute pass rewrites ~6.9M points over hours. With no
    output until it finishes, a run that is progressing and a run that is wedged
    look identical, and the only way to tell is to kill it."""

    def test_progress_is_reported_during_the_scan(self):
        client = _FakeClient([_Point(i, {"source_type": "paper", "doc_id": i})
                              for i in range(50)])
        lines = []

        migrate(client, "isolated_test_collection", execute=True, scan_batch=10,
                progress_every=10, log=lines.append)

        assert len(lines) >= 5, f"expected a line per 10 points scanned, got {lines}"
        assert "50" in lines[-1], f"last line must report the running total: {lines[-1]}"

    def test_progress_is_silent_by_default(self):
        client = _FakeClient(_sample_points())
        # No log= argument: must not crash and must not print anything itself.
        migrate(client, "isolated_test_collection", execute=False)

    def test_progress_reports_conversions_not_just_points_scanned(self):
        client = _FakeClient([_Point(i, {"source_type": "paper",
                                         "doc_id": i if i % 2 else str(i)})
                              for i in range(20)])
        lines = []

        migrate(client, "isolated_test_collection", execute=True, scan_batch=5,
                progress_every=5, log=lines.append)

        assert any("10" in line for line in lines), (
            f"the 10 int doc_ids converted must be visible in progress: {lines}")


class TestCollectionSafety:
    """The migration writes set_payload/delete_payload across an entire
    collection — --collection must be an explicit, deliberate choice."""

    def test_collection_argument_is_required(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_writing_to_papers_requires_an_explicit_acknowledgement(self):
        """`papers` is the 13.5M-point production collection. A slipped default
        or a copy-pasted invocation must not be able to write to it."""
        args = build_arg_parser().parse_args(
            ["--collection", "papers", "--execute"])
        with pytest.raises(SystemExit, match="confirm-production"):
            validate_args(args)

    def test_papers_writes_are_allowed_with_the_acknowledgement(self):
        args = build_arg_parser().parse_args(
            ["--collection", "papers", "--execute",
             "--confirm-production", PRODUCTION_ACK])
        validate_args(args)  # must not raise

    def test_a_wrong_acknowledgement_token_still_blocks(self):
        args = build_arg_parser().parse_args(
            ["--collection", "papers", "--execute", "--confirm-production", "yes"])
        with pytest.raises(SystemExit, match="confirm-production"):
            validate_args(args)

    def test_dry_run_against_papers_needs_no_acknowledgement(self):
        """Read-only sizing of production must stay easy — the gate is on the
        write, not on looking."""
        args = build_arg_parser().parse_args(["--collection", "papers"])
        validate_args(args)  # must not raise

    def test_acknowledgement_does_not_leak_to_execute_without_it(self):
        args = build_arg_parser().parse_args(["--collection", "papers", "--execute"])
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_isolated_collection_name_is_accepted(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--collection", "papers_migration_check"])
        assert args.collection == "papers_migration_check"
        assert args.execute is False  # dry-run by default
