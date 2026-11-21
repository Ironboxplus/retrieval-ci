"""Comparison rules for the serial final-API performance release gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_retrieval_release_api.py"
    spec = importlib.util.spec_from_file_location("retrieval_release_performance_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_performance_gate_rejects_tail_latency_or_throughput_regression():
    module = _module()
    baseline = {
        "schema_version": 2, "repetitions": 20, "concurrency": 1, "throughput_rps": 2.0,
        "cases": {"case": {"count": 20, "p50_ms": 100.0, "p95_ms": 120.0}},
    }
    same = {
        "schema_version": 2, "repetitions": 20, "concurrency": 1, "throughput_rps": 2.0,
        "cases": {"case": {"count": 20, "p50_ms": 104.0, "p95_ms": 125.0}},
    }
    assert module.compare_performance(baseline, same, max_ratio=1.10) == []

    slower = {**same, "cases": {"case": {"count": 20, "p50_ms": 111.0, "p95_ms": 125.0}}}
    assert "p50_ms ratio 1.110" in module.compare_performance(
        baseline, slower, max_ratio=1.10,
    )[0]

    lower_throughput = {**same, "throughput_rps": 1.7}
    assert "throughput ratio 0.850" in module.compare_performance(
        baseline, lower_throughput, max_ratio=1.10,
    )[0]



def test_performance_gate_requires_twenty_samples_per_case():
    module = _module()
    with pytest.raises(ValueError, match="20 repetitions"):
        module.benchmark("http://10.100.100.88:18080", repetitions=2, concurrency=1)


def test_serial_benchmark_stops_before_the_next_request_after_a_failure(monkeypatch):
    """Production validation must never build a request queue behind Retrieval."""
    module = _module()
    failed_case, slow_case = module.CASES[:2]
    monkeypatch.setattr(module, "CASES", (failed_case, slow_case))

    calls: list[str] = []
    def fake_request(_base_url, case, *, timeout_seconds):
        assert timeout_seconds == module.PERFORMANCE_REQUEST_TIMEOUT_SECONDS
        calls.append(case.name)
        if case is failed_case:
            if calls.count(case.name) == 1:
                return {"hits": [{"id": "warmup"}]}
            raise RuntimeError("upstream failed")
        return {"hits": [{"id": "warmup"}]}

    monkeypatch.setattr(module, "_request_json", fake_request)
    with pytest.raises(RuntimeError, match="upstream failed"):
        module.benchmark(
            "http://10.100.100.88:18080", repetitions=20, concurrency=1,
        )

    assert len(calls) == 3
    assert set(calls) == {failed_case.name, slow_case.name}


class TestTimeoutHasMarginOverMeasuredProductionLatency:
    """The 30s default had ~2x margin over a 14.5s warm `papers_reranker`
    (measured on .88, 2026-09-01), and 120 serial requests under that margin
    duly hit TimeoutError. The sibling capture tool uses 180s; the two walk the
    same six cases against the same host and must not disagree about how long
    an answer may take."""

    def test_default_matches_the_capture_tool(self):
        import importlib.util
        import inspect
        from pathlib import Path

        module = _module()
        capture_path = (Path(__file__).resolve().parents[1] / "scripts"
                        / "capture_retrieval_release_baseline.py")
        spec = importlib.util.spec_from_file_location("capture_under_test", capture_path)
        capture = importlib.util.module_from_spec(spec)
        # Register before exec: the module defines a dataclass, and resolving
        # its annotations looks the owning module up in sys.modules.
        sys.modules[spec.name] = capture
        spec.loader.exec_module(capture)

        capture_default = inspect.signature(
            capture._request_json).parameters["timeout_seconds"].default
        assert module.PERFORMANCE_REQUEST_TIMEOUT_SECONDS == capture_default

    def test_default_clears_the_slowest_measured_case_by_a_wide_margin(self):
        module = _module()
        slowest_measured_seconds = 15.5  # code_cuda_allocator cold, .88, 2026-09-01
        assert module.PERFORMANCE_REQUEST_TIMEOUT_SECONDS >= slowest_measured_seconds * 5

    def test_timeout_is_overridable_per_run(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module, "CASES", module.CASES[:1])
        seen: list[float] = []

        def fake_request(_base_url, case, *, timeout_seconds):
            seen.append(timeout_seconds)
            return {"hits": [{"id": "x"}]}

        monkeypatch.setattr(module, "_request_json", fake_request)
        module.benchmark("http://10.100.100.88:18080", repetitions=20,
                         concurrency=1, timeout_seconds=42)

        assert set(seen) == {42}


class TestAnAbortedRunStillReportsWhatItMeasured:
    """2026-09-01: the gate ran for ~20 minutes, hit one TimeoutError on request
    ~120, and wrote nothing at all — the operator was left with an exception
    type and no numbers.

    Stopping at the first failure is deliberate and stays (a production gate
    must never queue requests behind user traffic). Throwing away the
    measurements taken before it is the part that has no defence.
    """

    def test_partial_samples_survive_the_abort(self, monkeypatch):
        module = _module()
        good, doomed = module.CASES[:2]
        monkeypatch.setattr(module, "CASES", (good, doomed))
        calls: list[str] = []

        def fake_request(_base_url, case, *, timeout_seconds):
            calls.append(case.name)
            # 2 warmups + 5 full rounds, then the 6th `doomed` request fails.
            if case is doomed and calls.count(doomed.name) > 6:
                raise RuntimeError("timed out")
            return {"hits": [{"id": "x"}]}

        monkeypatch.setattr(module, "_request_json", fake_request)
        with pytest.raises(module.BenchmarkAborted) as caught:
            module.benchmark("http://10.100.100.88:18080", repetitions=20, concurrency=1)

        partial = caught.value.partial
        assert partial["cases"][good.name]["count"] == 6
        assert partial["cases"][doomed.name]["count"] == 5
        assert partial["cases"][good.name]["p50_ms"] >= 0
        assert partial["completed_requests"] == 11
        assert doomed.name in str(caught.value)
        assert "timed out" in str(caught.value)

    def test_an_aborted_capture_writes_no_baseline(self, monkeypatch, tmp_path, capsys):
        """A partial run must never become a baseline others compare against."""
        module = _module()
        monkeypatch.setattr(module, "CASES", module.CASES[:1])
        monkeypatch.setattr(
            module, "_request_json",
            lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("upstream down")))
        baseline = tmp_path / "baseline.json"
        monkeypatch.setattr(
            sys, "argv",
            ["benchmark", "capture", "--base-url", "http://10.100.100.88:18080",
             "--baseline", str(baseline)])

        assert module.main() == 1

        assert not baseline.exists()
        reported = capsys.readouterr().out
        assert "upstream down" in reported, "the operator must learn what failed"

    def test_a_warmup_failure_is_named_as_such(self, monkeypatch):
        """Cold-load failures and steady-state failures mean different things."""
        module = _module()
        monkeypatch.setattr(module, "CASES", module.CASES[:1])
        monkeypatch.setattr(
            module, "_request_json",
            lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("cold start blew up")))

        with pytest.raises(module.BenchmarkAborted, match="warmup"):
            module.benchmark("http://10.100.100.88:18080", repetitions=20, concurrency=1)

    def test_a_clean_run_is_unaffected(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module, "CASES", module.CASES[:2])
        monkeypatch.setattr(
            module, "_request_json",
            lambda _b, _c, *, timeout_seconds: {"hits": [{"id": "x"}], "results": [{"id": "x"}]})

        result = module.benchmark("http://10.100.100.88:18080", repetitions=20, concurrency=1)

        assert result["completed_requests"] == 40
        assert all(c["count"] == 20 for c in result["cases"].values())


def test_performance_gate_rejects_a_concurrent_production_run(monkeypatch):
    module = _module()
    monkeypatch.setattr(
        module,
        "_request_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not request")),
    )
    with pytest.raises(ValueError, match="concurrency one"):
        module.benchmark("http://10.100.100.88:18080", repetitions=20, concurrency=2)
