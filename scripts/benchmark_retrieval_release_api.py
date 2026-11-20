"""Measure serial final-API latency before and after a `.88` rollout.

This only calls the three read-only search surfaces through the fixed internal
compute door. It stores aggregate durations and statuses, never responses.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from capture_retrieval_release_baseline import CASES, _request_json, validate_release_target


# Twenty independent observations per final API case are the smallest sample
# that makes the p95 gate interpretable on this shared machine. Production runs
# them one at a time so the gate never creates a backlog behind user traffic.
# A p99 claim would need substantially more than this, so the release artifact
# deliberately reports and gates only p50/p95 plus throughput.
MINIMUM_REPETITIONS = 20

# Must match capture_retrieval_release_baseline._request_json's default: the two
# tools walk the same six cases against the same host, so disagreeing about how
# long an answer may take makes one of them lie. The former 30s gave only ~2x
# margin over a 14.5s warm `papers_reranker` (measured on .88, 2026-09-01), and
# 120 serial requests under that margin duly hit TimeoutError.
PERFORMANCE_REQUEST_TIMEOUT_SECONDS = 180


class BenchmarkAborted(RuntimeError):
    """A run that stopped at the first failure, carrying what it had measured.

    Stopping immediately is deliberate — a production gate must never build a
    request queue behind user traffic. Discarding the samples taken before the
    failure is not: on 2026-09-01 that turned ~20 minutes of measurement into a
    bare TimeoutError and no numbers at all.
    """

    def __init__(self, message: str, *, partial: dict[str, Any]) -> None:
        super().__init__(message)
        self.partial = partial


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _summarize(
    samples: dict[str, list[float]],
    *,
    validated_url: str,
    repetitions: int,
    concurrency: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    completed = sum(len(v) for v in samples.values())
    return {
        "schema_version": 2,
        "base_url": validated_url,
        "repetitions": repetitions,
        "concurrency": concurrency,
        "completed_requests": completed,
        "warmup_requests": len(samples),
        "wall_seconds": round(elapsed_seconds, 3),
        "throughput_rps": round(completed / elapsed_seconds, 4) if elapsed_seconds else 0.0,
        "cases": {
            name: {
                "count": len(values),
                "p50_ms": round(_percentile(values, 0.50), 3) if values else 0.0,
                "p95_ms": round(_percentile(values, 0.95), 3) if values else 0.0,
            }
            for name, values in sorted(samples.items())
        },
    }


def benchmark(
    base_url: str,
    *,
    repetitions: int,
    concurrency: int,
    timeout_seconds: float = PERFORMANCE_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    validated_url = validate_release_target(base_url)
    if repetitions < MINIMUM_REPETITIONS:
        raise ValueError(
            f"benchmark requires at least {MINIMUM_REPETITIONS} repetitions",
        )
    if concurrency != 1:
        raise ValueError("production benchmark requires concurrency one")

    def one(case):  # noqa: ANN001
        started = time.perf_counter()
        response = _request_json(
            validated_url,
            case,
            timeout_seconds=timeout_seconds,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        hits = response.get(case.result_key)
        if not isinstance(hits, list) or not hits:
            raise RuntimeError(f"{case.name}: final API returned no result list")
        return case.name, elapsed_ms

    samples: dict[str, list[float]] = {case.name: [] for case in CASES}

    def abort(message: str, *, elapsed_seconds: float) -> BenchmarkAborted:
        return BenchmarkAborted(message, partial=_summarize(
            samples, validated_url=validated_url, repetitions=repetitions,
            concurrency=concurrency, elapsed_seconds=elapsed_seconds,
        ))

    # Warm every exact final API case before timing. It is intentionally outside
    # the measurement window so a post-restart cold model load cannot masquerade
    # as an allocator-maintenance latency regression. A failure here is reported
    # as a warmup failure: a cold-load blowup and a steady-state one mean
    # different things.
    for case in CASES:
        try:
            one(case)
        except Exception as exc:  # noqa: BLE001 — reported, then re-raised
            raise abort(f"warmup failed on {case.name}: {exc}", elapsed_seconds=0.0) from exc

    started = time.perf_counter()
    for _ in range(repetitions):
        for case in CASES:
            try:
                name, elapsed_ms = one(case)
            except Exception as exc:  # noqa: BLE001 — reported, then re-raised
                elapsed = time.perf_counter() - started
                raise abort(
                    f"aborted during {case.name} after "
                    f"{sum(len(v) for v in samples.values())} timed requests: {exc}",
                    elapsed_seconds=elapsed,
                ) from exc
            samples[name].append(elapsed_ms)
    elapsed_seconds = time.perf_counter() - started
    return _summarize(
        samples, validated_url=validated_url, repetitions=repetitions,
        concurrency=concurrency, elapsed_seconds=elapsed_seconds,
    )


def compare_performance(expected: dict[str, Any], actual: dict[str, Any], *, max_ratio: float) -> list[str]:
    if expected.get("schema_version") != 2:
        return ["benchmark schema_version is unsupported"]
    if expected.get("repetitions") != actual.get("repetitions") or expected.get("concurrency") != actual.get("concurrency"):
        return ["benchmark shape differs from the pre-deploy baseline"]
    failures: list[str] = []
    expected_cases = expected.get("cases", {})
    actual_cases = actual.get("cases", {})
    if set(expected_cases) != set(actual_cases):
        return ["benchmark API case set changed during rollout"]
    for name in sorted(expected_cases):
        before = expected_cases[name]
        after = actual_cases[name]
        if before.get("count") != expected["repetitions"] or after.get("count") != actual["repetitions"]:
            failures.append(f"{name}: benchmark sample count is incomplete")
            continue
        for metric in ("p50_ms", "p95_ms"):
            ratio = after[metric] / before[metric]
            if ratio > max_ratio:
                failures.append(f"{name}: {metric} ratio {ratio:.3f} exceeds {max_ratio:.3f}")
    throughput_ratio = actual["throughput_rps"] / expected["throughput_rps"]
    if throughput_ratio < 1 / max_ratio:
        failures.append(
            f"throughput ratio {throughput_ratio:.3f} is below {1 / max_ratio:.3f}",
        )
    return failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "compare"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=MINIMUM_REPETITIONS)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-ratio", type=float, default=1.10)
    parser.add_argument(
        "--request-timeout", type=float, default=PERFORMANCE_REQUEST_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_ratio < 1:
        raise SystemExit("--max-ratio must be at least 1")
    try:
        actual = benchmark(
            args.base_url, repetitions=args.repetitions, concurrency=args.concurrency,
            timeout_seconds=args.request_timeout,
        )
    except BenchmarkAborted as exc:
        # Never write a baseline from a partial run — others compare against it.
        # But print what was measured: an operator who waited out 120 serial
        # requests should not be left with only an exception type.
        print(json.dumps(
            {"aborted": str(exc), "partial": exc.partial}, ensure_ascii=False))
        return 1
    if args.mode == "capture":
        if args.baseline.exists():
            raise SystemExit(f"refusing to overwrite existing baseline: {args.baseline}")
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"captured": actual}, ensure_ascii=False))
        return 0
    try:
        expected = json.loads(args.baseline.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read baseline: {exc}") from exc
    failures = compare_performance(expected, actual, max_ratio=args.max_ratio)
    print(json.dumps({"actual": actual, "failed": failures}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
