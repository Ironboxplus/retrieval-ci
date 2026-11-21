"""No-write, real-binary gate for `.88` CUDA reranker maintenance.

This intentionally validates the scorer primitive only. The companion HTTP
baseline tool compares the final Code/Public Code/Papers API responses before
and after deployment. Keeping these concerns separate avoids mistaking an
already-reranked paper response for a raw-candidate scorer test.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.gpu_memory import CudaAllocatorSnapshot, capture_cuda_allocator_snapshot
from core.retrieval.reranker import BgeReranker, RerankerCudaTelemetry


RELEASE_BASE_URL = "http://10.100.100.88:18080"
RELEASE_BATCH_SIZE = 128
# These are deliberately fixed rather than canary-tunable. They describe the
# one shared-card rollout approved for compute .88; a command run elsewhere
# must fail before importing torch or constructing a CUDA context.
RELEASE_COMPUTE_IP = "10.100.100.88"
RELEASE_GPU_INDEX = 0
RELEASE_GPU_UUID = "GPU-4ae88dce-1545-3658-b2b6-f08328c7d366"
RELEASE_GPU_PCI_BUS_ID = "00000000:01:00.0"
RELEASE_GPU_TOTAL_MIB = 12282
# The deployment uses this exact value. The combined real-model canary showed
# complete score/order equivalence, bounded cache/peak use, and only a
# user-accepted 0.7--2.4% managed latency increase at this cap.
RELEASE_MEMORY_FRACTION = 0.25
RELEASE_GPU_HEADROOM_MIB = 256
RELEASE_POST_TRIM_IDLE_CACHE_MIB = 64
RELEASE_MAX_PEACH_MEMORY_GROWTH_MIB = 256


@dataclass(frozen=True)
class CandidateCase:
    name: str
    path: str
    body: dict[str, Any]
    texts: Callable[[dict[str, Any]], list[str]]


@dataclass(frozen=True)
class GpuPreflight:
    """The exact physical device admitted to the `.88` canary."""

    index: int
    uuid: str
    pci_bus_id: str
    total_mib: int
    free_mib: int


def _code_texts(response: dict[str, Any]) -> list[str]:
    return [str(hit["code"]) for hit in response.get("hits", []) if hit.get("code")]


def _public_code_texts(response: dict[str, Any]) -> list[str]:
    return [str(hit["content"]) for hit in response.get("hits", []) if hit.get("content")]


def _paper_texts(response: dict[str, Any]) -> list[str]:
    return [str(hit["content"]) for hit in response.get("results", []) if hit.get("content")]


# Code/Public Code expose an un-reranked read mode. Papers do not; their final
# API equivalence is therefore checked by capture_retrieval_release_baseline.py.
CASES = (
    CandidateCase(
        name="code_raw",
        path="/api/code/search",
        body={
            "query": "CUDA allocator reserved memory cache and OOM retry handling",
            "top_k": 50,
            "mode": "hybrid",
            "rerank": False,
        },
        texts=_code_texts,
    ),
    CandidateCase(
        name="public_code_raw",
        path="/api/public-code/search",
        body={
            "query": "WebRTC peer connection ICE transport state error recovery",
            "top_k": 50,
            "mode": "hybrid",
            "rerank": False,
        },
        texts=_public_code_texts,
    ),
    CandidateCase(
        name="papers_final_sample",
        path="/api/search",
        body={
            "query": "cross encoder reranking for retrieval augmented generation long context question answering",
            "top_k": 50,
            "method": "hybrid",
        },
        texts=_paper_texts,
    ),
)


class _RejectRedirect(HTTPRedirectHandler):
    """A release verifier must never forward a request to a different origin."""

    def redirect_request(self, request, fp, code, msg, headers, new_url):  # noqa: ANN001
        raise RuntimeError(f"unexpected redirect while calling {request.full_url}: HTTP {code}")


def validate_release_target(base_url: str) -> str:
    """Allow only the private `.88` compute door; no credentials ever leave it."""
    parsed = urlparse(base_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("release canary base URL has an invalid port") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "10.100.100.88"
        or port != 18080
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "release canary only permits http://10.100.100.88:18080; "
            "it does not accept arbitrary hosts or credentials"
        )
    return RELEASE_BASE_URL


def _request_json(base_url: str, case: CandidateCase) -> dict[str, Any]:
    request = Request(
        f"{base_url}{case.path}",
        data=json.dumps(case.body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # This fixed private address must remain direct even if a shell inherited a
    # generic HTTP proxy for unrelated package/network work.
    opener = build_opener(_RejectRedirect(), ProxyHandler({}))
    try:
        with opener.open(request, timeout=120) as response:  # noqa: S310 - fixed private target
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"{case.name} API returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"{case.name} API request failed: {exc.reason}") from exc


def _snapshot_dict(snapshot: CudaAllocatorSnapshot | None) -> dict[str, int] | None:
    return dataclasses.asdict(snapshot) if snapshot is not None else None


def _telemetry_dict(telemetry: RerankerCudaTelemetry | None) -> dict[str, Any] | None:
    if telemetry is None:
        return None
    return {
        "before": _snapshot_dict(telemetry.before),
        "after": _snapshot_dict(telemetry.after),
        "post_trim": _snapshot_dict(telemetry.post_trim),
        "trim_reason": telemetry.trim_reason,
        "trim_executed": telemetry.trim_executed,
    }


def _validate_scores(scores: list[float], candidate_count: int, *, arm: str) -> str | None:
    if len(scores) != candidate_count:
        return f"{arm}: expected {candidate_count} scores, received {len(scores)}"
    if not all(math.isfinite(score) for score in scores):
        return f"{arm}: score vector contains a non-finite value"
    return None


def _rank_order(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _stats(
    *,
    scores: list[float],
    elapsed_ms: list[float],
    before: CudaAllocatorSnapshot | None,
    after: CudaAllocatorSnapshot | None,
    telemetry: list[RerankerCudaTelemetry | None],
) -> dict[str, Any]:
    return {
        "elapsed_ms": [round(value, 3) for value in elapsed_ms],
        "p50_elapsed_ms": round(statistics.median(elapsed_ms), 3),
        "p95_elapsed_ms": round(_percentile(elapsed_ms, 0.95), 3),
        "p99_elapsed_ms": round(_percentile(elapsed_ms, 0.99), 3),
        "score_count": len(scores),
        "before": _snapshot_dict(before),
        "after": _snapshot_dict(after),
        "maintenance": [_telemetry_dict(item) for item in telemetry],
    }


def _legacy_score_runs(
    reranker: BgeReranker, *, query: str, texts: list[str], iterations: int,
) -> tuple[list[float], dict[str, Any]]:
    model = reranker._load_model()
    before = capture_cuda_allocator_snapshot("cuda:0")
    elapsed_ms: list[float] = []
    scores: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        raw_scores = model.compute_score([[query, text] for text in texts], normalize=True)
        scores = [float(raw_scores)] if isinstance(raw_scores, (int, float)) else [
            float(score) for score in raw_scores
        ]
        elapsed_ms.append((time.perf_counter() - started) * 1000)
    after = capture_cuda_allocator_snapshot("cuda:0")
    return scores, _stats(
        scores=scores, elapsed_ms=elapsed_ms, before=before, after=after,
        telemetry=[],
    )


def _managed_score_runs(
    reranker: BgeReranker, *, query: str, texts: list[str], iterations: int,
) -> tuple[list[float], dict[str, Any]]:
    before = capture_cuda_allocator_snapshot("cuda:0")
    elapsed_ms: list[float] = []
    scores: list[float] = []
    telemetry: list[RerankerCudaTelemetry | None] = []
    for _ in range(iterations):
        started = time.perf_counter()
        scores = reranker.score_pairs(query, texts)
        elapsed_ms.append((time.perf_counter() - started) * 1000)
        telemetry.append(reranker.last_cuda_telemetry)
    after = capture_cuda_allocator_snapshot("cuda:0")
    return scores, _stats(
        scores=scores, elapsed_ms=elapsed_ms, before=before, after=after,
        telemetry=telemetry,
    )


def _gpu_processes(*, gpu_index: int = RELEASE_GPU_INDEX) -> dict[int, dict[str, Any]]:
    """Read only processes assigned to the approved physical GPU."""
    result = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu_index),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    processes: dict[int, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        pid, process_name, memory_mib = [part.strip() for part in line.split(",", maxsplit=2)]
        processes[int(pid)] = {"process_name": process_name, "memory_mib": int(memory_mib)}
    return processes


def _local_ipv4_addresses() -> set[str]:
    """Return local IPv4 addresses without relying on a mutable DNS name."""
    addresses: set[str] = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(item[4][0])
    except socket.gaierror:
        pass
    try:
        result = subprocess.run(
            ["hostname", "-I"], check=True, capture_output=True, text=True,
        )
        addresses.update(token for token in result.stdout.split() if token)
    except (OSError, subprocess.SubprocessError):
        pass
    return addresses


def _selected_gpu_preflight() -> GpuPreflight:
    """Read the driver view before any canary CUDA operation begins."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("release canary could not read nvidia-smi preflight") from exc
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            index = int(parts[0])
            total_mib = int(parts[3])
            free_mib = int(parts[4])
        except ValueError:
            continue
        if index == RELEASE_GPU_INDEX:
            return GpuPreflight(
                index=index,
                uuid=parts[1],
                pci_bus_id=parts[2],
                total_mib=total_mib,
                free_mib=free_mib,
            )
    raise RuntimeError("release canary could not find physical GPU0 in nvidia-smi")


def _validate_gpu_preflight(*, max_canary_peak_mib: int) -> GpuPreflight:
    """Fail closed unless the known `.88` GPU0 and its headroom are present."""
    gpu = _selected_gpu_preflight()
    if gpu.uuid != RELEASE_GPU_UUID:
        raise RuntimeError(
            f"release canary GPU0 UUID {gpu.uuid!r} does not match .88's approved GPU",
        )
    if gpu.pci_bus_id != RELEASE_GPU_PCI_BUS_ID:
        raise RuntimeError(
            f"release canary GPU0 PCI bus {gpu.pci_bus_id!r} does not match .88's approved GPU",
        )
    if gpu.total_mib != RELEASE_GPU_TOTAL_MIB:
        raise RuntimeError(
            f"release canary GPU0 total memory {gpu.total_mib} MiB does not match .88",
        )
    required_free_mib = max_canary_peak_mib + RELEASE_GPU_HEADROOM_MIB
    if gpu.free_mib < required_free_mib:
        raise RuntimeError(
            "release canary GPU0 free memory "
            f"{gpu.free_mib} MiB is below required {required_free_mib} MiB",
        )
    return gpu


def _peach_processes(processes: dict[int, dict[str, Any]]) -> dict[int, int]:
    """Return visible `.88` GPU0 PeachObj process memory by PID."""
    return {
        pid: int(item["memory_mib"])
        for pid, item in processes.items()
        if "peachobj" in str(item["process_name"]).lower()
    }


class _GpuSampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self._interval_seconds = interval_seconds
        self._snapshots: list[dict[int, dict[str, Any]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc_info) -> None:  # noqa: ANN001
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._snapshots.append(_gpu_processes())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self._interval_seconds)

    def peak_mib(self, pid: int) -> int | None:
        values = [snapshot[pid]["memory_mib"] for snapshot in self._snapshots if pid in snapshot]
        return max(values) if values else None


def _validate_runtime(*, memory_fraction: float) -> Path:
    if RELEASE_COMPUTE_IP not in _local_ipv4_addresses():
        raise RuntimeError(
            f"release canary must run on {RELEASE_COMPUTE_IP}; refusing to touch CUDA elsewhere",
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("release canary requires CUDA_VISIBLE_DEVICES=0 on .88")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("release canary requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if memory_fraction != RELEASE_MEMORY_FRACTION:
        raise ValueError(
            f"--memory-fraction must equal the frozen release value {RELEASE_MEMORY_FRACTION}",
        )
    model_path = Path(os.environ.get("BGE_RERANKER_PATH", ""))
    if not model_path.is_dir():
        raise RuntimeError("BGE_RERANKER_PATH must name the pre-existing local model directory")
    embedding_model_path = Path(os.environ.get("BGE_M3_PATH", ""))
    if not embedding_model_path.is_dir():
        raise RuntimeError("BGE_M3_PATH must name the pre-existing local model directory")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("release canary requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    return model_path


def _maintenance_failures(
    items: list[dict[str, Any] | None], *, max_post_trim_idle_cache_mib: int,
) -> list[str]:
    """Require an actual bounded post-trim idle allocator pool, not a no-op."""
    max_idle_cache_bytes = max_post_trim_idle_cache_mib * 1024 * 1024
    failures: list[str] = []
    for iteration, item in enumerate(items, start=1):
        if item is None or item["after"] is None:
            failures.append(f"managed iteration {iteration}: allocator telemetry is unavailable")
            continue
        after = item["after"]
        unused_after = after["reserved_bytes"] - after["allocated_bytes"]
        if unused_after <= max_idle_cache_bytes:
            continue
        if not item["trim_executed"]:
            failures.append(f"managed iteration {iteration}: unused CUDA cache was not trimmed")
            continue
        if item["post_trim"] is None:
            failures.append(
                f"managed iteration {iteration}: post-trim allocator snapshot is unavailable",
            )
            continue
        post_trim = item["post_trim"]
        unused_post_trim = post_trim["reserved_bytes"] - post_trim["allocated_bytes"]
        if post_trim["reserved_bytes"] >= after["reserved_bytes"]:
            failures.append(
                f"managed iteration {iteration}: trim did not reduce reserved CUDA memory",
            )
            continue
        if unused_post_trim > max_idle_cache_bytes:
            failures.append(
                f"managed iteration {iteration}: post-trim idle CUDA cache "
                f"{unused_post_trim / (1024 * 1024):.1f} MiB exceeds "
                f"{max_post_trim_idle_cache_mib} MiB",
            )
    return failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--batch-size", type=int, default=RELEASE_BATCH_SIZE)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-latency-ratio", type=float, default=1.05)
    parser.add_argument("--memory-fraction", required=True, type=float)
    parser.add_argument("--max-canary-peak-mib", required=True, type=int)
    parser.add_argument(
        "--max-post-trim-idle-cache-mib",
        type=int,
        default=RELEASE_POST_TRIM_IDLE_CACHE_MIB,
    )
    return parser.parse_args()


def _retired() -> None:
    """RETIRED: this script loads a model into its own process; every model is a vLLM service now."""
    from core.inference_retired import RERANKER_INSTEAD, RetiredInferencePath

    raise RetiredInferencePath('verify_reranker_cuda_canary.py (in-process torch reranker canary)', RERANKER_INSTEAD)


def main() -> int:
    _retired()
    args = _parse_args()
    base_url = validate_release_target(args.base_url)
    if args.batch_size != RELEASE_BATCH_SIZE:
        raise SystemExit("release canary only validates batch_size=128")
    if args.iterations < 5 or args.top_k < 1:
        raise SystemExit("--iterations must be at least 5 and --top-k must be positive")
    if args.max_latency_ratio < 1:
        raise SystemExit("--max-latency-ratio must be at least 1")
    if args.max_canary_peak_mib < 1:
        raise SystemExit("--max-canary-peak-mib must be positive")
    if args.max_post_trim_idle_cache_mib != RELEASE_POST_TRIM_IDLE_CACHE_MIB:
        raise SystemExit(
            "--max-post-trim-idle-cache-mib must equal the frozen release value "
            f"{RELEASE_POST_TRIM_IDLE_CACHE_MIB}",
        )
    _validate_runtime(memory_fraction=args.memory_fraction)
    gpu_preflight = _validate_gpu_preflight(max_canary_peak_mib=args.max_canary_peak_mib)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the release canary")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)
    before_processes = _gpu_processes()
    peach_before = _peach_processes(before_processes)
    if not peach_before:
        raise RuntimeError("expected a live PeachObj CUDA process before the canary")

    outcomes: list[dict[str, Any]] = []
    failed: list[str] = []
    with _GpuSampler() as sampler:
        # Do not import FlagEmbedding/BGE-M3 until host and physical-GPU
        # preflight above has succeeded and torch CUDA has been explicitly
        # admitted. bge_m3 imports FlagEmbedding, which imports torch.
        from core.indexing.embedders.bge_m3 import BgeM3Embedder

        # A production Retrieval process keeps both models resident. Load and
        # exercise BGE-M3 inside the same capped sampler rather than validating
        # a reranker-only process and calling its driver peak a service peak.
        embedder = BgeM3Embedder(device="cuda:0")
        embedding_result = embedder.embed_texts([case.body["query"] for case in CASES])
        if embedding_result.dense.shape[0] != len(CASES):
            failed.append("BGE-M3 canary embedding returned an unexpected vector count")
        reranker = BgeReranker(
            device="cuda:0",
            batch_size=RELEASE_BATCH_SIZE,
            idle_cache_limit_mib=0,
        )
        for case_index, case in enumerate(CASES):
            response = _request_json(base_url, case)
            texts = case.texts(response)
            if len(texts) != 50:
                failed.append(f"{case.name}: expected exactly 50 usable candidates, received {len(texts)}")
                continue
            # Alternate which arm runs first. This does not claim a service A/B;
            # it prevents a fixed warm-cache ordering from flattering one arm.
            if case_index % 2:
                managed_scores, managed = _managed_score_runs(
                    reranker, query=case.body["query"], texts=texts, iterations=args.iterations,
                )
                legacy_scores, legacy = _legacy_score_runs(
                    reranker, query=case.body["query"], texts=texts, iterations=args.iterations,
                )
                arm_order = ["managed", "legacy"]
            else:
                legacy_scores, legacy = _legacy_score_runs(
                    reranker, query=case.body["query"], texts=texts, iterations=args.iterations,
                )
                managed_scores, managed = _managed_score_runs(
                    reranker, query=case.body["query"], texts=texts, iterations=args.iterations,
                )
                arm_order = ["legacy", "managed"]
            for arm, scores in (("legacy", legacy_scores), ("managed", managed_scores)):
                error = _validate_scores(scores, len(texts), arm=arm)
                if error:
                    failed.append(f"{case.name}: {error}")
            if len(legacy_scores) == len(managed_scores) == len(texts):
                max_score_delta = max(abs(left - right) for left, right in zip(legacy_scores, managed_scores))
                full_order_equal = _rank_order(legacy_scores) == _rank_order(managed_scores)
                top_k_order_equal = (
                    _rank_order(legacy_scores)[:args.top_k]
                    == _rank_order(managed_scores)[:args.top_k]
                )
            else:
                max_score_delta = float("inf")
                full_order_equal = top_k_order_equal = False
            latency_ratio = managed["p50_elapsed_ms"] / legacy["p50_elapsed_ms"]
            maintenance_failures = _maintenance_failures(
                managed["maintenance"],
                max_post_trim_idle_cache_mib=args.max_post_trim_idle_cache_mib,
            )
            outcomes.append({
                "case": case.name,
                "candidate_count": len(texts),
                "arm_order": arm_order,
                "max_abs_score_delta": max_score_delta,
                "full_order_equal": full_order_equal,
                "top_k_order_equal": top_k_order_equal,
                "latency_ratio_p50": latency_ratio,
                "legacy": legacy,
                "managed": managed,
                "maintenance_failures": maintenance_failures,
            })
            if max_score_delta != 0 or not full_order_equal or not top_k_order_equal:
                failed.append(f"{case.name}: batch=128 managed scorer changed score/order")
            if latency_ratio > args.max_latency_ratio:
                failed.append(f"{case.name}: p50 latency ratio {latency_ratio:.3f} exceeds limit")
            failed.extend(f"{case.name}: {reason}" for reason in maintenance_failures)
    after_processes = _gpu_processes()
    peach_after = _peach_processes(after_processes)
    if not set(peach_before).issubset(peach_after):
        failed.append("PeachObj CUDA process disappeared during the canary")
    for pid, before_memory_mib in peach_before.items():
        after_memory_mib = peach_after.get(pid)
        if after_memory_mib is not None and after_memory_mib > (
            before_memory_mib + RELEASE_MAX_PEACH_MEMORY_GROWTH_MIB
        ):
            failed.append(
                f"PeachObj PID {pid} GPU0 memory grew from {before_memory_mib} MiB "
                f"to {after_memory_mib} MiB during the canary",
            )
    canary_peak_mib = sampler.peak_mib(os.getpid())
    if canary_peak_mib is None:
        failed.append("could not sample the canary CUDA process memory")
    elif canary_peak_mib > args.max_canary_peak_mib:
        failed.append(
            f"canary driver peak {canary_peak_mib} MiB exceeds {args.max_canary_peak_mib} MiB",
        )
    print(json.dumps({
        "batch_size": RELEASE_BATCH_SIZE,
        "gpu_preflight": dataclasses.asdict(gpu_preflight),
        "iterations": args.iterations,
        "memory_fraction": RELEASE_MEMORY_FRACTION,
        "bge_m3_query_count": len(CASES),
        "max_post_trim_idle_cache_mib": args.max_post_trim_idle_cache_mib,
        "canary_peak_mib": canary_peak_mib,
        "peach_gpu0_memory_before_mib": peach_before,
        "peach_gpu0_memory_after_mib": peach_after,
        "outcomes": outcomes,
        "failed": failed,
    }, ensure_ascii=False, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
