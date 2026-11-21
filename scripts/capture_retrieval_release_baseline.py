"""Capture or compare final `.88` retrieval API results without storing content.

Run ``capture`` before a service rollout, then ``compare`` immediately after it.
The artifact contains only request names, stable result identity fields and scores;
it never writes SQL/Qdrant/Journal data or candidate text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


RELEASE_BASE_URL = "http://10.100.100.88:18080"
_IDENTITY_FIELDS = (
    "id", "doc_id", "chunk_id", "paper_id", "path", "start_line", "end_line",
    "source_id", "revision", "language", "project", "host",
)


@dataclass(frozen=True)
class ApiCase:
    name: str
    path: str
    body: dict[str, Any]
    result_key: str


CASES = (
    ApiCase(
        "code_cuda_allocator", "/api/code/search",
        {"query": "CUDA allocator reserved memory cache and OOM retry handling", "top_k": 50,
         "mode": "hybrid", "rerank": True}, "hits",
    ),
    ApiCase(
        "code_concurrent_search", "/api/code/search",
        {"query": "concurrent search request scheduling and thread safety", "top_k": 50,
         "mode": "hybrid", "rerank": True}, "hits",
    ),
    ApiCase(
        "public_code_webrtc", "/api/public-code/search",
        {"query": "WebRTC peer connection ICE transport state error recovery", "top_k": 50,
         "mode": "hybrid", "rerank": True}, "hits",
    ),
    ApiCase(
        "public_code_chunking", "/api/public-code/search",
        {"query": "semantic code retrieval chunking and metadata filtering", "top_k": 50,
         "mode": "hybrid", "rerank": True}, "hits",
    ),
    ApiCase(
        "papers_reranker", "/api/search",
        {"query": "cross encoder reranking for retrieval augmented generation long context question answering",
         "top_k": 50, "method": "hybrid"}, "results",
    ),
    ApiCase(
        "papers_embedding", "/api/search",
        {"query": "dense sparse embedding retrieval with learned reranker evaluation", "top_k": 50,
         "method": "hybrid"}, "results",
    ),
)


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, new_url):  # noqa: ANN001
        raise RuntimeError(f"unexpected redirect while calling {request.full_url}: HTTP {code}")


def validate_release_target(base_url: str) -> str:
    parsed = urlparse(base_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("baseline URL has an invalid port") from exc
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
        raise ValueError("baseline tool only permits http://10.100.100.88:18080")
    return RELEASE_BASE_URL


def _request_json(
    base_url: str,
    case: ApiCase,
    *,
    timeout_seconds: float = 180,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("request timeout must be positive")
    request = Request(
        f"{base_url}{case.path}",
        data=json.dumps(case.body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with build_opener(_RejectRedirect(), ProxyHandler({})).open(  # noqa: S310
            request,
            timeout=timeout_seconds,
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"{case.name} API returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"{case.name} API request failed: {exc.reason}") from exc


def _identity(hit: dict[str, Any]) -> dict[str, Any]:
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    identity = {
        field: hit.get(field, metadata.get(field))
        for field in _IDENTITY_FIELDS
        if hit.get(field, metadata.get(field)) is not None
    }
    # Candidate content is intentionally absent. The hash helps distinguish
    # duplicate IDs without persisting source text in the release artifact.
    content = hit.get("content", hit.get("code", ""))
    identity["content_sha256"] = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
    non_content_payload = {
        key: value for key, value in hit.items() if key not in {"content", "code", "score"}
    }
    identity["metadata_sha256"] = hashlib.sha256(
        json.dumps(non_content_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()
    return identity


def canonical_case_result(case: ApiCase, response: dict[str, Any]) -> dict[str, Any]:
    hits = response.get(case.result_key)
    if not isinstance(hits, list):
        raise RuntimeError(f"{case.name}: response has no list field {case.result_key!r}")
    canonical_hits = []
    for hit in hits:
        if not isinstance(hit, dict):
            raise RuntimeError(f"{case.name}: response contains a non-object hit")
        score = hit.get("score")
        if not isinstance(score, (float, int)):
            raise RuntimeError(f"{case.name}: response hit has no numeric score")
        canonical_hits.append({"identity": _identity(hit), "score": float(score)})
    if not canonical_hits:
        raise RuntimeError(f"{case.name}: final API returned no hits")
    return {"name": case.name, "result_count": len(canonical_hits), "hits": canonical_hits}


def capture_baseline(base_url: str) -> dict[str, Any]:
    validated_url = validate_release_target(base_url)
    cases = [canonical_case_result(case, _request_json(validated_url, case)) for case in CASES]
    return {"schema_version": 1, "base_url": validated_url, "cases": cases}


def compare_baseline(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    if expected.get("schema_version") != 1:
        return ["baseline schema_version is unsupported"]
    expected_cases = expected.get("cases")
    actual_cases = actual.get("cases")
    if not isinstance(expected_cases, list) or not isinstance(actual_cases, list):
        return ["baseline or actual cases are malformed"]
    expected_by_name = {case.get("name"): case for case in expected_cases}
    actual_by_name = {case.get("name"): case for case in actual_cases}
    failures: list[str] = []
    if set(expected_by_name) != set(actual_by_name):
        failures.append("API case set changed during the rollout")
    for name in sorted(set(expected_by_name).intersection(actual_by_name)):
        if expected_by_name[name] != actual_by_name[name]:
            failures.append(f"{name}: final result identities, order, or scores changed")
    return failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "compare"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    actual = capture_baseline(args.base_url)
    if args.mode == "capture":
        if args.baseline.exists():
            raise SystemExit(f"refusing to overwrite existing baseline: {args.baseline}")
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"captured_cases": [case["name"] for case in actual["cases"]]}, ensure_ascii=False))
        return 0
    try:
        expected = json.loads(args.baseline.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read baseline: {exc}") from exc
    failures = compare_baseline(expected, actual)
    print(json.dumps({"compared_cases": [case["name"] for case in actual["cases"]], "failed": failures}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
