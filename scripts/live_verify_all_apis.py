#!/usr/bin/env python
"""
Strict live API verification helper for the retrieval repo.

Uses the live FastAPI service as the source of truth, exports the runtime
OpenAPI document to docs/openapi.json, and keeps running after business
failures so the final report shows the full exercised endpoint surface.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
REPORT_PATH = DOCS_DIR / "live_api_verification_v2.md"
OPENAPI_EXPORT_PATH = DOCS_DIR / "openapi.json"
BASE_URL = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
UNVERIFIED = "UNVERIFIED"
REAL_WEB_URL_PRIMARY = "https://www.python.org/about/"
REAL_WEB_URL_SECONDARY = "https://www.iana.org/help/example-domains"
REAL_PAPER_URL_PRIMARY = "https://arxiv.org/abs/1706.03762"
REAL_PAPER_URL_BATCH = "https://arxiv.org/abs/1810.04805"


def clear_proxy_env() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


def excerpt(data: Any, limit: int = 600) -> str:
    try:
        if isinstance(data, (dict, list)):
            text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        else:
            text = str(data)
    except Exception:
        text = repr(data)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


@dataclass
class StepResult:
    name: str
    method: str
    path: str
    spec_path: str
    status: str
    status_code: int | None = None
    detail: str = ""
    payload_excerpt: str = ""


@dataclass
class RunState:
    results: list[StepResult] = field(default_factory=list)
    created_doc_ids: list[int] = field(default_factory=list)
    created_task_ids: list[int] = field(default_factory=list)
    created_paper_ids: list[str] = field(default_factory=list)

    def add(
        self,
        *,
        name: str,
        method: str,
        path: str,
        spec_path: str | None,
        status: str,
        status_code: int | None = None,
        detail: str = "",
        payload_excerpt: str = "",
    ) -> None:
        self.results.append(
            StepResult(
                name=name,
                method=method.upper(),
                path=path,
                spec_path=spec_path or path,
                status=status,
                status_code=status_code,
                detail=detail,
                payload_excerpt=payload_excerpt,
            )
        )

    @property
    def failures(self) -> list[StepResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def blocked(self) -> list[StepResult]:
        return [r for r in self.results if r.status == BLOCKED]


def request_json(
    client: httpx.Client,
    state: RunState,
    name: str,
    method: str,
    path: str,
    *,
    spec_path: str | None = None,
    expected: int = 200,
    validator: Callable[[Any], str | None] | None = None,
    **kwargs: Any,
) -> Any | None:
    response = None
    try:
        response = client.request(method, path, **kwargs)
        try:
            payload: Any = response.json()
        except Exception:
            payload = response.text
        errors: list[str] = []
        if response.status_code != expected:
            errors.append(f"expected {expected}, got {response.status_code}")
        if validator:
            validation_error = validator(payload)
            if validation_error:
                errors.append(validation_error)
        state.add(
            name=name,
            method=method,
            path=path,
            spec_path=spec_path,
            status=PASS if not errors else FAIL,
            status_code=response.status_code,
            detail="OK" if not errors else "; ".join(errors),
            payload_excerpt=excerpt(payload),
        )
        return payload if not errors else None
    except Exception as exc:
        payload_excerpt = ""
        status_code = response.status_code if response is not None else None
        if response is not None:
            try:
                payload_excerpt = excerpt(response.json())
            except Exception:
                payload_excerpt = excerpt(response.text)
        state.add(
            name=name,
            method=method,
            path=path,
            spec_path=spec_path,
            status=FAIL,
            status_code=status_code,
            detail=str(exc),
            payload_excerpt=payload_excerpt,
        )
        return None


def request_check(
    client: httpx.Client,
    state: RunState,
    name: str,
    method: str,
    path: str,
    *,
    spec_path: str | None = None,
    expected: int = 200,
    validator: Callable[[httpx.Response], str | None] | None = None,
    **kwargs: Any,
) -> httpx.Response | None:
    response = None
    try:
        response = client.request(method, path, **kwargs)
        errors: list[str] = []
        if response.status_code != expected:
            errors.append(f"expected {expected}, got {response.status_code}")
        if validator:
            validation_error = validator(response)
            if validation_error:
                errors.append(validation_error)
        content_type = response.headers.get("content-type", "")
        body = response.text
        if "json" in content_type:
            try:
                body = response.json()
            except Exception:
                body = response.text
        state.add(
            name=name,
            method=method,
            path=path,
            spec_path=spec_path,
            status=PASS if not errors else FAIL,
            status_code=response.status_code,
            detail="OK" if not errors else "; ".join(errors),
            payload_excerpt=excerpt(body),
        )
        return response if not errors else None
    except Exception as exc:
        state.add(name=name, method=method, path=path, spec_path=spec_path, status=FAIL, detail=str(exc))
        return None


def request_sse(
    client: httpx.Client,
    state: RunState,
    name: str,
    path: str,
    *,
    spec_path: str | None = None,
    expected_events: tuple[str, ...] = ("snapshot",),
    max_lines: int = 80,
) -> str | None:
    response = None
    try:
        lines: list[str] = []
        with client.stream("GET", path, headers={"accept": "text/event-stream"}) as response:
            for line in response.iter_lines():
                if line is None:
                    continue
                lines.append(line)
                if len(lines) >= max_lines:
                    break
                if any(f"event: {event_name}" in line for event_name in expected_events):
                    if "event: final" in "\n".join(lines):
                        break
            payload = "\n".join(lines)
            errors: list[str] = []
            if response.status_code != 200:
                errors.append(f"expected 200, got {response.status_code}")
            for event_name in expected_events:
                if f"event: {event_name}" not in payload:
                    errors.append(f"missing SSE event: {event_name}")
            state.add(
                name=name,
                method="GET",
                path=path,
                spec_path=spec_path,
                status=PASS if not errors else FAIL,
                status_code=response.status_code,
                detail="OK" if not errors else "; ".join(errors),
                payload_excerpt=excerpt(payload),
            )
            return payload if not errors else None
    except Exception as exc:
        payload_excerpt = ""
        status_code = response.status_code if response is not None else None
        if response is not None:
            payload_excerpt = excerpt(response.text)
        state.add(
            name=name,
            method="GET",
            path=path,
            spec_path=spec_path,
            status=FAIL,
            status_code=status_code,
            detail=str(exc),
            payload_excerpt=payload_excerpt,
        )
        return None


def record_blocked(state: RunState, name: str, method: str, spec_path: str, detail: str) -> None:
    state.add(name=name, method=method, path=spec_path, spec_path=spec_path, status=BLOCKED, detail=detail)


def poll_task_status(
    client: httpx.Client,
    task_id: int,
    desired: set[str],
    timeout_s: float = 90.0,
    interval_s: float = 2.0,
) -> tuple[dict[str, Any] | None, str | None]:
    deadline = time.time() + timeout_s
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        response = client.get(f"/api/tasks/{task_id}")
        if response.status_code != 200:
            return None, f"poll returned HTTP {response.status_code}"
        payload = response.json()
        last_payload = payload
        if payload.get("status") in desired:
            return payload, None
        time.sleep(interval_s)
    return None, f"task {task_id} did not reach {sorted(desired)}; last={last_payload}"


def find_task_with_status(client: httpx.Client, status: str) -> int | None:
    response = client.get("/api/tasks", params={"status": status, "page_size": 100})
    if response.status_code != 200:
        return None
    tasks = response.json().get("tasks", [])
    return tasks[0]["id"] if tasks else None


def cleanup(client: httpx.Client, state: RunState) -> None:
    for paper_id in reversed(state.created_paper_ids):
        try:
            client.delete(f"/api/papers/{paper_id}")
        except Exception:
            pass
    for doc_id in reversed(state.created_doc_ids):
        try:
            client.delete(f"/api/documents/{doc_id}")
        except Exception:
            pass
    for task_id in reversed(state.created_task_ids):
        try:
            client.delete(f"/api/tasks/{task_id}")
        except Exception:
            pass


def write_report(state: RunState, commands: list[str], openapi: dict[str, Any]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    counts = {PASS: 0, FAIL: 0, BLOCKED: 0}
    for result in state.results:
        counts[result.status] = counts.get(result.status, 0) + 1

    paths = openapi.get("paths", {})
    result_by_endpoint = {(r.spec_path, r.method): r for r in state.results}

    lines = [
        "# Live API Verification V2",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Base URL: {BASE_URL}",
        f"Python: {sys.executable}",
        "",
        "## Summary",
        "",
        f"- Total checks: {len(state.results)}",
        f"- Passed: {counts.get(PASS, 0)}",
        f"- Failed: {counts.get(FAIL, 0)}",
        f"- Blocked: {counts.get(BLOCKED, 0)}",
        "",
        "## Commands Used",
        "",
    ]
    lines.extend(f"- `{command}`" for command in commands)
    lines.extend(
        [
            "",
            "## Live OpenAPI Export",
            "",
            f"- Exported to `docs/openapi.json`",
            f"- Paths discovered: {len(paths)}",
            "",
            "## Endpoint Matrix",
            "",
            "| Path | Method | Status | HTTP | Detail |",
            "|------|--------|--------|------|--------|",
        ]
    )

    for path in sorted(paths):
        for method in sorted(paths[path]):
            result = result_by_endpoint.get((path, method.upper()))
            if result is None:
                lines.append(f"| {path} | {method.upper()} | {UNVERIFIED} |  | No explicit live exercise recorded |")
                continue
            http = "" if result.status_code is None else str(result.status_code)
            lines.append(f"| {path} | {method.upper()} | {result.status} | {http} | {result.detail.replace(chr(10), ' ')} |")

    lines.extend(
        [
            "",
            "## Step Results",
            "",
            "| Name | Method | Actual Path | Spec Path | Status | HTTP | Detail |",
            "|------|--------|-------------|-----------|--------|------|--------|",
        ]
    )
    for result in state.results:
        http = "" if result.status_code is None else str(result.status_code)
        lines.append(
            f"| {result.name} | {result.method} | {result.path} | {result.spec_path} | {result.status} | {http} | {result.detail.replace(chr(10), ' ')} |"
        )

    if state.failures:
        lines.extend(["", "## Remaining Failures", ""])
        for result in state.failures:
            lines.extend(
                [
                    f"### {result.name}",
                    "",
                    f"- Method: `{result.method}`",
                    f"- Spec path: `{result.spec_path}`",
                    f"- Actual path: `{result.path}`",
                    f"- HTTP: `{result.status_code}`" if result.status_code is not None else "- HTTP: `n/a`",
                    f"- Detail: {result.detail}",
                ]
            )
            if result.payload_excerpt:
                lines.extend(["- Payload excerpt:", "```json", result.payload_excerpt, "```"])
            lines.append("")

    if state.blocked:
        lines.extend(["", "## Blocked Checks", ""])
        lines.extend(f"- `{result.method} {result.spec_path}`: {result.detail}" for result in state.blocked)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    clear_proxy_env()
    commands = [
        f"{sys.executable} scripts/live_verify_all_apis.py",
        f"{sys.executable} scripts/test_api.py --base-url {BASE_URL}",
        f"{sys.executable} scripts/test_e2e.py --api {BASE_URL}",
    ]
    state = RunState()
    openapi: dict[str, Any] = {"paths": {}}
    client = httpx.Client(base_url=BASE_URL, timeout=60.0, trust_env=False, follow_redirects=True)

    try:
        request_check(client, state, "live docs", "GET", "/docs")
        openapi_payload = request_json(
            client,
            state,
            "live openapi",
            "GET",
            "/openapi.json",
            validator=lambda p: None if isinstance(p, dict) and "paths" in p else "missing paths",
        )
        if isinstance(openapi_payload, dict) and "paths" in openapi_payload:
            openapi = openapi_payload
            OPENAPI_EXPORT_PATH.write_text(json.dumps(openapi, ensure_ascii=False, indent=2), encoding="utf-8")

        request_check(client, state, "home", "GET", "/", spec_path="/")
        request_json(client, state, "stats", "GET", "/api/stats", spec_path="/api/stats", validator=lambda p: None if isinstance(p, dict) and "total_documents" in p and "total_chunks" in p else "stats shape mismatch")
        request_json(client, state, "config", "GET", "/api/config", spec_path="/api/config", validator=lambda p: None if isinstance(p, dict) and "proxy" in p and "retrieval" in p else "config shape mismatch")
        request_json(client, state, "config proxy update", "POST", "/api/config/proxy", params={"enabled": "false"}, spec_path="/api/config/proxy", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "proxy update failed")
        request_json(client, state, "search post", "POST", "/api/search", spec_path="/api/search", json={"query": "verification", "method": "hybrid", "top_k": 3, "aggregate_by_doc": True}, validator=lambda p: None if isinstance(p, dict) and "results" in p and "aggregated" in p else "search response mismatch")
        request_json(client, state, "search multi method", "GET", "/api/search/multi-method", spec_path="/api/search/multi-method", params={"query": "verification", "top_k": 2}, validator=lambda p: None if isinstance(p, dict) and all(k in p for k in ("bm25", "vector", "hybrid", "enhanced_aggregated")) else "multi-method mismatch")

        request_json(client, state, "main upload", "POST", "/api/upload", spec_path="/api/upload", files={"file": ("verify-main-upload.txt", b"live verification main upload\n", "text/plain")}, data={"file_type": "txt"}, validator=lambda p: None if isinstance(p, dict) and p.get("success") is True and p.get("documents_count", 0) >= 1 else "main upload failed")
        document_upload = request_json(client, state, "documents upload", "POST", "/api/documents/upload", spec_path="/api/documents/upload", files={"file": ("verify-doc-upload.txt", b"live verification document upload\n", "text/plain")}, data={"title": "verify-doc-upload", "auto_index": "true"}, validator=lambda p: None if isinstance(p, dict) and p.get("doc_id") else "documents upload missing doc_id")
        doc_id = int(document_upload["doc_id"]) if isinstance(document_upload, dict) and document_upload.get("doc_id") else None
        if doc_id is not None:
            state.created_doc_ids.append(doc_id)

        documents_payload = request_json(client, state, "documents list", "GET", "/api/documents", spec_path="/api/documents", validator=lambda p: None if isinstance(p, dict) and "documents" in p and "total" in p else "document list mismatch")
        if doc_id is not None and isinstance(documents_payload, dict) and not any(doc.get("id") == doc_id for doc in documents_payload.get("documents", [])):
            state.add(name="documents list contains uploaded doc", method="GET", path="/api/documents", spec_path="/api/documents", status=FAIL, detail=f"uploaded doc {doc_id} missing from list page", payload_excerpt=excerpt(documents_payload))

        if doc_id is not None:
            request_json(client, state, "document detail", "GET", f"/api/documents/{doc_id}", spec_path="/api/documents/{doc_id}", validator=lambda p: None if isinstance(p, dict) and p.get("id") == doc_id else "wrong document detail")
            request_json(client, state, "document update", "PUT", f"/api/documents/{doc_id}", spec_path="/api/documents/{doc_id}", json={"title": "verify-doc-upload-updated", "metadata": {"verified": True}}, validator=lambda p: None if isinstance(p, dict) and p.get("metadata", {}).get("verified") is True else "document update mismatch")
            request_check(client, state, "document download", "GET", f"/api/documents/{doc_id}/download", spec_path="/api/documents/{doc_id}/download", validator=lambda r: None if "attachment" in r.headers.get("content-disposition", "").lower() else "download missing attachment header")
        else:
            record_blocked(state, "document detail", "GET", "/api/documents/{doc_id}", "documents upload did not return a usable doc_id")
            record_blocked(state, "document update", "PUT", "/api/documents/{doc_id}", "documents upload did not return a usable doc_id")
            record_blocked(state, "document download", "GET", "/api/documents/{doc_id}/download", "documents upload did not return a usable doc_id")

        crawl_main = request_json(client, state, "main crawl", "POST", "/api/crawl", spec_path="/api/crawl", json={"urls": [REAL_WEB_URL_PRIMARY], "use_proxy": False}, validator=lambda p: None if isinstance(p, dict) and p.get("task_id") and p.get("total_urls") == 1 else "main crawl mismatch")
        main_crawl_task_id = crawl_main.get("task_id") if isinstance(crawl_main, dict) else None
        if main_crawl_task_id:
            request_json(client, state, "main crawl status", "GET", f"/api/crawl/{main_crawl_task_id}", spec_path="/api/crawl/{task_id}", validator=lambda p: None if isinstance(p, dict) and "status" in p and "results" in p else "crawl status mismatch")
        else:
            record_blocked(state, "main crawl status", "GET", "/api/crawl/{task_id}", "main crawl did not produce a task_id")

        crawl_urls = request_json(client, state, "crawl urls", "POST", "/api/crawl/urls", spec_path="/api/crawl/urls", json={"urls": [REAL_WEB_URL_PRIMARY, "http://127.0.0.1:9"], "use_proxy": False, "auto_index": False, "max_concurrent": 2}, validator=lambda p: None if isinstance(p, dict) and p.get("success") is True and p.get("task_id") else "crawl urls failed")
        crawl_task_id = int(crawl_urls["task_id"]) if isinstance(crawl_urls, dict) and crawl_urls.get("task_id") else None
        if crawl_task_id is not None:
            state.created_task_ids.append(crawl_task_id)
            task_payload, poll_error = poll_task_status(client, crawl_task_id, {"completed", "failed", "cancelled", "timeout"})
            if poll_error:
                state.add(name="task polling", method="GET", path=f"/api/tasks/{crawl_task_id}", spec_path="/api/tasks/{task_id}", status=FAIL, detail=poll_error)
                task_payload = None
            request_json(client, state, "crawl records", "GET", "/api/crawl/records", spec_path="/api/crawl/records", params={"task_id": crawl_task_id}, validator=lambda p: None if isinstance(p, dict) and p.get("total", 0) >= 1 else "crawl records missing")
            request_json(client, state, "tasks list", "GET", "/api/tasks", spec_path="/api/tasks", validator=lambda p: None if isinstance(p, dict) and "tasks" in p and "total" in p else "tasks list mismatch")
            request_json(client, state, "task detail", "GET", f"/api/tasks/{crawl_task_id}", spec_path="/api/tasks/{task_id}", validator=lambda p: None if isinstance(p, dict) and p.get("id") == crawl_task_id else "task detail mismatch")

            retry_task_id = crawl_task_id if isinstance(task_payload, dict) and task_payload.get("status") == "failed" else find_task_with_status(client, "failed")
            if retry_task_id is not None:
                request_json(client, state, "task retry", "POST", f"/api/tasks/{retry_task_id}/retry", spec_path="/api/tasks/{task_id}/retry", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "task retry failed")
            else:
                record_blocked(state, "task retry", "POST", "/api/tasks/{task_id}/retry", "no failed task available for retry")
        else:
            record_blocked(state, "crawl records", "GET", "/api/crawl/records", "crawl urls did not produce a task_id")
            record_blocked(state, "tasks list", "GET", "/api/tasks", "crawl urls did not produce a task_id for downstream task checks")
            record_blocked(state, "task detail", "GET", "/api/tasks/{task_id}", "crawl urls did not produce a task_id")
            record_blocked(state, "task retry", "POST", "/api/tasks/{task_id}/retry", "crawl urls did not produce a task_id and no failed task was discovered")

        cancellable = request_json(client, state, "crawl html", "POST", "/api/crawl/html", spec_path="/api/crawl/html", files={"file": ("bookmarks.html", f'<html><body><a href=\"{REAL_WEB_URL_PRIMARY}\">python</a><a href=\"{REAL_WEB_URL_SECONDARY}\">iana</a></body></html>'.encode("utf-8"), "text/html")}, data={"use_proxy": "false", "auto_index": "false", "auto_crawl": "true"}, validator=lambda p: None if isinstance(p, dict) and p.get("success") is True and p.get("task_id") else "crawl html failed")
        cancel_task_id = int(cancellable["task_id"]) if isinstance(cancellable, dict) and cancellable.get("task_id") else None
        if cancel_task_id is not None:
            state.created_task_ids.append(cancel_task_id)
            request_json(client, state, "task cancel", "POST", f"/api/tasks/{cancel_task_id}/cancel", spec_path="/api/tasks/{task_id}/cancel", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "task cancel failed")
            deleted_task = request_json(client, state, "task delete", "DELETE", f"/api/tasks/{cancel_task_id}", spec_path="/api/tasks/{task_id}", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "task delete failed")
            if deleted_task is not None and cancel_task_id in state.created_task_ids:
                state.created_task_ids.remove(cancel_task_id)
        else:
            record_blocked(state, "task cancel", "POST", "/api/tasks/{task_id}/cancel", "crawl html did not produce a task_id")
            record_blocked(state, "task delete", "DELETE", "/api/tasks/{task_id}", "crawl html did not produce a task_id")

        request_json(client, state, "papers stats", "GET", "/api/papers/stats", spec_path="/api/papers/stats", validator=lambda p: None if isinstance(p, dict) and "collection_name" in p and "total_papers" in p else "papers stats mismatch")
        request_json(client, state, "papers list", "GET", "/api/papers/", spec_path="/api/papers/", validator=lambda p: None if isinstance(p, list) else "papers list mismatch")
        request_json(client, state, "papers search get", "GET", "/api/papers/search", spec_path="/api/papers/search", params={"query": "verification", "top_k": 5, "method": "hybrid"}, validator=lambda p: None if isinstance(p, dict) and "papers" in p else "papers search get mismatch")
        request_json(client, state, "papers search post", "POST", "/api/papers/search", spec_path="/api/papers/search", json={"query": "verification", "top_k": 5, "method": "hybrid", "aggregate_by_paper": True}, validator=lambda p: None if isinstance(p, dict) and "papers" in p and "total_results" in p else "papers search post mismatch")

        paper_id = "verify-paper-live-strict"
        batch_id = "verify-paper-live-strict-batch"
        paper_payload = {
            "paper_id": paper_id,
            "title": "Verification Paper",
            "abstract": "Verification abstract for live API testing.",
            "authors": ["Verifier One", "Verifier Two"],
            "source": "arxiv",
            "url": REAL_PAPER_URL_PRIMARY,
            "conference": "Other",
            "venue": "VerificationConf",
            "year": 2026,
            "paper_type": "unknown",
            "keywords": ["verification"],
            "content": "This is the body content for the verification paper.",
        }
        paper_index = request_json(client, state, "paper index", "POST", "/api/papers/index", spec_path="/api/papers/index", json=paper_payload, validator=lambda p: None if isinstance(p, dict) and p.get("success") is True and p.get("paper_id") == paper_id else "paper index failed")
        if isinstance(paper_index, dict) and paper_index.get("success") is True:
            state.created_paper_ids.append(paper_id)
        request_json(client, state, "paper detail", "GET", f"/api/papers/{paper_id}", spec_path="/api/papers/{paper_id}", validator=lambda p: None if isinstance(p, dict) and p.get("id") == paper_id else "paper detail mismatch")
        batch_index = request_json(client, state, "paper batch index", "POST", "/api/papers/index/batch", spec_path="/api/papers/index/batch", json={"papers": [{"paper_id": batch_id, "title": "Verification Batch Paper", "abstract": "Batch verification abstract.", "authors": ["Verifier Batch"], "source": "arxiv", "url": REAL_PAPER_URL_BATCH, "conference": "Other", "venue": "VerificationConf", "year": 2026, "paper_type": "unknown", "keywords": ["batch"], "content": "Batch paper content."}]}, validator=lambda p: None if isinstance(p, dict) and p.get("indexed", 0) >= 1 and p.get("results", {}).get(batch_id) else "paper batch index failed")
        if isinstance(batch_index, dict) and batch_index.get("results", {}).get(batch_id):
            state.created_paper_ids.append(batch_id)
        deleted_paper = request_json(client, state, "paper delete", "DELETE", f"/api/papers/{paper_id}", spec_path="/api/papers/{paper_id}", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True and p.get("paper_id") == paper_id else "paper delete failed")
        if deleted_paper is not None and paper_id in state.created_paper_ids:
            state.created_paper_ids.remove(paper_id)

        if doc_id is not None:
            deleted_doc = request_json(client, state, "document delete", "DELETE", f"/api/documents/{doc_id}", spec_path="/api/documents/{doc_id}", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True and p.get("doc_id") == doc_id else "document delete failed")
            if deleted_doc is not None and doc_id in state.created_doc_ids:
                state.created_doc_ids.remove(doc_id)
        else:
            record_blocked(state, "document delete", "DELETE", "/api/documents/{doc_id}", "documents upload did not return a usable doc_id")
    finally:
        write_report(state, commands, openapi)
        cleanup(client, state)
        client.close()

    if state.failures:
        print(f"FAILURES: {len(state.failures)}")
        print(f"REPORT: {REPORT_PATH}")
        for failure in state.failures:
            print(f"- {failure.method} {failure.spec_path}: {failure.detail}")
        if state.blocked:
            print(f"BLOCKED: {len(state.blocked)}")
            for blocked in state.blocked:
                print(f"- {blocked.method} {blocked.spec_path}: {blocked.detail}")
        return 1

    print("ALL LIVE API CHECKS PASSED")
    print(f"REPORT: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
