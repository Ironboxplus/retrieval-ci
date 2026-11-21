#!/usr/bin/env python
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "docs" / "live_api_contract_report.json"
OPENAPI_EXPORT_PATH = ROOT / "docs" / "openapi.json"
BASE_URL = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
LOCAL_WEB_URL_PRIMARY = f"{BASE_URL}/"
LOCAL_WEB_URL_SECONDARY = f"{BASE_URL}/docs"
FAILED_WEB_URL = "http://127.0.0.1:65534/"
REAL_PAPER_URL_PRIMARY = "https://arxiv.org/abs/1706.03762"
REAL_PAPER_URL_BATCH = "https://arxiv.org/abs/1810.04805"


def clear_proxy_env() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


def excerpt(data: Any, limit: int = 500) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, default=str) if isinstance(data, (dict, list)) else str(data)
    except Exception:
        text = repr(data)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def add(state: dict[str, Any], *, name: str, method: str, path: str, spec_path: str, status: str, status_code: int | None = None, detail: str = "", payload: Any = "") -> None:
    state["results"].append({"name": name, "method": method.upper(), "path": path, "spec_path": spec_path, "status": status, "status_code": status_code, "detail": detail, "payload_excerpt": excerpt(payload)})


def request_json(client: httpx.Client, state: dict[str, Any], name: str, method: str, path: str, *, spec_path: str | None = None, expected: int = 200, validator: Callable[[Any], str | None] | None = None, **kwargs: Any) -> Any | None:
    response = None
    try:
        response = client.request(method, path, **kwargs)
        try:
            payload: Any = response.json()
        except Exception:
            payload = response.text
        errors = []
        if response.status_code != expected:
            errors.append(f"expected {expected}, got {response.status_code}")
        if validator:
            error = validator(payload)
            if error:
                errors.append(error)
        add(state, name=name, method=method, path=path, spec_path=spec_path or path, status="PASS" if not errors else "FAIL", status_code=response.status_code, detail="OK" if not errors else "; ".join(errors), payload=payload)
        return payload if not errors else None
    except Exception as exc:
        add(state, name=name, method=method, path=path, spec_path=spec_path or path, status="FAIL", status_code=response.status_code if response is not None else None, detail=str(exc), payload=response.text if response is not None else "")
        return None


def request_check(client: httpx.Client, state: dict[str, Any], name: str, method: str, path: str, *, spec_path: str | None = None, expected: int = 200, validator: Callable[[httpx.Response], str | None] | None = None, **kwargs: Any) -> httpx.Response | None:
    response = None
    try:
        response = client.request(method, path, **kwargs)
        errors = []
        if response.status_code != expected:
            errors.append(f"expected {expected}, got {response.status_code}")
        if validator:
            error = validator(response)
            if error:
                errors.append(error)
        payload: Any = response.text
        if "json" in response.headers.get("content-type", ""):
            try:
                payload = response.json()
            except Exception:
                pass
        add(state, name=name, method=method, path=path, spec_path=spec_path or path, status="PASS" if not errors else "FAIL", status_code=response.status_code, detail="OK" if not errors else "; ".join(errors), payload=payload)
        return response if not errors else None
    except Exception as exc:
        add(state, name=name, method=method, path=path, spec_path=spec_path or path, status="FAIL", detail=str(exc))
        return None


def request_sse(client: httpx.Client, state: dict[str, Any], name: str, path: str, *, spec_path: str, expected_events: tuple[str, ...]) -> None:
    response = None
    try:
        lines: list[str] = []
        with client.stream("GET", path, headers={"accept": "text/event-stream"}) as response:
            for line in response.iter_lines():
                if line is None:
                    continue
                lines.append(line)
                joined = "\n".join(lines)
                if all(f"event: {event_name}" in joined for event_name in expected_events):
                    break
                if len(lines) >= 80:
                    break
        text = "\n".join(lines)
        errors = [f"missing SSE event: {event_name}" for event_name in expected_events if f"event: {event_name}" not in text]
        if response.status_code != 200:
            errors.insert(0, f"expected 200, got {response.status_code}")
        add(state, name=name, method="GET", path=path, spec_path=spec_path, status="PASS" if not errors else "FAIL", status_code=response.status_code, detail="OK" if not errors else "; ".join(errors), payload=text)
    except Exception as exc:
        add(state, name=name, method="GET", path=path, spec_path=spec_path, status="FAIL", detail=str(exc))


def record_blocked(state: dict[str, Any], name: str, method: str, spec_path: str, detail: str) -> None:
    add(state, name=name, method=method, path=spec_path, spec_path=spec_path, status="BLOCKED", detail=detail)


def poll_task_status(client: httpx.Client, task_id: int, desired: set[str], timeout_s: float = 90.0) -> dict[str, Any] | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f"/api/tasks/{task_id}")
        if response.status_code == 200:
            payload = response.json()
            if payload.get("status") in desired:
                return payload
        time.sleep(2.0)
    return None


def poll_main_crawl_status(client: httpx.Client, task_id: str, desired: set[str], timeout_s: float = 60.0) -> dict[str, Any] | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f"/api/crawl/{task_id}")
        if response.status_code == 200:
            payload = response.json()
            if payload.get("status") in desired:
                return payload
        time.sleep(1.0)
    return None


def cleanup(client: httpx.Client, state: dict[str, Any]) -> None:
    for paper_id in reversed(state["created_paper_ids"]):
        try:
            client.delete(f"/api/papers/{paper_id}")
        except Exception:
            pass
    for doc_id in reversed(state["created_doc_ids"]):
        try:
            client.delete(f"/api/documents/{doc_id}")
        except Exception:
            pass
    # Keep verification tasks as durable evidence and avoid leaving orphaned
    # queue jobs behind if a retry is still draining.


def main() -> int:
    clear_proxy_env()
    state: dict[str, Any] = {"results": [], "created_doc_ids": [], "created_task_ids": [], "created_paper_ids": []}
    openapi: dict[str, Any] = {"paths": {}}
    client = httpx.Client(base_url=BASE_URL, timeout=60.0, trust_env=False, follow_redirects=True)
    try:
        request_check(client, state, "docs", "GET", "/docs", spec_path="/docs")
        openapi_payload = request_json(client, state, "openapi", "GET", "/openapi.json", spec_path="/openapi.json", validator=lambda p: None if isinstance(p, dict) and "paths" in p else "missing paths")
        if isinstance(openapi_payload, dict):
            openapi = openapi_payload
            OPENAPI_EXPORT_PATH.write_text(json.dumps(openapi, ensure_ascii=False, indent=2), encoding="utf-8")
        request_check(client, state, "home", "GET", "/", spec_path="/")

        sample_paper_id = None
        sample_doc_id = None
        sample_query = "verification"
        sample_author = None
        sample_room = None

        paper_ids = request_json(client, state, "papers list", "GET", "/api/papers/", spec_path="/api/papers/", validator=lambda p: None if isinstance(p, list) and len(p) >= 1 else "papers list empty")
        if isinstance(paper_ids, list) and paper_ids:
            sample_paper_id = paper_ids[0]
            sample_paper = request_json(client, state, "sample paper detail", "GET", f"/api/papers/{sample_paper_id}", spec_path="/api/papers/{paper_id}", validator=lambda p: None if isinstance(p, dict) and p.get("id") == sample_paper_id and p.get("doc_id") and p.get("chunk_count", 0) >= 1 else "sample paper detail mismatch")
            if isinstance(sample_paper, dict):
                sample_doc_id = int(sample_paper["doc_id"])
                sample_query = " ".join(str(sample_paper.get("title") or "").split()[:6]) or sample_query
                metadata = sample_paper.get("metadata") or {}
                authors = metadata.get("authors") or []
                sample_author = str(authors[0]) if authors else None
                if metadata.get("conference") and metadata.get("year"):
                    sample_room = f"{metadata['conference']} {metadata['year']}"
        else:
            record_blocked(state, "sample paper detail", "GET", "/api/papers/{paper_id}", "no warmed sample paper is available")

        request_json(client, state, "papers stats", "GET", "/api/papers/stats", spec_path="/api/papers/stats", validator=lambda p: None if isinstance(p, dict) and p.get("total_papers", 0) >= 1 and p.get("total_documents", 0) >= 1 else "papers stats mismatch")
        request_json(client, state, "stats", "GET", "/api/stats", spec_path="/api/stats", validator=lambda p: None if isinstance(p, dict) and p.get("total_documents", 0) >= 1 and "total_chunks" in p else "stats mismatch")
        request_json(client, state, "config", "GET", "/api/config", spec_path="/api/config", validator=lambda p: None if isinstance(p, dict) and "proxy" in p and "retrieval" in p else "config mismatch")
        request_json(client, state, "config proxy update", "POST", "/api/config/proxy", spec_path="/api/config/proxy", params={"enabled": "false"}, validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "config proxy update failed")
        request_json(client, state, "search post", "POST", "/api/search", spec_path="/api/search", json={"query": sample_query, "method": "hybrid", "top_k": 3, "aggregate_by_doc": True}, validator=lambda p: None if isinstance(p, dict) and len(p.get("results", [])) >= 1 and isinstance(p.get("aggregated"), list) else "search post mismatch")
        request_json(client, state, "search multi method", "GET", "/api/search/multi-method", spec_path="/api/search/multi-method", params={"query": sample_query, "top_k": 2}, validator=lambda p: None if isinstance(p, dict) and all(key in p for key in ("bm25", "vector", "hybrid", "enhanced_aggregated")) else "search multi-method mismatch")

        request_json(client, state, "main upload", "POST", "/api/upload", spec_path="/api/upload", files={"file": ("verify-main-upload.txt", b"main upload\n", "text/plain")}, data={"file_type": "txt"}, validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "main upload failed")
        doc_upload = request_json(client, state, "documents upload", "POST", "/api/documents/upload", spec_path="/api/documents/upload", files={"file": ("verify-doc-upload.txt", b"doc upload\n", "text/plain")}, data={"title": "verify-doc-upload", "auto_index": "true"}, validator=lambda p: None if isinstance(p, dict) and p.get("doc_id") else "documents upload missing doc_id")
        upload_doc_id = int(doc_upload["doc_id"]) if isinstance(doc_upload, dict) and doc_upload.get("doc_id") else None
        if upload_doc_id is not None:
            state["created_doc_ids"].append(upload_doc_id)
        docs_payload = request_json(client, state, "documents list", "GET", "/api/documents", spec_path="/api/documents", validator=lambda p: None if isinstance(p, dict) and "documents" in p and "total" in p else "documents list mismatch")
        if upload_doc_id is not None and isinstance(docs_payload, dict) and not any(doc.get("id") == upload_doc_id for doc in docs_payload.get("documents", [])):
            add(state, name="documents list contains upload", method="GET", path="/api/documents", spec_path="/api/documents", status="FAIL", detail=f"uploaded doc {upload_doc_id} missing")
        if upload_doc_id is not None:
            request_json(client, state, "document detail", "GET", f"/api/documents/{upload_doc_id}", spec_path="/api/documents/{doc_id}", validator=lambda p: None if isinstance(p, dict) and p.get("id") == upload_doc_id else "document detail mismatch")
            request_json(client, state, "document update", "PUT", f"/api/documents/{upload_doc_id}", spec_path="/api/documents/{doc_id}", json={"title": "verify-doc-upload-updated", "metadata": {"verified": True}}, validator=lambda p: None if isinstance(p, dict) and p.get("metadata", {}).get("verified") is True else "document update mismatch")
            request_check(client, state, "document download", "GET", f"/api/documents/{upload_doc_id}/download", spec_path="/api/documents/{doc_id}/download", validator=lambda r: None if "attachment" in r.headers.get("content-disposition", "").lower() else "document download mismatch")
        else:
            for name, method, spec_path in [("document detail", "GET", "/api/documents/{doc_id}"), ("document update", "PUT", "/api/documents/{doc_id}"), ("document download", "GET", "/api/documents/{doc_id}/download")]:
                record_blocked(state, name, method, spec_path, "documents upload did not return a usable doc_id")

        if sample_doc_id is not None:
            request_json(client, state, "sample document detail", "GET", f"/api/documents/{sample_doc_id}", spec_path="/api/documents/{doc_id}", validator=lambda p: None if isinstance(p, dict) and p.get("id") == sample_doc_id and p.get("source_download_url") else "sample document detail mismatch")
            request_check(client, state, "sample markdown download", "GET", f"/api/documents/{sample_doc_id}/download", spec_path="/api/documents/{doc_id}/download", validator=lambda r: None if "markdown" in r.headers.get("content-type", "").lower() else "sample markdown download mismatch")
            request_check(client, state, "sample source download", "GET", f"/api/documents/{sample_doc_id}/source-download", spec_path="/api/documents/{doc_id}/source-download", validator=lambda r: None if "pdf" in r.headers.get("content-type", "").lower() else "sample source download mismatch")
        else:
            for name, spec_path in [("sample document detail", "/api/documents/{doc_id}"), ("sample markdown download", "/api/documents/{doc_id}/download"), ("sample source download", "/api/documents/{doc_id}/source-download")]:
                record_blocked(state, name, "GET", spec_path, "no warmed paper document is available")

        crawl_main = request_json(client, state, "main crawl", "POST", "/api/crawl", spec_path="/api/crawl", json={"urls": [LOCAL_WEB_URL_PRIMARY], "use_proxy": False}, validator=lambda p: None if isinstance(p, dict) and p.get("task_id") and p.get("total_urls") == 1 else "main crawl mismatch")
        main_task_id = crawl_main.get("task_id") if isinstance(crawl_main, dict) else None
        if main_task_id:
            main_crawl_status = poll_main_crawl_status(client, main_task_id, {"completed"}, timeout_s=60.0)
            if main_crawl_status is None:
                add(state, name="main crawl status", method="GET", path=f"/api/crawl/{main_task_id}", spec_path="/api/crawl/{task_id}", status="FAIL", detail="main crawl did not complete within timeout")
            else:
                add(state, name="main crawl status", method="GET", path=f"/api/crawl/{main_task_id}", spec_path="/api/crawl/{task_id}", status="PASS", status_code=200, detail="OK", payload=main_crawl_status)
        else:
            record_blocked(state, "main crawl status", "GET", "/api/crawl/{task_id}", "main crawl did not produce a task_id")

        crawl_urls = request_json(client, state, "crawl urls", "POST", "/api/crawl/urls", spec_path="/api/crawl/urls", json={"urls": [LOCAL_WEB_URL_PRIMARY, FAILED_WEB_URL], "use_proxy": False, "auto_index": False, "max_concurrent": 2}, validator=lambda p: None if isinstance(p, dict) and p.get("success") is True and p.get("task_id") else "crawl urls failed")
        crawl_task_id = int(crawl_urls["task_id"]) if isinstance(crawl_urls, dict) and crawl_urls.get("task_id") else None
        if crawl_task_id is not None:
            state["created_task_ids"].append(crawl_task_id)
            poll_task_status(client, crawl_task_id, {"completed", "failed", "cancelled", "timeout"})
            request_json(client, state, "crawl records", "GET", "/api/crawl/records", spec_path="/api/crawl/records", params={"task_id": crawl_task_id}, validator=lambda p: None if isinstance(p, dict) and p.get("total", 0) >= 1 else "crawl records mismatch")
            request_json(client, state, "tasks list", "GET", "/api/tasks", spec_path="/api/tasks", validator=lambda p: None if isinstance(p, dict) and any(task.get("id") == crawl_task_id for task in p.get("tasks", [])) else "tasks list mismatch")
            request_json(client, state, "task detail", "GET", f"/api/tasks/{crawl_task_id}", spec_path="/api/tasks/{task_id}", validator=lambda p: None if isinstance(p, dict) and p.get("id") == crawl_task_id and "health" in p else "task detail mismatch")
            request_json(client, state, "task events", "GET", f"/api/tasks/{crawl_task_id}/events", spec_path="/api/tasks/{task_id}/events", validator=lambda p: None if isinstance(p, dict) and p.get("total", 0) >= 1 else "task events mismatch")
            request_json(client, state, "task resources", "GET", f"/api/tasks/{crawl_task_id}/resources", spec_path="/api/tasks/{task_id}/resources", validator=lambda p: None if isinstance(p, dict) and "samples" in p else "task resources mismatch")
            subtasks_payload = request_json(client, state, "task subtasks", "GET", f"/api/tasks/{crawl_task_id}/subtasks", spec_path="/api/tasks/{task_id}/subtasks", validator=lambda p: None if isinstance(p, dict) and p.get("total", 0) >= 1 else "task subtasks mismatch")
            workflow_payload = request_json(client, state, "task workflow", "GET", f"/api/tasks/{crawl_task_id}/workflow", spec_path="/api/tasks/{task_id}/workflow", validator=lambda p: None if isinstance(p, dict) and len(p.get("nodes", [])) >= 1 else "task workflow mismatch")
            request_sse(client, state, "task stream", f"/api/tasks/{crawl_task_id}/stream?after_event_id=999999&after_resource_id=999999", spec_path="/api/tasks/{task_id}/stream", expected_events=("snapshot", "final"))
            request_sse(client, state, "task workflow stream", f"/api/tasks/{crawl_task_id}/workflow/stream?after_event_id=999999&after_resource_id=999999", spec_path="/api/tasks/{task_id}/workflow/stream", expected_events=("snapshot", "final"))

            selected_subtask_id = None
            if isinstance(subtasks_payload, dict):
                for subtask in subtasks_payload.get("subtasks", []):
                    if (subtask.get("metadata") or {}).get("workflow_action_type") == "crawl_url":
                        selected_subtask_id = subtask["id"]
                        break
                if selected_subtask_id is None and subtasks_payload.get("subtasks"):
                    selected_subtask_id = subtasks_payload["subtasks"][0]["id"]
            selected_edge_id = workflow_payload["edges"][0]["id"] if isinstance(workflow_payload, dict) and workflow_payload.get("edges") else None
            if selected_subtask_id is not None:
                request_json(client, state, "task subtask detail", "GET", f"/api/tasks/{crawl_task_id}/subtasks/{selected_subtask_id}", spec_path="/api/tasks/{task_id}/subtasks/{subtask_id}", validator=lambda p: None if isinstance(p, dict) and p.get("id") == selected_subtask_id else "task subtask detail mismatch")
                request_json(client, state, "task retry", "POST", f"/api/tasks/{crawl_task_id}/retry", spec_path="/api/tasks/{task_id}/retry", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "task retry failed")
                poll_task_status(client, crawl_task_id, {"completed", "failed", "cancelled", "timeout"})
                request_json(client, state, "task subtask retry", "POST", f"/api/tasks/{crawl_task_id}/subtasks/{selected_subtask_id}/retry", spec_path="/api/tasks/{task_id}/subtasks/{subtask_id}/retry", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "task subtask retry failed")
                poll_task_status(client, crawl_task_id, {"completed", "failed", "cancelled", "timeout"})
                request_json(client, state, "workflow node skip", "POST", f"/api/tasks/{crawl_task_id}/workflow/nodes/{selected_subtask_id}/actions/skip", spec_path="/api/tasks/{task_id}/workflow/nodes/{subtask_id}/actions/{action}", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "workflow node skip failed")
            else:
                for name, method, spec_path in [("task subtask detail", "GET", "/api/tasks/{task_id}/subtasks/{subtask_id}"), ("task retry", "POST", "/api/tasks/{task_id}/retry"), ("task subtask retry", "POST", "/api/tasks/{task_id}/subtasks/{subtask_id}/retry"), ("workflow node skip", "POST", "/api/tasks/{task_id}/workflow/nodes/{subtask_id}/actions/{action}")]:
                    record_blocked(state, name, method, spec_path, "no subtask is available")
            if selected_edge_id is not None:
                request_json(client, state, "workflow edge disable", "POST", f"/api/tasks/{crawl_task_id}/workflow/edges/{selected_edge_id}/actions/disable", spec_path="/api/tasks/{task_id}/workflow/edges/{edge_id}/actions/{action}", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "workflow edge disable failed")
                request_json(client, state, "workflow edge enable", "POST", f"/api/tasks/{crawl_task_id}/workflow/edges/{selected_edge_id}/actions/enable", spec_path="/api/tasks/{task_id}/workflow/edges/{edge_id}/actions/{action}", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "workflow edge enable failed")
            else:
                for name in ("workflow edge disable", "workflow edge enable"):
                    record_blocked(state, name, "POST", "/api/tasks/{task_id}/workflow/edges/{edge_id}/actions/{action}", "no workflow edge is available")
        else:
            for name, method, spec_path in [("crawl records", "GET", "/api/crawl/records"), ("tasks list", "GET", "/api/tasks"), ("task detail", "GET", "/api/tasks/{task_id}"), ("task events", "GET", "/api/tasks/{task_id}/events"), ("task resources", "GET", "/api/tasks/{task_id}/resources"), ("task subtasks", "GET", "/api/tasks/{task_id}/subtasks"), ("task subtask detail", "GET", "/api/tasks/{task_id}/subtasks/{subtask_id}"), ("task workflow", "GET", "/api/tasks/{task_id}/workflow"), ("task stream", "GET", "/api/tasks/{task_id}/stream"), ("task workflow stream", "GET", "/api/tasks/{task_id}/workflow/stream"), ("task retry", "POST", "/api/tasks/{task_id}/retry"), ("task subtask retry", "POST", "/api/tasks/{task_id}/subtasks/{subtask_id}/retry"), ("workflow node skip", "POST", "/api/tasks/{task_id}/workflow/nodes/{subtask_id}/actions/{action}"), ("workflow edge disable", "POST", "/api/tasks/{task_id}/workflow/edges/{edge_id}/actions/{action}"), ("workflow edge enable", "POST", "/api/tasks/{task_id}/workflow/edges/{edge_id}/actions/{action}")]:
                record_blocked(state, name, method, spec_path, "crawl urls did not produce a task_id")

        cancel_task = request_json(client, state, "crawl html", "POST", "/api/crawl/html", spec_path="/api/crawl/html", files={"file": ("bookmarks.html", f'<html><body><a href=\"{LOCAL_WEB_URL_PRIMARY}\">home</a><a href=\"{LOCAL_WEB_URL_SECONDARY}\">docs</a></body></html>'.encode("utf-8"), "text/html")}, data={"use_proxy": "false", "auto_index": "false", "auto_crawl": "true"}, validator=lambda p: None if isinstance(p, dict) and p.get("task_id") else "crawl html failed")
        cancel_task_id = int(cancel_task["task_id"]) if isinstance(cancel_task, dict) and cancel_task.get("task_id") else None
        if cancel_task_id is not None:
            state["created_task_ids"].append(cancel_task_id)
            request_json(client, state, "task cancel", "POST", f"/api/tasks/{cancel_task_id}/cancel", spec_path="/api/tasks/{task_id}/cancel", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "task cancel failed")
            deleted_task = request_json(client, state, "task delete", "DELETE", f"/api/tasks/{cancel_task_id}", spec_path="/api/tasks/{task_id}", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True else "task delete failed")
            if deleted_task is not None and cancel_task_id in state["created_task_ids"]:
                state["created_task_ids"].remove(cancel_task_id)
        else:
            record_blocked(state, "task cancel", "POST", "/api/tasks/{task_id}/cancel", "crawl html did not produce a task_id")
            record_blocked(state, "task delete", "DELETE", "/api/tasks/{task_id}", "crawl html did not produce a task_id")

        request_json(client, state, "papers search get", "GET", "/api/papers/search", spec_path="/api/papers/search", params={"query": sample_query, "top_k": 5, "method": "hybrid"}, validator=lambda p: None if isinstance(p, dict) and len(p.get("papers", [])) >= 1 else "papers search get mismatch")
        request_json(client, state, "papers search post", "POST", "/api/papers/search", spec_path="/api/papers/search", json={"query": sample_query, "top_k": 5, "method": "hybrid", "aggregate_by_paper": True}, validator=lambda p: None if isinstance(p, dict) and p.get("total_results", 0) >= 1 else "papers search post mismatch")

        paper_id = "verify-paper-live-strict"
        batch_id = "verify-paper-live-strict-batch"
        request_json(client, state, "paper index", "POST", "/api/papers/index", spec_path="/api/papers/index", json={"paper_id": paper_id, "title": "Verification Paper", "abstract": "Verification abstract.", "authors": ["Verifier One", "Verifier Two"], "source": "arxiv", "url": REAL_PAPER_URL_PRIMARY, "conference": "Other", "venue": "VerificationConf", "year": 2026, "paper_type": "unknown", "keywords": ["verification"], "content": "Verification body content."}, validator=lambda p: None if isinstance(p, dict) and p.get("success") is True and p.get("paper_id") == paper_id else "paper index failed")
        state["created_paper_ids"].append(paper_id)
        request_json(client, state, "paper detail", "GET", f"/api/papers/{paper_id}", spec_path="/api/papers/{paper_id}", validator=lambda p: None if isinstance(p, dict) and p.get("id") == paper_id and p.get("chunk_count", 0) >= 1 else "paper detail mismatch")
        request_json(client, state, "paper batch index", "POST", "/api/papers/index/batch", spec_path="/api/papers/index/batch", json={"papers": [{"paper_id": batch_id, "title": "Verification Batch Paper", "abstract": "Batch verification abstract.", "authors": ["Verifier Batch"], "source": "arxiv", "url": REAL_PAPER_URL_BATCH, "conference": "Other", "venue": "VerificationConf", "year": 2026, "paper_type": "unknown", "keywords": ["batch"], "content": "Batch paper content."}]}, validator=lambda p: None if isinstance(p, dict) and p.get("results", {}).get(batch_id) else "paper batch index failed")
        state["created_paper_ids"].append(batch_id)
        request_json(client, state, "paper delete", "DELETE", f"/api/papers/{paper_id}", spec_path="/api/papers/{paper_id}", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True and p.get("paper_id") == paper_id else "paper delete failed")
        if paper_id in state["created_paper_ids"]:
            state["created_paper_ids"].remove(paper_id)

        if upload_doc_id is not None:
            deleted_doc = request_json(client, state, "document delete", "DELETE", f"/api/documents/{upload_doc_id}", spec_path="/api/documents/{doc_id}", validator=lambda p: None if isinstance(p, dict) and p.get("success") is True and p.get("doc_id") == upload_doc_id else "document delete failed")
            if deleted_doc is not None and upload_doc_id in state["created_doc_ids"]:
                state["created_doc_ids"].remove(upload_doc_id)
        else:
            record_blocked(state, "document delete", "DELETE", "/api/documents/{doc_id}", "documents upload did not return a usable doc_id")
    finally:
        cleanup(client, state)
        client.close()

    result_by_endpoint = {(item["spec_path"], item["method"]): item for item in state["results"]}
    unverified = [f"{method.upper()} {path}" for path, methods in openapi.get("paths", {}).items() for method in methods if (path, method.upper()) not in result_by_endpoint]
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": BASE_URL,
        "total_checks": len(state["results"]),
        "failures": [item for item in state["results"] if item["status"] == "FAIL"],
        "blocked": [item for item in state["results"] if item["status"] == "BLOCKED"],
        "unverified": unverified,
        "results": state["results"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["failures"] or report["blocked"] or unverified:
        print(f"FAILURES: {len(report['failures'])}")
        print(f"BLOCKED: {len(report['blocked'])}")
        print(f"UNVERIFIED: {len(unverified)}")
        print(f"REPORT: {REPORT_PATH}")
        return 1
    print("ALL LIVE API CONTRACT CHECKS PASSED")
    print(f"REPORT: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
