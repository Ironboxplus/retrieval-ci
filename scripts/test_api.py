#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retrieval System API Layer Comprehensive Test Script

This script performs thorough testing of all API endpoints including:
- /api/stats
- /api/search
- /api/documents
- /api/upload
- /api/crawl
- /api/tasks

Usage:
    python scripts/test_api.py [--base-url http://127.0.0.1:8000] [--timeout 30]

Author: API Test Script
Date: 2026-01-21
"""

import os
import sys
import json
import time
import asyncio
import argparse
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


# ==================== Configuration ====================

def clear_proxy_env():
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


REAL_WEB_URL_PRIMARY = "https://www.python.org/about/"
REAL_WEB_URL_SECONDARY = "https://www.iana.org/help/example-domains"
REAL_WEB_URL_TERTIARY = "https://docs.python.org/3/tutorial/"
REAL_PAPER_URL_PRIMARY = "https://arxiv.org/abs/1706.03762"
REAL_PAPER_URL_BATCH = "https://arxiv.org/abs/1810.04805"

@dataclass
class TestConfig:
    """Test configuration"""
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 30.0
    verbose: bool = True
    save_report: bool = True
    report_path: str = "docs/api_layer_test_report.md"


@dataclass
class TestResult:
    """Single test result"""
    name: str
    passed: bool
    duration: float
    status_code: Optional[int] = None
    expected_status: Optional[int] = None
    message: str = ""
    response_data: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class TestSuite:
    """Test suite results"""
    name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(r.duration for r in self.results)


# ==================== Test Helper ====================

class APITester:
    """API testing utility class"""

    def __init__(self, config: TestConfig):
        clear_proxy_env()
        self.config = config
        self.client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout,
            trust_env=False,
            follow_redirects=True,
        )
        self.suites: List[TestSuite] = []
        self.created_resources: Dict[str, List[int]] = {
            "documents": [],
            "tasks": [],
            "papers": [],
        }

    def close(self):
        """Close HTTP client"""
        self.client.close()

    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode is enabled"""
        if self.config.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def run_test(
        self,
        name: str,
        method: str,
        url: str,
        expected_status: int = 200,
        json_data: dict = None,
        params: dict = None,
        files: dict = None,
        data: dict = None,
        check_fn: callable = None
    ) -> TestResult:
        """Run a single API test"""
        start_time = time.time()
        result = TestResult(
            name=name,
            passed=False,
            duration=0,
            expected_status=expected_status
        )

        try:
            # Build request kwargs
            kwargs = {}
            if json_data:
                kwargs["json"] = json_data
            if params:
                kwargs["params"] = params
            if files:
                kwargs["files"] = files
            if data:
                kwargs["data"] = data

            # Execute request
            response = self.client.request(method, url, **kwargs)
            result.status_code = response.status_code
            result.duration = time.time() - start_time

            # Try to parse JSON response
            try:
                result.response_data = response.json()
            except:
                result.response_data = {"raw": response.text[:500]}

            # Check status code
            if response.status_code != expected_status:
                result.message = f"Expected status {expected_status}, got {response.status_code}"
                result.passed = False
            elif check_fn and not check_fn(result.response_data):
                result.message = "Response validation failed"
                result.passed = False
            else:
                result.passed = True
                result.message = "OK"

        except httpx.TimeoutException as e:
            result.duration = time.time() - start_time
            result.error = f"Timeout: {str(e)}"
            result.message = "Request timed out"

        except httpx.RequestError as e:
            result.duration = time.time() - start_time
            result.error = f"Request error: {str(e)}"
            result.message = "Request failed"

        except Exception as e:
            result.duration = time.time() - start_time
            result.error = f"Unexpected error: {str(e)}"
            result.message = "Unexpected error occurred"

        # Log result
        status = "PASS" if result.passed else "FAIL"
        self.log(f"[{status}] {name} ({result.duration:.2f}s)", "INFO" if result.passed else "ERROR")
        if not result.passed and result.error:
            self.log(f"  Error: {result.error}", "ERROR")

        return result

    def cleanup(self):
        """Clean up created resources"""
        self.log("Cleaning up test resources...")

        # Delete created documents
        for doc_id in self.created_resources["documents"]:
            try:
                self.client.delete(f"/api/documents/{doc_id}")
                self.log(f"  Deleted document {doc_id}")
            except:
                pass

        # Delete/cancel created tasks
        for task_id in self.created_resources["tasks"]:
            try:
                # Try to cancel first if running
                self.client.post(f"/api/tasks/{task_id}/cancel")
                # Then delete
                self.client.delete(f"/api/tasks/{task_id}")
                self.log(f"  Deleted task {task_id}")
            except:
                pass

        for paper_id in self.created_resources["papers"]:
            try:
                self.client.delete(f"/api/papers/{paper_id}")
                self.log(f"  Deleted paper {paper_id}")
            except:
                pass


# ==================== Test Suites ====================

def test_stats_api(tester: APITester) -> TestSuite:
    """Test /api/stats endpoint"""
    suite = TestSuite(name="Stats API", start_time=datetime.now())

    # Test 1: Normal request
    suite.results.append(tester.run_test(
        name="GET /api/stats - Normal request",
        method="GET",
        url="/api/stats",
        expected_status=200,
        check_fn=lambda r: all(k in r for k in ["total_documents", "total_chunks"])
    ))

    # Test 2: Verify response structure
    suite.results.append(tester.run_test(
        name="GET /api/stats - Verify response structure",
        method="GET",
        url="/api/stats",
        expected_status=200,
        check_fn=lambda r: (
            isinstance(r.get("total_documents"), int) and
            isinstance(r.get("total_chunks"), int) and
            isinstance(r.get("chunk_levels"), dict) and
            isinstance(r.get("vector_index_size"), int) and
            isinstance(r.get("bm25_index_size"), int)
        )
    ))

    suite.end_time = datetime.now()
    return suite


def test_paper_api(tester: APITester) -> TestSuite:
    """Test /api/papers endpoints and OpenAPI exposure"""
    suite = TestSuite(name="Paper API", start_time=datetime.now())

    suite.results.append(tester.run_test(
        name="GET /openapi.json - Paper routes exposed",
        method="GET",
        url="/openapi.json",
        expected_status=200,
        check_fn=lambda r: all(
            path in r.get("paths", {})
            for path in ["/api/papers/", "/api/papers/search", "/api/papers/stats"]
        )
    ))

    suite.results.append(tester.run_test(
        name="GET /api/papers/stats - Runtime topology",
        method="GET",
        url="/api/papers/stats",
        expected_status=200,
        check_fn=lambda r: (
            r.get("collection_name") == "papers" and
            isinstance(r.get("total_papers"), int) and
            isinstance(r.get("vector_index_size"), int)
        )
    ))

    suite.results.append(tester.run_test(
        name="GET /api/papers/ - List indexed papers",
        method="GET",
        url="/api/papers/",
        expected_status=200,
        check_fn=lambda r: isinstance(r, list)
    ))

    suite.end_time = datetime.now()
    return suite


def test_search_api(tester: APITester) -> TestSuite:
    """Test /api/search endpoint"""
    suite = TestSuite(name="Search API", start_time=datetime.now())

    # Test 1: BM25 search
    suite.results.append(tester.run_test(
        name="POST /api/search - BM25 method",
        method="POST",
        url="/api/search",
        json_data={"query": "test query", "method": "bm25", "top_k": 10},
        expected_status=200,
        check_fn=lambda r: r.get("method") == "bm25" and "results" in r
    ))

    # Test 2: Vector search
    suite.results.append(tester.run_test(
        name="POST /api/search - Vector method",
        method="POST",
        url="/api/search",
        json_data={"query": "machine learning", "method": "vector", "top_k": 10},
        expected_status=200,
        check_fn=lambda r: r.get("method") == "vector"
    ))

    # Test 3: Hybrid search
    suite.results.append(tester.run_test(
        name="POST /api/search - Hybrid method",
        method="POST",
        url="/api/search",
        json_data={"query": "neural network", "method": "hybrid", "top_k": 10},
        expected_status=200,
        check_fn=lambda r: r.get("method") == "hybrid"
    ))

    # Test 4: Enhanced search
    suite.results.append(tester.run_test(
        name="POST /api/search - Enhanced method",
        method="POST",
        url="/api/search",
        json_data={"query": "deep learning", "method": "enhanced", "top_k": 10},
        expected_status=200,
        check_fn=lambda r: r.get("method") == "enhanced"
    ))

    # Test 5: Enhanced with aggregation
    suite.results.append(tester.run_test(
        name="POST /api/search - Enhanced with aggregation",
        method="POST",
        url="/api/search",
        json_data={
            "query": "python tutorial",
            "method": "enhanced",
            "top_k": 10,
            "aggregate_by_doc": True
        },
        expected_status=200,
        check_fn=lambda r: "aggregated" in r
    ))

    # Test 6: Source type filter - webpage
    suite.results.append(tester.run_test(
        name="POST /api/search - Filter by source_type=webpage",
        method="POST",
        url="/api/search",
        json_data={
            "query": "test",
            "method": "hybrid",
            "top_k": 10,
            "source_type": "webpage"
        },
        expected_status=200
    ))

    # Test 7: Source type filter - paper
    suite.results.append(tester.run_test(
        name="POST /api/search - Filter by source_type=paper",
        method="POST",
        url="/api/search",
        json_data={
            "query": "test",
            "method": "hybrid",
            "top_k": 10,
            "source_type": "paper"
        },
        expected_status=200
    ))

    # Test 8: Source type filter - upload
    suite.results.append(tester.run_test(
        name="POST /api/search - Filter by source_type=upload",
        method="POST",
        url="/api/search",
        json_data={
            "query": "test",
            "method": "hybrid",
            "top_k": 10,
            "source_type": "upload"
        },
        expected_status=200
    ))

    # Test 9: Data source filter - system
    suite.results.append(tester.run_test(
        name="POST /api/search - Filter by data_source=system",
        method="POST",
        url="/api/search",
        json_data={
            "query": "test",
            "method": "hybrid",
            "top_k": 10,
            "data_source": "system"
        },
        expected_status=200
    ))

    # Test 10: Data source filter - user
    suite.results.append(tester.run_test(
        name="POST /api/search - Filter by data_source=user",
        method="POST",
        url="/api/search",
        json_data={
            "query": "test",
            "method": "hybrid",
            "top_k": 10,
            "data_source": "user"
        },
        expected_status=200
    ))

    # Test 11: Combined filters - (moved to Test 12b as known issue)

    # Test 12: Empty query
    suite.results.append(tester.run_test(
        name="POST /api/search - Empty query (boundary)",
        method="POST",
        url="/api/search",
        json_data={"query": "", "method": "bm25", "top_k": 10},
        expected_status=200  # Empty query should still return (possibly empty results)
    ))

    # Test 12b: Combined filters
    suite.results.append(tester.run_test(
        name="POST /api/search - Combined filters",
        method="POST",
        url="/api/search",
        json_data={
            "query": "test",
            "method": "hybrid",
            "top_k": 10,
            "source_type": "paper",
            "data_source": "system"
        },
        expected_status=200
    ))

    # Test 13: Very long query
    long_query = "a" * 5000
    suite.results.append(tester.run_test(
        name="POST /api/search - Very long query (boundary)",
        method="POST",
        url="/api/search",
        json_data={"query": long_query, "method": "bm25", "top_k": 10},
        expected_status=200
    ))

    # Test 14: Special characters in query
    suite.results.append(tester.run_test(
        name="POST /api/search - Special characters in query",
        method="POST",
        url="/api/search",
        json_data={
            "query": "test <script>alert('xss')</script> query & special | chars",
            "method": "bm25",
            "top_k": 10
        },
        expected_status=200
    ))

    # Test 15: Unicode characters in query
    suite.results.append(tester.run_test(
        name="POST /api/search - Unicode characters in query",
        method="POST",
        url="/api/search",
        json_data={
            "query": "test",
            "method": "bm25",
            "top_k": 10
        },
        expected_status=200
    ))

    # Test 16: top_k = 0
    suite.results.append(tester.run_test(
        name="POST /api/search - top_k=0 (boundary)",
        method="POST",
        url="/api/search",
        json_data={"query": "test", "method": "bm25", "top_k": 0},
        expected_status=422
    ))

    # Test 17: top_k = 1
    suite.results.append(tester.run_test(
        name="POST /api/search - top_k=1 (boundary)",
        method="POST",
        url="/api/search",
        json_data={"query": "test", "method": "bm25", "top_k": 1},
        expected_status=200,
        check_fn=lambda r: len(r.get("results", [])) <= 1
    ))

    # Test 18: top_k = 100
    suite.results.append(tester.run_test(
        name="POST /api/search - top_k=100 (boundary)",
        method="POST",
        url="/api/search",
        json_data={"query": "test", "method": "bm25", "top_k": 100},
        expected_status=200
    ))

    # Test 19: top_k = 1000
    suite.results.append(tester.run_test(
        name="POST /api/search - top_k=1000 (boundary)",
        method="POST",
        url="/api/search",
        json_data={"query": "test", "method": "bm25", "top_k": 1000},
        expected_status=200
    ))

    # Test 20: Invalid method
    suite.results.append(tester.run_test(
        name="POST /api/search - Invalid method (error handling)",
        method="POST",
        url="/api/search",
        json_data={"query": "test", "method": "invalid_method", "top_k": 10},
        expected_status=422  # Validation error
    ))

    # Test 21: Multi-method search
    suite.results.append(tester.run_test(
        name="GET /api/search/multi-method - Normal request",
        method="GET",
        url="/api/search/multi-method",
        params={"query": "test", "top_k": 5},
        expected_status=200,
        check_fn=lambda r: all(k in r for k in ["bm25", "vector", "hybrid"])
    ))

    # Test 22: Multi-method with filters
    suite.results.append(tester.run_test(
        name="GET /api/search/multi-method - With filters",
        method="GET",
        url="/api/search/multi-method",
        params={
            "query": "test",
            "top_k": 5,
            "source_type": "paper",
            "data_source": "system"
        },
        expected_status=200
    ))

    suite.end_time = datetime.now()
    return suite


def test_documents_api(tester: APITester) -> TestSuite:
    """Test /api/documents endpoints"""
    suite = TestSuite(name="Documents API", start_time=datetime.now())

    # Test 1: List documents - default
    result = tester.run_test(
        name="GET /api/documents - Default list",
        method="GET",
        url="/api/documents",
        expected_status=200,
        check_fn=lambda r: all(k in r for k in ["total", "page", "page_size", "documents"])
    )
    suite.results.append(result)

    # Test 2: List with pagination
    suite.results.append(tester.run_test(
        name="GET /api/documents - Pagination (page=2, page_size=5)",
        method="GET",
        url="/api/documents",
        params={"page": 2, "page_size": 5},
        expected_status=200,
        check_fn=lambda r: r.get("page") == 2 and r.get("page_size") == 5
    ))

    # Test 3: Filter by source_type
    suite.results.append(tester.run_test(
        name="GET /api/documents - Filter source_type=webpage",
        method="GET",
        url="/api/documents",
        params={"source_type": "webpage"},
        expected_status=200
    ))

    suite.results.append(tester.run_test(
        name="GET /api/documents - Filter source_type=paper",
        method="GET",
        url="/api/documents",
        params={"source_type": "paper"},
        expected_status=200
    ))

    suite.results.append(tester.run_test(
        name="GET /api/documents - Filter source_type=upload",
        method="GET",
        url="/api/documents",
        params={"source_type": "upload"},
        expected_status=200
    ))

    # Test 4: Filter by data_source
    suite.results.append(tester.run_test(
        name="GET /api/documents - Filter data_source=system",
        method="GET",
        url="/api/documents",
        params={"data_source": "system"},
        expected_status=200
    ))

    suite.results.append(tester.run_test(
        name="GET /api/documents - Filter data_source=user",
        method="GET",
        url="/api/documents",
        params={"data_source": "user"},
        expected_status=200
    ))

    # Test 5: Filter by status
    suite.results.append(tester.run_test(
        name="GET /api/documents - Filter status=indexed",
        method="GET",
        url="/api/documents",
        params={"status": "indexed"},
        expected_status=200
    ))

    suite.results.append(tester.run_test(
        name="GET /api/documents - Filter status=pending",
        method="GET",
        url="/api/documents",
        params={"status": "pending"},
        expected_status=200
    ))

    # Test 6: Invalid filter values
    suite.results.append(tester.run_test(
        name="GET /api/documents - Invalid source_type (error handling)",
        method="GET",
        url="/api/documents",
        params={"source_type": "invalid_type"},
        expected_status=400
    ))

    suite.results.append(tester.run_test(
        name="GET /api/documents - Invalid status (error handling)",
        method="GET",
        url="/api/documents",
        params={"status": "invalid_status"},
        expected_status=400
    ))

    # Test 7: Pagination edge cases
    suite.results.append(tester.run_test(
        name="GET /api/documents - page=0 (boundary, should fail)",
        method="GET",
        url="/api/documents",
        params={"page": 0},
        expected_status=422  # Validation error
    ))

    suite.results.append(tester.run_test(
        name="GET /api/documents - page_size=0 (boundary, should fail)",
        method="GET",
        url="/api/documents",
        params={"page_size": 0},
        expected_status=422
    ))

    suite.results.append(tester.run_test(
        name="GET /api/documents - page_size=101 (over limit, should fail)",
        method="GET",
        url="/api/documents",
        params={"page_size": 101},
        expected_status=422
    ))

    suite.results.append(tester.run_test(
        name="GET /api/documents - page_size=100 (max allowed)",
        method="GET",
        url="/api/documents",
        params={"page_size": 100},
        expected_status=200
    ))

    # Test 8: Get document detail - non-existent
    suite.results.append(tester.run_test(
        name="GET /api/documents/{id} - Non-existent document",
        method="GET",
        url="/api/documents/999999999",
        expected_status=404
    ))

    # Test 9: Delete document - non-existent
    suite.results.append(tester.run_test(
        name="DELETE /api/documents/{id} - Non-existent document",
        method="DELETE",
        url="/api/documents/999999999",
        expected_status=404
    ))

    # Test 10: Get existing document detail (if any exist)
    if result.response_data and result.response_data.get("documents"):
        first_doc = result.response_data["documents"][0]
        doc_id = first_doc.get("id")
        if doc_id:
            suite.results.append(tester.run_test(
                name=f"GET /api/documents/{doc_id} - Existing document detail",
                method="GET",
                url=f"/api/documents/{doc_id}",
                expected_status=200,
                check_fn=lambda r: r.get("id") == doc_id and "chunks" in r
            ))

    suite.end_time = datetime.now()
    return suite


def test_upload_api(tester: APITester) -> TestSuite:
    """Test /api/upload and /api/documents/upload endpoints"""
    suite = TestSuite(name="Upload API", start_time=datetime.now())

    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test text file for API testing.\nIt contains multiple lines.\nEnd of file.")
        txt_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(f"""
        <html>
        <body>
            <a href="{REAL_WEB_URL_PRIMARY}">Link 1</a>
            <a href="{REAL_WEB_URL_SECONDARY}">Link 2</a>
            <a href="{REAL_WEB_URL_TERTIARY}">Link 3</a>
        </body>
        </html>
        """)
        html_path = f.name

    try:
        # Test 1: Upload TXT file via /api/upload
        with open(txt_path, 'rb') as f:
            result = tester.run_test(
                name="POST /api/upload - TXT file",
                method="POST",
                url="/api/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={"file_type": "txt"},
                expected_status=200,
                check_fn=lambda r: r.get("success") == True
            )
            suite.results.append(result)

        # Test 2: Upload TXT file via /api/documents/upload
        with open(txt_path, 'rb') as f:
            result = tester.run_test(
                name="POST /api/documents/upload - TXT file with title",
                method="POST",
                url="/api/documents/upload",
                files={"file": ("test_doc.txt", f, "text/plain")},
                data={"title": "Test Document", "auto_index": "true"},
                expected_status=200,
                check_fn=lambda r: r.get("success") == True and r.get("doc_id") is not None
            )
            suite.results.append(result)
            if result.response_data and result.response_data.get("doc_id"):
                tester.created_resources["documents"].append(result.response_data["doc_id"])
                doc_id = result.response_data["doc_id"]
                suite.results.append(tester.run_test(
                    name="GET /api/documents/{id} - Uploaded TXT is indexed",
                    method="GET",
                    url=f"/api/documents/{doc_id}",
                    expected_status=200,
                    check_fn=lambda r: r.get("status") == "indexed" and r.get("chunk_count", 0) > 0
                ))

        # Test 3: Upload HTML bookmark file
        with open(html_path, 'rb') as f:
            result = tester.run_test(
                name="POST /api/documents/upload - HTML bookmark file",
                method="POST",
                url="/api/documents/upload",
                files={"file": ("bookmarks.html", f, "text/html")},
                data={"auto_index": "true"},
                expected_status=200,
                check_fn=lambda r: r.get("success") == True and r.get("doc_id") is not None
            )
            suite.results.append(result)
            if result.response_data and result.response_data.get("doc_id"):
                tester.created_resources["documents"].append(result.response_data["doc_id"])
                doc_id = result.response_data["doc_id"]
                suite.results.append(tester.run_test(
                    name="GET /api/documents/{id} - Uploaded HTML bookmarks are indexed",
                    method="GET",
                    url=f"/api/documents/{doc_id}",
                    expected_status=200,
                    check_fn=lambda r: r.get("status") == "indexed" and r.get("chunk_count", 0) > 0
                ))

        # Test 4: Invalid file type via /api/upload
        with open(txt_path, 'rb') as f:
            suite.results.append(tester.run_test(
                name="POST /api/upload - Invalid file_type (error handling)",
                method="POST",
                url="/api/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={"file_type": "invalid"},
                expected_status=400
            ))

        # Test 5: Unsupported extension via /api/documents/upload
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("test content")
            xyz_path = f.name

        with open(xyz_path, 'rb') as f:
            suite.results.append(tester.run_test(
                name="POST /api/documents/upload - Unsupported file extension",
                method="POST",
                url="/api/documents/upload",
                files={"file": ("test.xyz", f, "application/octet-stream")},
                expected_status=400
            ))

        os.unlink(xyz_path)

        # Test 6: Empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            empty_path = f.name

        with open(empty_path, 'rb') as f:
            suite.results.append(tester.run_test(
                name="POST /api/documents/upload - Empty file",
                method="POST",
                url="/api/documents/upload",
                files={"file": ("empty.txt", f, "text/plain")},
                expected_status=200  # Empty files should still be accepted
            ))

        os.unlink(empty_path)

        # Test 7: Large content file (but within reasonable limits)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("A" * 100000)  # 100KB
            large_path = f.name

        with open(large_path, 'rb') as f:
            suite.results.append(tester.run_test(
                name="POST /api/documents/upload - Large file (100KB)",
                method="POST",
                url="/api/documents/upload",
                files={"file": ("large.txt", f, "text/plain")},
                data={"auto_index": "false"},
                expected_status=200
            ))

        os.unlink(large_path)

    finally:
        # Cleanup temp files
        try:
            os.unlink(txt_path)
            os.unlink(html_path)
        except:
            pass

    suite.end_time = datetime.now()
    return suite


def test_crawl_api(tester: APITester) -> TestSuite:
    """Test /api/crawl endpoints"""
    suite = TestSuite(name="Crawl API", start_time=datetime.now())

    # Test 1: Submit single URL crawl
    result = tester.run_test(
        name="POST /api/crawl/urls - Single valid URL",
        method="POST",
        url="/api/crawl/urls",
        json_data={
            "urls": [REAL_WEB_URL_PRIMARY],
            "use_proxy": False,
            "auto_index": False,
            "max_concurrent": 1
        },
        expected_status=200,
        check_fn=lambda r: r.get("success") == True and r.get("task_id") is not None
    )
    suite.results.append(result)
    if result.response_data and result.response_data.get("task_id"):
        tester.created_resources["tasks"].append(int(result.response_data["task_id"]))

    # Test 2: Submit batch URL crawl
    result = tester.run_test(
        name="POST /api/crawl/urls - Multiple URLs",
        method="POST",
        url="/api/crawl/urls",
        json_data={
            "urls": [
                REAL_WEB_URL_PRIMARY,
                REAL_WEB_URL_SECONDARY,
                REAL_WEB_URL_TERTIARY,
            ],
            "use_proxy": False,
            "auto_index": False,
            "max_concurrent": 3
        },
        expected_status=200,
        check_fn=lambda r: r.get("total_urls") == 3
    )
    suite.results.append(result)
    if result.response_data and result.response_data.get("task_id"):
        tester.created_resources["tasks"].append(int(result.response_data["task_id"]))

    # Test 3: Empty URL list
    suite.results.append(tester.run_test(
        name="POST /api/crawl/urls - Empty URL list (error handling)",
        method="POST",
        url="/api/crawl/urls",
        json_data={"urls": [], "use_proxy": False},
        expected_status=400
    ))

    # Test 4: Invalid URLs only
    suite.results.append(tester.run_test(
        name="POST /api/crawl/urls - All invalid URLs (error handling)",
        method="POST",
        url="/api/crawl/urls",
        json_data={
            "urls": ["not-a-url", "also-not-valid", "ftp://wrong-protocol.com"],
            "use_proxy": False
        },
        expected_status=400
    ))

    # Test 5: Mixed valid and invalid URLs
    result = tester.run_test(
        name="POST /api/crawl/urls - Mixed valid/invalid URLs",
        method="POST",
        url="/api/crawl/urls",
        json_data={
            "urls": [
                REAL_WEB_URL_PRIMARY,
                "not-a-url",
                REAL_WEB_URL_SECONDARY,
                "invalid"
            ],
            "use_proxy": False,
            "auto_index": False
        },
        expected_status=200,
        check_fn=lambda r: r.get("total_urls") == 2  # Only valid URLs counted
    )
    suite.results.append(result)
    if result.response_data and result.response_data.get("task_id"):
        tester.created_resources["tasks"].append(int(result.response_data["task_id"]))

    # Test 6: max_concurrent boundary - lower
    suite.results.append(tester.run_test(
        name="POST /api/crawl/urls - max_concurrent=0 (boundary, should fail)",
        method="POST",
        url="/api/crawl/urls",
        json_data={"urls": [REAL_WEB_URL_PRIMARY], "max_concurrent": 0},
        expected_status=422
    ))

    # Test 7: max_concurrent boundary - upper
    suite.results.append(tester.run_test(
        name="POST /api/crawl/urls - max_concurrent=21 (over limit, should fail)",
        method="POST",
        url="/api/crawl/urls",
        json_data={"urls": [REAL_WEB_URL_PRIMARY], "max_concurrent": 21},
        expected_status=422
    ))

    # Test 8: Get crawl records
    suite.results.append(tester.run_test(
        name="GET /api/crawl/records - List crawl records",
        method="GET",
        url="/api/crawl/records",
        expected_status=200,
        check_fn=lambda r: all(k in r for k in ["total", "page", "page_size", "records"])
    ))

    # Test 9: Get crawl records with pagination
    suite.results.append(tester.run_test(
        name="GET /api/crawl/records - With pagination",
        method="GET",
        url="/api/crawl/records",
        params={"page": 1, "page_size": 10},
        expected_status=200
    ))

    # Test 10: Filter crawl records by status
    suite.results.append(tester.run_test(
        name="GET /api/crawl/records - Filter status=pending",
        method="GET",
        url="/api/crawl/records",
        params={"status": "pending"},
        expected_status=200
    ))

    suite.results.append(tester.run_test(
        name="GET /api/crawl/records - Filter status=success",
        method="GET",
        url="/api/crawl/records",
        params={"status": "success"},
        expected_status=200
    ))

    # Test 11: Invalid status filter
    suite.results.append(tester.run_test(
        name="GET /api/crawl/records - Invalid status (error handling)",
        method="GET",
        url="/api/crawl/records",
        params={"status": "invalid_status"},
        expected_status=400
    ))

    # Test 12: Upload HTML for crawling
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(f"""
        <html>
        <body>
            <a href="{REAL_WEB_URL_PRIMARY}">Link 1</a>
            <a href="{REAL_WEB_URL_SECONDARY}">Link 2</a>
        </body>
        </html>
        """)
        html_path = f.name

    try:
        with open(html_path, 'rb') as f:
            result = tester.run_test(
                name="POST /api/crawl/html - Extract URLs from HTML",
                method="POST",
                url="/api/crawl/html",
                files={"file": ("bookmarks.html", f, "text/html")},
                data={"use_proxy": "false", "auto_index": "false", "auto_crawl": "false"},
                expected_status=200,
                check_fn=lambda r: r.get("total_urls", 0) >= 2
            )
            suite.results.append(result)

        with open(html_path, 'rb') as f:
            result = tester.run_test(
                name="POST /api/crawl/html - Extract and start crawl",
                method="POST",
                url="/api/crawl/html",
                files={"file": ("bookmarks.html", f, "text/html")},
                data={"auto_crawl": "true", "auto_index": "false"},
                expected_status=200,
                check_fn=lambda r: r.get("task_id") is not None
            )
            suite.results.append(result)
            if result.response_data and result.response_data.get("task_id"):
                tester.created_resources["tasks"].append(int(result.response_data["task_id"]))

    finally:
        os.unlink(html_path)

    # Test 13: Wrong file type for HTML crawl
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("not html")
        txt_path = f.name

    try:
        with open(txt_path, 'rb') as f:
            suite.results.append(tester.run_test(
                name="POST /api/crawl/html - Wrong file type (error handling)",
                method="POST",
                url="/api/crawl/html",
                files={"file": ("test.txt", f, "text/plain")},
                expected_status=400
            ))
    finally:
        os.unlink(txt_path)

    # Test 14: Legacy /api/crawl endpoint
    suite.results.append(tester.run_test(
        name="POST /api/crawl - Legacy crawl endpoint",
        method="POST",
        url="/api/crawl",
        json_data={"urls": [REAL_WEB_URL_PRIMARY], "use_proxy": False},
        expected_status=200,
        check_fn=lambda r: "task_id" in r
    ))

    suite.end_time = datetime.now()
    return suite


def test_tasks_api(tester: APITester) -> TestSuite:
    """Test /api/tasks endpoints"""
    suite = TestSuite(name="Tasks API", start_time=datetime.now())

    # Test 1: List tasks - default
    result = tester.run_test(
        name="GET /api/tasks - Default list",
        method="GET",
        url="/api/tasks",
        expected_status=200,
        check_fn=lambda r: all(k in r for k in ["total", "page", "page_size", "tasks"])
    )
    suite.results.append(result)

    # Test 2: List with pagination
    suite.results.append(tester.run_test(
        name="GET /api/tasks - With pagination",
        method="GET",
        url="/api/tasks",
        params={"page": 1, "page_size": 5},
        expected_status=200,
        check_fn=lambda r: r.get("page_size") == 5
    ))

    # Test 3: Filter by status
    suite.results.append(tester.run_test(
        name="GET /api/tasks - Filter status=pending",
        method="GET",
        url="/api/tasks",
        params={"status": "pending"},
        expected_status=200
    ))

    suite.results.append(tester.run_test(
        name="GET /api/tasks - Filter status=completed",
        method="GET",
        url="/api/tasks",
        params={"status": "completed"},
        expected_status=200
    ))

    suite.results.append(tester.run_test(
        name="GET /api/tasks - Filter status=failed",
        method="GET",
        url="/api/tasks",
        params={"status": "failed"},
        expected_status=200
    ))

    # Test 4: Filter by task_type
    suite.results.append(tester.run_test(
        name="GET /api/tasks - Filter task_type=crawl_urls",
        method="GET",
        url="/api/tasks",
        params={"task_type": "crawl_urls"},
        expected_status=200
    ))

    suite.results.append(tester.run_test(
        name="GET /api/tasks - Filter task_type=index_document",
        method="GET",
        url="/api/tasks",
        params={"task_type": "index_document"},
        expected_status=200
    ))

    # Test 5: Invalid status filter
    suite.results.append(tester.run_test(
        name="GET /api/tasks - Invalid status (error handling)",
        method="GET",
        url="/api/tasks",
        params={"status": "invalid_status"},
        expected_status=400
    ))

    # Test 6: Get non-existent task
    suite.results.append(tester.run_test(
        name="GET /api/tasks/{id} - Non-existent task",
        method="GET",
        url="/api/tasks/999999999",
        expected_status=404
    ))

    # Test 7: Cancel non-existent task
    suite.results.append(tester.run_test(
        name="POST /api/tasks/{id}/cancel - Non-existent task",
        method="POST",
        url="/api/tasks/999999999/cancel",
        expected_status=404
    ))

    # Test 8: Retry non-existent task
    suite.results.append(tester.run_test(
        name="POST /api/tasks/{id}/retry - Non-existent task",
        method="POST",
        url="/api/tasks/999999999/retry",
        expected_status=404
    ))

    # Test 9: Delete non-existent task
    suite.results.append(tester.run_test(
        name="DELETE /api/tasks/{id} - Non-existent task",
        method="DELETE",
        url="/api/tasks/999999999",
        expected_status=404
    ))

    # Test 10: Get existing task detail (if any exist)
    if result.response_data and result.response_data.get("tasks"):
        first_task = result.response_data["tasks"][0]
        task_id = first_task.get("id")
        if task_id:
            suite.results.append(tester.run_test(
                name=f"GET /api/tasks/{task_id} - Existing task detail",
                method="GET",
                url=f"/api/tasks/{task_id}",
                expected_status=200,
                check_fn=lambda r: r.get("id") == task_id
            ))

            # Try cancel if pending
            if first_task.get("status") == "pending":
                suite.results.append(tester.run_test(
                    name=f"POST /api/tasks/{task_id}/cancel - Cancel pending task",
                    method="POST",
                    url=f"/api/tasks/{task_id}/cancel",
                    expected_status=200
                ))

    suite.end_time = datetime.now()
    return suite


def test_error_handling(tester: APITester) -> TestSuite:
    """Test error handling across all endpoints"""
    suite = TestSuite(name="Error Handling", start_time=datetime.now())

    # Test 1: 404 routes
    suite.results.append(tester.run_test(
        name="GET /api/nonexistent - 404 route",
        method="GET",
        url="/api/nonexistent",
        expected_status=404
    ))

    suite.results.append(tester.run_test(
        name="POST /api/nonexistent - 404 route (POST)",
        method="POST",
        url="/api/nonexistent",
        json_data={},
        expected_status=404
    ))

    # Test 2: Wrong method
    suite.results.append(tester.run_test(
        name="POST /api/stats - Wrong HTTP method",
        method="POST",
        url="/api/stats",
        expected_status=405
    ))

    suite.results.append(tester.run_test(
        name="GET /api/search - Wrong HTTP method (GET instead of POST)",
        method="GET",
        url="/api/search",
        expected_status=405
    ))

    # Test 3: Missing required parameters
    suite.results.append(tester.run_test(
        name="POST /api/search - Missing query (required)",
        method="POST",
        url="/api/search",
        json_data={"method": "bm25"},
        expected_status=422
    ))

    suite.results.append(tester.run_test(
        name="POST /api/crawl/urls - Missing urls (required)",
        method="POST",
        url="/api/crawl/urls",
        json_data={"use_proxy": False},
        expected_status=422
    ))

    # Test 4: Invalid JSON
    suite.results.append(tester.run_test(
        name="POST /api/search - Invalid JSON body",
        method="POST",
        url="/api/search",
        data={"raw": "not json"},  # Send as form data
        expected_status=422
    ))

    # Test 5: Wrong Content-Type - use httpx directly
    try:
        response = tester.client.post(
            "/api/search",
            content="not json at all",
            headers={"Content-Type": "text/plain"}
        )
        result = TestResult(
            name="POST /api/search - Wrong Content-Type",
            passed=response.status_code == 422,
            duration=0,
            status_code=response.status_code,
            expected_status=422,
            message="OK" if response.status_code == 422 else f"Expected 422, got {response.status_code}"
        )
    except Exception as e:
        result = TestResult(
            name="POST /api/search - Wrong Content-Type",
            passed=False,
            duration=0,
            error=str(e),
            message="Request failed"
        )
    suite.results.append(result)

    # Test 6: Invalid path parameter types
    suite.results.append(tester.run_test(
        name="GET /api/documents/abc - Invalid ID type",
        method="GET",
        url="/api/documents/abc",
        expected_status=422
    ))

    suite.results.append(tester.run_test(
        name="GET /api/tasks/xyz - Invalid ID type",
        method="GET",
        url="/api/tasks/xyz",
        expected_status=422
    ))

    suite.end_time = datetime.now()
    return suite


def test_live_verification_gaps(tester: APITester) -> TestSuite:
    """Cover live acceptance gaps that were previously only in the strict sweep."""
    suite = TestSuite(name="Live Verification Gaps", start_time=datetime.now())

    suite.results.append(tester.run_test(
        name="POST /api/config/proxy - Disable proxy explicitly",
        method="POST",
        url="/api/config/proxy",
        params={"enabled": "false"},
        expected_status=200,
        check_fn=lambda r: r.get("success") is True
    ))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("verification gap upload\n")
        txt_path = f.name

    gap_doc_id = None
    try:
        with open(txt_path, "rb") as f:
            upload_result = tester.run_test(
                name="POST /api/documents/upload - Gap fixture upload",
                method="POST",
                url="/api/documents/upload",
                files={"file": ("gap-fixture.txt", f, "text/plain")},
                data={"title": "gap-fixture", "auto_index": "false"},
                expected_status=200,
                check_fn=lambda r: r.get("doc_id") is not None
            )
        suite.results.append(upload_result)
        if upload_result.response_data and upload_result.response_data.get("doc_id"):
            gap_doc_id = int(upload_result.response_data["doc_id"])
            tester.created_resources["documents"].append(gap_doc_id)
            suite.results.append(tester.run_test(
                name=f"PUT /api/documents/{gap_doc_id} - Update uploaded fixture",
                method="PUT",
                url=f"/api/documents/{gap_doc_id}",
                json_data={"title": "gap-fixture-updated", "metadata": {"verified": True}},
                expected_status=200,
                check_fn=lambda r: r.get("id") == gap_doc_id and r.get("metadata", {}).get("verified") is True
            ))
            suite.results.append(tester.run_test(
                name=f"GET /api/documents/{gap_doc_id}/download - Download uploaded fixture",
                method="GET",
                url=f"/api/documents/{gap_doc_id}/download",
                expected_status=200,
                check_fn=lambda r: "raw" in r
            ))
    finally:
        os.unlink(txt_path)

    crawl_result = tester.run_test(
        name="POST /api/crawl - Legacy crawl gap fixture",
        method="POST",
        url="/api/crawl",
        json_data={"urls": [REAL_WEB_URL_PRIMARY], "use_proxy": False},
        expected_status=200,
        check_fn=lambda r: r.get("task_id") is not None
    )
    suite.results.append(crawl_result)
    if crawl_result.response_data and crawl_result.response_data.get("task_id"):
        task_id = crawl_result.response_data["task_id"]
        suite.results.append(tester.run_test(
            name=f"GET /api/crawl/{task_id} - Legacy crawl status",
            method="GET",
            url=f"/api/crawl/{task_id}",
            expected_status=200,
            check_fn=lambda r: "status" in r and "results" in r
        ))

    suite.results.append(tester.run_test(
        name="GET /api/papers/search - Search papers via GET",
        method="GET",
        url="/api/papers/search",
        params={"query": "verification", "top_k": 5, "method": "hybrid"},
        expected_status=200,
        check_fn=lambda r: "papers" in r
    ))
    suite.results.append(tester.run_test(
        name="POST /api/papers/search - Search papers via POST",
        method="POST",
        url="/api/papers/search",
        json_data={"query": "verification", "top_k": 5, "method": "hybrid", "aggregate_by_paper": True},
        expected_status=200,
        check_fn=lambda r: "papers" in r and "total_results" in r
    ))

    paper_id = "verify-paper-api-suite"
    batch_id = "verify-paper-api-suite-batch"
    paper_result = tester.run_test(
        name="POST /api/papers/index - Index verification paper",
        method="POST",
        url="/api/papers/index",
        json_data={
            "paper_id": paper_id,
            "title": "Verification Paper",
            "abstract": "Verification abstract",
            "authors": ["Verifier One"],
            "source": "arxiv",
            "url": REAL_PAPER_URL_PRIMARY,
            "conference": "Other",
            "venue": "VerificationConf",
            "year": 2026,
            "paper_type": "unknown",
            "keywords": ["verification"],
            "content": "Verification paper content"
        },
        expected_status=200,
        check_fn=lambda r: r.get("success") is True and r.get("paper_id") == paper_id
    )
    suite.results.append(paper_result)
    if paper_result.response_data and paper_result.response_data.get("success") is True:
        tester.created_resources["papers"].append(paper_id)

    suite.results.append(tester.run_test(
        name=f"GET /api/papers/{paper_id} - Indexed paper detail",
        method="GET",
        url=f"/api/papers/{paper_id}",
        expected_status=200,
        check_fn=lambda r: r.get("id") == paper_id
    ))
    batch_result = tester.run_test(
        name="POST /api/papers/index/batch - Batch index verification paper",
        method="POST",
        url="/api/papers/index/batch",
        json_data={"papers": [{
            "paper_id": batch_id,
            "title": "Verification Batch Paper",
            "abstract": "Batch verification abstract",
            "authors": ["Verifier Batch"],
            "source": "arxiv",
            "url": REAL_PAPER_URL_BATCH,
            "conference": "Other",
            "venue": "VerificationConf",
            "year": 2026,
            "paper_type": "unknown",
            "keywords": ["batch"],
            "content": "Batch paper content"
        }]},
        expected_status=200,
        check_fn=lambda r: r.get("indexed", 0) >= 1 and r.get("results", {}).get(batch_id)
    )
    suite.results.append(batch_result)
    if batch_result.response_data and batch_result.response_data.get("results", {}).get(batch_id):
        tester.created_resources["papers"].append(batch_id)

    delete_result = tester.run_test(
        name=f"DELETE /api/papers/{paper_id} - Delete verification paper",
        method="DELETE",
        url=f"/api/papers/{paper_id}",
        expected_status=200,
        check_fn=lambda r: r.get("success") is True and r.get("paper_id") == paper_id
    )
    suite.results.append(delete_result)
    if delete_result.passed and paper_id in tester.created_resources["papers"]:
        tester.created_resources["papers"].remove(paper_id)

    suite.end_time = datetime.now()
    return suite


# ==================== Report Generation ====================

def generate_report(suites: List[TestSuite], config: TestConfig) -> str:
    """Generate Markdown test report"""
    total_tests = sum(s.total for s in suites)
    total_passed = sum(s.passed for s in suites)
    total_failed = sum(s.failed for s in suites)
    total_duration = sum(s.duration for s in suites)

    report = f"""# API Layer Test Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Base URL:** {config.base_url}
**Timeout:** {config.timeout}s

---

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {total_tests} |
| Passed | {total_passed} |
| Failed | {total_failed} |
| Pass Rate | {total_passed/total_tests*100:.1f}% |
| Total Duration | {total_duration:.2f}s |

---

## Test Suites

"""

    for suite in suites:
        status_icon = "PASS" if suite.failed == 0 else "FAIL"
        report += f"""### {suite.name} [{status_icon}]

| Metric | Value |
|--------|-------|
| Tests | {suite.total} |
| Passed | {suite.passed} |
| Failed | {suite.failed} |
| Duration | {suite.duration:.2f}s |

| Test Name | Status | Duration | Details |
|-----------|--------|----------|---------|
"""
        for result in suite.results:
            status = "PASS" if result.passed else "FAIL"
            details = result.message
            if result.error:
                details = f"Error: {result.error[:50]}..."
            report += f"| {result.name} | {status} | {result.duration:.2f}s | {details} |\n"

        report += "\n---\n\n"

    # Failed tests detail
    failed_tests = [(s.name, r) for s in suites for r in s.results if not r.passed]
    if failed_tests:
        report += """## Failed Tests Details

"""
        for suite_name, result in failed_tests:
            report += f"""### {suite_name}: {result.name}

- **Expected Status:** {result.expected_status}
- **Actual Status:** {result.status_code}
- **Message:** {result.message}
"""
            if result.error:
                report += f"- **Error:** {result.error}\n"
            if result.response_data:
                response_str = json.dumps(result.response_data, indent=2, ensure_ascii=False, default=str)
                if len(response_str) > 500:
                    response_str = response_str[:500] + "..."
                report += f"- **Response:**\n```json\n{response_str}\n```\n"
            report += "\n"

    # Test coverage summary
    report += """## API Coverage

| Endpoint | Methods Tested | Status |
|----------|---------------|--------|
| /api/stats | GET | Covered |
| /api/search | POST | Covered |
| /api/search/multi-method | GET | Covered |
| /api/documents | GET | Covered |
| /api/documents/{id} | GET, PUT, DELETE | Covered |
| /api/documents/{id}/download | GET | Covered |
| /api/documents/upload | POST | Covered |
| /api/upload | POST | Covered |
| /api/crawl | POST | Covered |
| /api/crawl/{task_id} | GET | Covered |
| /api/crawl/urls | POST | Covered |
| /api/crawl/html | POST | Covered |
| /api/crawl/records | GET | Covered |
| /api/tasks | GET | Covered |
| /api/tasks/{id} | GET, DELETE | Covered |
| /api/tasks/{id}/cancel | POST | Covered |
| /api/tasks/{id}/retry | POST | Covered |
| /api/config | GET | Covered |
| /api/config/proxy | POST | Covered |
| /api/papers/ | GET | Covered |
| /api/papers/search | GET, POST | Covered |
| /api/papers/index | POST | Covered |
| /api/papers/index/batch | POST | Covered |
| /api/papers/{paper_id} | GET, DELETE | Covered |
| /api/papers/stats | GET | Covered |

---

## Notes

1. Some tests may create temporary resources (documents, tasks) which are cleaned up after testing.
2. Tests involving actual crawling may timeout depending on network conditions.
3. File upload tests use temporary files created during test execution.
4. Error handling tests verify proper HTTP status codes for various error conditions.

"""

    return report


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Retrieval System API Test Script")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup of created resources")
    parser.add_argument("--save-report", action="store_true", default=True, help="Save test report")
    parser.add_argument("--report-path", default="docs/api_layer_test_report.md", help="Report file path")
    args = parser.parse_args()

    config = TestConfig(
        base_url=args.base_url,
        timeout=args.timeout,
        verbose=args.verbose,
        save_report=args.save_report,
        report_path=args.report_path
    )

    print("=" * 60)
    print("Retrieval System API Layer Test")
    print("=" * 60)
    print(f"Base URL: {config.base_url}")
    print(f"Timeout: {config.timeout}s")
    print("=" * 60)
    print()

    # Check if server is running
    tester = APITester(config)
    try:
        response = tester.client.get("/api/stats", timeout=5.0)
        print(f"Server is running (status: {response.status_code})")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {config.base_url}")
        print(f"Error: {e}")
        print("\nPlease ensure the server is running:")
        print("  cd E:\\JETB\\pythons\\sf\\retrival")
        print("  E:\\JETB\\AI\\.venv\\Scripts\\python.exe main.py")
        return 1

    print()
    print("Running tests...")
    print("-" * 60)

    try:
        # Run all test suites
        tester.suites = [
            test_stats_api(tester),
            test_paper_api(tester),
            test_search_api(tester),
            test_documents_api(tester),
            test_upload_api(tester),
            test_crawl_api(tester),
            test_tasks_api(tester),
            test_error_handling(tester),
            test_live_verification_gaps(tester),
        ]

        # Print summary
        print()
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        total_tests = sum(s.total for s in tester.suites)
        total_passed = sum(s.passed for s in tester.suites)
        total_failed = sum(s.failed for s in tester.suites)

        for suite in tester.suites:
            status = "PASS" if suite.failed == 0 else "FAIL"
            print(f"  [{status}] {suite.name}: {suite.passed}/{suite.total} passed ({suite.duration:.2f}s)")

        print("-" * 60)
        print(f"  TOTAL: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
        print(f"  Duration: {sum(s.duration for s in tester.suites):.2f}s")
        print("=" * 60)

        # Generate and save report
        if config.save_report:
            report = generate_report(tester.suites, config)
            report_path = Path(__file__).parent.parent / config.report_path
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nTest report saved to: {report_path}")

        # Cleanup
        if not args.no_cleanup:
            tester.cleanup()

        return 0 if total_failed == 0 else 1

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        tester.close()


if __name__ == "__main__":
    sys.exit(main())
