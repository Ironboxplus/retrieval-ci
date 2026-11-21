#!/usr/bin/env python
"""
Retrieval System - Automated API Testing Script

Features:
1. Cycles through all API endpoints
2. Detects errors and generates reports
3. Monitors Worker logs
4. All operations have timeout mechanisms
5. Outputs test reports

Usage:
    python scripts/auto_test.py [--rounds 10] [--interval 30] [--base-url http://localhost:8000]
"""

import sys
import os
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
except ImportError:
    print("ERROR: requests library not found. Please install: pip install requests")
    sys.exit(1)


# ==================== Configuration ====================

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 10  # seconds for HTTP requests
DEFAULT_ROUNDS = 10
DEFAULT_INTERVAL = 30  # seconds between rounds
SUBPROCESS_TIMEOUT = 15  # seconds for subprocess commands

# Log file patterns to check for errors
LOG_PATTERNS = [
    PROJECT_ROOT / "data" / "logs",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "worker.log",
]


class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SKIP = "SKIP"


@dataclass
class TestResult:
    """Single test result"""
    name: str
    endpoint: str
    method: str
    status: TestStatus
    response_time_ms: float = 0.0
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    response_preview: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RoundReport:
    """Report for a single test round"""
    round_number: int
    start_time: str
    end_time: str
    duration_seconds: float
    tests_passed: int
    tests_failed: int
    tests_error: int
    tests_total: int
    results: List[TestResult]
    worker_status: str
    log_errors: List[str]


@dataclass
class FinalReport:
    """Final summary report"""
    total_rounds: int
    total_tests: int
    total_passed: int
    total_failed: int
    total_errors: int
    success_rate: float
    average_response_time_ms: float
    rounds: List[RoundReport]
    start_time: str
    end_time: str


class APITester:
    """Automated API testing class"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Tuple[Optional[requests.Response], Optional[Exception], float]:
        """Make HTTP request with timeout, return (response, error, time_ms)"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.perf_counter()

        try:
            if method.upper() == "GET":
                resp = self.session.get(url, params=params, timeout=self.timeout)
            elif method.upper() == "POST":
                resp = self.session.post(url, json=data, params=params, timeout=self.timeout)
            elif method.upper() == "PUT":
                resp = self.session.put(url, json=data, params=params, timeout=self.timeout)
            elif method.upper() == "DELETE":
                resp = self.session.delete(url, params=params, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return resp, None, elapsed_ms

        except requests.exceptions.Timeout as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return None, e, elapsed_ms
        except requests.exceptions.ConnectionError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return None, e, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return None, e, elapsed_ms

    def test_get_documents(self) -> TestResult:
        """Test GET /api/documents"""
        name = "GET Documents"
        endpoint = "/api/documents"
        method = "GET"

        resp, error, elapsed_ms = self._make_request(method, endpoint, params={"page": 1, "page_size": 10})

        if error:
            if isinstance(error, requests.exceptions.Timeout):
                return TestResult(name, endpoint, method, TestStatus.TIMEOUT,
                                  elapsed_ms, error_message=str(error))
            return TestResult(name, endpoint, method, TestStatus.ERROR,
                              elapsed_ms, error_message=str(error))

        if resp.status_code == 200:
            try:
                data = resp.json()
                preview = f"total={data.get('total', 'N/A')}, page={data.get('page', 'N/A')}"
                return TestResult(name, endpoint, method, TestStatus.PASS,
                                  elapsed_ms, resp.status_code, response_preview=preview)
            except Exception as e:
                return TestResult(name, endpoint, method, TestStatus.ERROR,
                                  elapsed_ms, resp.status_code, error_message=f"JSON parse error: {e}")
        else:
            return TestResult(name, endpoint, method, TestStatus.FAIL,
                              elapsed_ms, resp.status_code, error_message=resp.text[:200])

    def test_post_search(self) -> TestResult:
        """Test POST /api/search"""
        name = "POST Search"
        endpoint = "/api/search"
        method = "POST"

        search_data = {
            "query": "test query",
            "method": "bm25",
            "top_k": 5
        }

        resp, error, elapsed_ms = self._make_request(method, endpoint, data=search_data)

        if error:
            if isinstance(error, requests.exceptions.Timeout):
                return TestResult(name, endpoint, method, TestStatus.TIMEOUT,
                                  elapsed_ms, error_message=str(error))
            return TestResult(name, endpoint, method, TestStatus.ERROR,
                              elapsed_ms, error_message=str(error))

        if resp.status_code == 200:
            try:
                data = resp.json()
                preview = f"total_results={data.get('total_results', 'N/A')}, method={data.get('method', 'N/A')}"
                return TestResult(name, endpoint, method, TestStatus.PASS,
                                  elapsed_ms, resp.status_code, response_preview=preview)
            except Exception as e:
                return TestResult(name, endpoint, method, TestStatus.ERROR,
                                  elapsed_ms, resp.status_code, error_message=f"JSON parse error: {e}")
        else:
            return TestResult(name, endpoint, method, TestStatus.FAIL,
                              elapsed_ms, resp.status_code, error_message=resp.text[:200])

    def test_get_tasks(self) -> TestResult:
        """Test GET /api/tasks"""
        name = "GET Tasks"
        endpoint = "/api/tasks"
        method = "GET"

        resp, error, elapsed_ms = self._make_request(method, endpoint, params={"page": 1, "page_size": 10})

        if error:
            if isinstance(error, requests.exceptions.Timeout):
                return TestResult(name, endpoint, method, TestStatus.TIMEOUT,
                                  elapsed_ms, error_message=str(error))
            return TestResult(name, endpoint, method, TestStatus.ERROR,
                              elapsed_ms, error_message=str(error))

        if resp.status_code == 200:
            try:
                data = resp.json()
                preview = f"total={data.get('total', 'N/A')}, tasks_count={len(data.get('tasks', []))}"
                return TestResult(name, endpoint, method, TestStatus.PASS,
                                  elapsed_ms, resp.status_code, response_preview=preview)
            except Exception as e:
                return TestResult(name, endpoint, method, TestStatus.ERROR,
                                  elapsed_ms, resp.status_code, error_message=f"JSON parse error: {e}")
        else:
            return TestResult(name, endpoint, method, TestStatus.FAIL,
                              elapsed_ms, resp.status_code, error_message=resp.text[:200])

    def test_get_stats(self) -> TestResult:
        """Test GET /api/stats"""
        name = "GET Stats"
        endpoint = "/api/stats"
        method = "GET"

        resp, error, elapsed_ms = self._make_request(method, endpoint)

        if error:
            if isinstance(error, requests.exceptions.Timeout):
                return TestResult(name, endpoint, method, TestStatus.TIMEOUT,
                                  elapsed_ms, error_message=str(error))
            return TestResult(name, endpoint, method, TestStatus.ERROR,
                              elapsed_ms, error_message=str(error))

        if resp.status_code == 200:
            try:
                data = resp.json()
                # Stats might have various fields, try to show key info
                preview_parts = []
                for key in ["total_documents", "total_chunks", "document_count"]:
                    if key in data:
                        preview_parts.append(f"{key}={data[key]}")
                preview = ", ".join(preview_parts) if preview_parts else "OK"
                return TestResult(name, endpoint, method, TestStatus.PASS,
                                  elapsed_ms, resp.status_code, response_preview=preview)
            except Exception as e:
                return TestResult(name, endpoint, method, TestStatus.ERROR,
                                  elapsed_ms, resp.status_code, error_message=f"JSON parse error: {e}")
        else:
            return TestResult(name, endpoint, method, TestStatus.FAIL,
                              elapsed_ms, resp.status_code, error_message=resp.text[:200])

    def test_get_config(self) -> TestResult:
        """Test GET /api/config"""
        name = "GET Config"
        endpoint = "/api/config"
        method = "GET"

        resp, error, elapsed_ms = self._make_request(method, endpoint)

        if error:
            if isinstance(error, requests.exceptions.Timeout):
                return TestResult(name, endpoint, method, TestStatus.TIMEOUT,
                                  elapsed_ms, error_message=str(error))
            return TestResult(name, endpoint, method, TestStatus.ERROR,
                              elapsed_ms, error_message=str(error))

        if resp.status_code == 200:
            try:
                data = resp.json()
                preview = f"embedding_model={data.get('embedding_model', 'N/A')}"
                return TestResult(name, endpoint, method, TestStatus.PASS,
                                  elapsed_ms, resp.status_code, response_preview=preview)
            except Exception as e:
                return TestResult(name, endpoint, method, TestStatus.ERROR,
                                  elapsed_ms, resp.status_code, error_message=f"JSON parse error: {e}")
        else:
            return TestResult(name, endpoint, method, TestStatus.FAIL,
                              elapsed_ms, resp.status_code, error_message=resp.text[:200])

    def test_health_check(self) -> TestResult:
        """Test basic connectivity (GET /)"""
        name = "Health Check"
        endpoint = "/"
        method = "GET"

        resp, error, elapsed_ms = self._make_request(method, endpoint)

        if error:
            if isinstance(error, requests.exceptions.Timeout):
                return TestResult(name, endpoint, method, TestStatus.TIMEOUT,
                                  elapsed_ms, error_message=str(error))
            return TestResult(name, endpoint, method, TestStatus.ERROR,
                              elapsed_ms, error_message=str(error))

        if resp.status_code in [200, 304]:
            return TestResult(name, endpoint, method, TestStatus.PASS,
                              elapsed_ms, resp.status_code, response_preview="Server is running")
        else:
            return TestResult(name, endpoint, method, TestStatus.FAIL,
                              elapsed_ms, resp.status_code, error_message=resp.text[:200])


def check_worker_process() -> Tuple[bool, str]:
    """Check if worker process is alive"""
    try:
        # Try to find huey worker process
        if sys.platform == "win32":
            cmd = ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"]
        else:
            cmd = ["pgrep", "-f", "worker"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT
        )

        output = result.stdout.lower()

        # Check for worker-related processes
        if "worker" in output or "huey" in output:
            return True, "Worker process found"

        # On Windows, check if any python process exists (basic check)
        if sys.platform == "win32" and "python" in output:
            return True, "Python processes running (worker status uncertain)"

        return False, "No worker process found"

    except subprocess.TimeoutExpired:
        return False, "Process check timed out"
    except FileNotFoundError:
        return False, "Process check command not found"
    except Exception as e:
        return False, f"Process check error: {e}"


def check_log_errors() -> List[str]:
    """Check log files for ERROR entries"""
    errors = []

    for log_path in LOG_PATTERNS:
        try:
            if log_path.is_file():
                # Check last 100 lines of log file
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    recent_lines = lines[-100:] if len(lines) > 100 else lines
                    for line in recent_lines:
                        if "ERROR" in line.upper() or "EXCEPTION" in line.upper():
                            errors.append(f"[{log_path.name}] {line.strip()[:150]}")
            elif log_path.is_dir():
                # Check all .log files in directory
                for log_file in log_path.glob("*.log"):
                    try:
                        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                            recent_lines = lines[-50:] if len(lines) > 50 else lines
                            for line in recent_lines:
                                if "ERROR" in line.upper() or "EXCEPTION" in line.upper():
                                    errors.append(f"[{log_file.name}] {line.strip()[:150]}")
                    except Exception:
                        pass
        except Exception:
            pass

    # Deduplicate and limit errors
    seen = set()
    unique_errors = []
    for err in errors:
        if err not in seen:
            seen.add(err)
            unique_errors.append(err)

    return unique_errors[:20]  # Limit to 20 errors


def run_single_round(tester: APITester, round_number: int) -> RoundReport:
    """Run a single test round"""
    start_time = datetime.now()
    results: List[TestResult] = []

    print(f"\n{'='*60}")
    print(f"Round {round_number} - Starting at {start_time.isoformat()}")
    print(f"{'='*60}")

    # Run all API tests
    tests = [
        ("Health Check", tester.test_health_check),
        ("GET Documents", tester.test_get_documents),
        ("POST Search", tester.test_post_search),
        ("GET Tasks", tester.test_get_tasks),
        ("GET Stats", tester.test_get_stats),
        ("GET Config", tester.test_get_config),
    ]

    for test_name, test_func in tests:
        print(f"  Testing: {test_name}...", end=" ", flush=True)
        try:
            result = test_func()
            results.append(result)

            # Print status
            status_str = result.status.value
            if result.status == TestStatus.PASS:
                print(f"[{status_str}] {result.response_time_ms:.0f}ms")
            else:
                print(f"[{status_str}] {result.error_message or 'Unknown error'}")
        except Exception as e:
            print(f"[ERROR] Unexpected: {e}")
            results.append(TestResult(
                test_name, "N/A", "N/A", TestStatus.ERROR,
                error_message=f"Test execution error: {e}"
            ))

    # Check worker process
    print(f"\n  Checking Worker process...", end=" ", flush=True)
    worker_alive, worker_status = check_worker_process()
    if worker_alive:
        print(f"[OK] {worker_status}")
    else:
        print(f"[WARNING] {worker_status}")

    # Check log errors
    print(f"  Checking logs for errors...", end=" ", flush=True)
    log_errors = check_log_errors()
    if log_errors:
        print(f"[FOUND] {len(log_errors)} error(s)")
        for err in log_errors[:5]:  # Show first 5
            print(f"    - {err}")
    else:
        print(f"[OK] No recent errors found")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Calculate statistics
    passed = sum(1 for r in results if r.status == TestStatus.PASS)
    failed = sum(1 for r in results if r.status == TestStatus.FAIL)
    errors = sum(1 for r in results if r.status in [TestStatus.ERROR, TestStatus.TIMEOUT])

    print(f"\n  Round {round_number} Summary: {passed}/{len(results)} passed, "
          f"{failed} failed, {errors} errors, {duration:.1f}s total")

    return RoundReport(
        round_number=round_number,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        duration_seconds=duration,
        tests_passed=passed,
        tests_failed=failed,
        tests_error=errors,
        tests_total=len(results),
        results=results,
        worker_status=worker_status,
        log_errors=log_errors
    )


def generate_final_report(rounds: List[RoundReport], start_time: datetime, end_time: datetime) -> FinalReport:
    """Generate final summary report"""
    total_tests = sum(r.tests_total for r in rounds)
    total_passed = sum(r.tests_passed for r in rounds)
    total_failed = sum(r.tests_failed for r in rounds)
    total_errors = sum(r.tests_error for r in rounds)

    # Calculate average response time
    all_response_times = []
    for r in rounds:
        for result in r.results:
            if result.response_time_ms > 0:
                all_response_times.append(result.response_time_ms)

    avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    return FinalReport(
        total_rounds=len(rounds),
        total_tests=total_tests,
        total_passed=total_passed,
        total_failed=total_failed,
        total_errors=total_errors,
        success_rate=success_rate,
        average_response_time_ms=avg_response_time,
        rounds=rounds,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat()
    )


def print_final_report(report: FinalReport):
    """Print final report to console"""
    print("\n" + "="*70)
    print("FINAL TEST REPORT")
    print("="*70)
    print(f"Start Time:      {report.start_time}")
    print(f"End Time:        {report.end_time}")
    print(f"Total Rounds:    {report.total_rounds}")
    print(f"Total Tests:     {report.total_tests}")
    print(f"Passed:          {report.total_passed}")
    print(f"Failed:          {report.total_failed}")
    print(f"Errors:          {report.total_errors}")
    print(f"Success Rate:    {report.success_rate:.1f}%")
    print(f"Avg Response:    {report.average_response_time_ms:.1f}ms")
    print("="*70)

    # Show failed tests summary
    failed_tests = []
    for r in report.rounds:
        for result in r.results:
            if result.status != TestStatus.PASS:
                failed_tests.append((r.round_number, result))

    if failed_tests:
        print("\nFAILED TESTS:")
        print("-"*50)
        for round_num, result in failed_tests[:20]:  # Limit to 20
            print(f"  Round {round_num}: {result.name} - [{result.status.value}] {result.error_message or ''}")

    print("\n" + "="*70)


def save_report_to_file(report: FinalReport, output_path: Path):
    """Save report to JSON file"""
    # Convert dataclasses to dict
    def convert_to_dict(obj):
        if hasattr(obj, "__dict__"):
            if hasattr(obj, "value"):  # Enum
                return obj.value
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, list):
                    result[key] = [convert_to_dict(v) for v in value]
                elif hasattr(value, "__dict__"):
                    result[key] = convert_to_dict(value)
                elif hasattr(value, "value"):  # Enum
                    result[key] = value.value
                else:
                    result[key] = value
            return result
        return obj

    report_dict = convert_to_dict(report)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated API Testing Script for Retrieval System"
    )
    parser.add_argument(
        "--base-url", "-u",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the API server (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Maximum number of test rounds (default: {DEFAULT_ROUNDS})"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Interval between rounds in seconds (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP request timeout in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path for JSON report (default: auto-generated)"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop testing if all tests in a round fail"
    )

    args = parser.parse_args()

    print("="*70)
    print("Retrieval System - Automated API Testing")
    print("="*70)
    print(f"Base URL:     {args.base_url}")
    print(f"Max Rounds:   {args.rounds}")
    print(f"Interval:     {args.interval}s")
    print(f"Timeout:      {args.timeout}s")
    print(f"Python:       {sys.executable}")
    print("="*70)

    tester = APITester(base_url=args.base_url, timeout=args.timeout)
    rounds: List[RoundReport] = []
    overall_start = datetime.now()

    try:
        for round_num in range(1, args.rounds + 1):
            round_report = run_single_round(tester, round_num)
            rounds.append(round_report)

            # Check if all tests failed (server might be down)
            if args.stop_on_error and round_report.tests_passed == 0:
                print(f"\n[STOP] All tests failed in round {round_num}. Stopping.")
                break

            # Wait for next round (except for last round)
            if round_num < args.rounds:
                print(f"\nWaiting {args.interval}s before next round...")
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Testing stopped by user.")
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
    finally:
        overall_end = datetime.now()

        # Generate and display final report
        if rounds:
            final_report = generate_final_report(rounds, overall_start, overall_end)
            print_final_report(final_report)

            # Save report to file
            if args.output:
                output_path = Path(args.output)
            else:
                timestamp = overall_start.strftime("%Y%m%d_%H%M%S")
                output_path = PROJECT_ROOT / "scripts" / f"test_report_{timestamp}.json"

            try:
                save_report_to_file(final_report, output_path)
            except Exception as e:
                print(f"Failed to save report: {e}")

            # Exit with error code if tests failed
            if final_report.success_rate < 100:
                sys.exit(1)
        else:
            print("\nNo test rounds completed.")
            sys.exit(1)


if __name__ == "__main__":
    main()
