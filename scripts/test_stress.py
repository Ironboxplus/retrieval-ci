"""
Retrieval System Stress Test
压力测试脚本 - 并发和性能测试

用法:
    python scripts/test_stress.py [--base-url URL] [--timeout SECONDS]

测试场景:
1. 并发搜索测试 (10/50/100 并发)
2. 并发写入测试 (文档上传)
3. 读写混合测试
4. 任务队列压力测试
5. 大数据量测试
6. 边界条件测试
"""

import asyncio
import aiohttp
import time
import json
import os
import sys
import random
import string
import statistics
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import tempfile


# ==================== 配置 ====================

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 30  # seconds


class TestStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class TestMetrics:
    """测试指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests + self.error_requests) / self.total_requests * 100

    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def p50_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)

    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def throughput(self) -> float:
        """请求/秒"""
        if not self.response_times:
            return 0.0
        total_time = sum(self.response_times)
        if total_time == 0:
            return 0.0
        return self.total_requests / total_time

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_requests": self.error_requests,
            "error_rate": f"{self.error_rate:.2f}%",
            "avg_response_time": f"{self.avg_response_time * 1000:.2f}ms",
            "p50_response_time": f"{self.p50_response_time * 1000:.2f}ms",
            "p95_response_time": f"{self.p95_response_time * 1000:.2f}ms",
            "p99_response_time": f"{self.p99_response_time * 1000:.2f}ms",
            "throughput": f"{self.throughput:.2f} req/s"
        }


@dataclass
class TestResult:
    """测试结果"""
    name: str
    status: TestStatus
    duration: float
    metrics: Optional[TestMetrics] = None
    message: str = ""
    details: dict = field(default_factory=dict)


class StressTestRunner:
    """压力测试运行器"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.results: List[TestResult] = []
        self.start_time = None

        # 测试查询列表
        self.test_queries = [
            "machine learning",
            "deep learning neural network",
            "natural language processing",
            "transformer architecture",
            "attention mechanism",
            "reinforcement learning",
            "computer vision",
            "generative adversarial network",
            "graph neural network",
            "federated learning",
            "contrastive learning",
            "self-supervised learning",
            "multi-modal learning",
            "knowledge distillation",
            "neural architecture search",
        ]

    async def check_server_health(self) -> bool:
        """检查服务器是否可用"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.base_url}/api/stats") as resp:
                    return resp.status == 200
        except Exception as e:
            print(f"[ERROR] Server health check failed: {e}")
            return False

    async def run_single_search(
        self,
        session: aiohttp.ClientSession,
        query: str,
        method: str = "enhanced"
    ) -> tuple[bool, float, Optional[str]]:
        """执行单次搜索请求"""
        start = time.time()
        try:
            payload = {
                "query": query,
                "method": method,
                "top_k": 10,
                "aggregate_by_doc": False
            }
            async with session.post(
                f"{self.base_url}/api/search",
                json=payload
            ) as resp:
                elapsed = time.time() - start
                if resp.status == 200:
                    data = await resp.json()
                    # 验证响应结构
                    if "results" in data:
                        return True, elapsed, None
                    return False, elapsed, "Invalid response structure"
                return False, elapsed, f"HTTP {resp.status}"
        except asyncio.TimeoutError:
            return False, time.time() - start, "Timeout"
        except Exception as e:
            return False, time.time() - start, str(e)

    async def run_concurrent_search_test(
        self,
        concurrency: int,
        total_requests: int = 100
    ) -> TestMetrics:
        """并发搜索测试"""
        metrics = TestMetrics()
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_search(session: aiohttp.ClientSession, query: str):
            async with semaphore:
                success, elapsed, error = await self.run_single_search(session, query)
                metrics.total_requests += 1
                metrics.response_times.append(elapsed)
                if success:
                    metrics.successful_requests += 1
                elif error and "Timeout" in str(error):
                    metrics.error_requests += 1
                    metrics.errors.append(f"Timeout: {query}")
                else:
                    metrics.failed_requests += 1
                    if error:
                        metrics.errors.append(f"{error}: {query}")

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            tasks = []
            for i in range(total_requests):
                query = random.choice(self.test_queries)
                tasks.append(bounded_search(session, query))

            await asyncio.gather(*tasks, return_exceptions=True)

        return metrics

    async def run_upload_test(
        self,
        session: aiohttp.ClientSession,
        content: bytes,
        filename: str
    ) -> tuple[bool, float, Optional[str], Optional[int]]:
        """执行单次上传请求"""
        start = time.time()
        try:
            data = aiohttp.FormData()
            data.add_field('file', content, filename=filename, content_type='text/plain')
            data.add_field('title', f"Test Document {filename}")
            data.add_field('auto_index', 'true')

            async with session.post(
                f"{self.base_url}/api/documents/upload",
                data=data
            ) as resp:
                elapsed = time.time() - start
                if resp.status == 200:
                    result = await resp.json()
                    doc_id = result.get("doc_id")
                    return True, elapsed, None, doc_id
                return False, elapsed, f"HTTP {resp.status}", None
        except asyncio.TimeoutError:
            return False, time.time() - start, "Timeout", None
        except Exception as e:
            return False, time.time() - start, str(e), None

    async def run_concurrent_upload_test(
        self,
        concurrency: int,
        num_files: int = 20
    ) -> TestMetrics:
        """并发上传测试"""
        metrics = TestMetrics()
        semaphore = asyncio.Semaphore(concurrency)
        uploaded_doc_ids = []

        # 生成测试文档内容
        def generate_doc_content(size_kb: int = 10) -> bytes:
            content = []
            while len('\n'.join(content).encode()) < size_kb * 1024:
                content.append(''.join(random.choices(
                    string.ascii_letters + string.digits + ' ' * 10,
                    k=random.randint(50, 200)
                )))
            return '\n'.join(content).encode('utf-8')

        async def bounded_upload(session: aiohttp.ClientSession, idx: int):
            async with semaphore:
                content = generate_doc_content(random.randint(5, 50))
                filename = f"stress_test_{idx}_{int(time.time())}.txt"
                success, elapsed, error, doc_id = await self.run_upload_test(
                    session, content, filename
                )
                metrics.total_requests += 1
                metrics.response_times.append(elapsed)
                if success:
                    metrics.successful_requests += 1
                    if doc_id:
                        uploaded_doc_ids.append(doc_id)
                else:
                    metrics.failed_requests += 1
                    if error:
                        metrics.errors.append(f"{error}: {filename}")

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            tasks = [bounded_upload(session, i) for i in range(num_files)]
            await asyncio.gather(*tasks, return_exceptions=True)

        # 清理上传的测试文档
        await self._cleanup_test_documents(uploaded_doc_ids)

        return metrics

    async def _cleanup_test_documents(self, doc_ids: List[int]):
        """清理测试文档"""
        if not doc_ids:
            return

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            for doc_id in doc_ids:
                try:
                    async with session.delete(
                        f"{self.base_url}/api/documents/{doc_id}"
                    ) as resp:
                        pass  # 忽略结果
                except Exception:
                    pass

    async def run_mixed_read_write_test(
        self,
        duration_seconds: int = 30,
        read_ratio: float = 0.8
    ) -> TestMetrics:
        """读写混合测试"""
        metrics = TestMetrics()
        uploaded_doc_ids = []
        end_time = time.time() + duration_seconds

        def generate_doc_content() -> bytes:
            content = '\n'.join([
                ''.join(random.choices(string.ascii_letters + ' ' * 5, k=100))
                for _ in range(50)
            ])
            return content.encode('utf-8')

        async def mixed_operation(session: aiohttp.ClientSession, idx: int):
            if random.random() < read_ratio:
                # 读操作 (搜索)
                query = random.choice(self.test_queries)
                success, elapsed, error = await self.run_single_search(session, query)
            else:
                # 写操作 (上传)
                content = generate_doc_content()
                filename = f"mixed_test_{idx}_{int(time.time())}.txt"
                success, elapsed, error, doc_id = await self.run_upload_test(
                    session, content, filename
                )
                if doc_id:
                    uploaded_doc_ids.append(doc_id)

            metrics.total_requests += 1
            metrics.response_times.append(elapsed)
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
                if error:
                    metrics.errors.append(str(error))

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            idx = 0
            tasks = []
            while time.time() < end_time:
                task = asyncio.create_task(mixed_operation(session, idx))
                tasks.append(task)
                idx += 1
                # 控制并发
                if len(tasks) >= 20:
                    await asyncio.gather(*tasks[:10], return_exceptions=True)
                    tasks = tasks[10:]
                await asyncio.sleep(0.05)

            # 等待剩余任务
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # 清理
        await self._cleanup_test_documents(uploaded_doc_ids)

        return metrics

    async def run_task_queue_test(
        self,
        num_tasks: int = 50
    ) -> TestMetrics:
        """任务队列压力测试"""
        metrics = TestMetrics()
        task_ids = []

        # 生成测试 URL 列表
        test_urls = [
            f"https://example.com/page/{i}"
            for i in range(num_tasks)
        ]

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # 批量提交任务
            start = time.time()
            try:
                payload = {
                    "urls": test_urls[:10],  # 限制每次提交的 URL 数
                    "use_proxy": False,
                    "auto_index": False,
                    "max_concurrent": 5
                }
                async with session.post(
                    f"{self.base_url}/api/crawl/urls",
                    json=payload
                ) as resp:
                    elapsed = time.time() - start
                    metrics.total_requests += 1
                    metrics.response_times.append(elapsed)

                    if resp.status == 200:
                        result = await resp.json()
                        task_id = result.get("task_id")
                        if task_id:
                            task_ids.append(task_id)
                        metrics.successful_requests += 1
                    else:
                        metrics.failed_requests += 1
                        text = await resp.text()
                        metrics.errors.append(f"HTTP {resp.status}: {text[:100]}")
            except Exception as e:
                metrics.error_requests += 1
                metrics.errors.append(str(e))

            # 检查任务状态
            await asyncio.sleep(1)
            for task_id in task_ids:
                start = time.time()
                try:
                    async with session.get(
                        f"{self.base_url}/api/tasks/{task_id}"
                    ) as resp:
                        elapsed = time.time() - start
                        metrics.total_requests += 1
                        metrics.response_times.append(elapsed)
                        if resp.status == 200:
                            metrics.successful_requests += 1
                        else:
                            metrics.failed_requests += 1
                except Exception as e:
                    metrics.error_requests += 1
                    metrics.errors.append(str(e))

        return metrics

    async def run_large_document_test(
        self,
        size_mb: float = 1.0
    ) -> TestMetrics:
        """大文档测试"""
        metrics = TestMetrics()

        # 生成大文档
        content_size = int(size_mb * 1024 * 1024)
        content = ('A' * 1000 + '\n') * (content_size // 1001)
        content = content.encode('utf-8')

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)  # 大文件需要更长超时
        ) as session:
            start = time.time()
            success, elapsed, error, doc_id = await self.run_upload_test(
                session, content, f"large_doc_{int(size_mb)}mb.txt"
            )

            metrics.total_requests += 1
            metrics.response_times.append(elapsed)

            if success:
                metrics.successful_requests += 1
                # 清理
                if doc_id:
                    await self._cleanup_test_documents([doc_id])
            else:
                metrics.failed_requests += 1
                if error:
                    metrics.errors.append(error)

        return metrics

    async def run_pagination_test(
        self,
        pages: int = 10,
        page_size: int = 20
    ) -> TestMetrics:
        """分页性能测试"""
        metrics = TestMetrics()

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            for page in range(1, pages + 1):
                start = time.time()
                try:
                    async with session.get(
                        f"{self.base_url}/api/documents",
                        params={"page": page, "page_size": page_size}
                    ) as resp:
                        elapsed = time.time() - start
                        metrics.total_requests += 1
                        metrics.response_times.append(elapsed)

                        if resp.status == 200:
                            data = await resp.json()
                            if "documents" in data:
                                metrics.successful_requests += 1
                            else:
                                metrics.failed_requests += 1
                        else:
                            metrics.failed_requests += 1
                except Exception as e:
                    metrics.error_requests += 1
                    metrics.errors.append(str(e))

        return metrics

    async def run_empty_search_test(self) -> TestMetrics:
        """空数据库搜索测试"""
        metrics = TestMetrics()

        # 使用不太可能匹配的查询
        random_query = ''.join(random.choices(string.ascii_letters, k=50))

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            start = time.time()
            success, elapsed, error = await self.run_single_search(
                session, random_query
            )

            metrics.total_requests += 1
            metrics.response_times.append(elapsed)

            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
                if error:
                    metrics.errors.append(error)

        return metrics

    async def run_all_tests(self) -> List[TestResult]:
        """运行所有测试"""
        self.start_time = datetime.now()
        self.results = []

        print("=" * 60)
        print("Retrieval System Stress Test")
        print(f"Base URL: {self.base_url}")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 检查服务器
        print("\n[*] Checking server health...")
        if not await self.check_server_health():
            print("[FATAL] Server is not available. Aborting tests.")
            return self.results
        print("[OK] Server is healthy.\n")

        # 测试 1: 并发搜索测试
        for concurrency in [10, 50, 100]:
            print(f"\n[TEST] Concurrent Search Test ({concurrency} concurrent)...")
            start = time.time()
            try:
                metrics = await self.run_concurrent_search_test(
                    concurrency=concurrency,
                    total_requests=concurrency * 2
                )
                duration = time.time() - start

                status = TestStatus.PASSED if metrics.error_rate < 10 else TestStatus.FAILED
                result = TestResult(
                    name=f"Concurrent Search ({concurrency})",
                    status=status,
                    duration=duration,
                    metrics=metrics,
                    message=f"Error rate: {metrics.error_rate:.2f}%"
                )
                self.results.append(result)
                print(f"  Status: {status.value}")
                print(f"  Duration: {duration:.2f}s")
                print(f"  Avg Response: {metrics.avg_response_time * 1000:.2f}ms")
                print(f"  P95 Response: {metrics.p95_response_time * 1000:.2f}ms")
                print(f"  Error Rate: {metrics.error_rate:.2f}%")
            except Exception as e:
                self.results.append(TestResult(
                    name=f"Concurrent Search ({concurrency})",
                    status=TestStatus.ERROR,
                    duration=time.time() - start,
                    message=str(e)
                ))
                print(f"  [ERROR] {e}")

        # 测试 2: 并发上传测试
        print("\n[TEST] Concurrent Upload Test...")
        start = time.time()
        try:
            metrics = await self.run_concurrent_upload_test(
                concurrency=5,
                num_files=10
            )
            duration = time.time() - start

            status = TestStatus.PASSED if metrics.error_rate < 20 else TestStatus.FAILED
            result = TestResult(
                name="Concurrent Upload",
                status=status,
                duration=duration,
                metrics=metrics,
                message=f"Uploaded {metrics.successful_requests} files"
            )
            self.results.append(result)
            print(f"  Status: {status.value}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Successful: {metrics.successful_requests}/{metrics.total_requests}")
        except Exception as e:
            self.results.append(TestResult(
                name="Concurrent Upload",
                status=TestStatus.ERROR,
                duration=time.time() - start,
                message=str(e)
            ))
            print(f"  [ERROR] {e}")

        # 测试 3: 读写混合测试
        print("\n[TEST] Mixed Read/Write Test (30s)...")
        start = time.time()
        try:
            metrics = await self.run_mixed_read_write_test(
                duration_seconds=30,
                read_ratio=0.8
            )
            duration = time.time() - start

            status = TestStatus.PASSED if metrics.error_rate < 15 else TestStatus.FAILED
            result = TestResult(
                name="Mixed Read/Write",
                status=status,
                duration=duration,
                metrics=metrics,
                message=f"Total ops: {metrics.total_requests}"
            )
            self.results.append(result)
            print(f"  Status: {status.value}")
            print(f"  Total Operations: {metrics.total_requests}")
            print(f"  Throughput: {metrics.total_requests / duration:.2f} ops/s")
        except Exception as e:
            self.results.append(TestResult(
                name="Mixed Read/Write",
                status=TestStatus.ERROR,
                duration=time.time() - start,
                message=str(e)
            ))
            print(f"  [ERROR] {e}")

        # 测试 4: 任务队列测试
        print("\n[TEST] Task Queue Stress Test...")
        start = time.time()
        try:
            metrics = await self.run_task_queue_test(num_tasks=20)
            duration = time.time() - start

            status = TestStatus.PASSED if metrics.error_rate < 20 else TestStatus.FAILED
            result = TestResult(
                name="Task Queue",
                status=status,
                duration=duration,
                metrics=metrics,
                message=f"Tasks processed: {metrics.successful_requests}"
            )
            self.results.append(result)
            print(f"  Status: {status.value}")
            print(f"  Duration: {duration:.2f}s")
        except Exception as e:
            self.results.append(TestResult(
                name="Task Queue",
                status=TestStatus.ERROR,
                duration=time.time() - start,
                message=str(e)
            ))
            print(f"  [ERROR] {e}")

        # 测试 5: 大文档测试
        print("\n[TEST] Large Document Test (1MB)...")
        start = time.time()
        try:
            metrics = await self.run_large_document_test(size_mb=1.0)
            duration = time.time() - start

            status = TestStatus.PASSED if metrics.successful_requests > 0 else TestStatus.FAILED
            result = TestResult(
                name="Large Document (1MB)",
                status=status,
                duration=duration,
                metrics=metrics,
                message=f"Upload time: {metrics.avg_response_time:.2f}s"
            )
            self.results.append(result)
            print(f"  Status: {status.value}")
            print(f"  Upload Time: {metrics.avg_response_time:.2f}s")
        except Exception as e:
            self.results.append(TestResult(
                name="Large Document (1MB)",
                status=TestStatus.ERROR,
                duration=time.time() - start,
                message=str(e)
            ))
            print(f"  [ERROR] {e}")

        # 测试 6: 分页性能测试
        print("\n[TEST] Pagination Test...")
        start = time.time()
        try:
            metrics = await self.run_pagination_test(pages=10, page_size=20)
            duration = time.time() - start

            status = TestStatus.PASSED if metrics.error_rate < 5 else TestStatus.FAILED
            result = TestResult(
                name="Pagination",
                status=status,
                duration=duration,
                metrics=metrics,
                message=f"Avg page load: {metrics.avg_response_time * 1000:.2f}ms"
            )
            self.results.append(result)
            print(f"  Status: {status.value}")
            print(f"  Avg Page Load: {metrics.avg_response_time * 1000:.2f}ms")
        except Exception as e:
            self.results.append(TestResult(
                name="Pagination",
                status=TestStatus.ERROR,
                duration=time.time() - start,
                message=str(e)
            ))
            print(f"  [ERROR] {e}")

        # 测试 7: 空搜索测试
        print("\n[TEST] Empty Search Test...")
        start = time.time()
        try:
            metrics = await self.run_empty_search_test()
            duration = time.time() - start

            status = TestStatus.PASSED if metrics.successful_requests > 0 else TestStatus.FAILED
            result = TestResult(
                name="Empty Search",
                status=status,
                duration=duration,
                metrics=metrics,
                message="Search handled gracefully"
            )
            self.results.append(result)
            print(f"  Status: {status.value}")
        except Exception as e:
            self.results.append(TestResult(
                name="Empty Search",
                status=TestStatus.ERROR,
                duration=time.time() - start,
                message=str(e)
            ))
            print(f"  [ERROR] {e}")

        # 汇总
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)

        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        print(f"Total Duration: {(datetime.now() - self.start_time).total_seconds():.2f}s")

        return self.results

    def generate_report(self, output_path: str) -> str:
        """生成 Markdown 格式的测试报告"""
        report = []
        report.append("# Retrieval System Stress Test Report\n")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Base URL**: {self.base_url}\n")
        report.append(f"**Start Time**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'N/A'}\n")
        report.append("")

        # 概要
        report.append("## Summary\n")
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)

        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Total Tests | {len(self.results)} |")
        report.append(f"| Passed | {passed} |")
        report.append(f"| Failed | {failed} |")
        report.append(f"| Errors | {errors} |")
        report.append("")

        # 详细结果
        report.append("## Test Results\n")

        for result in self.results:
            status_emoji = {
                TestStatus.PASSED: "[PASS]",
                TestStatus.FAILED: "[FAIL]",
                TestStatus.ERROR: "[ERROR]",
                TestStatus.SKIPPED: "[SKIP]"
            }.get(result.status, "")

            report.append(f"### {result.name} {status_emoji}\n")
            report.append(f"- **Status**: {result.status.value}")
            report.append(f"- **Duration**: {result.duration:.2f}s")
            report.append(f"- **Message**: {result.message}")

            if result.metrics:
                report.append("")
                report.append("**Metrics**:")
                report.append("")
                report.append("| Metric | Value |")
                report.append("|--------|-------|")
                metrics_dict = result.metrics.to_dict()
                for key, value in metrics_dict.items():
                    report.append(f"| {key.replace('_', ' ').title()} | {value} |")

                if result.metrics.errors:
                    report.append("")
                    report.append("**Errors** (first 5):")
                    for err in result.metrics.errors[:5]:
                        report.append(f"- `{err[:100]}`")

            report.append("")

        # 建议
        report.append("## Recommendations\n")

        recommendations = []
        for result in self.results:
            if result.status == TestStatus.FAILED:
                if "Search" in result.name and result.metrics:
                    if result.metrics.p95_response_time > 1.0:
                        recommendations.append(
                            f"- **{result.name}**: P95 latency ({result.metrics.p95_response_time * 1000:.0f}ms) exceeds 1s. "
                            "Consider optimizing index or adding caching."
                        )
                    if result.metrics.error_rate > 5:
                        recommendations.append(
                            f"- **{result.name}**: High error rate ({result.metrics.error_rate:.1f}%). "
                            "Check server logs for bottlenecks."
                        )
                elif "Upload" in result.name:
                    recommendations.append(
                        f"- **{result.name}**: Upload failures detected. "
                        "Check disk space and file handling."
                    )

        if not recommendations:
            recommendations.append("- All tests within acceptable thresholds. System is performing well.")

        report.extend(recommendations)
        report.append("")

        # 写入文件
        content = "\n".join(report)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path


async def main():
    parser = argparse.ArgumentParser(description="Retrieval System Stress Test")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the API (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the test report (default: docs/stress_test_report.md)"
    )

    args = parser.parse_args()

    # 确定输出路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_path = args.output or os.path.join(project_dir, "docs", "stress_test_report.md")

    runner = StressTestRunner(base_url=args.base_url, timeout=args.timeout)

    try:
        results = await runner.run_all_tests()

        # 生成报告
        report_path = runner.generate_report(output_path)
        print(f"\n[*] Report saved to: {report_path}")

        # 返回退出码
        failed_count = sum(1 for r in results if r.status in (TestStatus.FAILED, TestStatus.ERROR))
        sys.exit(1 if failed_count > 0 else 0)

    except KeyboardInterrupt:
        print("\n[!] Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
