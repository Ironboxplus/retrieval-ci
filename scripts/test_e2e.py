"""
Retrieval System - End-to-End Test Suite

完整的端到端测试脚本，覆盖所有核心场景：
- 场景1: 书签导入流程
- 场景2: PDF上传流程
- 场景3: 文本上传流程
- 场景4: 搜索全流程
- 场景5: 任务队列流程
- 场景6: 数据一致性验证

Python环境: E:\\JETB\\AI\\.venv\\Scripts\\python.exe
API地址: http://127.0.0.1:8000
"""
import os
import sys
import time
import json
import asyncio
import hashlib
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx

# ==================== 配置 ====================

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
TIMEOUT = 60  # 请求超时时间(秒)
POLL_INTERVAL = 2  # 轮询间隔(秒)
MAX_POLL_ATTEMPTS = 30  # 最大轮询次数
REAL_WEB_URL_PRIMARY = "https://www.python.org/about/"
REAL_WEB_URL_SECONDARY = "https://www.iana.org/help/example-domains"

# 测试数据目录
TEST_DATA_DIR = PROJECT_ROOT / "scripts" / "test_data"


def clear_proxy_env():
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


# ==================== 数据类 ====================

@dataclass
class TestResult:
    """单个测试结果"""
    name: str
    scenario: str
    status: str  # passed, failed, skipped, error
    message: str = ""
    duration: float = 0.0
    details: Dict = field(default_factory=dict)
    error: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class ScenarioResult:
    """场景测试结果"""
    name: str
    description: str
    status: str  # passed, failed, partial
    tests: List[TestResult] = field(default_factory=list)
    setup_result: Optional[TestResult] = None
    teardown_result: Optional[TestResult] = None
    duration: float = 0.0

    @property
    def passed_count(self) -> int:
        return sum(1 for t in self.tests if t.status == "passed")

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.tests if t.status in ("failed", "error"))


@dataclass
class TestReport:
    """完整测试报告"""
    start_time: datetime
    end_time: Optional[datetime] = None
    scenarios: List[ScenarioResult] = field(default_factory=list)
    api_base: str = API_BASE

    @property
    def total_tests(self) -> int:
        return sum(len(s.tests) for s in self.scenarios)

    @property
    def passed_tests(self) -> int:
        return sum(s.passed_count for s in self.scenarios)

    @property
    def failed_tests(self) -> int:
        return sum(s.failed_count for s in self.scenarios)

    @property
    def duration(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


# ==================== HTTP客户端 ====================

class APIClient:
    """API客户端"""

    def __init__(self, base_url: str = API_BASE):
        clear_proxy_env()
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=TIMEOUT, trust_env=False, follow_redirects=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def get(self, path: str, **kwargs) -> httpx.Response:
        return self.client.get(f"{self.base_url}{path}", **kwargs)

    def post(self, path: str, **kwargs) -> httpx.Response:
        return self.client.post(f"{self.base_url}{path}", **kwargs)

    def put(self, path: str, **kwargs) -> httpx.Response:
        return self.client.put(f"{self.base_url}{path}", **kwargs)

    def delete(self, path: str, **kwargs) -> httpx.Response:
        return self.client.delete(f"{self.base_url}{path}", **kwargs)

    def upload_file(self, path: str, file_path: str,
                   file_field: str = "file", **form_data) -> httpx.Response:
        """上传文件"""
        with open(file_path, "rb") as f:
            files = {file_field: (os.path.basename(file_path), f)}
            return self.client.post(
                f"{self.base_url}{path}",
                files=files,
                data=form_data
            )


# ==================== 测试工具函数 ====================

def create_test_bookmark_html(urls: List[Tuple[str, str]]) -> str:
    """创建测试用书签HTML"""
    links = "\n".join([
        f'<DT><A HREF="{url}">{title}</A>'
        for title, url in urls
    ])
    return f"""<!DOCTYPE NETSCAPE-Bookmark-file-1>
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DL><p>
    <DT><H3>Test Folder</H3>
    <DL><p>
        {links}
    </DL><p>
</DL><p>
"""


def create_test_txt_file(content: str) -> str:
    """创建临时TXT文件"""
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def create_test_html_file(content: str) -> str:
    """创建临时HTML文件"""
    fd, path = tempfile.mkstemp(suffix=".html")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def wait_for_task(client: APIClient, task_id: str,
                  max_attempts: int = MAX_POLL_ATTEMPTS) -> Dict:
    """等待任务完成"""
    for i in range(max_attempts):
        resp = client.get(f"/api/tasks/{task_id}")
        if resp.status_code == 200:
            task = resp.json()
            if task["status"] in ("completed", "failed", "cancelled", "timeout"):
                return task
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Task {task_id} did not complete in time")


def get_data_consistency_stats(client: APIClient) -> Dict:
    """获取数据一致性统计"""
    stats_resp = client.get("/api/stats")
    if stats_resp.status_code != 200:
        raise Exception(f"Failed to get stats: {stats_resp.text}")
    return stats_resp.json()


# ==================== 测试场景 ====================

class TestScenario:
    """测试场景基类"""

    name: str = "Base Scenario"
    description: str = ""

    def __init__(self, client: APIClient):
        self.client = client
        self.context: Dict = {}  # 存储测试上下文

    def setup(self) -> TestResult:
        """场景初始化"""
        return TestResult(
            name="setup",
            scenario=self.name,
            status="passed",
            message="No setup required"
        )

    def teardown(self) -> TestResult:
        """场景清理"""
        return TestResult(
            name="teardown",
            scenario=self.name,
            status="passed",
            message="No teardown required"
        )

    def run_tests(self) -> List[TestResult]:
        """运行测试（子类实现）"""
        return []

    def run(self) -> ScenarioResult:
        """运行完整场景"""
        start_time = time.time()
        result = ScenarioResult(
            name=self.name,
            description=self.description,
            status="passed"
        )

        # 初始化
        try:
            result.setup_result = self.setup()
            if result.setup_result.status != "passed":
                result.status = "failed"
                return result
        except Exception as e:
            result.setup_result = TestResult(
                name="setup",
                scenario=self.name,
                status="error",
                error=str(e),
                traceback=traceback.format_exc()
            )
            result.status = "failed"
            return result

        # 运行测试
        try:
            result.tests = self.run_tests()
        except Exception as e:
            result.tests.append(TestResult(
                name="run_tests",
                scenario=self.name,
                status="error",
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # 清理
        try:
            result.teardown_result = self.teardown()
        except Exception as e:
            result.teardown_result = TestResult(
                name="teardown",
                scenario=self.name,
                status="error",
                error=str(e),
                traceback=traceback.format_exc()
            )

        # 计算最终状态
        if result.failed_count > 0:
            result.status = "failed" if result.passed_count == 0 else "partial"

        result.duration = time.time() - start_time
        return result


class Scenario1_BookmarkImport(TestScenario):
    """场景1: 书签导入流程"""

    name = "Scenario 1: Bookmark Import"
    description = "测试HTML书签导入、URL提取、爬取和索引流程"

    def setup(self) -> TestResult:
        """创建测试书签文件"""
        start = time.time()

        # 创建书签HTML
        test_urls = [
            ("Python About", REAL_WEB_URL_PRIMARY),
            ("IANA Example Domains", REAL_WEB_URL_SECONDARY),
        ]
        html_content = create_test_bookmark_html(test_urls)
        self.context["bookmark_html"] = create_test_html_file(html_content)
        self.context["expected_urls"] = [url for _, url in test_urls]

        return TestResult(
            name="setup",
            scenario=self.name,
            status="passed",
            message=f"Created bookmark file with {len(test_urls)} URLs",
            duration=time.time() - start
        )

    def teardown(self) -> TestResult:
        """清理测试文件"""
        start = time.time()

        # 删除临时文件
        if "bookmark_html" in self.context:
            try:
                os.unlink(self.context["bookmark_html"])
            except:
                pass

        # 清理创建的文档
        if "created_doc_ids" in self.context:
            for doc_id in self.context["created_doc_ids"]:
                try:
                    self.client.delete(f"/api/documents/{doc_id}")
                except:
                    pass

        return TestResult(
            name="teardown",
            scenario=self.name,
            status="passed",
            duration=time.time() - start
        )

    def run_tests(self) -> List[TestResult]:
        results = []

        # Test 1.1: 上传书签HTML
        test_name = "1.1 Upload Bookmark HTML"
        start = time.time()
        try:
            resp = self.client.upload_file(
                "/api/crawl/html",
                self.context["bookmark_html"],
                file_field="file",
                use_proxy="false",
                auto_index="true",
                auto_crawl="false"  # 先不自动爬取
            )

            if resp.status_code == 200:
                data = resp.json()
                self.context["extracted_urls"] = data.get("extracted_urls", [])
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message=f"Extracted {data.get('total_urls', 0)} URLs",
                    duration=time.time() - start,
                    details=data
                ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start,
                    error=resp.text
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 1.2: 验证URL提取
        test_name = "1.2 Verify URL Extraction"
        start = time.time()
        try:
            extracted = set(self.context.get("extracted_urls", []))
            expected = set(self.context.get("expected_urls", []))

            if expected.issubset(extracted):
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message=f"All {len(expected)} expected URLs found",
                    duration=time.time() - start
                ))
            else:
                missing = expected - extracted
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"Missing URLs: {missing}",
                    duration=time.time() - start
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 1.3: 触发URL爬取
        test_name = "1.3 Trigger URL Crawl"
        start = time.time()
        try:
            # 只爬取一个简单URL测试
            resp = self.client.post(
                "/api/crawl/urls",
                json={
                    "urls": [REAL_WEB_URL_PRIMARY],
                    "use_proxy": False,
                    "auto_index": True,
                    "max_concurrent": 1
                }
            )

            if resp.status_code == 200:
                data = resp.json()
                self.context["crawl_task_id"] = data.get("task_id")
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message=f"Created crawl task: {data.get('task_id')}",
                    duration=time.time() - start,
                    details=data
                ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start,
                    error=resp.text
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 1.4: 等待爬取完成并验证
        if "crawl_task_id" in self.context:
            test_name = "1.4 Wait for Crawl Completion"
            start = time.time()
            try:
                task = wait_for_task(self.client, self.context["crawl_task_id"])

                if task["status"] == "completed":
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="passed",
                        message=f"Crawl completed: progress {task.get('progress', 0)}/{task.get('total', 0)}",
                        duration=time.time() - start,
                        details=task
                    ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="failed",
                        message=f"Task status: {task['status']}",
                        duration=time.time() - start,
                        details=task,
                        error=task.get("error_message")
                    ))
            except Exception as e:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="error",
                    duration=time.time() - start,
                    error=str(e),
                    traceback=traceback.format_exc()
                ))

        return results


class Scenario2_PDFUpload(TestScenario):
    """场景2: PDF上传流程"""

    name = "Scenario 2: PDF Upload"
    description = "测试PDF文件上传、解析、索引和删除流程"

    def run_tests(self) -> List[TestResult]:
        results = []

        # 检查是否有测试PDF文件
        test_pdf = PROJECT_ROOT / "scripts" / "test_data" / "test.pdf"

        # Test 2.1: PDF上传（如果有测试文件）
        if test_pdf.exists():
            test_name = "2.1 Upload PDF File"
            start = time.time()
            try:
                resp = self.client.upload_file(
                    "/api/documents/upload",
                    str(test_pdf),
                    file_field="file",
                    title="Test PDF Document",
                    auto_index="true"
                )

                if resp.status_code == 200:
                    data = resp.json()
                    self.context["pdf_doc_id"] = data.get("doc_id")
                    self.context["pdf_task_id"] = data.get("task_id")
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="passed",
                        message=f"Uploaded doc_id={data.get('doc_id')}",
                        duration=time.time() - start,
                        details=data
                    ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="failed",
                        message=f"HTTP {resp.status_code}",
                        duration=time.time() - start,
                        error=resp.text
                    ))
            except Exception as e:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="error",
                    duration=time.time() - start,
                    error=str(e),
                    traceback=traceback.format_exc()
                ))
        else:
            results.append(TestResult(
                name="2.1 Upload PDF File",
                scenario=self.name,
                status="skipped",
                message=f"No test PDF at {test_pdf}"
            ))

        # Test 2.2: 验证文档创建
        if "pdf_doc_id" in self.context:
            test_name = "2.2 Verify Document Created"
            start = time.time()
            try:
                resp = self.client.get(f"/api/documents/{self.context['pdf_doc_id']}")

                if resp.status_code == 200:
                    doc = resp.json()
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="passed",
                        message=f"Document exists: {doc.get('title')}",
                        duration=time.time() - start,
                        details=doc
                    ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="failed",
                        message=f"HTTP {resp.status_code}",
                        duration=time.time() - start
                    ))
            except Exception as e:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="error",
                    duration=time.time() - start,
                    error=str(e),
                    traceback=traceback.format_exc()
                ))

        # Test 2.3: 删除文档并验证级联删除
        if "pdf_doc_id" in self.context:
            test_name = "2.3 Delete Document (Cascade)"
            start = time.time()
            try:
                # 先获取文档信息
                doc_resp = self.client.get(f"/api/documents/{self.context['pdf_doc_id']}")
                chunk_count = 0
                if doc_resp.status_code == 200:
                    chunk_count = doc_resp.json().get("chunk_count", 0)

                # 删除文档
                resp = self.client.delete(f"/api/documents/{self.context['pdf_doc_id']}")

                if resp.status_code == 200:
                    # 验证文档已删除
                    verify_resp = self.client.get(f"/api/documents/{self.context['pdf_doc_id']}")
                    if verify_resp.status_code == 404:
                        results.append(TestResult(
                            name=test_name,
                            scenario=self.name,
                            status="passed",
                            message=f"Document and {chunk_count} chunks deleted",
                            duration=time.time() - start
                        ))
                    else:
                        results.append(TestResult(
                            name=test_name,
                            scenario=self.name,
                            status="failed",
                            message="Document still exists after deletion",
                            duration=time.time() - start
                        ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="failed",
                        message=f"HTTP {resp.status_code}",
                        duration=time.time() - start,
                        error=resp.text
                    ))
            except Exception as e:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="error",
                    duration=time.time() - start,
                    error=str(e),
                    traceback=traceback.format_exc()
                ))

        return results


class Scenario3_TextUpload(TestScenario):
    """场景3: 文本上传流程"""

    name = "Scenario 3: Text Upload"
    description = "测试TXT文件上传和索引流程"

    def setup(self) -> TestResult:
        """创建测试TXT文件"""
        start = time.time()

        test_content = """
        This is a test document for the retrieval system.

        Machine Learning and Artificial Intelligence

        Machine learning is a subset of artificial intelligence that focuses on
        building applications that learn from data and improve their accuracy
        over time without being programmed to do so.

        Deep learning is a subset of machine learning that uses neural networks
        with many layers to analyze various factors of data.

        Natural Language Processing (NLP) is a branch of AI that helps computers
        understand, interpret and manipulate human language.
        """

        self.context["txt_file"] = create_test_txt_file(test_content)
        self.context["test_content"] = test_content

        return TestResult(
            name="setup",
            scenario=self.name,
            status="passed",
            message="Created test TXT file",
            duration=time.time() - start
        )

    def teardown(self) -> TestResult:
        """清理测试文件和文档"""
        start = time.time()

        # 删除临时文件
        if "txt_file" in self.context:
            try:
                os.unlink(self.context["txt_file"])
            except:
                pass

        # 删除创建的文档
        if "txt_doc_id" in self.context:
            try:
                self.client.delete(f"/api/documents/{self.context['txt_doc_id']}")
            except:
                pass

        return TestResult(
            name="teardown",
            scenario=self.name,
            status="passed",
            duration=time.time() - start
        )

    def run_tests(self) -> List[TestResult]:
        results = []

        # Test 3.1: 上传TXT文件
        test_name = "3.1 Upload TXT File"
        start = time.time()
        try:
            resp = self.client.upload_file(
                "/api/documents/upload",
                self.context["txt_file"],
                file_field="file",
                title="Test ML Document",
                auto_index="true"
            )

            if resp.status_code == 200:
                data = resp.json()
                self.context["txt_doc_id"] = data.get("doc_id")
                self.context["txt_task_id"] = data.get("task_id")
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message=f"Uploaded doc_id={data.get('doc_id')}",
                    duration=time.time() - start,
                    details=data
                ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start,
                    error=resp.text
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 3.2: 验证同步上传后文档立即进入indexed状态
        if "txt_doc_id" in self.context:
            test_name = "3.2 Verify Indexed Status"
            start = time.time()
            try:
                resp = self.client.get(f"/api/documents/{self.context['txt_doc_id']}")
                if resp.status_code == 200:
                    doc = resp.json()
                    if doc.get("status") == "indexed":
                        results.append(TestResult(
                            name=test_name,
                            scenario=self.name,
                            status="passed",
                            message="Document status is indexed",
                            duration=time.time() - start,
                            details={"status": doc.get("status")}
                        ))
                    else:
                        results.append(TestResult(
                            name=test_name,
                            scenario=self.name,
                            status="failed",
                            message=f"Document status: {doc.get('status')}",
                            duration=time.time() - start,
                            details=doc
                        ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="failed",
                        message=f"HTTP {resp.status_code}",
                        duration=time.time() - start,
                    ))
            except Exception as e:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="error",
                    duration=time.time() - start,
                    error=str(e),
                    traceback=traceback.format_exc()
                ))

        # Test 3.3: 验证索引
        if "txt_doc_id" in self.context:
            test_name = "3.3 Verify Index Created"
            start = time.time()
            try:
                # 等待一小段时间确保索引完成
                time.sleep(1)

                resp = self.client.get(f"/api/documents/{self.context['txt_doc_id']}")

                if resp.status_code == 200:
                    doc = resp.json()
                    chunk_count = doc.get("chunk_count", 0)

                    if chunk_count > 0:
                        results.append(TestResult(
                            name=test_name,
                            scenario=self.name,
                            status="passed",
                            message=f"Document indexed with {chunk_count} chunks",
                            duration=time.time() - start,
                            details={"chunk_count": chunk_count}
                        ))
                    else:
                        results.append(TestResult(
                            name=test_name,
                            scenario=self.name,
                            status="failed",
                            message="No chunks created",
                            duration=time.time() - start
                        ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="failed",
                        message=f"HTTP {resp.status_code}",
                        duration=time.time() - start
                    ))
            except Exception as e:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="error",
                    duration=time.time() - start,
                    error=str(e),
                    traceback=traceback.format_exc()
                ))

        # Test 3.4: 搜索验证
        test_name = "3.4 Search Verification"
        start = time.time()
        try:
            # 搜索应该能找到刚上传的文档
            resp = self.client.post("/api/search", json={
                "query": "machine learning neural networks",
                "method": "hybrid",
                "top_k": 10
            })

            if resp.status_code == 200:
                data = resp.json()
                total_results = data.get("total_results", 0)

                if total_results > 0:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="passed",
                        message=f"Found {total_results} results",
                        duration=time.time() - start,
                        details={"total_results": total_results}
                    ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="failed",
                        message="No search results found",
                        duration=time.time() - start
                    ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start,
                    error=resp.text
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        return results


class Scenario4_SearchFlow(TestScenario):
    """场景4: 搜索全流程"""

    name = "Scenario 4: Search Flow"
    description = "测试各种搜索方法和结果一致性"

    def run_tests(self) -> List[TestResult]:
        results = []
        query = "machine learning deep neural network"

        # Test 4.1: BM25搜索
        test_name = "4.1 BM25 Search"
        start = time.time()
        try:
            resp = self.client.post("/api/search", json={
                "query": query,
                "method": "bm25",
                "top_k": 10
            })

            if resp.status_code == 200:
                data = resp.json()
                self.context["bm25_results"] = data
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message=f"Found {data.get('total_results', 0)} results",
                    duration=time.time() - start,
                    details={"total_results": data.get("total_results", 0)}
                ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start,
                    error=resp.text
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 4.2: 向量搜索
        test_name = "4.2 Vector Search"
        start = time.time()
        try:
            resp = self.client.post("/api/search", json={
                "query": query,
                "method": "vector",
                "top_k": 10
            })

            if resp.status_code == 200:
                data = resp.json()
                self.context["vector_results"] = data
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message=f"Found {data.get('total_results', 0)} results",
                    duration=time.time() - start,
                    details={"total_results": data.get("total_results", 0)}
                ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start,
                    error=resp.text
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 4.3: 混合搜索
        test_name = "4.3 Hybrid Search"
        start = time.time()
        try:
            resp = self.client.post("/api/search", json={
                "query": query,
                "method": "hybrid",
                "top_k": 10
            })

            if resp.status_code == 200:
                data = resp.json()
                self.context["hybrid_results"] = data
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message=f"Found {data.get('total_results', 0)} results",
                    duration=time.time() - start,
                    details={"total_results": data.get("total_results", 0)}
                ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start,
                    error=resp.text
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 4.4: 增强搜索
        test_name = "4.4 Enhanced Search"
        start = time.time()
        try:
            resp = self.client.post("/api/search", json={
                "query": query,
                "method": "enhanced",
                "top_k": 10,
                "aggregate_by_doc": True
            })

            if resp.status_code == 200:
                data = resp.json()
                self.context["enhanced_results"] = data
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message=f"Found {data.get('total_results', 0)} results, {len(data.get('aggregated', []))} aggregated",
                    duration=time.time() - start,
                    details={
                        "total_results": data.get("total_results", 0),
                        "aggregated_count": len(data.get("aggregated", []))
                    }
                ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start,
                    error=resp.text
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 4.5: 多方法对比
        test_name = "4.5 Multi-Method Comparison"
        start = time.time()
        try:
            resp = self.client.get("/api/search/multi-method", params={
                "query": query,
                "top_k": 10
            })

            if resp.status_code == 200:
                data = resp.json()
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message="Multi-method search successful",
                    duration=time.time() - start,
                    details={
                        "bm25_count": len(data.get("bm25", [])),
                        "vector_count": len(data.get("vector", [])),
                        "hybrid_count": len(data.get("hybrid", [])),
                        "enhanced_count": len(data.get("enhanced_aggregated", []))
                    }
                ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start,
                    error=resp.text
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        return results


class Scenario5_TaskQueue(TestScenario):
    """场景5: 任务队列流程"""

    name = "Scenario 5: Task Queue"
    description = "测试任务队列的创建、监控和完成流程"

    def run_tests(self) -> List[TestResult]:
        results = []

        # Test 5.1: 获取任务列表
        test_name = "5.1 List Tasks"
        start = time.time()
        try:
            resp = self.client.get("/api/tasks", params={
                "page": 1,
                "page_size": 10
            })

            if resp.status_code == 200:
                data = resp.json()
                self.context["initial_task_count"] = data.get("total", 0)
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message=f"Found {data.get('total', 0)} tasks",
                    duration=time.time() - start,
                    details=data
                ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start,
                    error=resp.text
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 5.2: 按状态筛选任务
        for status in ["pending", "running", "completed", "failed"]:
            test_name = f"5.2 Filter Tasks by Status ({status})"
            start = time.time()
            try:
                resp = self.client.get("/api/tasks", params={
                    "status": status,
                    "page": 1,
                    "page_size": 5
                })

                if resp.status_code == 200:
                    data = resp.json()
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="passed",
                        message=f"Found {data.get('total', 0)} {status} tasks",
                        duration=time.time() - start
                    ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="failed",
                        message=f"HTTP {resp.status_code}",
                        duration=time.time() - start,
                        error=resp.text
                    ))
            except Exception as e:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="error",
                    duration=time.time() - start,
                    error=str(e),
                    traceback=traceback.format_exc()
                ))

        # Test 5.3: 获取单个任务详情（如果有任务）
        test_name = "5.3 Get Task Detail"
        start = time.time()
        try:
            # 先获取任务列表
            resp = self.client.get("/api/tasks", params={"page": 1, "page_size": 1})

            if resp.status_code == 200:
                data = resp.json()
                if data.get("tasks"):
                    task_id = data["tasks"][0]["id"]
                    detail_resp = self.client.get(f"/api/tasks/{task_id}")

                    if detail_resp.status_code == 200:
                        task_detail = detail_resp.json()
                        results.append(TestResult(
                            name=test_name,
                            scenario=self.name,
                            status="passed",
                            message=f"Got task {task_id} detail",
                            duration=time.time() - start,
                            details=task_detail
                        ))
                    else:
                        results.append(TestResult(
                            name=test_name,
                            scenario=self.name,
                            status="failed",
                            message=f"HTTP {detail_resp.status_code}",
                            duration=time.time() - start
                        ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="skipped",
                        message="No tasks available",
                        duration=time.time() - start
                    ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        return results


class Scenario6_DataConsistency(TestScenario):
    """场景6: 数据一致性验证"""

    name = "Scenario 6: Data Consistency"
    description = "验证SQLite、ChromaDB和Whoosh之间的数据一致性"

    def run_tests(self) -> List[TestResult]:
        results = []

        # Test 6.1: 获取统计信息
        test_name = "6.1 Get Index Stats"
        start = time.time()
        try:
            resp = self.client.get("/api/stats")

            if resp.status_code == 200:
                stats = resp.json()
                self.context["stats"] = stats
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message="Stats retrieved successfully",
                    duration=time.time() - start,
                    details=stats
                ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start,
                    error=resp.text
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 6.2: 验证文档计数
        test_name = "6.2 Verify Document Count"
        start = time.time()
        try:
            # 获取文档列表
            resp = self.client.get("/api/documents", params={
                "page": 1,
                "page_size": 1
            })

            if resp.status_code == 200:
                doc_data = resp.json()
                doc_count = doc_data.get("total", 0)
                stats = self.context.get("stats", {})
                stats_doc_count = stats.get("total_documents", 0)

                if doc_count == stats_doc_count:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="passed",
                        message=f"Document count consistent: {doc_count}",
                        duration=time.time() - start
                    ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="failed",
                        message=f"Mismatch: API={doc_count}, Stats={stats_doc_count}",
                        duration=time.time() - start
                    ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 6.3: 验证向量索引和BM25索引大小
        test_name = "6.3 Verify Index Sizes"
        start = time.time()
        try:
            stats = self.context.get("stats", {})
            vector_size = stats.get("vector_index_size", 0)
            bm25_size = stats.get("bm25_index_size", 0)
            total_chunks = stats.get("total_chunks", 0)

            # 检查索引大小是否合理
            details = {
                "vector_index_size": vector_size,
                "bm25_index_size": bm25_size,
                "total_chunks": total_chunks
            }

            # 向量索引大小应该接近chunks数量
            if total_chunks > 0:
                if abs(vector_size - total_chunks) / total_chunks < 0.1:  # 10%容差
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="passed",
                        message="Index sizes are consistent with chunks",
                        duration=time.time() - start,
                        details=details
                    ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="failed",
                        message=f"Index size mismatch: vector={vector_size}, chunks={total_chunks}",
                        duration=time.time() - start,
                        details=details
                    ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message="No chunks to verify",
                    duration=time.time() - start,
                    details=details
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 6.4: 按来源类型统计
        test_name = "6.4 Verify Source Type Stats"
        start = time.time()
        try:
            stats = self.context.get("stats", {})
            source_type_stats = stats.get("source_type_stats", {})
            total_docs = stats.get("total_documents", 0)

            # 验证各来源类型数量之和等于总文档数
            sum_by_type = sum(source_type_stats.values())

            if sum_by_type == total_docs:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="passed",
                    message=f"Source type stats consistent: {source_type_stats}",
                    duration=time.time() - start,
                    details=source_type_stats
                ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"Mismatch: sum={sum_by_type}, total={total_docs}",
                    duration=time.time() - start,
                    details=source_type_stats
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        # Test 6.5: 验证无孤立数据
        test_name = "6.5 Check for Orphan Data"
        start = time.time()
        try:
            # 获取几个文档并验证其chunks存在
            resp = self.client.get("/api/documents", params={
                "page": 1,
                "page_size": 5,
                "status": "indexed"
            })

            orphan_found = False
            orphan_details = []

            if resp.status_code == 200:
                docs = resp.json().get("documents", [])

                for doc in docs:
                    doc_id = doc["id"]
                    chunk_count = doc.get("chunk_count", 0)

                    # 获取文档详情验证chunks
                    detail_resp = self.client.get(f"/api/documents/{doc_id}")
                    if detail_resp.status_code == 200:
                        detail = detail_resp.json()
                        actual_chunks = len(detail.get("chunks", []))

                        if actual_chunks != chunk_count:
                            orphan_found = True
                            orphan_details.append({
                                "doc_id": doc_id,
                                "expected_chunks": chunk_count,
                                "actual_chunks": actual_chunks
                            })

                if not orphan_found:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="passed",
                        message=f"No orphan data found in {len(docs)} checked documents",
                        duration=time.time() - start
                    ))
                else:
                    results.append(TestResult(
                        name=test_name,
                        scenario=self.name,
                        status="failed",
                        message="Orphan data found",
                        duration=time.time() - start,
                        details={"orphans": orphan_details}
                    ))
            else:
                results.append(TestResult(
                    name=test_name,
                    scenario=self.name,
                    status="failed",
                    message=f"HTTP {resp.status_code}",
                    duration=time.time() - start
                ))
        except Exception as e:
            results.append(TestResult(
                name=test_name,
                scenario=self.name,
                status="error",
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))

        return results


# ==================== 报告生成 ====================

def generate_markdown_report(report: TestReport) -> str:
    """生成Markdown格式的测试报告"""
    lines = [
        "# Retrieval System E2E Test Report",
        "",
        f"**Test Date:** {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**API Base:** {report.api_base}",
        f"**Total Duration:** {report.duration:.2f}s",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Tests | {report.total_tests} |",
        f"| Passed | {report.passed_tests} |",
        f"| Failed | {report.failed_tests} |",
        f"| Pass Rate | {(report.passed_tests/report.total_tests*100) if report.total_tests > 0 else 0:.1f}% |",
        "",
        "## Scenario Results",
        ""
    ]

    for scenario in report.scenarios:
        status_icon = {
            "passed": "[PASSED]",
            "failed": "[FAILED]",
            "partial": "[PARTIAL]"
        }.get(scenario.status, "[UNKNOWN]")

        lines.extend([
            f"### {scenario.name} {status_icon}",
            "",
            f"> {scenario.description}",
            "",
            f"**Duration:** {scenario.duration:.2f}s | **Passed:** {scenario.passed_count} | **Failed:** {scenario.failed_count}",
            "",
            "| Test | Status | Duration | Message |",
            "|------|--------|----------|---------|"
        ])

        for test in scenario.tests:
            status_badge = {
                "passed": "PASS",
                "failed": "FAIL",
                "skipped": "SKIP",
                "error": "ERROR"
            }.get(test.status, test.status.upper())

            message = test.message[:80] + "..." if len(test.message) > 80 else test.message
            lines.append(f"| {test.name} | {status_badge} | {test.duration:.2f}s | {message} |")

        # 添加失败/错误详情
        failed_tests = [t for t in scenario.tests if t.status in ("failed", "error")]
        if failed_tests:
            lines.extend(["", "**Failure Details:**", ""])
            for test in failed_tests:
                lines.extend([
                    f"<details>",
                    f"<summary>{test.name}</summary>",
                    "",
                    "```",
                    f"Error: {test.error or 'N/A'}",
                    "",
                    f"Traceback: {test.traceback or 'N/A'}",
                    "```",
                    "</details>",
                    ""
                ])

        lines.append("")

    # 添加环境信息
    lines.extend([
        "## Environment",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Python | {sys.version.split()[0]} |",
        f"| Platform | {sys.platform} |",
        f"| Project Root | {PROJECT_ROOT} |",
        ""
    ])

    return "\n".join(lines)


def generate_json_report(report: TestReport) -> str:
    """生成JSON格式的测试报告"""
    def serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        else:
            return obj

    return json.dumps(serialize(report), indent=2, ensure_ascii=False)


# ==================== 主测试运行器 ====================

def run_all_tests(api_base: str = API_BASE) -> TestReport:
    """运行所有测试"""
    report = TestReport(start_time=datetime.now(), api_base=api_base)

    scenarios = [
        Scenario1_BookmarkImport,
        Scenario2_PDFUpload,
        Scenario3_TextUpload,
        Scenario4_SearchFlow,
        Scenario5_TaskQueue,
        Scenario6_DataConsistency,
    ]

    with APIClient(api_base) as client:
        # 首先检查API是否可用
        try:
            health_resp = client.get("/")
            if health_resp.status_code not in (200, 307):
                print(f"[ERROR] API not available at {api_base}")
                return report
        except Exception as e:
            print(f"[ERROR] Cannot connect to API: {e}")
            return report

        print(f"\n{'='*60}")
        print(f"Running E2E Tests against {api_base}")
        print(f"{'='*60}\n")

        for scenario_class in scenarios:
            scenario = scenario_class(client)
            print(f"[RUNNING] {scenario.name}...")

            result = scenario.run()
            report.scenarios.append(result)

            # 打印结果
            status_icon = {
                "passed": "[PASSED]",
                "failed": "[FAILED]",
                "partial": "[PARTIAL]"
            }.get(result.status, "[UNKNOWN]")

            print(f"  {status_icon} {result.passed_count}/{len(result.tests)} tests passed ({result.duration:.2f}s)")

            # 打印失败的测试
            for test in result.tests:
                if test.status in ("failed", "error"):
                    print(f"    - {test.name}: {test.message}")
                    if test.error:
                        print(f"      Error: {test.error}")

    report.end_time = datetime.now()

    # 打印总结
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Total: {report.total_tests} | Passed: {report.passed_tests} | Failed: {report.failed_tests}")
    print(f"Duration: {report.duration:.2f}s")
    print(f"Pass Rate: {(report.passed_tests/report.total_tests*100) if report.total_tests > 0 else 0:.1f}%")

    return report


def main():
    """主函数"""
    import argparse

    clear_proxy_env()

    parser = argparse.ArgumentParser(description="Retrieval System E2E Tests")
    parser.add_argument("--api", default=API_BASE, help="API base URL")
    parser.add_argument("--output", "-o", default=None, help="Output report file")
    parser.add_argument("--format", choices=["md", "json", "both"], default="md",
                        help="Report format")

    args = parser.parse_args()

    # 运行测试
    report = run_all_tests(args.api)

    # 生成报告
    if args.output or True:  # 始终生成报告
        output_dir = PROJECT_ROOT / "docs"
        output_dir.mkdir(exist_ok=True)

        if args.format in ("md", "both"):
            md_path = output_dir / "e2e_test_report.md"
            md_content = generate_markdown_report(report)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            print(f"\nMarkdown report saved to: {md_path}")

        if args.format in ("json", "both"):
            json_path = output_dir / "e2e_test_report.json"
            json_content = generate_json_report(report)
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json_content)
            print(f"JSON report saved to: {json_path}")

    # 返回退出码
    return 0 if report.failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
