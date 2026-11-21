"""
服务层全面测试脚本
测试范围: IndexService, DocumentService, TaskService, CrawlService
"""
import sys
import os
import json
import time
import hashlib
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import tempfile
import traceback

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from core.db import get_db_session, init_db
from core.db.models import (
    Document, Chunk, Task, CrawlRecord, Paper,
    SourceType, DataSource, DocumentStatus,
    TaskStatus as DBTaskStatus, CrawlStatus as DBCrawlStatus
)
from app.services import DocumentService, IndexService, TaskService, CrawlService
from app.tasks.base import (
    TaskStatus, TaskRecord, TaskContext,
    create_task_record, get_task_record, update_task_record,
    cancel_task, cleanup_zombie_tasks, DEFAULT_TASK_TIMEOUT
)


# ================== 测试结果数据结构 ==================

@dataclass
class TestCase:
    """单个测试用例"""
    name: str
    category: str
    description: str
    input_params: Dict[str, Any]
    expected_result: str
    actual_result: str = ""
    status: str = "pending"  # pending, passed, failed, error
    error_message: str = ""
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TestReport:
    """测试报告"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    test_cases: List[TestCase] = field(default_factory=list)

    def add_test(self, test_case: TestCase):
        self.test_cases.append(test_case)
        self.total_tests += 1
        if test_case.status == "passed":
            self.passed_tests += 1
        elif test_case.status == "failed":
            self.failed_tests += 1
        elif test_case.status == "error":
            self.error_tests += 1

    def finish(self):
        self.end_time = datetime.now()

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests * 100


# ================== 测试工具函数 ==================

def run_test(test_func, test_case: TestCase) -> TestCase:
    """运行单个测试并捕获结果"""
    start_time = time.time()
    try:
        result = test_func()
        test_case.actual_result = str(result)
        test_case.status = "passed"
    except AssertionError as e:
        test_case.actual_result = str(e)
        test_case.status = "failed"
        test_case.error_message = str(e)
    except Exception as e:
        test_case.actual_result = str(e)
        test_case.status = "error"
        test_case.error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    finally:
        test_case.execution_time_ms = (time.time() - start_time) * 1000

    return test_case


def cleanup_test_data(doc_ids: List[int] = None, task_ids: List[int] = None):
    """清理测试数据"""
    with get_db_session() as session:
        if doc_ids:
            for doc_id in doc_ids:
                doc = session.query(Document).filter(Document.id == doc_id).first()
                if doc:
                    session.delete(doc)
        if task_ids:
            for task_id in task_ids:
                task = session.query(Task).filter(Task.id == task_id).first()
                if task:
                    session.delete(task)


# ================== DocumentService 测试 ==================

class DocumentServiceTests:
    """DocumentService 测试类"""

    def __init__(self):
        self.service = DocumentService()
        self.created_doc_ids = []

    def cleanup(self):
        """清理测试数据"""
        cleanup_test_data(doc_ids=self.created_doc_ids)
        self.created_doc_ids = []

    def test_create_document_normal(self) -> TestCase:
        """测试正常文档创建"""
        test_case = TestCase(
            name="create_document_normal",
            category="DocumentService",
            description="创建正常文档，包含所有必要字段",
            input_params={
                "title": "Test Document",
                "content": "This is a test document content.",
                "source": "test_source",
                "source_type": "upload"
            },
            expected_result="Document created with valid ID"
        )

        def test_func():
            doc = self.service.create_document(
                title="Test Document",
                content="This is a test document content.",
                source="test_source",
                source_type=SourceType.upload,
                data_source=DataSource.user
            )
            self.created_doc_ids.append(doc.id)
            assert doc.id is not None, "Document ID should not be None"
            assert doc.title == "Test Document", "Title mismatch"
            assert doc.status == DocumentStatus.pending, "Status should be pending"
            return f"Created document with ID: {doc.id}"

        return run_test(test_func, test_case)

    def test_create_document_empty_content(self) -> TestCase:
        """测试空内容文档创建"""
        test_case = TestCase(
            name="create_document_empty_content",
            category="DocumentService",
            description="创建空内容文档",
            input_params={
                "title": "Empty Document",
                "content": None,
            },
            expected_result="Document created without content hash"
        )

        def test_func():
            doc = self.service.create_document(
                title="Empty Document",
                content=None,
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)
            assert doc.id is not None, "Document ID should not be None"
            assert doc.content_hash is None, "Content hash should be None for empty content"
            return f"Created empty document with ID: {doc.id}"

        return run_test(test_func, test_case)

    def test_create_document_large_content(self) -> TestCase:
        """测试大文档创建"""
        test_case = TestCase(
            name="create_document_large_content",
            category="DocumentService",
            description="创建包含大量内容的文档 (100KB)",
            input_params={
                "title": "Large Document",
                "content_size": "100KB",
            },
            expected_result="Document created with large content"
        )

        def test_func():
            large_content = "x" * (100 * 1024)  # 100KB
            doc = self.service.create_document(
                title="Large Document",
                content=large_content,
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)
            assert doc.id is not None, "Document ID should not be None"
            assert doc.content_hash is not None, "Content hash should exist"
            return f"Created large document with ID: {doc.id}"

        return run_test(test_func, test_case)

    def test_create_document_special_characters(self) -> TestCase:
        """测试特殊字符文档创建"""
        test_case = TestCase(
            name="create_document_special_characters",
            category="DocumentService",
            description="创建包含特殊字符的文档 (中文、emoji、换行符)",
            input_params={
                "title": "特殊字符文档 Test",
                "content": "包含中文和特殊符号",
            },
            expected_result="Document created with special characters"
        )

        def test_func():
            special_content = "中文内容\n换行符\ttab\r\n\u0000null"
            doc = self.service.create_document(
                title="特殊字符文档 Test",
                content=special_content,
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)
            assert doc.id is not None, "Document ID should not be None"
            assert "特殊字符" in doc.title, "Title should contain Chinese characters"
            return f"Created document with special characters, ID: {doc.id}"

        return run_test(test_func, test_case)

    def test_get_document_exists(self) -> TestCase:
        """测试获取存在的文档"""
        test_case = TestCase(
            name="get_document_exists",
            category="DocumentService",
            description="获取已存在的文档",
            input_params={"doc_id": "created_doc_id"},
            expected_result="Document retrieved successfully"
        )

        def test_func():
            # 先创建文档
            doc = self.service.create_document(
                title="Test Get Document",
                content="Test content",
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            # 获取文档
            retrieved = self.service.get_document(doc.id)
            assert retrieved is not None, "Retrieved document should not be None"
            assert retrieved.id == doc.id, "Document ID mismatch"
            assert retrieved.title == doc.title, "Document title mismatch"
            return f"Retrieved document: {retrieved.title}"

        return run_test(test_func, test_case)

    def test_get_document_not_exists(self) -> TestCase:
        """测试获取不存在的文档"""
        test_case = TestCase(
            name="get_document_not_exists",
            category="DocumentService",
            description="获取不存在的文档",
            input_params={"doc_id": 999999},
            expected_result="None returned for non-existent document"
        )

        def test_func():
            doc = self.service.get_document(999999)
            assert doc is None, "Should return None for non-existent document"
            return "Correctly returned None"

        return run_test(test_func, test_case)

    def test_update_document_title(self) -> TestCase:
        """测试更新文档标题"""
        test_case = TestCase(
            name="update_document_title",
            category="DocumentService",
            description="更新文档标题",
            input_params={"title": "Updated Title"},
            expected_result="Document title updated"
        )

        def test_func():
            # 创建文档
            doc = self.service.create_document(
                title="Original Title",
                content="Test content",
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            # 更新文档
            updated = self.service.update_document(doc.id, title="Updated Title")
            assert updated is not None, "Updated document should not be None"
            assert updated.title == "Updated Title", "Title should be updated"
            return f"Title updated from 'Original Title' to '{updated.title}'"

        return run_test(test_func, test_case)

    def test_update_document_status(self) -> TestCase:
        """测试更新文档状态"""
        test_case = TestCase(
            name="update_document_status",
            category="DocumentService",
            description="更新文档状态为 indexed",
            input_params={"status": "indexed"},
            expected_result="Document status updated to indexed"
        )

        def test_func():
            doc = self.service.create_document(
                title="Test Status Update",
                content="Test",
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            updated = self.service.update_document(doc.id, status=DocumentStatus.indexed)
            assert updated is not None, "Updated document should not be None"
            assert updated.status == DocumentStatus.indexed, "Status should be indexed"
            return f"Status updated to {updated.status.value}"

        return run_test(test_func, test_case)

    def test_update_document_metadata(self) -> TestCase:
        """测试更新文档元数据"""
        test_case = TestCase(
            name="update_document_metadata",
            category="DocumentService",
            description="更新文档元数据（合并）",
            input_params={"metadata": {"key1": "value1", "key2": "value2"}},
            expected_result="Metadata merged correctly"
        )

        def test_func():
            doc = self.service.create_document(
                title="Test Metadata Update",
                content="Test",
                source="test",
                source_type=SourceType.upload,
                metadata={"existing_key": "existing_value"}
            )
            self.created_doc_ids.append(doc.id)

            updated = self.service.update_document(
                doc.id,
                metadata={"new_key": "new_value"}
            )
            assert updated is not None, "Updated document should not be None"
            meta = updated.metadata_json
            assert "existing_key" in meta, "Existing key should be preserved"
            assert "new_key" in meta, "New key should be added"
            return f"Metadata: {meta}"

        return run_test(test_func, test_case)

    def test_delete_document_exists(self) -> TestCase:
        """测试删除存在的文档"""
        test_case = TestCase(
            name="delete_document_exists",
            category="DocumentService",
            description="删除已存在的文档",
            input_params={"doc_id": "created_doc_id"},
            expected_result="Document deleted successfully"
        )

        def test_func():
            doc = self.service.create_document(
                title="Test Delete Document",
                content="Test",
                source="test",
                source_type=SourceType.upload
            )
            doc_id = doc.id

            result = self.service.delete_document(doc_id, delete_index=False)
            assert result is True, "Delete should return True"

            # 验证删除
            deleted_doc = self.service.get_document(doc_id)
            assert deleted_doc is None, "Document should be deleted"
            return f"Document {doc_id} deleted successfully"

        return run_test(test_func, test_case)

    def test_delete_document_not_exists(self) -> TestCase:
        """测试删除不存在的文档"""
        test_case = TestCase(
            name="delete_document_not_exists",
            category="DocumentService",
            description="删除不存在的文档",
            input_params={"doc_id": 999999},
            expected_result="Returns False for non-existent document"
        )

        def test_func():
            result = self.service.delete_document(999999, delete_index=False)
            assert result is False, "Should return False for non-existent document"
            return "Correctly returned False"

        return run_test(test_func, test_case)

    def test_list_documents_pagination(self) -> TestCase:
        """测试文档分页"""
        test_case = TestCase(
            name="list_documents_pagination",
            category="DocumentService",
            description="测试文档列表分页功能",
            input_params={"offset": 0, "limit": 5},
            expected_result="Pagination works correctly"
        )

        def test_func():
            # 创建多个文档
            for i in range(3):
                doc = self.service.create_document(
                    title=f"Pagination Test {i}",
                    content=f"Content {i}",
                    source="test",
                    source_type=SourceType.upload
                )
                self.created_doc_ids.append(doc.id)

            # 测试分页
            docs, total = self.service.list_documents(offset=0, limit=2)
            assert len(docs) <= 2, "Should return at most 2 documents"
            assert total >= 3, "Total should be at least 3"
            return f"Page 1: {len(docs)} docs, Total: {total}"

        return run_test(test_func, test_case)

    def test_list_documents_filter_source_type(self) -> TestCase:
        """测试按来源类型筛选"""
        test_case = TestCase(
            name="list_documents_filter_source_type",
            category="DocumentService",
            description="按来源类型筛选文档",
            input_params={"source_type": "upload"},
            expected_result="Only upload documents returned"
        )

        def test_func():
            doc = self.service.create_document(
                title="Filter Test Upload",
                content="Test",
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            docs, total = self.service.list_documents(source_type=SourceType.upload)
            assert all(d.source_type == SourceType.upload for d in docs), \
                "All documents should be upload type"
            return f"Found {len(docs)} upload documents"

        return run_test(test_func, test_case)

    def test_list_documents_keyword_search(self) -> TestCase:
        """测试关键词搜索"""
        test_case = TestCase(
            name="list_documents_keyword_search",
            category="DocumentService",
            description="按关键词搜索文档标题",
            input_params={"keyword": "UniqueKeyword"},
            expected_result="Only matching documents returned"
        )

        def test_func():
            doc = self.service.create_document(
                title="UniqueKeyword Document Test",
                content="Test",
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            docs, total = self.service.list_documents(keyword="UniqueKeyword")
            assert any("UniqueKeyword" in d.title for d in docs), \
                "Should find document with keyword"
            return f"Found {len(docs)} documents with keyword"

        return run_test(test_func, test_case)

    def test_check_duplicate_exists(self) -> TestCase:
        """测试重复内容检测-存在"""
        test_case = TestCase(
            name="check_duplicate_exists",
            category="DocumentService",
            description="检测已存在的重复内容",
            input_params={"content": "duplicate content"},
            expected_result="Duplicate detected"
        )

        def test_func():
            content = "This is duplicate content for testing"
            doc = self.service.create_document(
                title="Duplicate Test",
                content=content,
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            duplicate = self.service.check_duplicate(content)
            assert duplicate is not None, "Should detect duplicate"
            assert duplicate.id == doc.id, "Should return the same document"
            return f"Duplicate found: document {duplicate.id}"

        return run_test(test_func, test_case)

    def test_check_duplicate_not_exists(self) -> TestCase:
        """测试重复内容检测-不存在"""
        test_case = TestCase(
            name="check_duplicate_not_exists",
            category="DocumentService",
            description="检测不存在的内容",
            input_params={"content": "unique content"},
            expected_result="No duplicate found"
        )

        def test_func():
            unique_content = f"Unique content {datetime.now().isoformat()}"
            duplicate = self.service.check_duplicate(unique_content)
            assert duplicate is None, "Should not find duplicate"
            return "No duplicate found as expected"

        return run_test(test_func, test_case)

    def test_extract_links_from_html(self) -> TestCase:
        """测试从HTML提取链接"""
        test_case = TestCase(
            name="extract_links_from_html",
            category="DocumentService",
            description="从HTML内容提取链接",
            input_params={"html": "<a href='https://example.com'>Link</a>"},
            expected_result="Links extracted correctly"
        )

        def test_func():
            html = """
            <html>
            <body>
                <a href="https://example.com">Example</a>
                <a href="/relative">Relative</a>
                <a href="#">Anchor</a>
                <a href="javascript:void(0)">JS</a>
            </body>
            </html>
            """
            links = self.service.extract_links_from_html(
                html,
                base_url="https://base.com"
            )
            assert any(l["url"] == "https://example.com" for l in links), \
                "Should find absolute link"
            assert any(l["url"] == "https://base.com/relative" for l in links), \
                "Should resolve relative link"
            return f"Extracted {len(links)} links"

        return run_test(test_func, test_case)

    def run_all(self) -> List[TestCase]:
        """运行所有测试"""
        tests = [
            self.test_create_document_normal,
            self.test_create_document_empty_content,
            self.test_create_document_large_content,
            self.test_create_document_special_characters,
            self.test_get_document_exists,
            self.test_get_document_not_exists,
            self.test_update_document_title,
            self.test_update_document_status,
            self.test_update_document_metadata,
            self.test_delete_document_exists,
            self.test_delete_document_not_exists,
            self.test_list_documents_pagination,
            self.test_list_documents_filter_source_type,
            self.test_list_documents_keyword_search,
            self.test_check_duplicate_exists,
            self.test_check_duplicate_not_exists,
            self.test_extract_links_from_html,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                print(f"  [{result.status.upper()}] {result.name}")
            except Exception as e:
                print(f"  [ERROR] {test.__name__}: {e}")

        self.cleanup()
        return results


# ================== TaskService 测试 ==================

class TaskServiceTests:
    """TaskService 测试类"""

    def __init__(self):
        self.service = TaskService()
        self.created_task_ids = []

    def cleanup(self):
        """清理测试数据"""
        cleanup_test_data(task_ids=self.created_task_ids)
        self.created_task_ids = []
        # 清理任务记录文件
        from app.tasks.base import TASKS_DIR
        for task_id in self.created_task_ids:
            task_file = TASKS_DIR / f"{task_id}.json"
            if task_file.exists():
                task_file.unlink()

    def test_create_task_basic(self) -> TestCase:
        """测试基本任务创建"""
        test_case = TestCase(
            name="create_task_basic",
            category="TaskService",
            description="创建基本任务",
            input_params={
                "task_type": "test_task",
                "params": {"key": "value"}
            },
            expected_result="Task created with valid ID"
        )

        def test_func():
            task = self.service.create_task(
                task_type="test_task",
                params={"key": "value"}
            )
            self.created_task_ids.append(task.id)
            assert task.id is not None, "Task ID should not be None"
            assert task.status == DBTaskStatus.pending, "Status should be pending"
            assert task.task_type == "test_task", "Task type mismatch"
            return f"Created task with ID: {task.id}"

        return run_test(test_func, test_case)

    def test_create_task_with_priority(self) -> TestCase:
        """测试带优先级的任务创建"""
        test_case = TestCase(
            name="create_task_with_priority",
            category="TaskService",
            description="创建高优先级任务",
            input_params={
                "task_type": "priority_task",
                "priority": 10
            },
            expected_result="Task created with priority 10"
        )

        def test_func():
            task = self.service.create_task(
                task_type="priority_task",
                priority=10
            )
            self.created_task_ids.append(task.id)
            assert task.priority == 10, "Priority should be 10"
            return f"Created task with priority: {task.priority}"

        return run_test(test_func, test_case)

    def test_create_task_with_timeout(self) -> TestCase:
        """测试带超时的任务创建"""
        test_case = TestCase(
            name="create_task_with_timeout",
            category="TaskService",
            description="创建自定义超时时间的任务",
            input_params={
                "task_type": "timeout_task",
                "timeout_seconds": 300
            },
            expected_result="Task created with 300s timeout"
        )

        def test_func():
            task = self.service.create_task(
                task_type="timeout_task",
                timeout_seconds=300
            )
            self.created_task_ids.append(task.id)
            assert task.timeout_seconds == 300, "Timeout should be 300"
            return f"Created task with timeout: {task.timeout_seconds}s"

        return run_test(test_func, test_case)

    def test_get_task_exists(self) -> TestCase:
        """测试获取存在的任务"""
        test_case = TestCase(
            name="get_task_exists",
            category="TaskService",
            description="获取已存在的任务",
            input_params={"task_id": "created_task_id"},
            expected_result="Task retrieved successfully"
        )

        def test_func():
            task = self.service.create_task(
                task_type="get_test",
                params={"test": True}
            )
            self.created_task_ids.append(task.id)

            retrieved = self.service.get_task(task.id)
            assert retrieved is not None, "Task should be found"
            assert retrieved.id == task.id, "Task ID mismatch"
            return f"Retrieved task: {retrieved.task_type}"

        return run_test(test_func, test_case)

    def test_get_task_not_exists(self) -> TestCase:
        """测试获取不存在的任务"""
        test_case = TestCase(
            name="get_task_not_exists",
            category="TaskService",
            description="获取不存在的任务",
            input_params={"task_id": 999999},
            expected_result="None returned"
        )

        def test_func():
            task = self.service.get_task(999999)
            assert task is None, "Should return None"
            return "Correctly returned None"

        return run_test(test_func, test_case)

    def test_get_task_status(self) -> TestCase:
        """测试获取任务状态详情"""
        test_case = TestCase(
            name="get_task_status",
            category="TaskService",
            description="获取任务状态详细信息",
            input_params={"task_id": "created_task_id"},
            expected_result="Status info returned"
        )

        def test_func():
            task = self.service.create_task(
                task_type="status_test",
                params={"test": True}
            )
            self.created_task_ids.append(task.id)

            status_info = self.service.get_task_status(task.id)
            assert status_info is not None, "Status info should not be None"
            assert "status" in status_info, "Should have status field"
            assert "task_type" in status_info, "Should have task_type field"
            return f"Status: {status_info.get('status')}"

        return run_test(test_func, test_case)

    def test_list_tasks_pagination(self) -> TestCase:
        """测试任务列表分页"""
        test_case = TestCase(
            name="list_tasks_pagination",
            category="TaskService",
            description="测试任务列表分页",
            input_params={"offset": 0, "limit": 5},
            expected_result="Pagination works correctly"
        )

        def test_func():
            # 创建多个任务
            for i in range(3):
                task = self.service.create_task(
                    task_type=f"pagination_test_{i}"
                )
                self.created_task_ids.append(task.id)

            tasks, total = self.service.list_tasks(offset=0, limit=2)
            assert len(tasks) <= 2, "Should return at most 2 tasks"
            return f"Page 1: {len(tasks)} tasks, Total: {total}"

        return run_test(test_func, test_case)

    def test_list_tasks_filter_status(self) -> TestCase:
        """测试按状态筛选任务"""
        test_case = TestCase(
            name="list_tasks_filter_status",
            category="TaskService",
            description="按状态筛选任务",
            input_params={"status": "pending"},
            expected_result="Only pending tasks returned"
        )

        def test_func():
            task = self.service.create_task(task_type="filter_test")
            self.created_task_ids.append(task.id)

            tasks, total = self.service.list_tasks(status=DBTaskStatus.pending)
            assert all(t.status == DBTaskStatus.pending for t in tasks), \
                "All tasks should be pending"
            return f"Found {len(tasks)} pending tasks"

        return run_test(test_func, test_case)

    def test_cancel_task_pending(self) -> TestCase:
        """测试取消待处理任务"""
        test_case = TestCase(
            name="cancel_task_pending",
            category="TaskService",
            description="取消待处理状态的任务",
            input_params={"task_id": "pending_task_id"},
            expected_result="Task cancelled successfully"
        )

        def test_func():
            task = self.service.create_task(task_type="cancel_test")
            self.created_task_ids.append(task.id)

            result = self.service.cancel_task(task.id)
            assert result is True, "Cancel should return True"

            updated = self.service.get_task(task.id)
            assert updated.status == DBTaskStatus.cancelled, "Status should be cancelled"
            return f"Task {task.id} cancelled"

        return run_test(test_func, test_case)

    def test_cancel_task_completed(self) -> TestCase:
        """测试取消已完成任务（应失败）"""
        test_case = TestCase(
            name="cancel_task_completed",
            category="TaskService",
            description="尝试取消已完成的任务（应失败）",
            input_params={"task_id": "completed_task_id"},
            expected_result="Cancel returns False"
        )

        def test_func():
            task = self.service.create_task(task_type="cancel_completed_test")
            self.created_task_ids.append(task.id)

            # 手动更新为已完成状态
            with get_db_session() as session:
                db_task = session.query(Task).filter(Task.id == task.id).first()
                db_task.status = DBTaskStatus.completed

            result = self.service.cancel_task(task.id)
            assert result is False, "Cannot cancel completed task"
            return "Correctly returned False for completed task"

        return run_test(test_func, test_case)

    def test_retry_task_failed(self) -> TestCase:
        """测试重试失败任务"""
        test_case = TestCase(
            name="retry_task_failed",
            category="TaskService",
            description="重试失败状态的任务",
            input_params={"task_id": "failed_task_id"},
            expected_result="New task created for retry"
        )

        def test_func():
            # 创建并设为失败状态
            task = self.service.create_task(
                task_type="retry_test",
                params={"original": True},
                max_retries=3
            )
            self.created_task_ids.append(task.id)

            with get_db_session() as session:
                db_task = session.query(Task).filter(Task.id == task.id).first()
                db_task.status = DBTaskStatus.failed

            new_task = self.service.retry_task(task.id)
            if new_task:
                self.created_task_ids.append(new_task.id)
            assert new_task is not None, "Retry should create new task"
            assert new_task.retry_count == 1, "Retry count should be 1"
            return f"Retry created task {new_task.id}"

        return run_test(test_func, test_case)

    def test_retry_task_max_retries(self) -> TestCase:
        """测试超过最大重试次数"""
        test_case = TestCase(
            name="retry_task_max_retries",
            category="TaskService",
            description="测试超过最大重试次数后无法重试",
            input_params={"max_retries": 1, "retry_count": 1},
            expected_result="Retry returns None"
        )

        def test_func():
            task = self.service.create_task(
                task_type="max_retry_test",
                max_retries=1
            )
            self.created_task_ids.append(task.id)

            with get_db_session() as session:
                db_task = session.query(Task).filter(Task.id == task.id).first()
                db_task.status = DBTaskStatus.failed
                db_task.retry_count = 1  # 已达最大重试

            new_task = self.service.retry_task(task.id)
            assert new_task is None, "Should not allow retry"
            return "Correctly prevented retry after max attempts"

        return run_test(test_func, test_case)

    def test_update_task_progress(self) -> TestCase:
        """测试更新任务进度"""
        test_case = TestCase(
            name="update_task_progress",
            category="TaskService",
            description="更新任务进度",
            input_params={"progress": 50, "total": 100},
            expected_result="Progress updated to 50/100"
        )

        def test_func():
            task = self.service.create_task(task_type="progress_test")
            self.created_task_ids.append(task.id)

            self.service.update_task_progress(task.id, progress=50, total=100)

            updated = self.service.get_task(task.id)
            assert updated.progress == 50, "Progress should be 50"
            assert updated.total == 100, "Total should be 100"
            return f"Progress: {updated.progress}/{updated.total}"

        return run_test(test_func, test_case)

    def test_get_pending_tasks(self) -> TestCase:
        """测试获取待处理任务"""
        test_case = TestCase(
            name="get_pending_tasks",
            category="TaskService",
            description="获取待处理的任务列表",
            input_params={"limit": 10},
            expected_result="Pending tasks retrieved"
        )

        def test_func():
            task = self.service.create_task(task_type="pending_list_test")
            self.created_task_ids.append(task.id)

            pending = self.service.get_pending_tasks(limit=10)
            assert any(t.id == task.id for t in pending), "Should find created task"
            return f"Found {len(pending)} pending tasks"

        return run_test(test_func, test_case)

    def test_delete_task_pending(self) -> TestCase:
        """测试删除待处理任务"""
        test_case = TestCase(
            name="delete_task_pending",
            category="TaskService",
            description="删除待处理状态的任务",
            input_params={"task_id": "pending_task_id"},
            expected_result="Task deleted successfully"
        )

        def test_func():
            task = self.service.create_task(task_type="delete_test")
            task_id = task.id

            result = self.service.delete_task(task_id)
            assert result is True, "Delete should return True"

            deleted = self.service.get_task(task_id)
            assert deleted is None, "Task should be deleted"
            return f"Task {task_id} deleted"

        return run_test(test_func, test_case)

    def test_delete_task_running(self) -> TestCase:
        """测试删除运行中任务（应失败）"""
        test_case = TestCase(
            name="delete_task_running",
            category="TaskService",
            description="尝试删除运行中的任务（应失败）",
            input_params={"task_id": "running_task_id"},
            expected_result="Delete returns False"
        )

        def test_func():
            task = self.service.create_task(task_type="delete_running_test")
            self.created_task_ids.append(task.id)

            with get_db_session() as session:
                db_task = session.query(Task).filter(Task.id == task.id).first()
                db_task.status = DBTaskStatus.running

            result = self.service.delete_task(task.id)
            assert result is False, "Cannot delete running task"
            return "Correctly prevented deleting running task"

        return run_test(test_func, test_case)

    def test_cleanup_zombie_tasks(self) -> TestCase:
        """测试僵尸任务清理"""
        test_case = TestCase(
            name="cleanup_zombie_tasks",
            category="TaskService",
            description="清理僵尸任务（心跳超时）",
            input_params={},
            expected_result="Zombie tasks cleaned"
        )

        def test_func():
            # 创建一个模拟僵尸任务
            task = self.service.create_task(task_type="zombie_test")
            self.created_task_ids.append(task.id)

            # 设置为运行状态并设置过期心跳
            with get_db_session() as session:
                db_task = session.query(Task).filter(Task.id == task.id).first()
                db_task.status = DBTaskStatus.running
                db_task.last_heartbeat = datetime.now() - timedelta(hours=1)

            result = self.service.cleanup_all_zombies()
            assert "db_records" in result, "Should have db_records in result"
            return f"Cleaned: {result}"

        return run_test(test_func, test_case)

    def test_get_task_health(self) -> TestCase:
        """测试获取任务健康状态"""
        test_case = TestCase(
            name="get_task_health",
            category="TaskService",
            description="获取任务健康状态信息",
            input_params={"task_id": "task_id"},
            expected_result="Health info returned"
        )

        def test_func():
            task = self.service.create_task(task_type="health_test")
            self.created_task_ids.append(task.id)

            health = self.service.get_task_health(task.id)
            # 可能返回 not_found 如果文件记录不存在
            assert health is not None, "Health info should not be None"
            return f"Health status: {health.get('status', 'unknown')}"

        return run_test(test_func, test_case)

    def run_all(self) -> List[TestCase]:
        """运行所有测试"""
        tests = [
            self.test_create_task_basic,
            self.test_create_task_with_priority,
            self.test_create_task_with_timeout,
            self.test_get_task_exists,
            self.test_get_task_not_exists,
            self.test_get_task_status,
            self.test_list_tasks_pagination,
            self.test_list_tasks_filter_status,
            self.test_cancel_task_pending,
            self.test_cancel_task_completed,
            self.test_retry_task_failed,
            self.test_retry_task_max_retries,
            self.test_update_task_progress,
            self.test_get_pending_tasks,
            self.test_delete_task_pending,
            self.test_delete_task_running,
            self.test_cleanup_zombie_tasks,
            self.test_get_task_health,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                print(f"  [{result.status.upper()}] {result.name}")
            except Exception as e:
                print(f"  [ERROR] {test.__name__}: {e}")

        self.cleanup()
        return results


# ================== IndexService 测试 ==================

class IndexServiceTests:
    """IndexService 测试类"""

    def __init__(self):
        self.service = IndexService(collection_name="test_collection")
        self.doc_service = DocumentService()
        self.created_doc_ids = []

    def cleanup(self):
        """清理测试数据"""
        for doc_id in self.created_doc_ids:
            try:
                self.service.delete_document_index(doc_id)
                self.doc_service.delete_document(doc_id, delete_index=False)
            except:
                pass
        self.created_doc_ids = []

    def test_index_document_normal(self) -> TestCase:
        """测试正常文档索引"""
        test_case = TestCase(
            name="index_document_normal",
            category="IndexService",
            description="索引正常文档内容",
            input_params={
                "content": "This is a test document for indexing.",
                "title": "Test Document"
            },
            expected_result="Document indexed with chunks"
        )

        def test_func():
            # 创建文档
            doc = self.doc_service.create_document(
                title="Index Test Document",
                content="This is a test document for indexing. It has multiple sentences.",
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            # 索引文档
            result = self.service.index_document(
                doc_id=doc.id,
                content="This is a test document for indexing. It has multiple sentences.",
                title="Index Test Document",
                source="test"
            )

            assert result["success"], f"Indexing failed: {result.get('error')}"
            assert result["chunk_count"] > 0, "Should have at least one chunk"
            return f"Indexed with {result['chunk_count']} chunks"

        return run_test(test_func, test_case)

    def test_index_document_empty(self) -> TestCase:
        """测试空文档索引"""
        test_case = TestCase(
            name="index_document_empty",
            category="IndexService",
            description="索引空内容文档",
            input_params={"content": ""},
            expected_result="Indexing fails or returns 0 chunks"
        )

        def test_func():
            doc = self.doc_service.create_document(
                title="Empty Index Test",
                content="",
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            result = self.service.index_document(
                doc_id=doc.id,
                content="",
                title="Empty Index Test"
            )

            # 空内容应该失败或返回0 chunks
            if result["success"]:
                assert result["chunk_count"] == 0, "Empty content should have 0 chunks"
                return "Indexed with 0 chunks"
            else:
                return f"Correctly failed: {result.get('error')}"

        return run_test(test_func, test_case)

    def test_index_document_large(self) -> TestCase:
        """测试大文档索引"""
        test_case = TestCase(
            name="index_document_large",
            category="IndexService",
            description="索引大文档 (50KB)",
            input_params={"content_size": "50KB"},
            expected_result="Large document indexed with multiple chunks"
        )

        def test_func():
            # 生成50KB的内容
            large_content = " ".join(["This is a test sentence."] * 3000)

            doc = self.doc_service.create_document(
                title="Large Index Test",
                content=large_content,
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            result = self.service.index_document(
                doc_id=doc.id,
                content=large_content,
                title="Large Index Test"
            )

            assert result["success"], f"Indexing failed: {result.get('error')}"
            assert result["chunk_count"] > 1, "Large document should have multiple chunks"
            return f"Indexed with {result['chunk_count']} chunks"

        return run_test(test_func, test_case)

    def test_index_document_special_chars(self) -> TestCase:
        """测试特殊字符文档索引"""
        test_case = TestCase(
            name="index_document_special_chars",
            category="IndexService",
            description="索引包含特殊字符的文档",
            input_params={"content": "中文内容 + special chars"},
            expected_result="Document with special chars indexed"
        )

        def test_func():
            special_content = "这是中文内容。This is English. Numbers: 12345. Symbols: @#$%^&*()"

            doc = self.doc_service.create_document(
                title="Special Chars Index Test",
                content=special_content,
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            result = self.service.index_document(
                doc_id=doc.id,
                content=special_content,
                title="Special Chars Index Test"
            )

            assert result["success"], f"Indexing failed: {result.get('error')}"
            return f"Indexed with {result['chunk_count']} chunks"

        return run_test(test_func, test_case)

    def test_delete_document_index_exists(self) -> TestCase:
        """测试删除存在的索引"""
        test_case = TestCase(
            name="delete_document_index_exists",
            category="IndexService",
            description="删除已存在的文档索引",
            input_params={"doc_id": "indexed_doc_id"},
            expected_result="Index deleted successfully"
        )

        def test_func():
            # 创建并索引文档
            doc = self.doc_service.create_document(
                title="Delete Index Test",
                content="Test content for deletion",
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            self.service.index_document(
                doc_id=doc.id,
                content="Test content for deletion",
                title="Delete Index Test"
            )

            # 删除索引
            result = self.service.delete_document_index(doc.id)
            assert result is True, "Delete should return True"
            return f"Index deleted for doc {doc.id}"

        return run_test(test_func, test_case)

    def test_delete_document_index_not_exists(self) -> TestCase:
        """测试删除不存在的索引"""
        test_case = TestCase(
            name="delete_document_index_not_exists",
            category="IndexService",
            description="删除不存在的文档索引",
            input_params={"doc_id": 999999},
            expected_result="Returns True (no error)"
        )

        def test_func():
            # 删除不存在的文档索引应该不报错
            result = self.service.delete_document_index(999999)
            # 即使文档不存在，也应该返回True（没有报错）
            assert result is True, "Should return True even for non-existent doc"
            return "Correctly handled non-existent index"

        return run_test(test_func, test_case)

    def test_reindex_document(self) -> TestCase:
        """测试重新索引"""
        test_case = TestCase(
            name="reindex_document",
            category="IndexService",
            description="重新索引已索引的文档",
            input_params={"force_reindex": True},
            expected_result="Document reindexed successfully"
        )

        def test_func():
            doc = self.doc_service.create_document(
                title="Reindex Test",
                content="Original content",
                source="test",
                source_type=SourceType.upload
            )
            self.created_doc_ids.append(doc.id)

            # 首次索引
            result1 = self.service.index_document(
                doc_id=doc.id,
                content="Original content",
                title="Reindex Test"
            )

            # 重新索引（带不同内容）
            result2 = self.service.index_document(
                doc_id=doc.id,
                content="Updated content for reindexing test",
                title="Reindex Test",
                force_reindex=True
            )

            assert result2["success"], f"Reindex failed: {result2.get('error')}"
            return f"Reindexed with {result2['chunk_count']} chunks"

        return run_test(test_func, test_case)

    def test_rebuild_index(self) -> TestCase:
        """测试批量重建索引"""
        test_case = TestCase(
            name="rebuild_index",
            category="IndexService",
            description="批量重建多个文档的索引",
            input_params={"doc_ids": ["id1", "id2"]},
            expected_result="Batch rebuild completed"
        )

        def test_func():
            # 创建多个文档
            doc_ids = []
            for i in range(2):
                doc = self.doc_service.create_document(
                    title=f"Rebuild Test {i}",
                    content=f"Content for rebuild test {i}",
                    source="test",
                    source_type=SourceType.upload
                )
                self.created_doc_ids.append(doc.id)
                doc_ids.append(doc.id)

                # 索引文档
                self.service.index_document(
                    doc_id=doc.id,
                    content=f"Content for rebuild test {i}",
                    title=f"Rebuild Test {i}"
                )

            # 重建索引
            result = self.service.rebuild_index(doc_ids=doc_ids)
            assert result["success"], "Rebuild should succeed"
            return f"Rebuilt {result['success_count']}/{result['total']} documents"

        return run_test(test_func, test_case)

    def test_get_index_stats(self) -> TestCase:
        """测试获取索引统计"""
        test_case = TestCase(
            name="get_index_stats",
            category="IndexService",
            description="获取索引统计信息",
            input_params={},
            expected_result="Stats returned successfully"
        )

        def test_func():
            stats = self.service.get_index_stats()
            assert "error" not in stats or stats.get("error") is None, \
                f"Stats error: {stats.get('error')}"
            return f"Vector: {stats.get('vector_index_size', 0)}, BM25: {stats.get('bm25_index_size', 0)}"

        return run_test(test_func, test_case)

    def run_all(self) -> List[TestCase]:
        """运行所有测试"""
        tests = [
            self.test_index_document_normal,
            self.test_index_document_empty,
            self.test_index_document_large,
            self.test_index_document_special_chars,
            self.test_delete_document_index_exists,
            self.test_delete_document_index_not_exists,
            self.test_reindex_document,
            self.test_rebuild_index,
            self.test_get_index_stats,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                print(f"  [{result.status.upper()}] {result.name}")
            except Exception as e:
                print(f"  [ERROR] {test.__name__}: {e}")
                traceback.print_exc()

        self.cleanup()
        return results


# ================== CrawlService 测试 ==================

class CrawlServiceTests:
    """CrawlService 测试类"""

    def __init__(self):
        # 不使用代理，不使用增强爬虫（避免浏览器依赖）
        self.service = CrawlService(use_proxy=False, use_enhanced=False)

    def cleanup(self):
        """清理测试数据"""
        pass

    def test_parse_html_links_basic(self) -> TestCase:
        """测试基本HTML链接解析"""
        test_case = TestCase(
            name="parse_html_links_basic",
            category="CrawlService",
            description="解析HTML中的基本链接",
            input_params={"html": "<a href='...'>...</a>"},
            expected_result="Links parsed correctly"
        )

        def test_func():
            html = """
            <html>
            <body>
                <a href="https://example.com/page1">Page 1</a>
                <a href="https://example.com/page2">Page 2</a>
                <a href="/relative">Relative</a>
            </body>
            </html>
            """
            links = self.service.parse_html_links(html, base_url="https://example.com")
            assert len(links) >= 2, "Should find at least 2 links"
            assert any(l["url"] == "https://example.com/page1" for l in links), \
                "Should find page1 link"
            return f"Found {len(links)} links"

        return run_test(test_func, test_case)

    def test_parse_html_links_same_domain(self) -> TestCase:
        """测试同域名链接过滤"""
        test_case = TestCase(
            name="parse_html_links_same_domain",
            category="CrawlService",
            description="只返回同域名的链接",
            input_params={"same_domain_only": True},
            expected_result="Only same domain links returned"
        )

        def test_func():
            html = """
            <html>
            <body>
                <a href="https://example.com/page1">Same Domain</a>
                <a href="https://other.com/page2">Other Domain</a>
            </body>
            </html>
            """
            links = self.service.parse_html_links(
                html,
                base_url="https://example.com",
                same_domain_only=True
            )
            assert all("example.com" in l["url"] for l in links), \
                "All links should be same domain"
            return f"Found {len(links)} same-domain links"

        return run_test(test_func, test_case)

    def test_parse_html_links_skip_anchors(self) -> TestCase:
        """测试跳过锚点链接"""
        test_case = TestCase(
            name="parse_html_links_skip_anchors",
            category="CrawlService",
            description="跳过锚点和JavaScript链接",
            input_params={"html": "<a href='#'>...</a>"},
            expected_result="Anchors and JS links skipped"
        )

        def test_func():
            html = """
            <html>
            <body>
                <a href="https://example.com/valid">Valid</a>
                <a href="#">Anchor</a>
                <a href="javascript:void(0)">JS</a>
                <a href="mailto:test@test.com">Email</a>
            </body>
            </html>
            """
            links = self.service.parse_html_links(html)
            assert len(links) == 1, "Should only find 1 valid link"
            assert links[0]["url"] == "https://example.com/valid", \
                "Should only keep valid link"
            return f"Correctly filtered to {len(links)} valid link"

        return run_test(test_func, test_case)

    def test_filter_uncrawled_urls(self) -> TestCase:
        """测试过滤未爬取URL"""
        test_case = TestCase(
            name="filter_uncrawled_urls",
            category="CrawlService",
            description="过滤出未爬取的URL",
            input_params={"urls": ["url1", "url2", "url3"]},
            expected_result="Only uncrawled URLs returned"
        )

        def test_func():
            # 测试空列表
            result = self.service.filter_uncrawled_urls([])
            assert result == [], "Empty input should return empty list"

            # 测试正常列表
            urls = [
                "https://test.com/unique1",
                "https://test.com/unique2"
            ]
            result = self.service.filter_uncrawled_urls(urls)
            # 这些URL应该都未爬取过
            return f"Filtered to {len(result)} uncrawled URLs"

        return run_test(test_func, test_case)

    def test_check_url_crawled(self) -> TestCase:
        """测试检查URL是否已爬取"""
        test_case = TestCase(
            name="check_url_crawled",
            category="CrawlService",
            description="检查URL是否已被爬取",
            input_params={"url": "https://test.com/page"},
            expected_result="Returns boolean correctly"
        )

        def test_func():
            # 检查一个不存在的URL
            result = self.service.check_url_crawled("https://nonexistent.example.com/page")
            assert result is False, "Should return False for uncrawled URL"
            return "Correctly returned False for uncrawled URL"

        return run_test(test_func, test_case)

    def test_get_crawl_stats(self) -> TestCase:
        """测试获取爬取统计"""
        test_case = TestCase(
            name="get_crawl_stats",
            category="CrawlService",
            description="获取爬取统计信息",
            input_params={},
            expected_result="Stats returned with all fields"
        )

        def test_func():
            stats = self.service.get_crawl_stats()
            assert "total" in stats, "Should have total field"
            assert "success" in stats, "Should have success field"
            assert "failed" in stats, "Should have failed field"
            return f"Stats: total={stats['total']}, success={stats['success']}, failed={stats['failed']}"

        return run_test(test_func, test_case)

    def test_get_crawl_records_pagination(self) -> TestCase:
        """测试爬取记录分页"""
        test_case = TestCase(
            name="get_crawl_records_pagination",
            category="CrawlService",
            description="测试爬取记录分页查询",
            input_params={"offset": 0, "limit": 10},
            expected_result="Pagination works correctly"
        )

        def test_func():
            records, total = self.service.get_crawl_records(offset=0, limit=10)
            assert isinstance(records, list), "Should return a list"
            assert isinstance(total, int), "Total should be an integer"
            return f"Found {len(records)} records, total: {total}"

        return run_test(test_func, test_case)

    def run_all(self) -> List[TestCase]:
        """运行所有测试"""
        tests = [
            self.test_parse_html_links_basic,
            self.test_parse_html_links_same_domain,
            self.test_parse_html_links_skip_anchors,
            self.test_filter_uncrawled_urls,
            self.test_check_url_crawled,
            self.test_get_crawl_stats,
            self.test_get_crawl_records_pagination,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                print(f"  [{result.status.upper()}] {result.name}")
            except Exception as e:
                print(f"  [ERROR] {test.__name__}: {e}")

        self.cleanup()
        return results


# ================== 测试报告生成 ==================

def generate_markdown_report(report: TestReport) -> str:
    """生成Markdown格式的测试报告"""
    lines = []
    lines.append("# 服务层测试报告")
    lines.append("")
    lines.append(f"**测试时间**: {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试耗时**: {report.duration_seconds:.2f} 秒")
    lines.append("")
    lines.append("## 测试概要")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 总测试数 | {report.total_tests} |")
    lines.append(f"| 通过 | {report.passed_tests} |")
    lines.append(f"| 失败 | {report.failed_tests} |")
    lines.append(f"| 错误 | {report.error_tests} |")
    lines.append(f"| 通过率 | {report.pass_rate:.1f}% |")
    lines.append("")

    # 按类别分组
    categories = {}
    for tc in report.test_cases:
        if tc.category not in categories:
            categories[tc.category] = []
        categories[tc.category].append(tc)

    for category, tests in categories.items():
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| 测试用例 | 状态 | 耗时(ms) | 描述 |")
        lines.append("|----------|------|----------|------|")

        for tc in tests:
            status_icon = {
                "passed": "PASS",
                "failed": "FAIL",
                "error": "ERROR",
                "pending": "PENDING"
            }.get(tc.status, "?")

            lines.append(
                f"| {tc.name} | {status_icon} | {tc.execution_time_ms:.1f} | {tc.description} |"
            )

        lines.append("")

        # 失败和错误的详细信息
        failed_tests = [tc for tc in tests if tc.status in ("failed", "error")]
        if failed_tests:
            lines.append("### 失败/错误详情")
            lines.append("")
            for tc in failed_tests:
                lines.append(f"#### {tc.name}")
                lines.append("")
                lines.append(f"**输入参数**: `{json.dumps(tc.input_params, ensure_ascii=False)}`")
                lines.append("")
                lines.append(f"**预期结果**: {tc.expected_result}")
                lines.append("")
                lines.append(f"**实际结果**: {tc.actual_result}")
                lines.append("")
                if tc.error_message:
                    lines.append("**错误信息**:")
                    lines.append("```")
                    lines.append(tc.error_message)
                    lines.append("```")
                    lines.append("")

    return "\n".join(lines)


# ================== 主函数 ==================

def main():
    """主测试函数"""
    print("=" * 60)
    print("服务层全面测试")
    print("=" * 60)
    print()

    # 初始化数据库
    init_db()

    report = TestReport()

    # 运行各服务测试
    test_classes = [
        ("DocumentService", DocumentServiceTests),
        ("TaskService", TaskServiceTests),
        ("IndexService", IndexServiceTests),
        ("CrawlService", CrawlServiceTests),
    ]

    for name, test_class in test_classes:
        print(f"\n>>> 测试 {name}")
        print("-" * 40)
        try:
            tester = test_class()
            results = tester.run_all()
            for result in results:
                report.add_test(result)
        except Exception as e:
            print(f"[ERROR] {name} 测试初始化失败: {e}")
            traceback.print_exc()

    report.finish()

    # 打印总结
    print()
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"总测试数: {report.total_tests}")
    print(f"通过: {report.passed_tests}")
    print(f"失败: {report.failed_tests}")
    print(f"错误: {report.error_tests}")
    print(f"通过率: {report.pass_rate:.1f}%")
    print(f"总耗时: {report.duration_seconds:.2f}秒")

    # 生成报告
    markdown_report = generate_markdown_report(report)
    report_path = Path(__file__).parent.parent / "docs" / "service_layer_test_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    print(f"\n测试报告已保存至: {report_path}")

    # 返回退出码
    return 0 if report.failed_tests == 0 and report.error_tests == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
