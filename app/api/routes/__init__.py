"""
API路由模块
"""
from fastapi import APIRouter

from app.api.routes.tasks import router as tasks_router
from app.api.routes.documents import router as documents_router
from app.api.routes.crawl import router as crawl_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.journal import router as journal_router
from app.api.routes.index_catalog import router as index_catalog_router
from app.api.routes.playbook import router as playbook_router
from app.api.routes.code_kb import router as code_kb_router
from app.api.routes.public_code import router as public_code_router
from app.api.routes.admin_console import router as admin_console_router


def register_routes(app):
    """
    注册所有API路由到FastAPI应用

    Args:
        app: FastAPI应用实例
    """
    app.include_router(tasks_router, prefix="/api/tasks", tags=["任务管理"])
    app.include_router(documents_router, prefix="/api/documents", tags=["文档管理"])
    app.include_router(crawl_router, prefix="/api/crawl", tags=["爬虫"])
    app.include_router(journal_router, prefix="/api/journal", tags=["ACE Journal"])
    app.include_router(index_catalog_router, prefix="/api/index", tags=["Index Catalog"])
    app.include_router(playbook_router, prefix="/api/playbook", tags=["ACE Playbook"])
    app.include_router(code_kb_router, prefix="/api/code", tags=["Code KB"])
    app.include_router(public_code_router, prefix="/api/public-code", tags=["Public Code KB"])
    app.include_router(admin_console_router, prefix="/api/admin/console", tags=["Admin Console"])
    app.include_router(dashboard_router)

__all__ = [
    "tasks_router",
    "documents_router",
    "crawl_router",
    "journal_router",
    "index_catalog_router",
    "playbook_router",
    "code_kb_router",
    "public_code_router",
    "admin_console_router",
    "dashboard_router",
    "register_routes",
]
