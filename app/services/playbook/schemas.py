"""Pydantic contracts for the server-side ACE playbook."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PlaybookRecordRequest(BaseModel):
    project_slug: str = Field(..., min_length=1, max_length=2000,
                              description="项目命名空间 = 索引目录绝对路径（MCP 客户端强制）；"
                                          "同一 (用户, slug) 下 bullet 互相可检索。够长以容纳绝对路径。")
    content: str = Field(..., min_length=1, description="要沉淀的 lesson 正文（一句可复用的经验）。")
    section: str = Field(default="insights", min_length=1, max_length=50,
                         description="patterns|best_practices|insights|failure_mode。")
    domain: str | None = Field(default=None, max_length=200, description="可选领域标签。")
    # owner is resolved server-side from the gateway header / LAN identity
    # (see app/api/identity.resolve_owner) — never trusted from the request body.

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "project_slug": "C:/Users/Arc/Prd/retrival",
                "content": "Qdrant upsert 必须 wait=True，wait=False 在写承压时静默丢 batch。",
                "section": "failure_mode",
                "domain": "retrieval",
            }]
        }
    )


class PlaybookSearchRequest(BaseModel):
    project_slug: str = Field(..., min_length=1, max_length=2000)
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    domain: str | None = Field(default=None, max_length=200)


class PlaybookMarkRequest(BaseModel):
    helpful: bool = Field(default=True, description="True 记正反馈，False 记负反馈。")


class PlaybookBulletResponse(BaseModel):
    id: int
    bullet_key: str
    owner_user_id: str
    project_slug: str
    content: str
    section: str
    domain: str | None = None
    helpful: int = 0
    harmful: int = 0
    score: float | None = None   # cosine similarity; set only on search results
    deduped: bool = False        # True when record() merged into an existing bullet
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PlaybookListResponse(BaseModel):
    bullets: list[PlaybookBulletResponse]
    total: int


class PlaybookDeleteResponse(BaseModel):
    deleted: bool
    bullet_id: int
