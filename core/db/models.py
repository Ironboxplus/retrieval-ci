"""
数据库模型定义
"""
from sqlalchemy import (
    Boolean, Column, Integer, String, Text, DateTime, ForeignKey, Enum, JSON,
    Float, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from core.db.database import Base


class SourceType(enum.Enum):
    """文档来源类型"""
    webpage = "webpage"
    paper = "paper"
    upload = "upload"
    review = "review"  # OpenReview 审稿意见（text-only，按 paper_id 关联到论文）


class DataSource(enum.Enum):
    """数据来源"""
    system = "system"
    user = "user"


class DocumentStatus(enum.Enum):
    """文档状态"""
    pending = "pending"
    processing = "processing"
    indexed = "indexed"
    failed = "failed"


class TaskStatus(enum.Enum):
    """任务状态"""
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    timeout = "timeout"  # 新增：超时状态


class CrawlStatus(enum.Enum):
    """爬取状态"""
    pending = "pending"
    success = "success"
    failed = "failed"
    skipped = "skipped"


class TaskSubtaskStatus(enum.Enum):
    """任务子任务状态"""
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    skipped = "skipped"


class TaskWorkflowEdgeStatus(enum.Enum):
    """工作流边状态"""
    active = "active"
    traversed = "traversed"
    blocked = "blocked"
    disabled = "disabled"


class Document(Base):
    """文档表"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=True, comment="文档标题")
    source = Column(String(500), nullable=True, comment="来源描述")
    source_type = Column(
        Enum(SourceType),
        default=SourceType.upload,
        comment="来源类型: webpage|paper|upload"
    )
    data_source = Column(
        Enum(DataSource),
        default=DataSource.user,
        comment="数据来源: system|user"
    )
    content_hash = Column(String(64), nullable=True, index=True, comment="内容哈希，用于去重")
    file_path = Column(String(1000), nullable=True, comment="本地文件路径")
    url = Column(String(2000), nullable=True, comment="网页或论文URL")
    status = Column(
        Enum(DocumentStatus),
        default=DocumentStatus.pending,
        comment="文档状态"
    )
    metadata_json = Column(JSON, nullable=True, comment="额外元数据")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )

    # 关系
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    paper = relationship("Paper", back_populates="document", uselist=False)

    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}', source_type={self.source_type})>"


class Chunk(Base):
    """分块表"""
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属文档ID"
    )
    content = Column(Text, nullable=False, comment="分块内容")
    level = Column(Integer, default=0, comment="分块层级(多粒度分块)")
    chunk_index = Column(Integer, default=0, comment="在文档中的序号")
    start_idx = Column(Integer, nullable=True, comment="在原文中的起始位置")
    end_idx = Column(Integer, nullable=True, comment="在原文中的结束位置")
    token_count = Column(Integer, nullable=True, comment="Token数量")
    metadata_json = Column(JSON, nullable=True, comment="额外元数据")

    # 关系
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk(id={self.id}, doc_id={self.doc_id}, level={self.level}, index={self.chunk_index})>"


class JournalProject(Base):
    """ACE journal project namespace.

    Journal entries are append-only work memory. They are deliberately separate
    from Document/Chunk/Paper so paper ingest and user/work logs cannot pollute
    each other's storage contracts.
    """
    __tablename__ = "journal_projects"
    __table_args__ = (
        UniqueConstraint("owner_user_id", "slug", name="uq_journal_projects_owner_slug"),
    )

    project_id = Column(Integer, primary_key=True, autoincrement=True)
    slug = Column(String(200), nullable=False, unique=True, index=True, comment="人类可读稳定项目名")
    display_name = Column(String(500), nullable=True, comment="展示名称")
    workspace_hint = Column(String(1000), nullable=True, comment="客户端工作区提示")
    owner_user_id = Column(String(200), nullable=True, index=True, comment="公网用户标识")
    metadata_json = Column(JSON, nullable=True, comment="项目级扩展元数据")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )

    entries = relationship("JournalEntry", back_populates="project", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<JournalProject(project_id={self.project_id}, slug='{self.slug}')>"


# Historical draft code marked `slug` as globally unique. Journal projects are
# private user namespaces, so only the `(owner_user_id, slug)` pair is unique.
for _constraint in list(JournalProject.__table__.constraints):
    if (
        isinstance(_constraint, UniqueConstraint)
        and [column.name for column in _constraint.columns] == ["slug"]
    ):
        JournalProject.__table__.constraints.remove(_constraint)
JournalProject.__table__.c.slug.unique = False
for _index in list(JournalProject.__table__.indexes):
    if _index.name == "ix_journal_projects_slug" and _index.unique:
        JournalProject.__table__.indexes.remove(_index)


class JournalEntry(Base):
    """Append-only ACE journal entry."""
    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_id = Column(String(100), nullable=False, unique=True, index=True, comment="稳定 entry id")
    project_id = Column(
        Integer,
        ForeignKey("journal_projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属 journal project"
    )
    client = Column(String(100), nullable=False, index=True, comment="codex|claude|frontend|http|mcp")
    source_kind = Column(String(100), nullable=False, index=True, comment="memory|handoff|session_summary|manual_note")
    title = Column(String(500), nullable=True, comment="短标题")
    summary = Column(Text, nullable=True, comment="摘要")
    body = Column(Text, nullable=False, comment="原始 journal 正文")
    tags_json = Column(JSON, nullable=True, comment="标签")
    metadata_json = Column(JSON, nullable=True, comment="扩展元数据")
    content_hash = Column(String(64), nullable=False, index=True, comment="正文 SHA256")
    indexed_at = Column(DateTime, nullable=True, comment="向量索引完成时间")
    index_error = Column(Text, nullable=True, comment="最近一次索引错误")
    deleted_at = Column(DateTime, nullable=True, index=True, comment="软删除时间")
    deleted_reason = Column(String(500), nullable=True, comment="软删除原因")
    created_at = Column(DateTime, server_default=func.now(), index=True, comment="创建时间")
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )

    project = relationship("JournalProject", back_populates="entries")
    chunks = relationship(
        "JournalChunk",
        back_populates="entry",
        cascade="all, delete-orphan",
        order_by="JournalChunk.chunk_index",
    )

    def __repr__(self):
        return f"<JournalEntry(entry_id='{self.entry_id}', project_id={self.project_id})>"


class JournalChunk(Base):
    """Chunk table for ACE journal entries."""
    __tablename__ = "journal_chunks"
    __table_args__ = (
        UniqueConstraint("entry_id", "chunk_index", name="uq_journal_chunks_entry_index"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(100), nullable=False, unique=True, index=True, comment="稳定 chunk id")
    entry_id = Column(
        String(100),
        ForeignKey("journal_entries.entry_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属 journal entry"
    )
    project_id = Column(Integer, nullable=False, index=True, comment="冗余 project_id，便于过滤")
    chunk_index = Column(Integer, nullable=False, comment="entry 内序号")
    level = Column(Integer, nullable=False, default=2, index=True, comment="1=summary,2=body,3=detail")
    heading = Column(String(500), nullable=True, comment="Markdown heading path")
    content = Column(Text, nullable=False, comment="分块正文")
    token_count = Column(Integer, nullable=True, comment="粗略 token/word 数")
    metadata_json = Column(JSON, nullable=True, comment="扩展元数据")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")

    entry = relationship("JournalEntry", back_populates="chunks")

    def __repr__(self):
        return f"<JournalChunk(chunk_id='{self.chunk_id}', entry_id='{self.entry_id}', level={self.level})>"


class IndexCatalog(Base):
    """Structured registry of *what a user indexed, where*.

    One row per (owner_user_id, host, project, rel_path, kind) — the minimal shape
    that keeps directory/project separated per the requirement. It is a client-reported
    inventory (the machine where the code lives walks + reports), not the vectors
    themselves (those live in Qdrant). Powers the admin console's
    user → host → project → directory → file drill-down.
    """
    __tablename__ = "index_catalog"
    __table_args__ = (
        UniqueConstraint(
            "owner_user_id", "host", "project", "rel_path", "kind",
            name="uq_index_catalog_file",
        ),
        Index("ix_index_catalog_scope", "owner_user_id", "host", "project"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_user_id = Column(String(200), nullable=False, index=True, comment="所属用户（gateway id 或 internal-admin）")
    host = Column(String(200), nullable=False, index=True, comment="索引所在机器 hostname")
    project = Column(String(200), nullable=False, index=True, comment="项目/仓库 slug")
    root_dir = Column(String(1000), nullable=True, comment="该机器上的项目根绝对路径")
    rel_path = Column(String(1000), nullable=False, comment="相对 root_dir 的文件路径")
    kind = Column(String(50), nullable=False, default="code", index=True, comment="code|journal|doc")
    collection = Column(String(100), nullable=True, comment="向量所在 Qdrant collection")
    language = Column(String(50), nullable=True, comment="编程语言（可选）")
    chunk_count = Column(Integer, nullable=False, default=0, comment="该文件切出的块数")
    content_hash = Column(String(64), nullable=True, comment="文件内容 SHA256（变更检测）")
    size_bytes = Column(Integer, nullable=True, comment="文件字节数")
    metadata_json = Column(JSON, nullable=True, comment="扩展元数据")
    indexed_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="最近索引时间")
    deleted_at = Column(DateTime, nullable=True, index=True, comment="软删除时间")
    created_at = Column(DateTime, server_default=func.now(), comment="首次登记时间")

    def __repr__(self):
        return (
            f"<IndexCatalog(owner='{self.owner_user_id}', host='{self.host}', "
            f"project='{self.project}', rel_path='{self.rel_path}')>"
        )


class IndexScopePolicy(Base):
    """Persistent local-scan policy for one indexed scope."""
    __tablename__ = "index_scope_policies"
    __table_args__ = (
        UniqueConstraint(
            "owner_user_id", "host", "project", "kind",
            name="uq_index_scope_policy",
        ),
        Index("ix_index_scope_policy_scope", "owner_user_id", "host", "project"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_user_id = Column(String(200), nullable=False, index=True)
    host = Column(String(200), nullable=False, index=True)
    project = Column(String(2000), nullable=False, index=True)
    kind = Column(String(50), nullable=False, default="code", index=True)
    root_dir = Column(String(2000), nullable=False)
    auto_refresh = Column(Integer, nullable=False, default=1)
    respect_gitignore = Column(Integer, nullable=False, default=1)
    exclude_globs_json = Column(JSON, nullable=True)
    min_check_interval_s = Column(Integer, nullable=False, default=10)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class CodeScopeVisibility(Base):
    """Durable visibility fence for in-place Code KB mutations.

    Code vectors do not yet have an immutable active-revision pointer.  This small
    control-plane row is therefore written *before* the first Qdrant mutation and is
    cleared to ``completed`` only after vectors and catalog publishing both succeed.
    It is scoped by owner/project because host is display metadata for Code KB, not a
    vector namespace key; that also covers cross-scope fan-out targets.
    """
    __tablename__ = "code_scope_visibility"
    __table_args__ = (
        UniqueConstraint("owner_user_id", "project", name="uq_code_scope_visibility"),
        Index("ix_code_scope_visibility_owner_state", "owner_user_id", "state"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_user_id = Column(String(200), nullable=False, index=True)
    project = Column(String(2000), nullable=False, index=True)
    host = Column(String(200), nullable=False, default="")
    # preparing = marker persisted before mutation / interrupted worker; recovery_required
    # = a known failed mutation; completed = last complete durable snapshot.
    state = Column(String(32), nullable=False, default="completed", index=True)
    job_id = Column(String(64), nullable=True, index=True)
    previous_completed_snapshot_available = Column(Boolean, nullable=False, default=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class PlaybookBullet(Base):
    """Server-side ACE playbook bullet — a reusable lesson (strategy / convention /
    failure-mode / insight) curated per (owner_user_id, project_slug).

    Mirrors the client-side ``mcp_server/playbook.py`` store but centralized so the
    admin console can view/manage it and multiple clients share one playbook per
    (user, project). Embedding stored inline (JSON dense vector) for brute-force
    cosine — a playbook lives at the hundreds-of-bullets scale, per scope.
    """
    __tablename__ = "playbook_bullets"
    __table_args__ = (
        UniqueConstraint(
            "owner_user_id", "project_slug", "bullet_key",
            name="uq_playbook_bullet",
        ),
        Index("ix_playbook_scope", "owner_user_id", "project_slug"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_user_id = Column(String(200), nullable=False, index=True, comment="所属用户（gateway id 或 internal-admin）")
    project_slug = Column(String(200), nullable=False, index=True, comment="项目命名空间")
    bullet_key = Column(String(32), nullable=False, comment="内容 SHA256 前 16（scope 内幂等键）")
    section = Column(String(50), nullable=False, default="insights", comment="patterns|best_practices|insights|failure_mode")
    content = Column(Text, nullable=False, comment="lesson 正文")
    domain = Column(String(200), nullable=True, index=True, comment="可选领域标签")
    helpful = Column(Integer, nullable=False, default=0, comment="正反馈计数（含 grow-and-refine 合并）")
    harmful = Column(Integer, nullable=False, default=0, comment="负反馈计数")
    embedding_json = Column(JSON, nullable=True, comment="dense 向量（brute-force cosine）")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="最近更新时间")
    deleted_at = Column(DateTime, nullable=True, index=True, comment="软删除时间")

    def __repr__(self):
        return (
            f"<PlaybookBullet(owner='{self.owner_user_id}', project='{self.project_slug}', "
            f"section='{self.section}', key='{self.bullet_key}')>"
        )


class Task(Base):
    """任务表 - 增强版"""
    __tablename__ = "tasks"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(String(50), nullable=False, index=True, comment="任务类型")
    status = Column(
        Enum(TaskStatus),
        default=TaskStatus.pending,
        index=True,
        comment="任务状态"
    )
    priority = Column(Integer, default=0, comment="优先级，数字越大优先级越高")
    params_json = Column(JSON, nullable=True, comment="任务参数")
    result_json = Column(JSON, nullable=True, comment="任务结果")
    progress = Column(Integer, default=0, comment="当前进度")
    total = Column(Integer, default=0, comment="总数")
    error_message = Column(Text, nullable=True, comment="错误信息")
    retry_count = Column(Integer, default=0, comment="已重试次数")
    max_retries = Column(Integer, default=3, comment="最大重试次数")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    started_at = Column(DateTime, nullable=True, comment="开始时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")
    # 新增字段
    worker_id = Column(String(100), nullable=True, comment="处理Worker的ID")
    last_heartbeat = Column(DateTime, nullable=True, comment="最后心跳时间")
    timeout_seconds = Column(Integer, default=3600, comment="任务超时时间(秒)")
    # 提交者。公网多用户下 task_id 是连续整数，没有它任何人都能轮询/取消别人的任务。
    # 内网单用户任务留空；只有按用户隔离的路由（如 match_paper）才写入。
    owner_user_id = Column(String(255), nullable=True, index=True, comment="任务提交者")

    # 关系
    crawl_records = relationship("CrawlRecord", back_populates="task", cascade="all, delete-orphan")
    events = relationship("TaskEvent", back_populates="task", cascade="all, delete-orphan")
    subtasks = relationship("TaskSubtask", back_populates="task", cascade="all, delete-orphan")
    resource_samples = relationship("TaskResourceSample", back_populates="task", cascade="all, delete-orphan")
    workflow_edges = relationship("TaskWorkflowEdge", back_populates="task", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Task(id={self.id}, type='{self.task_type}', status={self.status})>"


class PublicCodeSourceRecord(Base):
    """Persistent state for one curated public-code source repository."""
    __tablename__ = "public_code_sources"

    source_id = Column(String(100), primary_key=True)
    domain = Column(String(100), nullable=False, index=True)
    relative_path = Column(String(500), nullable=False, unique=True)
    upstream_url = Column(String(2000), nullable=False)
    default_ref = Column(String(255), nullable=False, default="HEAD")
    proxy_profile = Column(String(100), nullable=True)
    credential_profile = Column(String(100), nullable=True)
    language_overrides = Column(JSON, nullable=False, default=dict)
    source_origin = Column(String(32), nullable=False, default="bootstrap")
    created_by_user_id = Column(String(255), nullable=True)
    update_policy = Column(String(32), nullable=False, default="manual")
    update_interval_minutes = Column(Integer, nullable=True)
    active_revision = Column(String(128), nullable=True, index=True)
    last_mirrored_revision = Column(String(128), nullable=True)
    last_remote_revision = Column(String(128), nullable=True)
    last_check_error = Column(Text, nullable=True)
    last_checked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


class PublicCodeRevision(Base):
    """One staged, active, failed, or superseded source revision."""
    __tablename__ = "public_code_revisions"
    __table_args__ = (
        UniqueConstraint("source_id", "revision", name="uq_public_code_source_revision"),
        Index("ix_public_code_revision_source_state", "source_id", "state"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(
        String(100),
        ForeignKey("public_code_sources.source_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    revision = Column(String(128), nullable=False)
    state = Column(String(32), nullable=False, default="candidate", index=True)
    manifest_file_count = Column(Integer, nullable=False, default=0)
    chunk_count = Column(Integer, nullable=False, default=0)
    vector_count = Column(Integer, nullable=False, default=0)
    cleanup_pending = Column(Boolean, nullable=False, default=False, index=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    promoted_at = Column(DateTime, nullable=True)


class PublicCodeJobRecord(Base):
    """Durable snapshot for a public-code control-plane background job."""
    __tablename__ = "public_code_jobs"

    job_id = Column(String(32), primary_key=True)
    operation = Column(String(32), nullable=False, index=True)
    source_ids = Column(JSON, nullable=False, default=list)
    status = Column(String(32), nullable=False, index=True)
    snapshot = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


class CrawlRecord(Base):
    """爬取记录表"""
    __tablename__ = "crawl_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="所属任务ID"
    )
    url = Column(String(2000), nullable=False, index=True, comment="爬取的URL")
    status = Column(
        Enum(CrawlStatus),
        default=CrawlStatus.pending,
        comment="爬取状态"
    )
    title = Column(String(500), nullable=True, comment="页面标题")
    content_length = Column(Integer, nullable=True, comment="内容长度")
    doc_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        comment="生成的文档ID"
    )
    error_message = Column(Text, nullable=True, comment="错误信息")
    # W41 (2026-05-20): EnhancedCrawler captures these but the records UI was
    # blind to retry count / HTTP status / proxy used. Surface them via the
    # CrawlRecord table + API response.
    status_code = Column(Integer, nullable=True, comment="HTTP 状态码")
    retry_count = Column(Integer, nullable=True, comment="重试次数")
    crawl_time = Column(Float, nullable=True, comment="爬取耗时(秒)")
    proxy_used = Column(String(500), nullable=True, comment="实际使用的代理 URL")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")

    # 关系
    task = relationship("Task", back_populates="crawl_records")

    def __repr__(self):
        return f"<CrawlRecord(id={self.id}, url='{self.url[:50]}...', status={self.status})>"


class TaskSubtask(Base):
    """任务子任务表，用于更细粒度地跟踪批量/分阶段任务。"""
    __tablename__ = "task_subtasks"
    __table_args__ = (
        UniqueConstraint("task_id", "subtask_key", name="uq_task_subtasks_task_key"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属任务ID"
    )
    subtask_key = Column(String(500), nullable=False, comment="任务内唯一的子任务键")
    subtask_type = Column(String(100), nullable=False, comment="子任务类型")
    name = Column(String(500), nullable=True, comment="展示名称")
    status = Column(
        Enum(TaskSubtaskStatus),
        default=TaskSubtaskStatus.pending,
        index=True,
        comment="子任务状态"
    )
    progress = Column(Integer, default=0, comment="子任务进度")
    total = Column(Integer, default=0, comment="子任务总量")
    retry_count = Column(Integer, default=0, comment="手动重试次数")
    max_retries = Column(Integer, default=3, comment="最大手动重试次数")
    message = Column(Text, nullable=True, comment="最新状态消息")
    stage = Column(String(100), nullable=True, comment="当前阶段")
    result_json = Column(JSON, nullable=True, comment="子任务结果")
    error_message = Column(Text, nullable=True, comment="错误信息")
    metadata_json = Column(JSON, nullable=True, comment="额外元数据")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )
    started_at = Column(DateTime, nullable=True, comment="开始时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")

    task = relationship("Task", back_populates="subtasks")
    events = relationship("TaskEvent", back_populates="subtask", cascade="all, delete-orphan")
    resource_samples = relationship("TaskResourceSample", back_populates="subtask", cascade="all, delete-orphan")

    def __repr__(self):
        return (
            f"<TaskSubtask(id={self.id}, task_id={self.task_id}, "
            f"key='{self.subtask_key}', status={self.status})>"
        )


class TaskEvent(Base):
    """任务事件表，用于提供流式过程和详细审计。"""
    __tablename__ = "task_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属任务ID"
    )
    subtask_id = Column(
        Integer,
        ForeignKey("task_subtasks.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="关联的子任务ID"
    )
    event_type = Column(String(100), nullable=False, index=True, comment="事件类型")
    level = Column(String(20), default="info", comment="日志级别")
    stage = Column(String(100), nullable=True, comment="阶段")
    message = Column(Text, nullable=True, comment="事件消息")
    payload_json = Column(JSON, nullable=True, comment="事件负载")
    created_at = Column(DateTime, server_default=func.now(), index=True, comment="创建时间")

    task = relationship("Task", back_populates="events")
    subtask = relationship("TaskSubtask", back_populates="events")

    def __repr__(self):
        return f"<TaskEvent(id={self.id}, task_id={self.task_id}, type='{self.event_type}')>"


class TaskResourceSample(Base):
    """任务资源采样表，记录任务运行期的进程级资源使用。"""
    __tablename__ = "task_resource_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属任务ID"
    )
    subtask_id = Column(
        Integer,
        ForeignKey("task_subtasks.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="关联的子任务ID"
    )
    phase = Column(String(100), nullable=True, comment="资源采样阶段")
    pid = Column(Integer, nullable=True, comment="进程ID")
    cpu_percent = Column(Float, nullable=True, comment="CPU百分比")
    rss_bytes = Column(Integer, nullable=True, comment="RSS内存字节数")
    vms_bytes = Column(Integer, nullable=True, comment="VMS内存字节数")
    thread_count = Column(Integer, nullable=True, comment="线程数")
    open_file_count = Column(Integer, nullable=True, comment="打开文件数")
    extra_json = Column(JSON, nullable=True, comment="扩展资源信息")
    created_at = Column(DateTime, server_default=func.now(), index=True, comment="采样时间")

    task = relationship("Task", back_populates="resource_samples")
    subtask = relationship("TaskSubtask", back_populates="resource_samples")

    def __repr__(self):
        return f"<TaskResourceSample(id={self.id}, task_id={self.task_id}, phase='{self.phase}')>"


class TaskWorkflowEdge(Base):
    """任务工作流边，用于表达节点之间的依赖和可视化状态。"""
    __tablename__ = "task_workflow_edges"
    __table_args__ = (
        UniqueConstraint("task_id", "edge_key", name="uq_task_workflow_edges_task_key"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属任务ID"
    )
    edge_key = Column(String(500), nullable=False, comment="任务内唯一边键")
    from_key = Column(String(500), nullable=False, comment="起始节点键")
    to_key = Column(String(500), nullable=False, comment="目标节点键")
    status = Column(
        Enum(TaskWorkflowEdgeStatus),
        default=TaskWorkflowEdgeStatus.active,
        index=True,
        comment="边状态"
    )
    label = Column(String(200), nullable=True, comment="边标签")
    metadata_json = Column(JSON, nullable=True, comment="边元数据")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )

    task = relationship("Task", back_populates="workflow_edges")

    def __repr__(self):
        return (
            f"<TaskWorkflowEdge(id={self.id}, task_id={self.task_id}, "
            f"from='{self.from_key}', to='{self.to_key}', status={self.status})>"
        )


class Paper(Base):
    """论文表

    Schema 2026-05-17 砸了重来后（方案 X）：
      - paper_id (string) 直接做 PRIMARY KEY，全栈业务唯一 key
      - doc_id (int) NOT NULL + UNIQUE FK → documents.id，1:1 inheritance
      - 删除了原 `id` 自增列（业务零引用，dead weight）
    """
    __tablename__ = "papers"

    # Business paper_id (e.g. "aaai-2022-20281"). PRIMARY KEY — single source of
    # truth across SQL / Qdrant payload / FastAPI routes (/papers/:paper_id) /
    # frontend URLs / MCP. No more rsplit() hacks.
    paper_id = Column(String(100), primary_key=True, comment="业务 paper_id（如 aaai-2022-20281），全局唯一 PK")
    doc_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        comment="关联的文档ID（1:1 inheritance，NOT NULL UNIQUE FK）"
    )
    arxiv_id = Column(String(50), nullable=True, index=True, comment="arXiv ID")
    doi = Column(String(100), nullable=True, index=True, comment="DOI")
    title = Column(String(1000), nullable=False, comment="论文标题")
    abstract = Column(Text, nullable=True, comment="摘要")
    authors_json = Column(JSON, nullable=True, comment="作者列表")
    year = Column(Integer, nullable=True, index=True, comment="发表年份")
    venue = Column(String(200), nullable=True, comment="发表venue/会议/期刊")
    paper_source = Column(String(50), nullable=True, comment="来源: arxiv|semantic_scholar|acl等")
    pdf_path = Column(String(1000), nullable=True, comment="PDF本地路径")
    txt_path = Column(String(1000), nullable=True, comment="TXT/MD转换后路径")
    citation_count = Column(Integer, default=0, comment="引用次数")
    track_type = Column(String(50), nullable=True, comment="main|workshop|demo|findings|tutorial")
    impact_level = Column(String(50), nullable=True, comment="oral|spotlight|poster|regular")
    venue_tier = Column(String(10), nullable=True, comment="CCF-A|CCF-B|CCF-C|other")
    venue_source = Column(String(100), nullable=True, comment="proceedings来源标识")

    # 关系
    document = relationship("Document", back_populates="paper")

    def __repr__(self):
        return f"<Paper(paper_id={self.paper_id}, doc_id={self.doc_id}, title='{(self.title or '')[:50]}...', year={self.year})>"


class SystemConfig(Base):
    """系统配置表"""
    __tablename__ = "system_configs"

    key = Column(String(100), primary_key=True, comment="配置键")
    value = Column(Text, nullable=True, comment="配置值")
    description = Column(String(500), nullable=True, comment="配置描述")

    def __repr__(self):
        return f"<SystemConfig(key='{self.key}', value='{self.value[:50] if self.value else None}...')>"
