"""
配置文件 - 代理、模型、索引参数等
"""
import os
from pathlib import Path

from pydantic import BaseModel, Field, FiniteFloat
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.indexing.embedding_profile import DEFAULT_MODEL_ID, PROFILES


class ProxySettings(BaseSettings):
    """代理配置 —— 强制走代理，无直连回退。

    外网抓取（USENIX/DBLP/ACM 等）必须经 10808 代理。曾有 ``enabled`` 开关 +
    ``... if enabled else None`` 直连回退，在 4090 盒（PROXY__ENABLED=false）上把
    流量静默改直连 → 开放会议站点 "Network is unreachable"。改为强制代理后，代理不可达
    时显式失败（强制抓取跑在有代理的 Arc 上），不再静默直连（2026-06-16）。
    """
    http_proxy: str = "http://127.0.0.1:10808"
    https_proxy: str = "http://127.0.0.1:10808"

    def get_proxy_dict(self) -> dict:
        return {
            "http": self.http_proxy,
            "https": self.https_proxy
        }

    def get_browser_proxy(self) -> dict:
        """browser-use格式的代理配置"""
        return {
            "server": self.http_proxy
        }


_BGE_M3 = PROFILES[DEFAULT_MODEL_ID]
# Longest query the reranker scores against a chunk (see ``RerankerSettings.max_length``).
RERANK_QUERY_TOKENS = 512

EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {"dimension": 384, "max_tokens": 128},
    "BAAI/bge-base-zh-v1.5": {"dimension": 768, "max_tokens": 512},
    "BAAI/bge-small-zh-v1.5": {"dimension": 512, "max_tokens": 512},
    # The bge-m3 numbers are its embedding profile's (core/indexing/embedding_profile.py).
    DEFAULT_MODEL_ID: {
        "dimension": _BGE_M3.dimension,
        "max_tokens": _BGE_M3.max_seq_len,
        "supports_sparse": _BGE_M3.sparse,
    },
    "Qwen/Qwen3-Embedding-0.6B": {"dimension": 1024, "max_tokens": 8192},
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {"dimension": 1536, "max_tokens": 8192},
}


class EmbeddingSettings(BaseSettings):
    """嵌入模型配置"""
    model_name: str = DEFAULT_MODEL_ID
    local_files_only: bool = True
    device: str = "cuda"
    # Where the vLLM service that runs the model listens (env ``EMBEDDING__SERVICE_URL``).
    service_url: str = "http://127.0.0.1:18200"
    # How long the backend's start-up waits for that service to answer (env ``EMBEDDING__SERVICE_WAIT_S``):
    # the unit loads its model for 50-75 s after the backend's ``After=`` is satisfied.
    service_wait_s: float = Field(default=180.0, ge=0.0)
    # dtype, sequence limit and encode batch size are the model's embedding profile.

    @property
    def dimension(self) -> int:
        return EMBEDDING_MODELS.get(self.model_name, {}).get("dimension", 384)

    @property
    def max_tokens(self) -> int:
        return EMBEDDING_MODELS.get(self.model_name, {}).get("max_tokens", 256)

    @property
    def supports_sparse(self) -> bool:
        return EMBEDDING_MODELS.get(self.model_name, {}).get("supports_sparse", False)


class RerankerSettings(BaseModel):
    """Cross-encoder resource controls.

    Deployments with a measured golden corpus can opt into another batch size through
    ``RERANKER__BATCH_SIZE`` without changing every host's ranking behaviour.
    """

    # 32, not FlagEmbedding's 128: with whole 2040-token chunks scored (max_length below),
    # a batch of 128 peaks at 6.10 GiB of CUDA memory against 2.44 GiB for the old 512-token
    # pairs; a batch of 32 peaks at 2.45 GiB (RTX 3090, fp16, measured 2026-11-14).
    batch_size: int = Field(default=32, ge=1, le=128)
    # Where the vLLM service that runs the cross-encoder listens (env ``RERANKER__SERVICE_URL``).
    service_url: str = "http://127.0.0.1:18201"
    # Token length of a (query, passage) pair the cross-encoder scores: a whole stored chunk
    # (the embedding profile's chunk budget) plus a query of up to ``RERANK_QUERY_TOKENS``.
    # FlagReranker's own default of 512 scored only the first ~480 tokens of every chunk.
    max_length: int = Field(default=_BGE_M3.chunk_max_tokens + RERANK_QUERY_TOKENS, ge=64)
    # Zero is deliberately aggressive: once a rerank is complete, return every
    # unused allocator block to the shared GPU. It does not free model weights.
    idle_cache_limit_mib: int = Field(default=0, ge=0)
    # Opt-in `.88` process cap. It limits all Retrieval CUDA allocator work and
    # must be applied before BGE-M3/reranker model construction.
    cuda_memory_fraction: float | None = Field(default=None, gt=0, le=1)


class ChunkSettings(BaseSettings):
    """分块配置"""
    # 多粒度分块：[L2 段落级, L3 句子级]（L1 摘要走 summary_max_length，不在此列）
    chunk_sizes: list = [2048, 512]
    chunk_overlap: int = 50
    # 文档摘要长度
    summary_max_length: int = 500


class RetrievalSettings(BaseSettings):
    """检索配置"""
    # 初始召回数量
    initial_top_k: int = 1000
    # 最终返回数量
    final_top_n: int = 20
    # 混合检索权重
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    # Match-doc preserves the legacy query_points semantics but may run this
    # many independent requests concurrently.  The production Qdrant has four
    # search threads, so two leave capacity for interactive searches.
    match_doc_qdrant_concurrency: int = Field(default=2, ge=1, le=4)


class CrawlerSettings(BaseSettings):
    """爬虫配置"""
    headless: bool = True
    timeout: int = 30000  # ms
    max_concurrent: int = 5
    retry_times: int = 3
    delay_between_requests: float = 1.0  # seconds


class PublicCodeSettings(BaseSettings):
    """Local source mirrors and collection settings for the public code corpus."""

    catalog_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config",
        "public_sources",
        "first_wave_repositories.json",
    )
    mirror_root: str = str(
        Path(__file__).resolve().parent.parent / "retrieval-public-sources" / "git-mirrors"
    )
    git_command_timeout_seconds: FiniteFloat = Field(default=30.0, gt=0)
    git_network_timeout_seconds: FiniteFloat = Field(default=900.0, gt=0)
    git_network_retry_attempts: int = Field(default=3, ge=1)
    git_network_retry_backoff_seconds: FiniteFloat = Field(default=2.0, ge=0)
    native_chunk_timeout_seconds: FiniteFloat = Field(default=60.0, gt=0)
    # Uploaded git bundles that seed a mirror without a network clone; keep them on
    # the mirror disk, since the mirror is built from the bundle there.
    import_root: str = str(
        Path(__file__).resolve().parent.parent / "retrieval-public-sources" / "imports"
    )
    max_import_bytes: int = Field(default=32 * 1024**3, ge=1)
    import_git_timeout_seconds: FiniteFloat = Field(default=3600.0, gt=0)


class SchedulerSettings(BaseSettings):
    """Recurring-job scheduler (app/services/scheduler)."""

    # Bare field names would read the generic env vars ENABLED / LOCK_PATH.
    model_config = SettingsConfigDict(env_prefix="SCHEDULER_")

    enabled: bool = True
    # Held (non-blocking) by the one process that runs the scheduler; a second
    # backend process on the same data dir stays idle instead of double-firing.
    lock_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "db", "scheduler.lock",
    )


class CodeAnalysisSettings(BaseSettings):
    """Background duplicate-code analysis after code indexing (app/services/code_kb/analysis.py)."""

    # CODE_ANALYSIS_MIN_INTERVAL_SECONDS / CODE_ANALYSIS_MAX_CHUNKS
    model_config = SettingsConfigDict(env_prefix="CODE_ANALYSIS_")

    # Two analyses of one scope are at least this far apart; index runs that finish inside
    # the interval coalesce into a single later analysis.
    min_interval_seconds: FiniteFloat = Field(default=600.0, ge=0)
    # One analysis queries the neighbours of the changed files' chunks only. A change
    # bigger than this is reported as too large instead of occupying Qdrant for minutes.
    max_chunks: int = Field(default=2000, ge=1)


class Settings(BaseSettings):
    """主配置"""
    # 项目路径
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = os.path.join(base_dir, "data")
    upload_dir: str = os.path.join(data_dir, "uploads")
    crawled_dir: str = os.path.join(data_dir, "crawled")
    index_dir: str = os.path.join(data_dir, "indices")
    chroma_dir: str = os.path.join(index_dir, "chroma")
    whoosh_dir: str = os.path.join(index_dir, "whoosh")

    # 数据分离存储路径
    system_data_dir: str = os.path.join(data_dir, "system")  # 预置论文库（系统数据）
    user_data_dir: str = os.path.join(data_dir, "user")  # 用户上传的数据

    # 论文 artifact 解析根（Arc/Windows → 4090/Linux 迁移后，papers.txt_path/pdf_path
    # 仍是 Windows 路径，Linux 打不开。resolver 按 <venue>/<year>/<basename> 在这些
    # POSIX 根下回查实际文件。POSIX 路径，可经 PAPER_MARKDOWN_ROOTS / PAPER_PDF_ROOTS
    # env 覆盖（JSON 列表）。Arc 上这些根不存在也无妨——原始路径本就可直接打开。
    paper_markdown_roots: list[str] = [
        "/mnt/lsb/retrival/papers/markdown",
        os.path.join(data_dir, "papers", "ccf_a_main_conference_corpus", "markdown"),
    ]
    paper_pdf_roots: list[str] = [
        "/mnt/lsb/retrival/papers/pdf",
        os.path.join(data_dir, "papers", "ccf_a_main_conference_corpus", "pdf"),
    ]

    # 子配置
    proxy: ProxySettings = ProxySettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    chunk: ChunkSettings = ChunkSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    crawler: CrawlerSettings = CrawlerSettings()
    public_code: PublicCodeSettings = Field(default_factory=PublicCodeSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    code_analysis: CodeAnalysisSettings = Field(default_factory=CodeAnalysisSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
    )


# 全局配置实例
settings = Settings()

# 确保目录存在
for dir_path in [settings.upload_dir, settings.crawled_dir,
                 settings.chroma_dir, settings.whoosh_dir,
                 settings.system_data_dir, settings.user_data_dir]:
    os.makedirs(dir_path, exist_ok=True)
