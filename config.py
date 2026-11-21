"""
配置文件 - 代理、模型、索引参数等
"""
import os
from pathlib import Path

from pydantic import BaseModel, Field, FiniteFloat
from pydantic_settings import BaseSettings, SettingsConfigDict


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


EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {"dimension": 384, "max_tokens": 128},
    "BAAI/bge-base-zh-v1.5": {"dimension": 768, "max_tokens": 512},
    "BAAI/bge-small-zh-v1.5": {"dimension": 512, "max_tokens": 512},
    "BAAI/bge-m3": {"dimension": 1024, "max_tokens": 8192, "supports_sparse": True},
    "Qwen/Qwen3-Embedding-0.6B": {"dimension": 1024, "max_tokens": 8192},
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {"dimension": 1536, "max_tokens": 8192},
}


class EmbeddingSettings(BaseSettings):
    """嵌入模型配置"""
    model_name: str = "BAAI/bge-m3"
    local_files_only: bool = True
    batch_size: int = 64
    device: str = "cuda"
    use_fp16: bool = True

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

    The default keeps FlagEmbedding's historical batch size. Deployments with a
    measured golden corpus can opt into a smaller value through
    ``RERANKER__BATCH_SIZE`` without changing every host's ranking behaviour.
    """

    batch_size: int = Field(default=128, ge=1, le=128)
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
