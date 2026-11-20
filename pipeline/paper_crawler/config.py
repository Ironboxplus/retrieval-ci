"""
Config - 配置文件

使用 pydantic-settings 管理项目配置。
支持从 .env 文件和环境变量读取配置。

配置项包括：
- 代理设置
- 下载并发数
- 重试次数
- 超时设置
- 各数据源的API密钥
- 存储路径
- 日志级别
"""

from pathlib import Path
from typing import Optional, List, Literal
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class ProxySettings(BaseSettings):
    """代理配置"""

    model_config = SettingsConfigDict(
        env_prefix="PROXY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 是否启用代理
    enabled: bool = Field(default=False, description="是否启用代理")

    # HTTP 代理
    http: Optional[str] = Field(
        default=None,
        description="HTTP 代理地址",
        examples=["http://127.0.0.1:7890"],
    )

    # HTTPS 代理
    https: Optional[str] = Field(
        default=None,
        description="HTTPS 代理地址",
        examples=["http://127.0.0.1:7890"],
    )

    # SOCKS5 代理
    socks5: Optional[str] = Field(
        default=None,
        description="SOCKS5 代理地址",
        examples=["socks5://127.0.0.1:7891"],
    )

    # 不使用代理的地址
    no_proxy: List[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "::1"],
        description="不使用代理的地址列表",
    )

    def get_proxies(self) -> Optional[dict]:
        """获取代理字典，用于 requests/httpx"""
        if not self.enabled:
            return None

        proxies = {}
        if self.http:
            proxies["http://"] = self.http
        if self.https:
            proxies["https://"] = self.https
        elif self.http:
            # 如果没有配置 HTTPS 代理，使用 HTTP 代理
            proxies["https://"] = self.http

        return proxies if proxies else None


class DownloadSettings(BaseSettings):
    """下载配置"""

    model_config = SettingsConfigDict(
        env_prefix="DOWNLOAD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 并发下载数
    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="最大并发下载数",
    )

    # 重试次数
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="最大重试次数",
    )

    # 重试延迟（秒）
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="重试延迟（秒）",
    )

    # 重试延迟倍数（指数退避）
    retry_backoff: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="重试延迟倍数",
    )

    # 连接超时（秒）
    connect_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="连接超时（秒）",
    )

    # 读取超时（秒）
    read_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="读取超时（秒）",
    )

    # 总超时（秒）
    total_timeout: float = Field(
        default=300.0,
        ge=1.0,
        le=1800.0,
        description="总超时（秒）",
    )

    # 请求间隔（秒）
    request_delay: float = Field(
        default=1.0,
        ge=0.0,
        le=60.0,
        description="请求间隔（秒）",
    )

    # 下载块大小（字节）
    chunk_size: int = Field(
        default=8192,
        ge=1024,
        le=1048576,
        description="下载块大小（字节）",
    )

    # 是否覆盖已存在的文件
    overwrite_existing: bool = Field(
        default=False,
        description="是否覆盖已存在的文件",
    )

    def get_timeout(self) -> tuple:
        """获取超时元组，用于 requests/httpx"""
        return (self.connect_timeout, self.read_timeout)


class APIKeySettings(BaseSettings):
    """API 密钥配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Semantic Scholar API Key
    semantic_scholar_api_key: Optional[str] = Field(
        default=None,
        alias="SEMANTIC_SCHOLAR_API_KEY",
        description="Semantic Scholar API 密钥",
    )

    # arXiv API Key（目前 arXiv 不需要 API Key，保留字段）
    arxiv_api_key: Optional[str] = Field(
        default=None,
        alias="ARXIV_API_KEY",
        description="arXiv API 密钥",
    )

    # Crossref API Key / Polite Pool
    crossref_api_key: Optional[str] = Field(
        default=None,
        alias="CROSSREF_API_KEY",
        description="Crossref API 密钥",
    )

    # Crossref 邮箱（用于 Polite Pool）
    crossref_email: Optional[str] = Field(
        default=None,
        alias="CROSSREF_EMAIL",
        description="Crossref Polite Pool 邮箱",
    )

    # Unpaywall 邮箱
    unpaywall_email: Optional[str] = Field(
        default=None,
        alias="UNPAYWALL_EMAIL",
        description="Unpaywall API 邮箱",
    )

    # CORE API Key
    core_api_key: Optional[str] = Field(
        default=None,
        alias="CORE_API_KEY",
        description="CORE API 密钥",
    )

    # Elsevier API Key
    elsevier_api_key: Optional[str] = Field(
        default=None,
        alias="ELSEVIER_API_KEY",
        description="Elsevier API 密钥",
    )

    # Springer API Key
    springer_api_key: Optional[str] = Field(
        default=None,
        alias="SPRINGER_API_KEY",
        description="Springer Nature API 密钥",
    )

    # IEEE API Key
    ieee_api_key: Optional[str] = Field(
        default=None,
        alias="IEEE_API_KEY",
        description="IEEE Xplore API 密钥",
    )

    # OpenAI API Key（用于 PDF 解析等）
    openai_api_key: Optional[str] = Field(
        default=None,
        alias="OPENAI_API_KEY",
        description="OpenAI API 密钥",
    )

    # Mathpix API（用于公式识别）
    mathpix_app_id: Optional[str] = Field(
        default=None,
        alias="MATHPIX_APP_ID",
        description="Mathpix App ID",
    )
    mathpix_app_key: Optional[str] = Field(
        default=None,
        alias="MATHPIX_APP_KEY",
        description="Mathpix App Key",
    )


class StorageSettings(BaseSettings):
    """存储路径配置"""

    model_config = SettingsConfigDict(
        env_prefix="STORAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 数据根目录
    data_dir: Path = Field(
        default=DATA_DIR,
        description="数据根目录",
    )

    # PDF 存储目录
    pdf_dir: Path = Field(
        default=DATA_DIR / "papers" / "pdf",
        description="PDF 文件存储目录",
    )

    # Markdown 存储目录
    markdown_dir: Path = Field(
        default=DATA_DIR / "papers" / "markdown",
        description="Markdown 文件存储目录",
    )

    # 元数据存储目录
    metadata_dir: Path = Field(
        default=DATA_DIR / "papers" / "metadata",
        description="论文元数据存储目录",
    )

    # 日志目录
    log_dir: Path = Field(
        default=DATA_DIR / "logs",
        description="日志文件目录",
    )

    # 缓存目录
    cache_dir: Path = Field(
        default=DATA_DIR / "cache",
        description="缓存目录",
    )

    # 临时目录
    temp_dir: Path = Field(
        default=DATA_DIR / "temp",
        description="临时文件目录",
    )

    @field_validator(
        "data_dir", "pdf_dir", "markdown_dir", "metadata_dir",
        "log_dir", "cache_dir", "temp_dir",
        mode="before",
    )
    @classmethod
    def parse_path(cls, v):
        """将字符串转换为 Path 对象"""
        if isinstance(v, str):
            return Path(v)
        return v

    def ensure_directories(self):
        """确保所有目录存在"""
        for attr in [
            "data_dir", "pdf_dir", "markdown_dir", "metadata_dir",
            "log_dir", "cache_dir", "temp_dir"
        ]:
            path = getattr(self, attr)
            path.mkdir(parents=True, exist_ok=True)


class LoggingSettings(BaseSettings):
    """日志配置"""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 日志级别
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="日志级别",
    )

    # 是否输出到控制台
    console: bool = Field(
        default=True,
        description="是否输出到控制台",
    )

    # 是否输出到文件
    file: bool = Field(
        default=True,
        description="是否输出到文件",
    )

    # 日志轮转时间
    rotation: str = Field(
        default="00:00",
        description="日志轮转时间",
    )

    # 日志保留时间
    retention: str = Field(
        default="30 days",
        description="日志保留时间",
    )

    # 日志压缩格式
    compression: str = Field(
        default="zip",
        description="日志压缩格式",
    )


class GoogleScholarSettings(BaseSettings):
    """Google Scholar 爬虫配置"""

    model_config = SettingsConfigDict(
        env_prefix="GOOGLE_SCHOLAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 是否启用 Google Scholar 引用数获取
    enabled: bool = Field(
        default=True,
        description="是否启用 Google Scholar 引用数获取",
    )

    # 请求延迟范围 (秒)
    min_delay: float = Field(
        default=3.0,
        ge=1.0,
        le=60.0,
        description="最小请求延迟（秒）",
    )

    max_delay: float = Field(
        default=8.0,
        ge=2.0,
        le=120.0,
        description="最大请求延迟（秒）",
    )

    # 批量请求间隔 (秒)
    batch_delay: float = Field(
        default=10.0,
        ge=5.0,
        le=300.0,
        description="批量请求间隔（秒）",
    )

    # 重试配置
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="最大重试次数",
    )

    retry_delay: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="重试延迟（秒）",
    )

    # 标题相似度阈值
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="标题相似度阈值",
    )

    # 代理配置 (专用于 Google Scholar)
    proxy: Optional[str] = Field(
        default=None,
        description="Google Scholar 代理地址",
    )

    # 是否启用 CAPTCHA 检测
    captcha_detection: bool = Field(
        default=True,
        description="是否启用 CAPTCHA 检测",
    )

    # CAPTCHA 等待时间 (秒)
    captcha_wait_time: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="CAPTCHA 等待时间（秒）",
    )

    # 优先使用的获取方式: requests, scholarly, chrome_mcp
    preferred_method: str = Field(
        default="requests",
        description="优先使用的获取方式",
    )

    # 引用数获取源选择: google_scholar, openalex, both
    citation_source: str = Field(
        default="openalex",
        description="引用数获取源: google_scholar, openalex, both",
    )


class ScraperSettings(BaseSettings):
    """爬虫配置"""

    model_config = SettingsConfigDict(
        env_prefix="SCRAPER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # User-Agent
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        description="User-Agent 字符串",
    )

    # 是否遵守 robots.txt
    respect_robots_txt: bool = Field(
        default=True,
        description="是否遵守 robots.txt",
    )

    # 默认搜索结果数量
    default_max_results: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="默认最大搜索结果数量",
    )

    # 是否自动下载 PDF
    auto_download_pdf: bool = Field(
        default=True,
        description="是否自动下载 PDF",
    )

    # 是否自动转换为 Markdown
    auto_convert_markdown: bool = Field(
        default=True,
        description="是否自动转换为 Markdown",
    )


class Settings(BaseSettings):
    """主配置类

    整合所有子配置，提供统一的配置访问接口。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 子配置
    proxy: ProxySettings = Field(default_factory=ProxySettings)
    download: DownloadSettings = Field(default_factory=DownloadSettings)
    api_keys: APIKeySettings = Field(default_factory=APIKeySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    scraper: ScraperSettings = Field(default_factory=ScraperSettings)
    google_scholar: GoogleScholarSettings = Field(default_factory=GoogleScholarSettings)

    # 应用名称
    app_name: str = Field(
        default="PaperCrawler",
        description="应用名称",
    )

    # 应用版本
    app_version: str = Field(
        default="0.1.0",
        description="应用版本",
    )

    # 调试模式
    debug: bool = Field(
        default=False,
        alias="DEBUG",
        description="调试模式",
    )

    @model_validator(mode="after")
    def ensure_storage_directories(self):
        """确保存储目录存在"""
        self.storage.ensure_directories()
        return self

    def get_semantic_scholar_headers(self) -> dict:
        """获取 Semantic Scholar API 请求头"""
        headers = {"User-Agent": self.scraper.user_agent}
        if self.api_keys.semantic_scholar_api_key:
            headers["x-api-key"] = self.api_keys.semantic_scholar_api_key
        return headers

    def get_crossref_headers(self) -> dict:
        """获取 Crossref API 请求头"""
        headers = {"User-Agent": self.scraper.user_agent}
        if self.api_keys.crossref_email:
            headers["User-Agent"] = f"{self.app_name}/1.0 (mailto:{self.api_keys.crossref_email})"
        return headers


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例

    使用 lru_cache 确保全局只有一个配置实例。

    Returns:
        Settings 配置实例
    """
    return Settings()


# 全局配置实例（便捷访问）
settings = get_settings()

# 向后兼容的 Config 类
class Config:
    """向后兼容的配置类

    保持与原有代码的兼容性。
    """

    def __init__(self):
        self._settings = get_settings()

    @property
    def base_dir(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return self._settings.storage.data_dir

    @property
    def pdf_dir(self) -> Path:
        return self._settings.storage.pdf_dir

    @property
    def markdown_dir(self) -> Path:
        return self._settings.storage.markdown_dir

    @property
    def metadata_dir(self) -> Path:
        return self._settings.storage.metadata_dir

    @property
    def log_dir(self) -> Path:
        return self._settings.storage.log_dir

    @property
    def semantic_scholar_api_key(self) -> Optional[str]:
        return self._settings.api_keys.semantic_scholar_api_key

    @property
    def request_delay(self) -> float:
        return self._settings.download.request_delay

    @property
    def max_retries(self) -> int:
        return self._settings.download.max_retries

    @property
    def timeout(self) -> int:
        return int(self._settings.download.total_timeout)

    @property
    def max_concurrent_downloads(self) -> int:
        return self._settings.download.max_concurrent


# 向后兼容的全局配置实例
config = Config()
