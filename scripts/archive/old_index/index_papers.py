#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文索引脚本 - 将 Markdown 论文文件索引到 GPU 向量数据库和全文索引

功能:
1. 读取 data/papers/markdown/ 目录下的所有 .md 文件
2. 使用 GPU embedding (BAAI/bge-m3) 生成向量
3. 将论文索引到 ChromaDB 向量数据库
4. 同时创建 Whoosh 全文索引用于 BM25 搜索
5. 支持增量索引（跳过已索引的文件）
6. 显示进度条和统计信息

Usage:
    python scripts/index_papers.py [--force] [--batch-size 32]
"""

import sys
import os
import re
import json
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from config import settings
from app.models.schemas import Document, Chunk, FileType
from core.indexing.embedder import Embedder
from core.indexing.chunker import TextChunker
from core.storage.chroma_store import ChromaStore
from core.storage.whoosh_store import WhooshStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "data" / "logs" / f"index_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """论文元数据"""
    paper_id: str
    title: str = ""
    source_pdf: str = ""
    conversion_method: str = ""
    conversion_time: str = ""
    page_count: int = 0
    has_equations: bool = False
    has_tables: bool = False
    file_path: str = ""
    content_hash: str = ""


@dataclass
class IndexStats:
    """索引统计信息"""
    total_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    chunks_by_level: Dict[int, int] = field(default_factory=dict)
    processing_time: float = 0.0
    failed_list: List[str] = field(default_factory=list)


class IndexStateManager:
    """索引状态管理器 - 用于增量索引"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state: Dict[str, str] = {}  # {paper_id: content_hash}
        self._load_state()

    def _load_state(self):
        """加载索引状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self.state = json.load(f)
                logger.info(f"Loaded index state: {len(self.state)} indexed papers")
            except Exception as e:
                logger.warning(f"Failed to load index state: {e}")
                self.state = {}

    def save_state(self):
        """保存索引状态"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2)

    def is_indexed(self, paper_id: str, content_hash: str) -> bool:
        """检查论文是否已索引（且内容未变更）"""
        return self.state.get(paper_id) == content_hash

    def mark_indexed(self, paper_id: str, content_hash: str):
        """标记论文为已索引"""
        self.state[paper_id] = content_hash

    def remove(self, paper_id: str):
        """移除索引记录"""
        self.state.pop(paper_id, None)


class PaperIndexer:
    """论文索引器"""

    def __init__(self,
                 markdown_dir: Path,
                 collection_name: str = "papers",
                 batch_size: int = 32,
                 force_reindex: bool = False):
        """
        初始化论文索引器

        Args:
            markdown_dir: Markdown 文件目录
            collection_name: ChromaDB 集合名称
            batch_size: 批处理大小
            force_reindex: 是否强制重新索引
        """
        self.markdown_dir = Path(markdown_dir)
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.force_reindex = force_reindex

        # 状态文件
        self.state_file = PROJECT_ROOT / "data" / "indices" / "paper_index_state.json"
        self.state_manager = IndexStateManager(self.state_file)

        # 初始化组件
        logger.info("Initializing indexer components...")
        self.embedder = Embedder()
        self.chunker = TextChunker()
        self.chroma_store = ChromaStore(collection_name=collection_name)
        self.whoosh_store = WhooshStore()

        # 统计信息
        self.stats = IndexStats()

    def compute_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def parse_markdown_frontmatter(self, content: str) -> Tuple[Dict, str]:
        """
        解析 Markdown frontmatter (YAML 头部)

        Args:
            content: Markdown 文件内容

        Returns:
            (metadata_dict, body_content)
        """
        metadata = {}
        body = content

        # 匹配 YAML frontmatter
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if match:
            yaml_content = match.group(1)
            body = match.group(2)

            # 简单解析 YAML (避免引入 pyyaml 依赖)
            for line in yaml_content.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    # 处理布尔值
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    # 处理数字
                    elif value.isdigit():
                        value = int(value)

                    metadata[key] = value

        return metadata, body

    def extract_title_from_content(self, content: str) -> str:
        """从内容中提取标题"""
        # 尝试匹配 Markdown 标题
        patterns = [
            r'^##\s+(.+?)(?:\n|$)',  # ## Title
            r'^#\s+(.+?)(?:\n|$)',    # # Title
            r'^(.+?)\n[=]+\n',        # Title\n===
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                title = match.group(1).strip()
                # 清理标题
                title = re.sub(r'\*+', '', title)  # 移除加粗标记
                title = re.sub(r'\s+', ' ', title)  # 规范化空白
                if len(title) > 10:  # 标题应该有一定长度
                    return title[:200]  # 限制长度

        return ""

    def parse_paper(self, file_path: Path) -> Optional[PaperMetadata]:
        """
        解析论文文件

        Args:
            file_path: Markdown 文件路径

        Returns:
            PaperMetadata 或 None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"Empty file: {file_path}")
                return None

            # 解析 frontmatter
            metadata_dict, body = self.parse_markdown_frontmatter(content)

            # 提取论文 ID (使用文件名)
            paper_id = file_path.stem

            # 提取标题
            title = self.extract_title_from_content(body)
            if not title:
                title = metadata_dict.get('title', paper_id)

            return PaperMetadata(
                paper_id=paper_id,
                title=title,
                source_pdf=metadata_dict.get('source_pdf', ''),
                conversion_method=metadata_dict.get('conversion_method', ''),
                conversion_time=metadata_dict.get('conversion_time', ''),
                page_count=metadata_dict.get('page_count', 0),
                has_equations=metadata_dict.get('has_equations', False),
                has_tables=metadata_dict.get('has_tables', False),
                file_path=str(file_path),
                content_hash=self.compute_content_hash(content)
            )

        except Exception as e:
            logger.error(f"Failed to parse paper {file_path}: {e}")
            return None

    def get_paper_content(self, file_path: Path) -> str:
        """获取论文内容（去除 frontmatter）"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        _, body = self.parse_markdown_frontmatter(content)
        return body.strip()

    def scan_papers(self) -> List[Tuple[Path, PaperMetadata]]:
        """
        扫描所有论文文件

        Returns:
            [(file_path, metadata), ...]
        """
        papers = []
        md_files = list(self.markdown_dir.rglob("*.md"))

        logger.info(f"Found {len(md_files)} markdown files in {self.markdown_dir} (recursive)")
        self.stats.total_files = len(md_files)

        for file_path in tqdm(md_files, desc="Scanning papers"):
            metadata = self.parse_paper(file_path)
            if metadata:
                papers.append((file_path, metadata))

        return papers

    def filter_papers_to_index(self,
                                papers: List[Tuple[Path, PaperMetadata]]
                                ) -> List[Tuple[Path, PaperMetadata]]:
        """
        过滤需要索引的论文（增量索引）

        Args:
            papers: 论文列表

        Returns:
            需要索引的论文列表
        """
        if self.force_reindex:
            logger.info("Force reindex enabled, will index all papers")
            return papers

        to_index = []
        for file_path, metadata in papers:
            if not self.state_manager.is_indexed(metadata.paper_id, metadata.content_hash):
                to_index.append((file_path, metadata))
            else:
                self.stats.skipped_files += 1

        logger.info(f"Papers to index: {len(to_index)}, skipped: {self.stats.skipped_files}")
        return to_index

    def create_document(self, metadata: PaperMetadata, content: str) -> Document:
        """创建文档对象"""
        return Document(
            id=metadata.paper_id,
            title=metadata.title,
            source=metadata.file_path,
            source_type=FileType.TXT,  # 使用 TXT 类型表示 Markdown
            content=content,
            metadata={
                "source_pdf": metadata.source_pdf,
                "conversion_method": metadata.conversion_method,
                "page_count": metadata.page_count,
                "has_equations": metadata.has_equations,
                "has_tables": metadata.has_tables,
                "content_hash": metadata.content_hash
            }
        )

    def process_batch(self,
                      papers: List[Tuple[Path, PaperMetadata]],
                      chunk_batch_size: int = 100) -> int:
        """
        处理一批论文 - 使用流式分块避免内存溢出

        Args:
            papers: 论文列表
            chunk_batch_size: 每批写入的chunk数量

        Returns:
            成功处理的数量
        """
        import gc
        import torch

        processed_count = 0

        for file_path, metadata in papers:
            try:
                content = self.get_paper_content(file_path)
                if not content:
                    continue

                # 创建文档
                doc = self.create_document(metadata, content)

                # 用于收集当前文档所有chunks（供Whoosh使用）
                all_chunks_for_whoosh = []
                paper_chunk_count = 0

                # 流式分块处理
                batch_chunks = []
                for chunk in self.chunker.chunk_document_iter(
                    doc_id=doc.id,
                    text=content,
                    include_summary=True
                ):
                    # 添加额外元数据
                    chunk.metadata["title"] = metadata.title
                    chunk.metadata["paper_id"] = metadata.paper_id
                    batch_chunks.append(chunk)

                    # 达到批次大小时处理
                    if len(batch_chunks) >= chunk_batch_size:
                        # 生成嵌入
                        batch_chunks = self.embedder.embed_chunks(batch_chunks, batch_size=32)
                        # 写入 ChromaDB
                        self.chroma_store.add_chunks(batch_chunks)
                        # 收集供Whoosh使用
                        all_chunks_for_whoosh.extend(batch_chunks)
                        # 更新统计
                        for c in batch_chunks:
                            self.stats.chunks_by_level[c.level] = self.stats.chunks_by_level.get(c.level, 0) + 1
                        paper_chunk_count += len(batch_chunks)
                        # 清理当前批次
                        batch_chunks = []
                        gc.collect()

                # 处理剩余的chunks
                if batch_chunks:
                    batch_chunks = self.embedder.embed_chunks(batch_chunks, batch_size=32)
                    self.chroma_store.add_chunks(batch_chunks)
                    all_chunks_for_whoosh.extend(batch_chunks)
                    for c in batch_chunks:
                        self.stats.chunks_by_level[c.level] = self.stats.chunks_by_level.get(c.level, 0) + 1
                    paper_chunk_count += len(batch_chunks)
                    batch_chunks = []

                # 存储到 Whoosh（需要所有chunks）
                self.whoosh_store.add_document(doc, all_chunks_for_whoosh)

                # 更新状态
                self.state_manager.mark_indexed(metadata.paper_id, metadata.content_hash)
                self.stats.indexed_files += 1
                self.stats.total_chunks += paper_chunk_count

                processed_count += 1

                # 清理内存
                del all_chunks_for_whoosh, doc, content
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                self.stats.failed_files += 1
                self.stats.failed_list.append(str(file_path))
                continue

        return processed_count

    def run(self):
        """运行索引流程"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Paper Indexing Started")
        logger.info("=" * 60)

        # 显示配置
        logger.info(f"Markdown directory: {self.markdown_dir}")
        logger.info(f"Collection name: {self.collection_name}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Force reindex: {self.force_reindex}")

        # 加载 embedding 模型
        logger.info("Loading embedding model...")
        self.embedder.load_model()
        device_info = self.embedder.get_device_info()
        logger.info(f"Device info: {device_info}")

        # 扫描论文
        logger.info("Scanning papers...")
        papers = self.scan_papers()

        if not papers:
            logger.warning("No papers found!")
            return

        # 过滤需要索引的论文
        papers_to_index = self.filter_papers_to_index(papers)

        if not papers_to_index:
            logger.info("All papers are already indexed. Nothing to do.")
            return

        # 分批处理
        total_batches = (len(papers_to_index) + self.batch_size - 1) // self.batch_size
        logger.info(f"Processing {len(papers_to_index)} papers in {total_batches} batches...")

        import gc
        import torch

        with tqdm(total=len(papers_to_index), desc="Indexing papers") as pbar:
            for i in range(0, len(papers_to_index), self.batch_size):
                batch = papers_to_index[i:i + self.batch_size]
                processed = self.process_batch(batch)
                pbar.update(len(batch))

                # 定期保存状态
                if (i // self.batch_size + 1) % 5 == 0:
                    self.state_manager.save_state()

                # 清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 保存最终状态
        self.state_manager.save_state()

        # 计算处理时间
        end_time = datetime.now()
        self.stats.processing_time = (end_time - start_time).total_seconds()

        # 打印统计信息
        self._print_stats()

    def _print_stats(self):
        """打印统计信息"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("Indexing Statistics")
        logger.info("=" * 60)
        logger.info(f"Total files scanned:  {self.stats.total_files}")
        logger.info(f"Files indexed:        {self.stats.indexed_files}")
        logger.info(f"Files skipped:        {self.stats.skipped_files}")
        logger.info(f"Files failed:         {self.stats.failed_files}")
        logger.info(f"Total chunks created: {self.stats.total_chunks}")
        logger.info("")
        logger.info("Chunks by level:")
        for level, count in sorted(self.stats.chunks_by_level.items()):
            level_name = {1: "Summary", 2: "Paragraph", 3: "Sentence"}.get(level, f"Level {level}")
            logger.info(f"  {level_name}: {count}")
        logger.info("")
        logger.info(f"Processing time:      {self.stats.processing_time:.2f} seconds")
        logger.info(f"ChromaDB count:       {self.chroma_store.count()}")
        logger.info(f"Whoosh count:         {self.whoosh_store.count()}")

        if self.stats.failed_list:
            logger.info("")
            logger.info("Failed files:")
            for f in self.stats.failed_list[:10]:  # 只显示前10个
                logger.info(f"  - {f}")
            if len(self.stats.failed_list) > 10:
                logger.info(f"  ... and {len(self.stats.failed_list) - 10} more")

        logger.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Index papers from markdown files to vector and full-text search indices"
    )
    parser.add_argument(
        "--markdown-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "papers" / "markdown"),
        help="Directory containing markdown files"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="papers",
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex all papers (ignore incremental state)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing indices before indexing"
    )

    args = parser.parse_args()

    # 确保日志目录存在
    log_dir = PROJECT_ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 检查目录
    markdown_dir = Path(args.markdown_dir)
    if not markdown_dir.exists():
        logger.error(f"Markdown directory does not exist: {markdown_dir}")
        sys.exit(1)

    # 清空索引（如果需要）
    if args.clear:
        logger.info("Clearing existing indices...")
        chroma = ChromaStore(collection_name=args.collection)
        whoosh = WhooshStore()
        chroma.clear()
        whoosh.clear()

        # 清空状态文件
        state_file = PROJECT_ROOT / "data" / "indices" / "paper_index_state.json"
        if state_file.exists():
            state_file.unlink()

        logger.info("Indices cleared.")

    # 创建索引器并运行
    indexer = PaperIndexer(
        markdown_dir=markdown_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
        force_reindex=args.force
    )

    try:
        indexer.run()
    except KeyboardInterrupt:
        logger.info("\nIndexing interrupted by user.")
        indexer.state_manager.save_state()
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Indexing failed with error: {e}")
        indexer.state_manager.save_state()
        sys.exit(1)


if __name__ == "__main__":
    main()
