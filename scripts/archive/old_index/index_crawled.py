"""
将已爬取的网页内容索引到检索系统中

批量处理 data/crawled/*.json 文件，导入到数据库并建立向量索引
"""
import json
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings
from core.db import get_db_session, init_db
from core.db.models import Document, DocumentStatus, SourceType, DataSource
from app.services.index_service import IndexService


def compute_content_hash(content: str) -> str:
    """计算内容哈希用于去重"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def load_crawled_files(crawled_dir: str, batch_size: int = 100) -> List[Dict[str, Any]]:
    """
    批量加载爬取的JSON文件

    Args:
        crawled_dir: 爬取文件目录
        batch_size: 每批处理的文件数量

    Yields:
        批次数据列表
    """
    json_files = list(Path(crawled_dir).glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process")

    batch = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_file_path'] = str(json_file)
                batch.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def get_indexed_urls(session) -> set:
    """获取已索引的URL集合"""
    docs = session.query(Document.url).filter(
        Document.source_type == SourceType.webpage,
        Document.status == DocumentStatus.indexed
    ).all()
    return {doc.url for doc in docs if doc.url}


def create_document(
    session,
    url: str,
    title: str,
    content: str,
    content_hash: str,
    crawled_at: str = None
) -> Optional[Document]:
    """
    创建文档记录

    Returns:
        创建的Document对象，如果已存在则返回None
    """
    # 检查URL是否已存在
    existing = session.query(Document).filter(Document.url == url).first()
    if existing:
        return None

    # 检查内容哈希是否已存在（去重）
    existing_hash = session.query(Document).filter(
        Document.content_hash == content_hash
    ).first()
    if existing_hash:
        print(f"  Skipping duplicate content: {url[:60]}...")
        return None

    doc = Document(
        title=title[:500] if title else "Untitled",
        source=url[:500] if url else "",
        source_type=SourceType.webpage,
        data_source=DataSource.system,
        content_hash=content_hash,
        url=url,
        status=DocumentStatus.pending,
        metadata_json={
            "crawled_at": crawled_at,
            "import_time": datetime.now().isoformat()
        }
    )
    session.add(doc)
    session.flush()  # 获取ID
    return doc


def index_batch(
    index_service: IndexService,
    session,
    batch: List[Dict[str, Any]],
    indexed_urls: set
) -> Dict[str, int]:
    """
    索引一批文档

    Returns:
        统计信息：success, skipped, failed
    """
    stats = {"success": 0, "skipped": 0, "failed": 0}

    for item in batch:
        url = item.get("url", "")
        title = item.get("title", "")
        content = item.get("content", "")
        status = item.get("status", "")
        crawled_at = item.get("crawled_at", "")

        # 跳过失败的爬取
        if status != "success":
            stats["skipped"] += 1
            continue

        # 跳过空内容
        if not content or len(content.strip()) < 50:
            stats["skipped"] += 1
            continue

        # 跳过已索引的URL
        if url in indexed_urls:
            stats["skipped"] += 1
            continue

        try:
            content_hash = compute_content_hash(content)

            # 创建文档记录
            doc = create_document(
                session=session,
                url=url,
                title=title,
                content=content,
                content_hash=content_hash,
                crawled_at=crawled_at
            )

            if doc is None:
                stats["skipped"] += 1
                continue

            # 建立索引
            result = index_service.index_document(
                doc_id=doc.id,
                content=content,
                title=title,
                source=url,
                metadata={"url": url, "crawled_at": crawled_at},
                include_summary=False,  # 网页不需要摘要
                db=session,
                source_type="webpage",
                data_source="system"
            )

            if result["success"]:
                doc.status = DocumentStatus.indexed
                stats["success"] += 1
                indexed_urls.add(url)  # 更新缓存
            else:
                doc.status = DocumentStatus.failed
                stats["failed"] += 1
                print(f"  Index failed: {url[:60]}... - {result.get('error', 'Unknown')}")

        except Exception as e:
            stats["failed"] += 1
            print(f"  Error processing {url[:60]}...: {e}")

    return stats


def main():
    """主函数"""
    print("=" * 60)
    print("Crawled Content Indexer")
    print("=" * 60)

    crawled_dir = settings.crawled_dir
    print(f"Crawled directory: {crawled_dir}")

    # 初始化数据库
    print("Initializing database...")
    init_db()

    # 初始化索引服务
    print("Initializing index service...")
    index_service = IndexService(collection_name="webpages")

    # 统计
    total_stats = {"success": 0, "skipped": 0, "failed": 0}
    batch_num = 0

    with get_db_session() as session:
        # 获取已索引的URL
        print("Loading indexed URLs...")
        indexed_urls = get_indexed_urls(session)
        print(f"Already indexed: {len(indexed_urls)} URLs")

        # 批量处理
        for batch in load_crawled_files(crawled_dir, batch_size=50):
            batch_num += 1
            print(f"\nProcessing batch {batch_num} ({len(batch)} items)...")

            stats = index_batch(
                index_service=index_service,
                session=session,
                batch=batch,
                indexed_urls=indexed_urls
            )

            # 累计统计
            for key in stats:
                total_stats[key] += stats[key]

            # 每批次提交
            session.commit()

            print(f"  Batch {batch_num}: +{stats['success']} indexed, "
                  f"{stats['skipped']} skipped, {stats['failed']} failed")
            print(f"  Total: {total_stats['success']} indexed, "
                  f"{total_stats['skipped']} skipped, {total_stats['failed']} failed")

    # 最终统计
    print("\n" + "=" * 60)
    print("Indexing Complete!")
    print("=" * 60)
    print(f"Successfully indexed: {total_stats['success']}")
    print(f"Skipped: {total_stats['skipped']}")
    print(f"Failed: {total_stats['failed']}")

    # 打印索引统计
    try:
        index_stats = index_service.get_index_stats()
        print(f"\nIndex Statistics:")
        print(f"  Vector index size: {index_stats.get('vector_index_size', 'N/A')}")
        print(f"  BM25 index size: {index_stats.get('bm25_index_size', 'N/A')}")
    except Exception as e:
        print(f"Could not get index stats: {e}")


if __name__ == "__main__":
    main()
