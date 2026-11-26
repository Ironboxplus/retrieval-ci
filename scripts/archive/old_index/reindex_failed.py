"""
重新索引失败的文档
"""
import json
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings
from core.db import get_db_session, init_db
from core.db.models import Document, Chunk, DocumentStatus, SourceType
from app.services.index_service import IndexService


def find_crawled_file(url: str, crawled_dir: str) -> dict | None:
    """根据URL查找对应的爬取文件"""
    json_files = Path(crawled_dir).glob("*.json")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("url") == url:
                    return data
        except Exception:
            continue
    return None


def main():
    print("=" * 60)
    print("Re-indexing Failed Documents")
    print("=" * 60)

    init_db()
    index_service = IndexService(collection_name="webpages")

    stats = {"success": 0, "skipped": 0, "failed": 0}

    with get_db_session() as session:
        # 获取所有失败的文档
        failed_docs = session.query(Document).filter(
            Document.source_type == SourceType.webpage,
            Document.status == DocumentStatus.failed
        ).all()

        print(f"Found {len(failed_docs)} failed documents to re-index")

        for i, doc in enumerate(failed_docs):
            if i % 50 == 0:
                print(f"\nProcessing {i}/{len(failed_docs)}...")
                session.commit()

            # 查找爬取文件获取内容
            crawled_data = find_crawled_file(doc.url, settings.crawled_dir)

            if not crawled_data:
                stats["skipped"] += 1
                continue

            content = crawled_data.get("content", "")
            if not content or len(content.strip()) < 50:
                stats["skipped"] += 1
                continue

            try:
                # 删除旧的chunks
                session.query(Chunk).filter(Chunk.doc_id == doc.id).delete()

                # 重新索引
                result = index_service.index_document(
                    doc_id=doc.id,
                    content=content,
                    title=doc.title,
                    source=doc.url,
                    metadata={"url": doc.url},
                    include_summary=False,
                    db=session,
                    source_type="webpage",
                    data_source="system"
                )

                if result["success"]:
                    doc.status = DocumentStatus.indexed
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                    print(f"  Failed: {doc.url[:60]}... - {result.get('error', 'Unknown')}")

            except Exception as e:
                stats["failed"] += 1
                print(f"  Error: {doc.url[:60]}... - {e}")

        session.commit()

    print("\n" + "=" * 60)
    print("Re-indexing Complete!")
    print("=" * 60)
    print(f"Successfully indexed: {stats['success']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")

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
