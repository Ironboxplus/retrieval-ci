"""
修复ChromaDB与SQLite数据不一致问题 - 完整版

问题分析:
1. papers collection: 62,995条向量 - SQLite中没有对应chunks (论文数据)
2. webpages collection: 94,732条向量 - 每个doc_id都有2倍的重复向量
3. SQLite chunks: 42,438条 - 这是正确的数据源

修复策略:
1. 清空 papers collection (没有对应SQLite数据)
2. 清空 webpages collection 中的重复向量，只保留与SQLite匹配的
3. 重建 webpages 向量确保一致性
"""
import sys
import os
import time
from typing import List, Dict, Set
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
from chromadb.config import Settings as ChromaSettings
from sqlalchemy import func

from config import settings
from core.db.database import get_db_session
from core.db.models import Chunk, Document, DocumentStatus


def get_chroma_client():
    """获取ChromaDB客户端"""
    return chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False)
    )


def analyze_all_collections():
    """分析所有collections的状态"""
    print("=" * 60)
    print("数据状态分析")
    print("=" * 60)

    client = get_chroma_client()
    collections = client.list_collections()

    print(f"\n[ChromaDB] 找到 {len(collections)} 个collections:")
    total_vectors = 0
    for col in collections:
        count = col.count()
        total_vectors += count
        print(f"  - {col.name}: {count} 条向量")
    print(f"  总计: {total_vectors} 条向量")

    with get_db_session() as db:
        total_chunks = db.query(Chunk).count()
        indexed_docs = db.query(Document).filter(
            Document.status == DocumentStatus.indexed
        ).count()

    print(f"\n[SQLite]")
    print(f"  indexed文档数: {indexed_docs}")
    print(f"  chunks总数: {total_chunks}")

    print(f"\n[差异]")
    print(f"  ChromaDB总向量 - SQLite chunks = {total_vectors - total_chunks}")

    return {
        "chroma_total": total_vectors,
        "sqlite_chunks": total_chunks,
    }


def get_valid_doc_ids() -> Set[int]:
    """获取SQLite中有效的doc_id集合"""
    with get_db_session() as db:
        # 获取所有有chunks的doc_id
        doc_ids = db.query(Chunk.doc_id).distinct().all()
        return {d.doc_id for d in doc_ids}


def get_sqlite_chunk_ids() -> Set[str]:
    """获取SQLite中所有chunk的ID（用于匹配ChromaDB）"""
    with get_db_session() as db:
        chunks = db.query(Chunk.id).all()
        # ChromaDB中的ID格式可能是 "chunk_{id}" 或者其他格式
        return {f"chunk_{c.id}" for c in chunks}


def analyze_webpages_collection():
    """分析webpages collection中的数据"""
    print("\n[分析] webpages collection...")

    client = get_chroma_client()
    try:
        col = client.get_collection("webpages")
    except:
        print("  webpages collection不存在")
        return None, []

    total = col.count()
    print(f"  向量总数: {total}")

    # 获取所有向量的元数据
    all_data = col.get(include=["metadatas"], limit=total)

    # 按doc_id分组，记录每个向量的ID
    doc_id_vectors = defaultdict(list)
    for i, vector_id in enumerate(all_data["ids"]):
        metadata = all_data["metadatas"][i]
        doc_id = metadata.get("doc_id")
        try:
            doc_id = int(doc_id)
        except (ValueError, TypeError):
            doc_id = -1
        doc_id_vectors[doc_id].append(vector_id)

    print(f"  不同doc_id数: {len(doc_id_vectors)}")

    # 获取SQLite中每个doc_id应有的chunk数
    with get_db_session() as db:
        chunk_counts = db.query(
            Chunk.doc_id, func.count(Chunk.id).label('count')
        ).group_by(Chunk.doc_id).all()
        sqlite_counts = {row.doc_id: row.count for row in chunk_counts}

    # 找出需要删除的向量
    vectors_to_delete = []
    valid_doc_ids = set(sqlite_counts.keys())

    for doc_id, vector_ids in doc_id_vectors.items():
        if doc_id not in valid_doc_ids:
            # doc_id在SQLite中不存在，删除所有向量
            vectors_to_delete.extend(vector_ids)
        else:
            # doc_id存在，但可能有重复
            expected_count = sqlite_counts[doc_id]
            current_count = len(vector_ids)
            if current_count > expected_count:
                # 有多余的向量，删除多余的部分
                # 保留前expected_count个
                vectors_to_delete.extend(vector_ids[expected_count:])

    print(f"  需要删除的向量数: {len(vectors_to_delete)}")

    return col, vectors_to_delete


def delete_vectors_in_batches(collection, vector_ids: List[str], batch_size: int = 5000):
    """分批删除向量"""
    total = len(vector_ids)
    if total == 0:
        return 0

    deleted = 0
    for i in range(0, total, batch_size):
        batch = vector_ids[i:i + batch_size]
        try:
            collection.delete(ids=batch)
            deleted += len(batch)
            if deleted % 10000 == 0 or deleted == total:
                print(f"  已删除 {deleted}/{total} ({100*deleted/total:.1f}%)")
        except Exception as e:
            print(f"  删除批次失败: {e}")

    return deleted


def clear_papers_collection():
    """清空papers collection"""
    print("\n[清理] papers collection...")

    client = get_chroma_client()
    try:
        col = client.get_collection("papers")
        old_count = col.count()
        if old_count > 0:
            client.delete_collection("papers")
            # 重新创建空集合
            client.get_or_create_collection(
                name="papers",
                metadata={"hnsw:space": "cosine"}
            )
        print(f"  已清空: {old_count} -> 0")
        return old_count
    except Exception as e:
        print(f"  papers collection不存在或清理失败: {e}")
        return 0


def verify_sync():
    """验证同步结果"""
    print("\n[验证] 同步结果...")

    client = get_chroma_client()

    with get_db_session() as db:
        sqlite_chunks = db.query(Chunk).count()

    webpages_count = 0
    papers_count = 0
    try:
        webpages_count = client.get_collection("webpages").count()
    except:
        pass
    try:
        papers_count = client.get_collection("papers").count()
    except:
        pass

    total_chroma = webpages_count + papers_count

    print(f"\n[结果]")
    print(f"  SQLite chunks: {sqlite_chunks}")
    print(f"  ChromaDB webpages: {webpages_count}")
    print(f"  ChromaDB papers: {papers_count}")
    print(f"  ChromaDB总计: {total_chroma}")
    print(f"  差异: {total_chroma - sqlite_chunks}")

    if total_chroma == sqlite_chunks:
        print("\n[成功] 数据已完全同步!")
    else:
        print("\n[注意] 仍有差异，可能需要进一步检查")


def main():
    print("=" * 60)
    print("ChromaDB与SQLite数据同步修复工具 (完整版)")
    print("=" * 60)

    start_time = time.time()

    # 1. 分析当前状态
    state = analyze_all_collections()

    # 2. 分析webpages collection
    webpages_col, vectors_to_delete = analyze_webpages_collection()

    # 3. 确认操作
    print("\n" + "=" * 60)
    print("修复方案:")
    print("  1. 清空 papers collection (论文向量，SQLite无对应数据)")
    print(f"  2. 删除 webpages collection 中的 {len(vectors_to_delete)} 条孤立/重复向量")
    print("=" * 60)

    confirm = input("\n确认执行? (输入 'yes' 确认): ")
    if confirm.lower() != 'yes':
        print("[取消] 用户取消操作")
        return

    # 4. 清空papers collection
    clear_papers_collection()

    # 5. 删除webpages中的多余向量
    if webpages_col and vectors_to_delete:
        print("\n[删除] webpages collection中的孤立/重复向量...")
        deleted = delete_vectors_in_batches(webpages_col, vectors_to_delete)
        print(f"  共删除 {deleted} 条向量")

    # 6. 验证结果
    verify_sync()

    elapsed = time.time() - start_time
    print(f"\n[完成] 总耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
