"""
GPU-based Paper Index Builder
使用GPU和轻量级embedding模型构建论文索引

Features:
- 使用CUDA GPU加速embedding生成
- 支持BGE-small或Qwen模型
- 重新计算impact_score考虑Best Paper奖项
- 保存embeddings到numpy文件
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# 检查GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Impact score weights
PAPER_TYPE_WEIGHTS = {
    "best_paper": 100,
    "outstanding": 90,
    "outstanding_paper": 90,
    "oral": 70,
    "spotlight": 50,
    "poster": 30,
    "workshop": 20,
    "unknown": 25,
}

CURRENT_YEAR = 2026


def calculate_impact_score(paper: dict) -> float:
    """计算论文影响力分数

    公式: type_score * 0.5 + year_score * 0.3 + citation_score * 0.2
    """
    # Paper type score (0-100)
    paper_type = paper.get("paper_type", "unknown")
    type_score = PAPER_TYPE_WEIGHTS.get(paper_type, 25)

    # Year score: newer papers get higher scores (0-100)
    year = paper.get("year", 2024)
    years_old = CURRENT_YEAR - year
    year_score = max(0, 100 - years_old * 10)  # -10 per year

    # Citation score (0-100, logarithmic)
    citations = paper.get("citation_count", 0) or 0
    if citations > 0:
        import math
        citation_score = min(100, math.log10(citations + 1) * 30)
    else:
        citation_score = 0

    # Weighted combination
    impact_score = type_score * 0.5 + year_score * 0.3 + citation_score * 0.2

    return round(impact_score, 2)


def get_impact_tier(score: float) -> str:
    """根据分数确定影响力等级"""
    if score >= 80:
        return "S"
    elif score >= 60:
        return "A"
    elif score >= 40:
        return "B"
    else:
        return "C"


def load_embedding_model(model_name: str = "BAAI/bge-small-zh-v1.5"):
    """加载embedding模型到GPU"""
    from sentence_transformers import SentenceTransformer

    print(f"\n加载embedding模型: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    model = SentenceTransformer(model_name, device=device)
    print(f"模型加载完成，维度: {model.get_sentence_embedding_dimension()}")

    return model


def main():
    print("=" * 60)
    print("GPU-based Paper Index Builder")
    print("=" * 60)

    base_path = Path("E:/JETB/pythons/sf/retrival/data/papers/metadata")
    papers_file = base_path / "top_1000_papers.json"
    output_dir = Path("E:/JETB/pythons/sf/retrival/data/index")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载论文数据
    print(f"\n加载论文数据: {papers_file}")
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    print(f"  共 {len(papers)} 篇论文")

    # 1. 重新计算impact scores
    print("\n1. 重新计算影响力分数...")
    for paper in papers:
        paper["impact_score"] = calculate_impact_score(paper)
        paper["impact_tier"] = get_impact_tier(paper["impact_score"])

    # 按分数排序并添加排名
    papers.sort(key=lambda x: x["impact_score"], reverse=True)
    for i, paper in enumerate(papers):
        paper["impact_rank"] = i + 1

    # 打印统计
    type_counts = {}
    tier_counts = {"S": 0, "A": 0, "B": 0, "C": 0}
    for paper in papers:
        pt = paper.get("paper_type", "unknown")
        type_counts[pt] = type_counts.get(pt, 0) + 1
        tier_counts[paper["impact_tier"]] += 1

    print("\n  论文类型分布:")
    for pt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {pt}: {count}")

    print("\n  影响力等级分布:")
    for tier, count in tier_counts.items():
        print(f"    {tier}级: {count}")

    print("\n  Top 10 论文:")
    for i, paper in enumerate(papers[:10]):
        print(f"    {i+1}. [{paper['impact_tier']}] {paper['paper_type']}: {paper['title'][:50]}...")

    # 保存更新后的论文数据
    with open(papers_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"\n  已保存到: {papers_file}")

    # 2. 生成embeddings
    print("\n2. 生成Embeddings...")

    # 尝试不同的模型
    model_options = [
        "BAAI/bge-small-zh-v1.5",  # 中文优化，384维
        "sentence-transformers/all-MiniLM-L6-v2",  # 英文，384维
    ]

    model = None
    for model_name in model_options:
        try:
            model = load_embedding_model(model_name)
            break
        except Exception as e:
            print(f"  加载 {model_name} 失败: {e}")
            continue

    if model is None:
        print("  所有模型加载失败，跳过embedding生成")
        return

    # 准备文本
    texts = []
    paper_ids = []
    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")[:500]  # 截断
        text = f"{title}. {abstract}"
        texts.append(text)
        paper_ids.append(paper.get("id", ""))

    print(f"\n  开始生成 {len(texts)} 个embeddings...")
    start_time = datetime.now()

    # 批量生成embeddings
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embeddings)

        if (i + batch_size) % 100 == 0 or i + batch_size >= len(texts):
            elapsed = (datetime.now() - start_time).total_seconds()
            done = min(i + batch_size, len(texts))
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    [{done}/{len(texts)}] {rate:.1f} papers/s")

    # 合并embeddings
    embeddings_array = np.vstack(all_embeddings)
    print(f"\n  Embeddings shape: {embeddings_array.shape}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"  耗时: {elapsed:.1f}s")

    # 保存embeddings
    embeddings_file = output_dir / "paper_embeddings.npy"
    np.save(embeddings_file, embeddings_array)
    print(f"  Embeddings 保存到: {embeddings_file}")

    # 保存paper_ids映射
    ids_file = output_dir / "paper_ids.json"
    with open(ids_file, 'w', encoding='utf-8') as f:
        json.dump(paper_ids, f)
    print(f"  Paper IDs 保存到: {ids_file}")

    # 3. 构建简单的搜索索引
    print("\n3. 构建搜索索引...")

    search_index = {
        "metadata": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_papers": len(papers),
            "embedding_dim": embeddings_array.shape[1],
            "model_name": model_name,
        },
        "papers": [],
    }

    for i, paper in enumerate(papers):
        search_index["papers"].append({
            "id": paper.get("id"),
            "title": paper.get("title"),
            "conference": paper.get("conference"),
            "year": paper.get("year"),
            "paper_type": paper.get("paper_type"),
            "impact_score": paper.get("impact_score"),
            "impact_tier": paper.get("impact_tier"),
            "impact_rank": paper.get("impact_rank"),
            "is_high_impact": paper.get("is_high_impact", False),
            "authors": [a.get("name", "") for a in paper.get("authors", [])[:3]],
            "keywords": paper.get("keywords", [])[:5],
        })

    index_file = output_dir / "search_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(search_index, f, indent=2, ensure_ascii=False)
    print(f"  搜索索引保存到: {index_file}")

    # 4. 测试搜索
    print("\n4. 测试搜索功能...")

    test_queries = [
        "transformer attention mechanism",
        "large language model",
        "diffusion model image generation",
    ]

    for query in test_queries:
        query_embedding = model.encode([query], convert_to_numpy=True)

        # 计算余弦相似度
        similarities = np.dot(embeddings_array, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:5]

        print(f"\n  查询: '{query}'")
        for idx in top_indices:
            paper = papers[idx]
            sim = similarities[idx]
            print(f"    [{sim:.3f}] [{paper['impact_tier']}] {paper['title'][:50]}...")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)

    # 打印文件大小
    for f in [embeddings_file, ids_file, index_file]:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
