#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文优先级排序脚本
结合 Google Scholar 和 OpenAlex 获取引用数，计算影响力分数并排序

功能:
1. 读取论文元数据
2. 按优先级获取引用数 (Google Scholar 优先，OpenAlex 回退)
3. 计算影响力分数
4. 生成排序结果并划分等级

使用方法:
    python scripts/update_paper_priority.py
    python scripts/update_paper_priority.py --source google_scholar --limit 50
    python scripts/update_paper_priority.py --force --source both
"""

import argparse
import json
import math
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置输出编码为 utf-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 导入引用获取器
from citation_fetcher import CitationFetcher, CitationFetcherConfig

# ==================== 配置 ====================

# 文件路径
INPUT_FILE = PROJECT_ROOT / "data/papers/metadata/top_1000_papers.json"
OUTPUT_FILE = PROJECT_ROOT / "data/papers/metadata/papers_ranked_by_impact.json"
PROGRESS_FILE = PROJECT_ROOT / "data/papers/metadata/priority_progress.json"

# 论文类型权重 (复用 update_citations.py)
TYPE_WEIGHTS = {
    "best_paper": 100,
    "oral": 80,
    "spotlight": 60,
    "poster": 40,
    "workshop": 20,
}

# Impact 等级阈值 (复用 update_citations.py)
TIER_THRESHOLDS = {
    "S": 80,
    "A": 60,
    "B": 40,
    "C": 0,
}

# 引用来源类型
SourceType = Literal["google_scholar", "openalex", "both"]


@dataclass
class FetchResult:
    """引用获取结果"""
    citations: int
    source: str  # "google_scholar", "openalex", "fallback", "cached"
    success: bool


def calculate_impact_score(paper: dict) -> float:
    """计算影响力分数

    公式: impact_score = type_score * 0.5 + log10(citations + 1) * 15 * 0.4 + (year - 2018) * 2 * 0.1

    Args:
        paper: 论文数据字典

    Returns:
        影响力分数
    """
    paper_type = paper.get("paper_type", "poster")
    type_score = TYPE_WEIGHTS.get(paper_type, 40)

    citations = paper.get("citations", 0) or 0
    citation_score = math.log10(citations + 1) * 15

    year = paper.get("year", 2024)
    recency_score = (year - 2018) * 2

    impact_score = type_score * 0.5 + citation_score * 0.4 + recency_score * 0.1

    return round(impact_score, 2)


def get_impact_tier(score: float) -> str:
    """根据分数获取等级

    Args:
        score: 影响力分数

    Returns:
        等级 (S/A/B/C)
    """
    for tier, threshold in TIER_THRESHOLDS.items():
        if score >= threshold:
            return tier
    return "C"


def load_progress() -> dict:
    """加载进度文件"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": 0, "citations": {}, "sources": {}}


def save_progress(processed: int, citations: dict, sources: dict):
    """保存进度文件"""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "processed": processed,
            "citations": citations,
            "sources": sources,
            "updated_at": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)


def fetch_citation_with_fallback(
    title: str,
    paper_id: str,
    fetcher_gs: Optional[CitationFetcher],
    fetcher_oa: CitationFetcher,
    source_mode: SourceType,
) -> FetchResult:
    """按优先级获取引用数

    策略:
    - google_scholar: 仅使用 Google Scholar
    - openalex: 仅使用 OpenAlex
    - both: 优先 Google Scholar，失败时回退到 OpenAlex

    Args:
        title: 论文标题
        paper_id: 论文 ID
        fetcher_gs: Google Scholar 获取器
        fetcher_oa: OpenAlex 获取器
        source_mode: 来源模式

    Returns:
        FetchResult 包含引用数、来源和成功状态
    """
    if source_mode == "openalex":
        count = fetcher_oa.get_citation_count(title)
        return FetchResult(
            citations=count,
            source="openalex",
            success=count > 0
        )

    elif source_mode == "google_scholar":
        if fetcher_gs is None:
            print("    [警告] Google Scholar 不可用，跳过")
            return FetchResult(citations=0, source="unavailable", success=False)

        count = fetcher_gs.get_citation_count(title)
        return FetchResult(
            citations=count,
            source="google_scholar",
            success=count > 0
        )

    elif source_mode == "both":
        # 优先 Google Scholar
        if fetcher_gs is not None:
            try:
                gs_count = fetcher_gs.get_citation_count(title)
                if gs_count is not None and gs_count > 0:
                    return FetchResult(
                        citations=gs_count,
                        source="google_scholar",
                        success=True
                    )
            except Exception as e:
                print(f"    [GS失败] {e}")

        # 回退到 OpenAlex
        oa_count = fetcher_oa.get_citation_count(title)
        return FetchResult(
            citations=oa_count,
            source="openalex" if oa_count > 0 else "fallback",
            success=oa_count > 0
        )

    return FetchResult(citations=0, source="unknown", success=False)


def print_top_papers(papers: List[dict], count: int = 20):
    """打印 Top N 高影响力论文

    Args:
        papers: 已排序的论文列表
        count: 显示数量
    """
    print(f"\n{'='*120}")
    print(f"Top {count} 高影响力论文:")
    print(f"{'='*120}")
    print(f"{'排名':<5} {'引用':<8} {'分数':<8} {'等级':<5} {'来源':<15} {'类型':<12} {'年份':<6} {'标题'}")
    print(f"{'-'*120}")

    for paper in papers[:count]:
        title = paper.get("title", "")
        if len(title) > 50:
            title = title[:50] + "..."

        print(
            f"{paper.get('impact_rank', 'N/A'):<5} "
            f"{paper.get('citations', 0):<8} "
            f"{paper.get('impact_score', 0):<8.2f} "
            f"{paper.get('impact_tier', 'N/A'):<5} "
            f"{paper.get('citation_source', 'N/A'):<15} "
            f"{paper.get('paper_type', 'N/A'):<12} "
            f"{paper.get('year', 'N/A'):<6} "
            f"{title}"
        )

    print(f"{'='*120}")


def print_tier_statistics(papers: List[dict]):
    """打印等级分布统计

    Args:
        papers: 论文列表
    """
    total = len(papers)

    # 等级统计
    tier_counts = {}
    for paper in papers:
        tier = paper.get("impact_tier", "C")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print(f"\n等级分布:")
    for tier in ["S", "A", "B", "C"]:
        count = tier_counts.get(tier, 0)
        bar = "#" * int(count / total * 50)
        print(f"  {tier}级: {count:>4} 篇 ({count/total*100:>5.1f}%) {bar}")

    # 来源统计
    source_counts = {}
    for paper in papers:
        source = paper.get("citation_source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"\n引用来源分布:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        bar = "#" * int(count / total * 50)
        print(f"  {source:<15}: {count:>4} 篇 ({count/total*100:>5.1f}%) {bar}")

    # 引用统计
    citations_list = [p.get("citations", 0) for p in papers]
    print(f"\n引用统计:")
    print(f"  总引用数: {sum(citations_list):,}")
    print(f"  平均引用: {sum(citations_list)/len(citations_list):.1f}")
    print(f"  最高引用: {max(citations_list):,}")
    print(f"  中位数引用: {sorted(citations_list)[len(citations_list)//2]:,}")

    # 高影响力论文统计
    high_impact = sum(1 for p in papers if p.get("is_high_impact", False))
    print(f"\n高影响力论文 (S/A级): {high_impact} 篇 ({high_impact/total*100:.1f}%)")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="论文优先级排序脚本 - 获取引用数并计算影响力分数"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["google_scholar", "openalex", "both"],
        default="both",
        help="引用来源 (默认: both - 优先 Google Scholar，回退 OpenAlex)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新获取所有引用 (忽略缓存)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="限制处理数量 (用于测试，0 表示全部)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_FILE),
        help=f"输入文件路径 (默认: {INPUT_FILE})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help=f"输出文件路径 (默认: {OUTPUT_FILE})"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    print("=" * 70)
    print("论文优先级排序脚本")
    print(f"引用来源: {args.source}")
    print(f"强制刷新: {args.force}")
    print(f"处理限制: {'全部' if args.limit == 0 else args.limit}")
    print("=" * 70)

    # [1/6] 读取论文数据
    print(f"\n[1/6] 读取论文数据: {input_file}")

    if not input_file.exists():
        print(f"  [错误] 文件不存在: {input_file}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    total_papers = len(papers)
    if args.limit > 0:
        papers = papers[:args.limit]
        print(f"  共 {total_papers} 篇论文，限制处理 {len(papers)} 篇")
    else:
        print(f"  共 {total_papers} 篇论文")

    # 加载进度
    progress = load_progress()

    if args.force:
        print("  [强制刷新] 清空缓存...")
        citations_cache = {}
        sources_cache = {}
        start_idx = 0
    else:
        citations_cache = progress.get("citations", {})
        sources_cache = progress.get("sources", {})
        start_idx = progress.get("processed", 0)
        if start_idx > 0 and start_idx < len(papers):
            print(f"  发现进度文件，从第 {start_idx + 1} 篇继续...")

    # [2/6] 初始化引用获取器
    print(f"\n[2/6] 初始化引用获取器...")

    # OpenAlex 获取器 (始终可用)
    fetcher_oa = CitationFetcher(source="openalex")
    print("  OpenAlex 获取器已就绪")

    # Google Scholar 获取器 (可能不可用)
    fetcher_gs = None
    if args.source in ["google_scholar", "both"]:
        try:
            fetcher_gs = CitationFetcher(source="google_scholar")
            print("  Google Scholar 获取器已就绪")
        except Exception as e:
            print(f"  [警告] Google Scholar 获取器初始化失败: {e}")
            if args.source == "google_scholar":
                print("  [错误] 指定了 google_scholar 但无法初始化")
                sys.exit(1)

    # [3/6] 获取引用数
    print(f"\n[3/6] 获取引用数...")

    success_count = sum(1 for c in citations_cache.values() if c and c > 0)
    failed_count = 0
    batch_size = 50

    try:
        for i in range(start_idx, len(papers)):
            paper = papers[i]
            paper_id = paper.get("id", f"paper_{i}")
            title = paper.get("title", "")

            # 检查缓存
            if paper_id in citations_cache and not args.force:
                continue

            short_title = title[:45] + "..." if len(title) > 45 else title
            print(f"  [{i + 1}/{len(papers)}] {short_title}")

            # 获取引用数
            result = fetch_citation_with_fallback(
                title=title,
                paper_id=paper_id,
                fetcher_gs=fetcher_gs,
                fetcher_oa=fetcher_oa,
                source_mode=args.source,
            )

            citations_cache[paper_id] = result.citations
            sources_cache[paper_id] = result.source

            if result.success:
                success_count += 1
                print(f"    -> 引用: {result.citations} (来源: {result.source})")
            else:
                failed_count += 1
                print(f"    -> 未找到 (来源: {result.source})")

            # 定期保存进度
            if (i + 1) % batch_size == 0:
                print(f"\n  === 进度: {i + 1}/{len(papers)} ({(i+1)/len(papers)*100:.1f}%) ===")
                print(f"  有引用: {success_count}, 无引用: {len(papers) - success_count}")
                save_progress(i + 1, citations_cache, sources_cache)
                print(f"  进度已保存\n")

            # 请求间隔
            time.sleep(random.uniform(0.1, 0.3))

    except KeyboardInterrupt:
        print(f"\n\n[中断] 用户中断，保存进度...")
        save_progress(i, citations_cache, sources_cache)
        print(f"进度已保存到 {PROGRESS_FILE}")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n[错误] {e}")
        save_progress(i, citations_cache, sources_cache)
        print(f"进度已保存到 {PROGRESS_FILE}")
        raise

    # 保存最终进度
    save_progress(len(papers), citations_cache, sources_cache)
    print(f"\n  完成! 有引用: {success_count}, 无引用: {len(papers) - success_count}")

    # [4/6] 计算影响力分数
    print(f"\n[4/6] 计算影响力分数...")

    for paper in papers:
        paper_id = paper.get("id", "")
        paper["citations"] = citations_cache.get(paper_id, 0) or 0
        paper["citation_source"] = sources_cache.get(paper_id, "unknown")
        paper["impact_score"] = calculate_impact_score(paper)

    # [5/6] 排序并划分等级
    print(f"\n[5/6] 按影响力分数排序并划分等级...")

    papers.sort(key=lambda x: x.get("impact_score", 0), reverse=True)

    for rank, paper in enumerate(papers, 1):
        paper["impact_rank"] = rank
        paper["impact_tier"] = get_impact_tier(paper["impact_score"])
        paper["is_high_impact"] = paper["impact_tier"] in ["S", "A"]

    # 保存结果
    print(f"\n  保存排序结果...")

    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {output_file}")

    # [6/6] 输出统计结果
    print(f"\n[6/6] 统计结果")

    # 打印 Top 20
    print_top_papers(papers, count=20)

    # 打印等级分布
    print_tier_statistics(papers)

    # 清理资源
    fetcher_oa.close()
    if fetcher_gs:
        fetcher_gs.close()

    print(f"\n{'='*70}")
    print("完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
