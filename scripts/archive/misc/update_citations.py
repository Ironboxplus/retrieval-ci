"""
从OpenAlex API获取论文引用次数并更新影响力排名
增加标题匹配验证，避免错误匹配
"""

import json
import time
import math
import requests
from pathlib import Path
from typing import Optional
import random
import re
import sys

# 设置输出编码为utf-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 配置
INPUT_FILE = Path("data/papers/metadata/top_1000_papers.json")
OUTPUT_FILE = Path("data/papers/metadata/top_1000_papers.json")
RANKED_OUTPUT_FILE = Path("data/papers/metadata/papers_ranked_by_impact.json")
PROGRESS_FILE = Path("data/papers/metadata/citation_progress.json")

# OpenAlex API配置
OPENALEX_BASE = "https://api.openalex.org/works"
BASE_DELAY = 0.3  # 稍微增加延迟
MAX_RETRIES = 3
BATCH_SIZE = 100

# 论文类型权重
TYPE_WEIGHTS = {
    "best_paper": 100,
    "oral": 80,
    "spotlight": 60,
    "poster": 40,
    "workshop": 20,
}

# Impact等级阈值
TIER_THRESHOLDS = {
    "S": 80,
    "A": 60,
    "B": 40,
    "C": 0,
}


def normalize_title(title: str) -> str:
    """规范化标题用于比较"""
    # 转小写
    t = title.lower()
    # 移除特殊字符
    t = re.sub(r'[^a-z0-9\s]', '', t)
    # 规范化空格
    t = ' '.join(t.split())
    return t


def title_similarity(title1: str, title2: str) -> float:
    """计算标题相似度（简单的词重叠比例）"""
    words1 = set(normalize_title(title1).split())
    words2 = set(normalize_title(title2).split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def get_citation_count_openalex(title: str, session: requests.Session) -> Optional[int]:
    """
    使用OpenAlex API搜索论文并获取引用次数
    增加标题验证
    """
    for attempt in range(MAX_RETRIES):
        try:
            # 清理标题 - 移除可能导致API错误的特殊字符
            clean_title = title
            for char in ['$', '\\', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}', ',', '.', '+', '*', '&', '%', '#', '@', '/', '-']:
                clean_title = clean_title.replace(char, ' ')
            clean_title = ' '.join(clean_title.split())
            clean_title = clean_title[:150]

            params = {
                "search": clean_title,
                "per-page": 3,  # 获取前3个结果以找最佳匹配
                "select": "title,cited_by_count"
            }

            response = session.get(
                OPENALEX_BASE,
                params=params,
                timeout=30
            )

            if response.status_code == 429:
                wait_time = 5 * (attempt + 1)
                print(f"    [限流] 等待{wait_time}秒...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            if data.get("results") and len(data["results"]) > 0:
                # 找最匹配的结果
                best_match = None
                best_similarity = 0.0

                for work in data["results"]:
                    work_title = work.get("title", "")
                    similarity = title_similarity(title, work_title)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = work

                # 只有相似度足够高才返回引用数
                if best_match and best_similarity >= 0.5:
                    return best_match.get("cited_by_count", 0)
                else:
                    # 相似度不够，可能匹配错误
                    return 0

            return 0

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                print(f"    [超时] 重试...")
                time.sleep(2)
            else:
                return None
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [重试] {e}")
                time.sleep(2)
            else:
                return None
        except json.JSONDecodeError:
            return None

    return None


def calculate_impact_score(paper: dict) -> float:
    """计算影响力分数"""
    paper_type = paper.get("paper_type", "poster")
    type_score = TYPE_WEIGHTS.get(paper_type, 40)

    citations = paper.get("citations", 0) or 0
    citation_score = math.log10(citations + 1) * 15

    year = paper.get("year", 2024)
    recency_score = (year - 2018) * 2

    impact_score = type_score * 0.5 + citation_score * 0.4 + recency_score * 0.1

    return round(impact_score, 2)


def get_impact_tier(score: float) -> str:
    for tier, threshold in TIER_THRESHOLDS.items():
        if score >= threshold:
            return tier
    return "C"


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": 0, "citations": {}}


def save_progress(processed: int, citations: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed": processed, "citations": citations}, f)


def main():
    print("=" * 60)
    print("OpenAlex 引用次数获取与影响力排名更新")
    print("=" * 60)

    # 读取论文数据
    print(f"\n[1/6] 读取论文数据: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        papers = json.load(f)

    total_papers = len(papers)
    print(f"  共 {total_papers} 篇论文")

    # 加载进度
    progress = load_progress()
    start_idx = progress["processed"]
    citations_cache = progress["citations"]

    if start_idx > 0:
        print(f"  发现进度文件，从第 {start_idx + 1} 篇继续...")

    # 获取引用次数
    print(f"\n[2/6] 从OpenAlex API获取引用次数...")
    print(f"  基础延迟: {BASE_DELAY}秒")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "AcademicResearchTool/1.0 (mailto:research@example.com)",
        "Accept": "application/json"
    })

    success_count = len([c for c in citations_cache.values() if c is not None and c > 0])
    failed_count = 0

    try:
        for i in range(start_idx, total_papers):
            paper = papers[i]
            paper_id = paper.get("id", "")
            title = paper.get("title", "")

            if paper_id in citations_cache:
                continue

            short_title = title[:50] + "..." if len(title) > 50 else title
            print(f"  [{i + 1}/{total_papers}] {short_title}")

            # 使用OpenAlex获取引用
            citation_count = get_citation_count_openalex(title, session)

            if citation_count is not None:
                citations_cache[paper_id] = citation_count
                if citation_count > 0:
                    success_count += 1
                print(f"    -> 引用: {citation_count}")
            else:
                citations_cache[paper_id] = 0
                failed_count += 1
                print(f"    -> 未找到，设为0")

            # 打印进度并保存
            if (i + 1) % BATCH_SIZE == 0:
                print(f"\n  === 进度: {i + 1}/{total_papers} ({(i+1)/total_papers*100:.1f}%) ===")
                print(f"  有引用: {success_count}, 无引用/未找到: {total_papers - success_count}")
                save_progress(i + 1, citations_cache)
                print(f"  进度已保存\n")

            # 请求延迟
            time.sleep(BASE_DELAY + random.uniform(0, 0.1))

    except KeyboardInterrupt:
        print(f"\n\n用户中断，保存进度...")
        save_progress(i, citations_cache)
        print(f"进度已保存到 {PROGRESS_FILE}")
        return

    except Exception as e:
        print(f"\n\n发生错误: {e}")
        save_progress(i, citations_cache)
        print(f"进度已保存到 {PROGRESS_FILE}")
        raise

    # 最终进度保存
    save_progress(total_papers, citations_cache)
    print(f"\n  完成! 有引用论文: {success_count}, 无引用/未找到: {total_papers - success_count}")

    # 更新论文引用数据
    print(f"\n[3/6] 更新论文引用数据并计算影响力分数...")
    for paper in papers:
        paper_id = paper.get("id", "")
        paper["citations"] = citations_cache.get(paper_id, 0) or 0
        paper["impact_score"] = calculate_impact_score(paper)

    # 按影响力分数排名
    print(f"\n[4/6] 按影响力分数排名...")
    papers.sort(key=lambda x: x.get("impact_score", 0), reverse=True)

    # 更新排名和等级
    for rank, paper in enumerate(papers, 1):
        paper["impact_rank"] = rank
        paper["impact_tier"] = get_impact_tier(paper["impact_score"])
        paper["is_high_impact"] = paper["impact_tier"] in ["S", "A"]

    # 保存更新后的数据
    print(f"\n[5/6] 保存更新后的数据...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {OUTPUT_FILE}")

    with open(RANKED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {RANKED_OUTPUT_FILE}")

    # 打印Top 20
    print(f"\n[6/6] Top 20 高影响力论文:")
    print("=" * 110)
    print(f"{'排名':<5} {'引用':<8} {'分数':<8} {'等级':<5} {'类型':<12} {'年份':<6} {'标题'}")
    print("-" * 110)

    for paper in papers[:20]:
        title = paper.get("title", "")[:55] + "..." if len(paper.get("title", "")) > 58 else paper.get("title", "")
        print(f"{paper['impact_rank']:<5} "
              f"{paper.get('citations', 0):<8} "
              f"{paper['impact_score']:<8.2f} "
              f"{paper['impact_tier']:<5} "
              f"{paper.get('paper_type', 'N/A'):<12} "
              f"{paper.get('year', 'N/A'):<6} "
              f"{title}")

    print("=" * 110)

    # 统计信息
    tier_counts = {}
    for paper in papers:
        tier = paper.get("impact_tier", "C")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print(f"\n等级分布:")
    for tier in ["S", "A", "B", "C"]:
        count = tier_counts.get(tier, 0)
        print(f"  {tier}级: {count} 篇 ({count/total_papers*100:.1f}%)")

    citations_list = [p.get("citations", 0) for p in papers]
    print(f"\n引用统计:")
    print(f"  总引用: {sum(citations_list)}")
    print(f"  平均引用: {sum(citations_list)/len(citations_list):.1f}")
    print(f"  最高引用: {max(citations_list)}")
    print(f"  中位数引用: {sorted(citations_list)[len(citations_list)//2]}")

    print("\n完成!")


if __name__ == "__main__":
    main()
