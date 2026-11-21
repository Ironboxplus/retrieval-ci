"""
Full Paper Crawler Script
完整论文爬取脚本 - 获取前1000篇高含金量论文

目标会议和年份:
- ICLR: 2024-2020
- ICML: 2024-2020
- NeurIPS: 2024-2020

论文类型优先级:
1. Best Paper / Outstanding Paper
2. Oral Papers
3. Spotlight Papers

使用方式:
    python paper_crawler/crawl_top_papers.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

# 获取项目根目录
script_dir = Path(__file__).parent
project_root = script_dir.parent

# 配置日志目录
log_dir = project_root / "data" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"full_crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 输出目录
output_dir = project_root / "data" / "papers" / "metadata"
output_dir.mkdir(parents=True, exist_ok=True)


def log(msg: str, level: str = "INFO"):
    """简单日志函数"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] [{level}] {msg}"
    print(log_line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


@dataclass
class TopPaper:
    """高含金量论文数据类"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    venue: str
    conference: str
    year: int
    paper_type: str  # best_paper, oral, spotlight, poster
    url: str
    pdf_url: str
    keywords: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TopPaperCrawler:
    """高含金量论文爬虫"""

    # 会议 venue 模板
    VENUE_TEMPLATES = {
        "ICLR": "ICLR.cc/{year}/Conference",
        "ICML": "ICML.cc/{year}/Conference",
        "NEURIPS": "NeurIPS.cc/{year}/Conference",
    }

    # 支持的年份范围
    SUPPORTED_YEARS = {
        "ICLR": range(2020, 2027),
        "ICML": range(2020, 2027),
        "NEURIPS": range(2020, 2027),
    }

    def __init__(self):
        from openreview.api import OpenReviewClient
        self.client = OpenReviewClient(baseurl="https://api2.openreview.net")
        log("Connected to OpenReview API")

        # 缓存已获取的论文（用于去重）
        self._paper_cache: Dict[str, TopPaper] = {}

    def _extract_field(self, content: Dict, field: str) -> Optional[str]:
        """从内容中提取字段值"""
        if field not in content:
            return None
        value = content[field]
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return str(value) if value else None

    def _extract_list_field(self, content: Dict, field: str) -> List[str]:
        """从内容中提取列表字段"""
        if field not in content:
            return []
        value = content[field]
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, dict) and "value" in value:
            val = value["value"]
            return [str(v) for v in val] if isinstance(val, list) else []
        return []

    def get_venue(self, conference: str, year: int) -> str:
        """获取会议的 venue 标识符"""
        conf_upper = conference.upper()
        if conf_upper not in self.VENUE_TEMPLATES:
            raise ValueError(f"Unsupported conference: {conference}")
        return self.VENUE_TEMPLATES[conf_upper].format(year=year)

    def _classify_paper_type(self, venue_str: str) -> str:
        """根据 venue 字符串分类论文类型"""
        venue_lower = venue_str.lower()

        # Best Paper / Outstanding Paper
        if any(kw in venue_lower for kw in ["best paper", "outstanding", "award"]):
            return "best_paper"

        # Oral
        if "oral" in venue_lower:
            return "oral"

        # Spotlight
        if "spotlight" in venue_lower:
            return "spotlight"

        # Poster
        if "poster" in venue_lower:
            return "poster"

        # Withdrawn or Rejected
        if "withdrawn" in venue_lower or "reject" in venue_lower:
            return "rejected"

        return "unknown"

    def get_all_papers(self, conference: str, year: int) -> List[TopPaper]:
        """获取指定会议年份的所有论文"""
        conf_upper = conference.upper()

        # 检查年份支持
        if year not in self.SUPPORTED_YEARS.get(conf_upper, []):
            log(f"Year {year} not supported for {conf_upper}", "WARN")
            return []

        venue = self.get_venue(conference, year)
        invitation = f"{venue}/-/Submission"

        log(f"Fetching {conf_upper} {year} papers...")

        try:
            notes = self.client.get_all_notes(invitation=invitation, details="original")
        except Exception as e:
            log(f"Primary invitation failed: {e}, trying alternatives...", "WARN")
            # 尝试备用格式
            alt_invitations = [
                f"{venue}/-/Blind_Submission",
                f"{venue}/Paper/-/Submission",
            ]
            notes = None
            for alt_inv in alt_invitations:
                try:
                    notes = self.client.get_all_notes(invitation=alt_inv)
                    if notes:
                        log(f"Found papers with invitation: {alt_inv}")
                        break
                except Exception:
                    continue

            if not notes:
                log(f"No papers found for {conf_upper} {year}", "ERROR")
                return []

        log(f"Found {len(notes) if notes else 0} submissions for {conf_upper} {year}")

        papers = []
        for note in (notes or []):
            try:
                content = note.content

                title = self._extract_field(content, "title")
                if not title:
                    continue

                # 跳过已缓存的论文
                if note.id in self._paper_cache:
                    continue

                abstract = self._extract_field(content, "abstract") or ""
                authors = self._extract_list_field(content, "authors")
                keywords = self._extract_list_field(content, "keywords")
                venue_str = self._extract_field(content, "venue") or f"{conf_upper} {year}"

                paper_type = self._classify_paper_type(venue_str)

                # 跳过被拒绝/撤回的论文
                if paper_type == "rejected":
                    continue

                paper = TopPaper(
                    id=note.id,
                    title=title,
                    authors=authors,
                    abstract=abstract[:1000] if len(abstract) > 1000 else abstract,
                    venue=venue_str,
                    conference=conf_upper,
                    year=year,
                    paper_type=paper_type,
                    url=f"https://openreview.net/forum?id={note.id}",
                    pdf_url=f"https://openreview.net/pdf?id={note.id}",
                    keywords=keywords,
                )

                papers.append(paper)
                self._paper_cache[note.id] = paper

            except Exception as e:
                log(f"Failed to parse note {getattr(note, 'id', 'unknown')}: {e}", "WARN")
                continue

        return papers

    def get_top_papers(self, conference: str, year: int) -> Dict[str, List[TopPaper]]:
        """获取高含金量论文（按类型分组）"""
        all_papers = self.get_all_papers(conference, year)

        grouped = {
            "best_paper": [],
            "oral": [],
            "spotlight": [],
            "poster": [],
        }

        for paper in all_papers:
            if paper.paper_type in grouped:
                grouped[paper.paper_type].append(paper)

        log(f"{conference} {year} - Best: {len(grouped['best_paper'])}, Oral: {len(grouped['oral'])}, "
            f"Spotlight: {len(grouped['spotlight'])}, Poster: {len(grouped['poster'])}")

        return grouped


def crawl_top_1000_papers():
    """爬取前 1000 篇高含金量论文"""
    log("=" * 60)
    log("Starting Full Paper Crawl - Target: 1000 Top Papers")
    log("=" * 60)

    crawler = TopPaperCrawler()

    # 目标会议和年份
    conferences = ["ICLR", "ICML", "NEURIPS"]
    years = [2024, 2023, 2022, 2021, 2020]

    # 收集论文
    all_best_papers: List[TopPaper] = []
    all_oral_papers: List[TopPaper] = []
    all_spotlight_papers: List[TopPaper] = []
    all_poster_papers: List[TopPaper] = []

    total_conf_years = len(conferences) * len(years)
    current = 0

    for conf in conferences:
        for year in years:
            current += 1
            log(f"[{current}/{total_conf_years}] Processing {conf} {year}...")

            try:
                grouped = crawler.get_top_papers(conf, year)

                all_best_papers.extend(grouped["best_paper"])
                all_oral_papers.extend(grouped["oral"])
                all_spotlight_papers.extend(grouped["spotlight"])
                all_poster_papers.extend(grouped["poster"])

                # 添加延迟避免请求过快
                time.sleep(1)

            except Exception as e:
                log(f"Failed to crawl {conf} {year}: {e}", "ERROR")
                continue

    # 统计结果
    log("")
    log("=" * 60)
    log("Crawl Statistics")
    log("=" * 60)
    log(f"Best Papers: {len(all_best_papers)}")
    log(f"Oral Papers: {len(all_oral_papers)}")
    log(f"Spotlight Papers: {len(all_spotlight_papers)}")
    log(f"Poster Papers: {len(all_poster_papers)}")

    # 按优先级组合论文列表
    # 优先级: Best Paper > Oral > Spotlight > (Poster)
    top_papers: List[TopPaper] = []

    # 1. 添加所有 Best Papers
    top_papers.extend(all_best_papers)
    log(f"Added {len(all_best_papers)} best papers")

    # 2. 添加所有 Oral Papers
    top_papers.extend(all_oral_papers)
    log(f"Added {len(all_oral_papers)} oral papers")

    # 3. 添加所有 Spotlight Papers
    top_papers.extend(all_spotlight_papers)
    log(f"Added {len(all_spotlight_papers)} spotlight papers")

    # 4. 如果不足 1000 篇，从 Poster 中补充
    remaining = 1000 - len(top_papers)
    if remaining > 0:
        # 按年份排序 Poster（优先新论文）
        poster_sorted = sorted(all_poster_papers, key=lambda p: p.year, reverse=True)
        top_papers.extend(poster_sorted[:remaining])
        log(f"Added {min(remaining, len(poster_sorted))} poster papers to reach target")

    # 最终去重
    seen_ids = set()
    unique_papers = []
    for paper in top_papers:
        if paper.id not in seen_ids:
            seen_ids.add(paper.id)
            unique_papers.append(paper)

    log(f"\nTotal unique papers: {len(unique_papers)}")

    # 截取前 1000 篇
    final_papers = unique_papers[:1000]
    log(f"Final selection: {len(final_papers)} papers")

    return final_papers


def save_papers(papers: List[TopPaper], filename: str = "top_1000_papers.json"):
    """保存论文到 JSON 文件"""
    output_file = output_dir / filename

    papers_data = [p.to_dict() for p in papers]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers_data, f, ensure_ascii=False, indent=2)

    log(f"Papers saved to: {output_file}")
    return output_file


def generate_summary(papers: List[TopPaper]) -> Dict:
    """生成爬取结果摘要"""
    summary = {
        "total_papers": len(papers),
        "by_conference": {},
        "by_year": {},
        "by_type": {},
        "crawl_time": datetime.now().isoformat(),
    }

    for paper in papers:
        # 按会议统计
        conf = paper.conference
        if conf not in summary["by_conference"]:
            summary["by_conference"][conf] = 0
        summary["by_conference"][conf] += 1

        # 按年份统计
        year = str(paper.year)
        if year not in summary["by_year"]:
            summary["by_year"][year] = 0
        summary["by_year"][year] += 1

        # 按类型统计
        ptype = paper.paper_type
        if ptype not in summary["by_type"]:
            summary["by_type"][ptype] = 0
        summary["by_type"][ptype] += 1

    return summary


def main():
    """主函数"""
    start_time = datetime.now()

    print("=" * 60)
    print("Top Paper Crawler - 获取前1000篇高含金量论文")
    print("=" * 60)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志文件: {log_file}")
    print()

    try:
        # 爬取论文
        papers = crawl_top_1000_papers()

        # 保存结果
        output_file = save_papers(papers)

        # 生成摘要
        summary = generate_summary(papers)
        summary_file = output_dir / "crawl_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log(f"Summary saved to: {summary_file}")

        # 打印摘要
        print("\n" + "=" * 60)
        print("爬取结果摘要")
        print("=" * 60)
        print(f"总论文数: {summary['total_papers']}")

        print("\n按会议统计:")
        for conf, count in summary["by_conference"].items():
            print(f"  {conf}: {count}")

        print("\n按年份统计:")
        for year, count in sorted(summary["by_year"].items(), reverse=True):
            print(f"  {year}: {count}")

        print("\n按类型统计:")
        for ptype, count in summary["by_type"].items():
            print(f"  {ptype}: {count}")

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n耗时: {duration}")
        print(f"输出文件: {output_file}")
        print(f"日志文件: {log_file}")
        print("=" * 60)

        log(f"Crawl completed in {duration}")
        return True

    except Exception as e:
        log(f"Crawl failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
