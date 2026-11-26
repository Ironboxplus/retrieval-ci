"""
Standalone Quick Test Script for OpenReview Paper Crawler
独立测试脚本 - 直接使用 openreview-py SDK

此脚本不依赖项目中可能不兼容的模块，直接测试 OpenReview API。

使用方式:
    python paper_crawler/test_crawl.py
"""

import sys
import json
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
log_file = log_dir / f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def log(msg: str, level: str = "INFO"):
    """简单日志函数"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] [{level}] {msg}"
    print(log_line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


@dataclass
class SimplePaper:
    """简化的论文数据类"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    venue: str
    year: int
    url: str
    pdf_url: str
    keywords: List[str]
    award: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SimpleOpenReviewCrawler:
    """简化的 OpenReview 爬虫，直接使用 openreview-py SDK"""

    def __init__(self):
        from openreview.api import OpenReviewClient
        self.client = OpenReviewClient(baseurl="https://api2.openreview.net")
        log("Connected to OpenReview API")

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
        templates = {
            "ICLR": "ICLR.cc/{year}/Conference",
            "ICML": "ICML.cc/{year}/Conference",
            "NEURIPS": "NeurIPS.cc/{year}/Conference",
        }
        conf_upper = conference.upper()
        if conf_upper not in templates:
            raise ValueError(f"Unsupported conference: {conference}")
        return templates[conf_upper].format(year=year)

    def get_papers(self, conference: str, year: int, limit: int = 100) -> List[SimplePaper]:
        """获取指定会议年份的论文"""
        venue = self.get_venue(conference, year)
        invitation = f"{venue}/-/Submission"

        log(f"Fetching {conference} {year} papers (invitation: {invitation})...")

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
                log("No papers found with any invitation format", "ERROR")
                return []

        log(f"Found {len(notes) if notes else 0} submissions")

        papers = []
        for note in (notes or [])[:limit]:
            try:
                content = note.content

                title = self._extract_field(content, "title")
                if not title:
                    continue

                abstract = self._extract_field(content, "abstract") or ""
                authors = self._extract_list_field(content, "authors")
                keywords = self._extract_list_field(content, "keywords")
                venue_str = self._extract_field(content, "venue") or f"{conference} {year}"

                # 检查是否是 Best Paper / Award
                award = None
                venue_lower = venue_str.lower()
                if any(kw in venue_lower for kw in ["best paper", "outstanding", "award"]):
                    award = venue_str

                paper = SimplePaper(
                    id=note.id,
                    title=title,
                    authors=authors,
                    abstract=abstract[:500] + "..." if len(abstract) > 500 else abstract,
                    venue=venue_str,
                    year=year,
                    url=f"https://openreview.net/forum?id={note.id}",
                    pdf_url=f"https://openreview.net/pdf?id={note.id}",
                    keywords=keywords,
                    award=award,
                )
                papers.append(paper)

            except Exception as e:
                log(f"Failed to parse note {getattr(note, 'id', 'unknown')}: {e}", "WARN")
                continue

        log(f"Parsed {len(papers)} papers successfully")
        return papers

    def get_best_papers(self, conference: str, year: int) -> List[SimplePaper]:
        """获取 Best Paper / Outstanding Paper"""
        log(f"Searching for Best Papers in {conference} {year}...")

        # 获取所有论文，然后筛选有奖项标记的
        all_papers = self.get_papers(conference, year, limit=2000)

        best_keywords = [
            "best paper", "outstanding", "award", "honorable mention",
            "best student", "notable", "top paper"
        ]

        best_papers = []
        for paper in all_papers:
            venue_lower = paper.venue.lower()
            if any(kw in venue_lower for kw in best_keywords):
                paper.award = paper.venue
                best_papers.append(paper)
                log(f"Found award paper: {paper.title[:50]}... ({paper.venue})")

        log(f"Found {len(best_papers)} best/award papers")
        return best_papers

    def get_oral_papers(self, conference: str, year: int) -> List[SimplePaper]:
        """获取 Oral Papers"""
        log(f"Searching for Oral Papers in {conference} {year}...")

        all_papers = self.get_papers(conference, year, limit=2000)

        oral_papers = []
        for paper in all_papers:
            venue_lower = paper.venue.lower()
            if "oral" in venue_lower:
                oral_papers.append(paper)

        log(f"Found {len(oral_papers)} oral papers")
        return oral_papers

    def get_spotlight_papers(self, conference: str, year: int) -> List[SimplePaper]:
        """获取 Spotlight Papers"""
        log(f"Searching for Spotlight Papers in {conference} {year}...")

        all_papers = self.get_papers(conference, year, limit=2000)

        spotlight_papers = []
        for paper in all_papers:
            venue_lower = paper.venue.lower()
            if "spotlight" in venue_lower:
                spotlight_papers.append(paper)

        log(f"Found {len(spotlight_papers)} spotlight papers")
        return spotlight_papers


def test_connection():
    """测试 API 连接"""
    print("=" * 60)
    print("Test 1: OpenReview API Connection")
    print("=" * 60)

    try:
        from openreview.api import OpenReviewClient
        client = OpenReviewClient(baseurl="https://api2.openreview.net")
        log("Successfully connected to OpenReview API!")
        print("[PASS] OpenReview API 连接成功")
        return True
    except Exception as e:
        log(f"Failed to connect: {e}", "ERROR")
        print(f"[FAIL] OpenReview API 连接失败: {e}")
        return False


def test_fetch_papers():
    """测试获取论文"""
    print("\n" + "=" * 60)
    print("Test 2: Fetch ICLR 2024 Papers")
    print("=" * 60)

    try:
        crawler = SimpleOpenReviewCrawler()
        papers = crawler.get_papers("ICLR", 2024, limit=5)

        if papers:
            print(f"\n[PASS] 成功获取 {len(papers)} 篇 ICLR 2024 论文")
            print("\n论文列表:")
            for i, paper in enumerate(papers, 1):
                print(f"\n[{i}] {paper.title}")
                print(f"    作者: {', '.join(paper.authors[:3])}" + (" et al." if len(paper.authors) > 3 else ""))
                print(f"    Venue: {paper.venue}")
                print(f"    URL: {paper.url}")
            return papers
        else:
            print("[WARN] 未找到论文")
            return []
    except Exception as e:
        print(f"[FAIL] 获取论文失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_fetch_best_papers():
    """测试获取 Best Papers"""
    print("\n" + "=" * 60)
    print("Test 3: Fetch ICLR 2024 Best Papers")
    print("=" * 60)

    try:
        crawler = SimpleOpenReviewCrawler()
        best_papers = crawler.get_best_papers("ICLR", 2024)

        if best_papers:
            print(f"\n[PASS] 成功获取 {len(best_papers)} 篇 ICLR 2024 Best Papers")
            print("\nBest Papers:")
            for i, paper in enumerate(best_papers, 1):
                print(f"\n[{i}] {paper.title}")
                print(f"    Award: {paper.award}")
                print(f"    URL: {paper.url}")
        else:
            print("[INFO] ICLR 2024 尚未公布 Best Paper 或未在 venue 中标记")

        return best_papers
    except Exception as e:
        print(f"[FAIL] 获取 Best Papers 失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_fetch_oral_papers():
    """测试获取 Oral Papers"""
    print("\n" + "=" * 60)
    print("Test 4: Fetch ICLR 2024 Oral Papers")
    print("=" * 60)

    try:
        crawler = SimpleOpenReviewCrawler()
        oral_papers = crawler.get_oral_papers("ICLR", 2024)

        if oral_papers:
            print(f"\n[PASS] 成功获取 {len(oral_papers)} 篇 ICLR 2024 Oral Papers")
            print("\nOral Papers (前5篇):")
            for i, paper in enumerate(oral_papers[:5], 1):
                title_preview = paper.title[:60] + "..." if len(paper.title) > 60 else paper.title
                print(f"  [{i}] {title_preview}")
        else:
            print("[INFO] 未找到标记为 Oral 的论文")

        return oral_papers
    except Exception as e:
        print(f"[FAIL] 获取 Oral Papers 失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_results(papers: List[SimplePaper], filename: str):
    """保存结果到 JSON 文件"""
    if not papers:
        return

    output_dir = project_root / "data" / "papers" / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / filename

    papers_data = [p.to_dict() for p in papers]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers_data, f, ensure_ascii=False, indent=2)

    log(f"Results saved to: {output_file}")
    print(f"\n结果已保存: {output_file}")


def main():
    """主测试函数"""
    print("=" * 60)
    print("Paper Crawler 快速测试 (Standalone)")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志文件: {log_file}")
    print()

    log("Starting Paper Crawler Tests...")

    # 测试 1: API 连接
    if not test_connection():
        print("\n[ERROR] API 连接失败，测试终止")
        return False

    # 测试 2: 获取论文
    papers = test_fetch_papers()

    # 测试 3: 获取 Best Papers
    best_papers = test_fetch_best_papers()

    # 测试 4: 获取 Oral Papers
    oral_papers = test_fetch_oral_papers()

    # 保存测试结果
    all_papers = papers + best_papers + oral_papers
    if all_papers:
        # 去重
        seen = set()
        unique_papers = []
        for p in all_papers:
            if p.id not in seen:
                seen.add(p.id)
                unique_papers.append(p)
        save_results(unique_papers, "test_results.json")

    print("\n" + "=" * 60)
    print("测试完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志文件: {log_file}")
    print("=" * 60)

    log("All tests completed!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
