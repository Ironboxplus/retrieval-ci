"""
NeurIPSScraper - NeurIPS爬虫

从NeurIPS官网获取论文信息。
"""

from typing import List, Optional, Dict, Any

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None  # type: ignore

from pipeline.paper_crawler.scrapers.base_scraper import BaseScraper
from pipeline.paper_crawler.models.paper import Paper, Author, PaperMetadata, PaperSource


class NeurIPSScraper(BaseScraper):
    """NeurIPS论文爬虫"""
    
    BASE_URL = "https://proceedings.neurips.cc"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def search(self, query: str, limit: int = 100) -> List[Paper]:
        """搜索NeurIPS论文
        
        注意: NeurIPS官网不支持直接搜索，建议使用OpenReview或Semantic Scholar
        
        Args:
            query: 搜索关键词
            limit: 最大返回数量
            
        Returns:
            论文列表
        """
        self.logger.warning("NeurIPS official site does not support search. Use OpenReview or Semantic Scholar instead.")
        return []
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """获取单篇NeurIPS论文
        
        Args:
            paper_id: 论文哈希ID
            
        Returns:
            论文对象或None
        """
        url = f"{self.BASE_URL}/paper_files/paper/{paper_id}"
        response = self.get(url)
        
        if response is None:
            return None
        
        return self._parse_paper_page(response.text, paper_id)
    
    def get_year_papers(self, year: int) -> List[Paper]:
        """获取特定年份的所有NeurIPS论文
        
        Args:
            year: 年份
            
        Returns:
            论文列表
        """
        url = f"{self.BASE_URL}/paper/{year}"
        response = self.get(url)
        
        if response is None:
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        papers = []
        
        for link in soup.select("a[href*='/paper/']"):
            href = link.get("href", "")
            if f"/paper/{year}/" in href and "/hash/" in href:
                paper_hash = href.split("/hash/")[-1].split("-")[0]
                paper = self.get_paper(f"{year}/hash/{paper_hash}")
                if paper:
                    papers.append(paper)
        
        return papers
    
    def _parse_paper_page(self, html: str, paper_id: str) -> Optional[Paper]:
        """解析论文详情页"""
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            title_elem = soup.select_one("h4")
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # 解析作者
            author_elem = soup.select_one("h4 + p i")
            author_text = author_elem.get_text(strip=True) if author_elem else ""
            authors = [Author(name=name.strip()) for name in author_text.split(",")]
            
            # 解析摘要
            abstract_elem = soup.select_one("h4:contains('Abstract') + p")
            abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
            
            # 提取年份
            year = None
            if "/" in paper_id:
                year_str = paper_id.split("/")[0]
                if year_str.isdigit():
                    year = int(year_str)
            
            # PDF链接
            pdf_link = soup.select_one("a[href*='.pdf']")
            pdf_url = None
            if pdf_link:
                pdf_url = pdf_link.get("href", "")
                if not pdf_url.startswith("http"):
                    pdf_url = f"{self.BASE_URL}{pdf_url}"
            
            metadata = PaperMetadata(
                venue="NeurIPS",
                year=year,
            )
            
            return Paper(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                source=PaperSource.NEURIPS,
                url=f"{self.BASE_URL}/paper_files/paper/{paper_id}",
                pdf_url=pdf_url,
                metadata=metadata,
            )
        except Exception as e:
            self.logger.error(f"Failed to parse NeurIPS paper page: {e}")
            return None