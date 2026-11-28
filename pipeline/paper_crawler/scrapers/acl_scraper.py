"""
ACLScraper - ACL Anthology Paper Scraper

Scrapes papers from ACL Anthology (https://aclanthology.org/).
Supports ACL, EMNLP, NAACL, EACL, COLING, and other *CL venues.

Features:
- Fetch papers by venue and year
- Get Best Paper / Outstanding Paper awards
- Support for official acl-anthology Python library (optional)
- Web scraping fallback with BeautifulSoup
- Async-first design with sync helper methods

Installation for official library support:
    pip install acl-anthology

Usage:
    scraper = ACLScraper()
    papers = await scraper.get_paper_list("ACL", 2024)
    best_papers = await scraper.get_best_papers("ACL", 2024)
"""

import re
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None  # type: ignore

try:
    from acl_anthology import Anthology
    ACL_ANTHOLOGY_AVAILABLE = True
except ImportError:
    ACL_ANTHOLOGY_AVAILABLE = False
    Anthology = None  # type: ignore

from pipeline.paper_crawler.scrapers.base_scraper import BaseScraper, ScraperError, NotFoundError
from pipeline.paper_crawler.models.paper import Paper, Author, PaperType, Conference


class ACLScraper(BaseScraper):
    """ACL Anthology Paper Scraper

    Scrapes papers from ACL Anthology with support for:
    - Multiple venues (ACL, EMNLP, NAACL, EACL, COLING, etc.)
    - Best Paper / Outstanding Paper detection
    - Official acl-anthology library (optional)
    - Web scraping fallback

    Attributes:
        BASE_URL: ACL Anthology base URL
        VENUE_MAPPING: Mapping of conference names to venue identifiers
        TRACK_MAPPING: Mapping of track names to paper types
    """

    BASE_URL = "https://aclanthology.org"

    # Mapping of conference names to venue identifiers used in paper IDs
    VENUE_MAPPING = {
        "ACL": "acl",
        "EMNLP": "emnlp",
        "NAACL": "naacl",
        "EACL": "eacl",
        "COLING": "coling",
        "TACL": "tacl",
        "CL": "cl",
        "AACL": "aacl",
        "IJCNLP": "ijcnlp",
        "CONLL": "conll",
        "FINDINGS": "findings",
    }

    # Mapping of track identifiers to paper types
    TRACK_MAPPING = {
        "long": PaperType.ORAL,  # Long papers are often orals
        "short": PaperType.POSTER,  # Short papers are often posters
        "main": PaperType.ORAL,
        "demos": PaperType.POSTER,
        "srw": PaperType.WORKSHOP,  # Student Research Workshop
        "tutorials": PaperType.WORKSHOP,
        "findings": PaperType.POSTER,
        "industry": PaperType.POSTER,
    }

    # Conference enum mapping
    CONFERENCE_MAPPING = {
        "acl": Conference.ACL,
        "emnlp": Conference.EMNLP,
        "naacl": Conference.NAACL,
        "eacl": Conference.OTHER,  # EACL not in Conference enum
        "coling": Conference.OTHER,
        "tacl": Conference.OTHER,
        "cl": Conference.OTHER,
    }

    def __init__(
        self,
        use_official_library: bool = False,
        log_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize ACL Scraper

        Args:
            use_official_library: Whether to use the official acl-anthology library.
                                  Requires pip install acl-anthology and git.
            log_dir: Directory for log files
            **kwargs: Additional arguments passed to BaseScraper
        """
        super().__init__(**kwargs)

        self.use_official_library = use_official_library and ACL_ANTHOLOGY_AVAILABLE
        self._anthology = None

        # Set up logging
        self.log_dir = log_dir or Path("data/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._setup_file_logger()

        if self.use_official_library:
            self.logger.info("Using official acl-anthology library")
        else:
            self.logger.info("Using web scraping mode")
            if not BS4_AVAILABLE:
                self.logger.warning("BeautifulSoup not available. Install with: pip install beautifulsoup4")

    def _setup_file_logger(self):
        """Set up file logging for debug mode"""
        log_file = self.log_dir / f"acl_scraper_{datetime.now().strftime('%Y-%m-%d')}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        self.logger.addHandler(file_handler)

    def _log(self, level: str, message: str, **kwargs):
        """Log message with additional context"""
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        full_message = f"{message} | {extra}" if extra else message
        getattr(self.logger, level)(full_message)

    # ==================== Abstract Property Implementations ====================

    @property
    def source_name(self) -> str:
        """Data source name."""
        return "acl_anthology"

    @property
    def base_url(self) -> str:
        """API base URL."""
        return self.BASE_URL

    @property
    def supported_conferences(self) -> List[str]:
        """List of supported conferences."""
        return list(self.VENUE_MAPPING.keys())

    # ==================== Abstract Method Implementations ====================

    async def get_paper_list(
        self,
        conference: str,
        year: int,
        paper_type: Optional[PaperType] = None,
        track: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Paper]:
        """Get list of papers for a conference/year.

        Args:
            conference: Conference name (e.g., "ACL", "EMNLP")
            year: Publication year
            paper_type: Filter by paper type (optional)
            track: Filter by track (e.g., "long", "short", "findings")
            limit: Maximum number of papers to return
            offset: Number of papers to skip

        Returns:
            List of Paper objects
        """
        self._log("info", f"Fetching paper list", conference=conference, year=year, track=track)

        venue = self.VENUE_MAPPING.get(conference.upper(), conference.lower())

        if self.use_official_library:
            return await self._get_papers_via_library(venue, year, track, limit, offset)
        else:
            return self._get_papers_via_web(venue, year, track, limit, offset)

    async def get_paper_details(self, paper_id: str) -> Paper:
        """Get paper details by ID.

        Args:
            paper_id: ACL Anthology paper ID (e.g., "2024.acl-long.1")

        Returns:
            Paper object with full details

        Raises:
            NotFoundError: If paper not found
        """
        self._log("debug", f"Fetching paper details", paper_id=paper_id)

        if self.use_official_library:
            paper = await self._get_paper_via_library(paper_id)
        else:
            paper = self._get_paper_via_web(paper_id)

        if paper is None:
            raise NotFoundError(f"Paper not found: {paper_id}", source=self.source_name)

        return paper

    async def get_pdf_url(self, paper: Paper) -> str:
        """Get PDF download URL for a paper.

        Args:
            paper: Paper object

        Returns:
            PDF download URL
        """
        if paper.pdf_url:
            return str(paper.pdf_url)
        return f"{self.BASE_URL}/{paper.id}.pdf"

    # ==================== Best Paper Methods ====================

    async def get_best_papers(
        self,
        conference: str,
        year: int,
        include_honorable_mentions: bool = True
    ) -> List[Paper]:
        """Get Best Paper and Outstanding Paper awards.

        Args:
            conference: Conference name (e.g., "ACL", "EMNLP")
            year: Publication year
            include_honorable_mentions: Include honorable mentions

        Returns:
            List of award-winning papers
        """
        self._log("info", f"Fetching best papers", conference=conference, year=year)

        venue = self.VENUE_MAPPING.get(conference.upper(), conference.lower())

        # Get all papers for the venue
        all_papers = await self.get_paper_list(conference, year)

        best_papers = []
        award_keywords = [
            "best paper",
            "outstanding paper",
            "best long paper",
            "best short paper",
            "best resource paper",
            "best demo",
            "best student paper",
        ]

        if include_honorable_mentions:
            award_keywords.extend([
                "honorable mention",
                "notable paper",
                "runner-up",
            ])

        for paper in all_papers:
            # Check if paper has award by fetching full details
            try:
                full_paper = await self.get_paper_details(paper.id)
                if full_paper.paper_type in [PaperType.BEST_PAPER, PaperType.OUTSTANDING]:
                    best_papers.append(full_paper)
            except Exception as e:
                self._log("warning", f"Failed to check paper for awards", paper_id=paper.id, error=str(e))
                continue

        self._log("info", f"Found best papers", count=len(best_papers), conference=conference, year=year)
        return best_papers

    async def get_best_papers_from_page(
        self,
        conference: str,
        year: int
    ) -> List[Paper]:
        """Get best papers by scanning event page for award markers.

        This method is more efficient as it only fetches papers that are likely
        to have awards, rather than checking every paper.

        Args:
            conference: Conference name
            year: Publication year

        Returns:
            List of papers with awards
        """
        self._log("info", f"Scanning for best papers", conference=conference, year=year)

        venue = self.VENUE_MAPPING.get(conference.upper(), conference.lower())
        url = f"{self.BASE_URL}/events/{venue}-{year}/"

        response = self._sync_get(url)
        if response is None:
            return []

        if not BS4_AVAILABLE:
            self._log("error", "BeautifulSoup not available for web scraping")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        best_papers = []

        # Find all paper entries and check for award markers
        paper_id_pattern = re.compile(rf"{year}\.{venue}-\w+\.\d+$")

        for link in soup.select("a"):
            href = link.get("href", "").strip("/")
            parts = href.split("/")
            if not parts:
                continue

            paper_id = parts[-1]
            if not paper_id_pattern.match(paper_id):
                continue

            # Check if there's an award marker near this link
            parent = link.parent
            if parent:
                parent_text = parent.get_text().lower()
                if any(kw in parent_text for kw in ["award", "best", "outstanding"]):
                    paper = self._get_paper_via_web(paper_id)
                    if paper:
                        best_papers.append(paper)

        self._log("info", f"Found best papers from page scan", count=len(best_papers))
        return best_papers

    # ==================== Official Library Methods ====================

    def _get_anthology(self):
        """Get or create Anthology instance (lazy loading)."""
        if not ACL_ANTHOLOGY_AVAILABLE:
            raise ScraperError("acl-anthology library not installed", source=self.source_name)

        if self._anthology is None:
            self._log("info", "Initializing ACL Anthology library (this may take a while on first run)")
            self._anthology = Anthology.from_repo()
            self._log("info", "ACL Anthology library initialized")

        return self._anthology

    async def _get_papers_via_library(
        self,
        venue: str,
        year: int,
        track: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Paper]:
        """Get papers using the official acl-anthology library."""
        anthology = self._get_anthology()

        papers = []
        # The library uses different access patterns
        # We need to iterate through volumes

        volume_pattern = f"{year}.{venue}"
        if track:
            volume_pattern = f"{year}.{venue}-{track}"

        try:
            # Get all matching volumes
            for volume_id in anthology.volumes:
                if not volume_id.startswith(volume_pattern):
                    continue

                volume = anthology.get_volume(volume_id)
                if volume is None:
                    continue

                for item in volume.items():
                    if offset > 0:
                        offset -= 1
                        continue

                    paper = self._convert_anthology_paper(item, venue, year)
                    if paper:
                        papers.append(paper)

                    if limit and len(papers) >= limit:
                        return papers

        except Exception as e:
            self._log("error", f"Error fetching papers via library", error=str(e))

        return papers

    async def _get_paper_via_library(self, paper_id: str) -> Optional[Paper]:
        """Get a single paper using the official library."""
        anthology = self._get_anthology()

        try:
            item = anthology.get(paper_id)
            if item is None:
                return None

            # Extract venue and year from paper_id
            parts = paper_id.split(".")
            year = int(parts[0]) if parts[0].isdigit() else 2024
            venue = parts[1].split("-")[0] if len(parts) > 1 else "acl"

            return self._convert_anthology_paper(item, venue, year)
        except Exception as e:
            self._log("error", f"Error fetching paper via library", paper_id=paper_id, error=str(e))
            return None

    def _convert_anthology_paper(self, item: Any, venue: str, year: int) -> Optional[Paper]:
        """Convert an acl-anthology library item to Paper object."""
        try:
            paper_id = str(item.id) if hasattr(item, 'id') else str(item)

            # Extract basic info
            title = str(item.title) if hasattr(item, 'title') else ""
            abstract = str(item.abstract) if hasattr(item, 'abstract') else ""

            # Extract authors
            authors = []
            if hasattr(item, 'authors'):
                for author in item.authors:
                    name = str(author.name) if hasattr(author, 'name') else str(author)
                    authors.append(Author(name=name))

            # Determine conference
            conference = self.CONFERENCE_MAPPING.get(venue.lower(), Conference.ACL)

            # Determine paper type from track
            paper_type = PaperType.UNKNOWN
            if "-" in paper_id:
                track = paper_id.split("-")[1].split(".")[0] if "." in paper_id.split("-")[1] else paper_id.split("-")[1]
                paper_type = self.TRACK_MAPPING.get(track, PaperType.UNKNOWN)

            # Check for awards
            if hasattr(item, 'award') and item.award:
                award_text = str(item.award).lower()
                if "best" in award_text:
                    paper_type = PaperType.BEST_PAPER
                elif "outstanding" in award_text:
                    paper_type = PaperType.OUTSTANDING

            # Build URLs
            pdf_url = f"{self.BASE_URL}/{paper_id}.pdf"
            paper_url = f"{self.BASE_URL}/{paper_id}/"

            return Paper(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                conference=conference,
                paper_type=paper_type,
                pdf_url=pdf_url,
                paper_page_url=paper_url,
                source=self.source_name,
            )
        except Exception as e:
            self._log("error", f"Error converting anthology paper", error=str(e))
            return None

    # ==================== Web Scraping Methods ====================

    def _sync_get(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Synchronous GET request."""
        import requests
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response
            self._log("warning", f"HTTP request failed", url=url, status=response.status_code)
            return None
        except Exception as e:
            self._log("error", f"GET request failed", url=url, error=str(e))
            return None

    def _get_papers_via_web(
        self,
        venue: str,
        year: int,
        track: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Paper]:
        """Get papers by scraping the ACL Anthology website."""
        if not BS4_AVAILABLE:
            self._log("error", "BeautifulSoup not available for web scraping")
            return []

        url = f"{self.BASE_URL}/events/{venue}-{year}/"
        response = self._sync_get(url)

        if response is None:
            self._log("warning", f"Failed to fetch event page", url=url)
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        papers = []

        # Build pattern for paper IDs
        if track:
            pattern = re.compile(rf"{year}\.{venue}-{track}\.\d+$")
        else:
            pattern = re.compile(rf"{year}\.{venue}-\w+\.\d+$")

        seen_ids: Set[str] = set()

        # Extract all paper IDs from links
        for link in soup.select("a"):
            href = link.get("href", "").strip("/")
            parts = href.split("/")
            if not parts:
                continue

            paper_id = parts[-1]
            if pattern.match(paper_id) and paper_id not in seen_ids:
                # Skip paper 0 which is usually the proceedings frontmatter
                if paper_id.endswith(".0"):
                    continue
                seen_ids.add(paper_id)

        self._log("info", f"Found paper IDs", count=len(seen_ids), venue=venue, year=year)

        # Apply offset and limit
        paper_ids = sorted(seen_ids)
        if offset > 0:
            paper_ids = paper_ids[offset:]
        if limit:
            paper_ids = paper_ids[:limit]

        # Fetch paper details (can be slow for many papers)
        for paper_id in paper_ids:
            paper = self._get_paper_via_web(paper_id)
            if paper:
                papers.append(paper)

        return papers

    def _get_paper_via_web(self, paper_id: str) -> Optional[Paper]:
        """Get a single paper by scraping its detail page."""
        if not BS4_AVAILABLE:
            self._log("error", "BeautifulSoup not available")
            return None

        url = f"{self.BASE_URL}/{paper_id}/"
        response = self._sync_get(url)

        if response is None:
            return None

        return self._parse_paper_page(response.text, paper_id)

    def _parse_paper_page(self, html: str, paper_id: str) -> Optional[Paper]:
        """Parse paper detail page HTML."""
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Extract title - try multiple selectors
            title = ""
            for selector in ["h2#title", "h2.card-title", ".acl-paper-title"]:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break

            if not title:
                # Try getting title from page title
                page_title = soup.select_one("title")
                if page_title:
                    title = page_title.get_text(strip=True).split(" - ")[0]

            # Extract authors
            authors = []
            author_elems = soup.select("p.lead a")
            for a in author_elems:
                name = a.get_text(strip=True)
                if name and not name.startswith("http"):
                    authors.append(Author(name=name))

            # Extract abstract
            abstract = ""
            abstract_elem = soup.select_one(".acl-abstract span")
            if not abstract_elem:
                abstract_elem = soup.select_one(".card-body .abstract")
            if abstract_elem:
                abstract = abstract_elem.get_text(strip=True)

            # Extract year from paper_id
            year = 2024
            if paper_id and len(paper_id) >= 4 and paper_id[:4].isdigit():
                year = int(paper_id[:4])

            # Determine venue and conference from paper_id
            venue = "acl"
            conference = Conference.ACL
            if "." in paper_id:
                parts = paper_id.split(".")
                if len(parts) > 1:
                    venue_part = parts[1].split("-")[0]
                    venue = venue_part
                    conference = self.CONFERENCE_MAPPING.get(venue_part, Conference.ACL)

            # Determine paper type from track
            paper_type = PaperType.UNKNOWN
            if "-" in paper_id:
                track_part = paper_id.split("-")
                if len(track_part) > 1:
                    track = track_part[1].split(".")[0]
                    paper_type = self.TRACK_MAPPING.get(track, PaperType.UNKNOWN)

            # Check for award
            award_text = ""
            for text in soup.stripped_strings:
                if "Award:" in text or "award" in text.lower():
                    award_text = text.lower()
                    break

            # Also look for explicit Award field
            award_label = soup.find(string=re.compile(r"Award:", re.IGNORECASE))
            if award_label:
                award_parent = award_label.parent
                if award_parent:
                    # Get the next sibling text
                    next_text = award_parent.find_next_sibling(string=True)
                    if next_text:
                        award_text = next_text.strip().lower()

            if "best paper" in award_text or "best long" in award_text or "best short" in award_text:
                paper_type = PaperType.BEST_PAPER
            elif "outstanding" in award_text:
                paper_type = PaperType.OUTSTANDING

            # Extract DOI
            doi = None
            doi_elem = soup.select_one("a[href*='doi.org']")
            if doi_elem:
                doi_url = doi_elem.get("href", "")
                if "doi.org/" in doi_url:
                    doi = doi_url.split("doi.org/")[1]

            # Build URLs
            pdf_url = f"{self.BASE_URL}/{paper_id}.pdf"
            paper_url = f"{self.BASE_URL}/{paper_id}/"

            return Paper(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                conference=conference,
                paper_type=paper_type,
                pdf_url=pdf_url,
                paper_page_url=paper_url,
                doi=doi,
                source=self.source_name,
            )

        except Exception as e:
            self._log("error", f"Failed to parse paper page", paper_id=paper_id, error=str(e))
            return None

    # ==================== Search Methods ====================

    async def search(
        self,
        query: str,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 100,
    ) -> List[Paper]:
        """Search papers by keyword.

        Args:
            query: Search query
            conference: Filter by conference (optional)
            year: Filter by year (optional)
            limit: Maximum results

        Returns:
            List of matching papers
        """
        self._log("info", f"Searching papers", query=query, conference=conference, year=year)

        if not BS4_AVAILABLE:
            self._log("error", "BeautifulSoup not available for search")
            return []

        url = f"{self.BASE_URL}/search/"
        params = {"q": query}

        response = self._sync_get(url, params=params)
        if response is None:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        papers = []

        # Parse search results
        for item in soup.select(".paper-abstract, .search-result")[:limit]:
            paper = self._parse_search_result(item)
            if paper:
                # Apply filters
                if conference and paper.conference.value.upper() != conference.upper():
                    continue
                if year and paper.year != year:
                    continue
                papers.append(paper)

        self._log("info", f"Search completed", query=query, results=len(papers))
        return papers

    def _parse_search_result(self, item) -> Optional[Paper]:
        """Parse a search result item."""
        try:
            # Find title link
            title_elem = item.select_one(".paper-title a, a.paper-link")
            if not title_elem:
                # Try finding any link that looks like a paper
                for link in item.select("a"):
                    href = link.get("href", "")
                    if re.match(r"/\d{4}\.\w+-\w+\.\d+/?$", href):
                        title_elem = link
                        break

            if not title_elem:
                return None

            title = title_elem.get_text(strip=True)
            paper_url = title_elem.get("href", "")
            paper_id = paper_url.strip("/").split("/")[-1]

            # Extract year from paper_id
            year = 2024
            if paper_id and len(paper_id) >= 4 and paper_id[:4].isdigit():
                year = int(paper_id[:4])

            # Determine conference
            conference = Conference.ACL
            for venue_name, venue_id in self.VENUE_MAPPING.items():
                if venue_id in paper_id.lower():
                    conference = self.CONFERENCE_MAPPING.get(venue_id, Conference.ACL)
                    break

            # Get authors if available
            authors = []
            author_elems = item.select(".paper-authors a, .authors a")
            for a in author_elems:
                name = a.get_text(strip=True)
                if name:
                    authors.append(Author(name=name))

            # Get abstract if available
            abstract = ""
            abstract_elem = item.select_one(".abstract, .paper-abstract-text")
            if abstract_elem:
                abstract = abstract_elem.get_text(strip=True)

            return Paper(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                conference=conference,
                paper_type=PaperType.UNKNOWN,
                pdf_url=f"{self.BASE_URL}/{paper_id}.pdf",
                paper_page_url=f"{self.BASE_URL}/{paper_id}/",
                source=self.source_name,
            )

        except Exception as e:
            self._log("error", f"Failed to parse search result", error=str(e))
            return None

    # ==================== Utility Methods ====================

    def get_venue_url(self, conference: str, year: int) -> str:
        """Get the URL for a venue's event page."""
        venue = self.VENUE_MAPPING.get(conference.upper(), conference.lower())
        return f"{self.BASE_URL}/events/{venue}-{year}/"

    def get_paper_url(self, paper_id: str) -> str:
        """Get the URL for a paper's detail page."""
        return f"{self.BASE_URL}/{paper_id}/"

    def parse_paper_id(self, paper_id: str) -> Tuple[int, str, str, int]:
        """Parse a paper ID into components.

        Args:
            paper_id: Paper ID like "2024.acl-long.1"

        Returns:
            Tuple of (year, venue, track, number)
        """
        parts = paper_id.split(".")
        year = int(parts[0]) if parts[0].isdigit() else 0

        venue_track = parts[1] if len(parts) > 1 else ""
        venue_parts = venue_track.split("-")
        venue = venue_parts[0]
        track = venue_parts[1] if len(venue_parts) > 1 else ""

        number = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0

        return year, venue, track, number


# ==================== Factory Function ====================

def get_acl_scraper(**kwargs) -> ACLScraper:
    """Factory function to get ACL Scraper instance.

    Args:
        **kwargs: Arguments passed to ACLScraper constructor

    Returns:
        ACLScraper instance
    """
    return ACLScraper(**kwargs)


# ==================== Convenience Functions ====================

async def fetch_acl_papers(year: int, limit: Optional[int] = None) -> List[Paper]:
    """Fetch ACL main conference papers."""
    async with ACLScraper() as scraper:
        return await scraper.get_paper_list("ACL", year, limit=limit)


async def fetch_emnlp_papers(year: int, limit: Optional[int] = None) -> List[Paper]:
    """Fetch EMNLP papers."""
    async with ACLScraper() as scraper:
        return await scraper.get_paper_list("EMNLP", year, limit=limit)


async def fetch_naacl_papers(year: int, limit: Optional[int] = None) -> List[Paper]:
    """Fetch NAACL papers."""
    async with ACLScraper() as scraper:
        return await scraper.get_paper_list("NAACL", year, limit=limit)


# ==================== Main (for testing) ====================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main():
        scraper = ACLScraper()

        print("Fetching ACL 2024 papers (limit 5)...")
        papers = await scraper.get_paper_list("ACL", 2024, limit=5)

        for paper in papers:
            print(f"\nTitle: {paper.title}")
            print(f"Authors: {', '.join(a.name for a in paper.authors[:3])}...")
            print(f"Type: {paper.paper_type.value}")
            print(f"URL: {paper.paper_page_url}")
            print(f"PDF: {paper.pdf_url}")

        print(f"\n\nTotal papers fetched: {len(papers)}")

        await scraper.close()

    asyncio.run(main())
