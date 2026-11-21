"""
Paper Crawler Data Models

Core data models for the paper crawling system using Pydantic.
Supports paper metadata, collections with filtering/sorting, download tracking, and crawl tasks.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


# =============================================================================
# Enumerations
# =============================================================================


class Conference(str, Enum):
    """Supported machine learning conferences."""

    ICLR = "ICLR"
    ICML = "ICML"
    NEURIPS = "NeurIPS"
    AAAI = "AAAI"
    CVPR = "CVPR"
    ICCV = "ICCV"
    ECCV = "ECCV"
    ACL = "ACL"
    EMNLP = "EMNLP"
    NAACL = "NAACL"
    KDD = "KDD"
    WWW = "WWW"
    SIGIR = "SIGIR"
    SIGMOD = "SIGMOD"
    IJCAI = "IJCAI"
    CCS = "CCS"
    ACM_MM = "ACM MM"
    VLDB = "VLDB"
    USENIX_SECURITY = "USENIX Security"
    NDSS = "NDSS"
    CSUR = "CSUR"
    # DS — 计算机体系结构/并行/存储 (open-access full-text only; IEEE-paywalled
    # MICRO/HPCA/ISCA<=2024 deferred — W89). Values MUST match conferences.yaml `title`.
    ASPLOS = "ASPLOS"
    ISCA = "ISCA"
    PPOPP = "PPoPP"
    EUROSYS = "EuroSys"
    DAC = "DAC"
    FAST = "FAST"
    USENIX_ATC = "USENIX ATC"
    # Tier-1 expansion. A venue missing here becomes Conference.OTHER and its
    # SQL `venue` is written as "Other", so every registry title needs a member.
    TPAMI = "TPAMI"
    JMLR = "JMLR"
    IJCV = "IJCV"
    IEEE_SP = "IEEE S&P"
    OSDI = "OSDI"
    SOSP = "SOSP"
    COLT = "COLT"
    AISTATS = "AISTATS"
    UAI = "UAI"
    ICRA = "ICRA"
    CORL = "CoRL"
    # Preprints from arXiv (HTML harvest). Not a conference: rows carry
    # venue_tier/track_type "preprint" so they stay distinguishable (AGENTS 2.2).
    ARXIV = "arXiv"
    OTHER = "Other"


class PaperType(str, Enum):
    """Paper acceptance/award types."""

    BEST_PAPER = "best_paper"
    OUTSTANDING = "outstanding"
    ORAL = "oral"
    SPOTLIGHT = "spotlight"
    POSTER = "poster"
    WORKSHOP = "workshop"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class TaskStatus(str, Enum):
    """Crawl task status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Alias for backward compatibility
CrawlStatus = TaskStatus


class SortField(str, Enum):
    """Available fields for sorting papers."""

    CITATIONS = "citations"
    YEAR = "year"
    TITLE = "title"
    CONFERENCE = "conference"
    PAPER_TYPE = "paper_type"
    CREATED_AT = "created_at"


class SortOrder(str, Enum):
    """Sort order direction."""

    ASC = "asc"
    DESC = "desc"


class PaperSource(str, Enum):
    """Paper data source."""

    OPENREVIEW = "openreview"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    ACL_ANTHOLOGY = "acl_anthology"
    ACL = "acl"  # Alias for ACL_ANTHOLOGY
    NEURIPS = "neurips"
    NEURIPS_PROC = "neurips_proc"
    CVPR = "cvpr"
    IEEE = "ieee"
    ACM = "acm"
    OTHER = "other"


# =============================================================================
# PaperMetadata Model
# =============================================================================


class PaperMetadata(BaseModel):
    """Additional metadata for a paper.

    Used by scrapers to store venue-specific metadata that doesn't fit
    in the main Paper model.
    """

    venue: Optional[str] = Field(default=None, description="Publication venue")
    year: Optional[int] = Field(default=None, ge=1900, le=2100, description="Publication year")
    keywords: list[str] = Field(default_factory=list, description="Paper keywords")
    citations: Optional[int] = Field(default=None, ge=0, description="Citation count")
    decision: Optional[str] = Field(default=None, description="Paper decision (accept, reject, etc.)")
    track: Optional[str] = Field(default=None, description="Conference track")

    class Config:
        extra = "allow"  # Allow additional fields for flexibility


# =============================================================================
# Author Model
# =============================================================================


class Author(BaseModel):
    """Author information."""

    name: str = Field(..., min_length=1, description="Author's full name")
    email: Optional[str] = Field(default=None, description="Author's email address")
    affiliation: Optional[str] = Field(default=None, description="Author's institution")
    orcid: Optional[str] = Field(default=None, description="ORCID identifier")
    google_scholar_id: Optional[str] = Field(default=None, description="Google Scholar ID")

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name.lower())


class PaperMetadata(BaseModel):
    """Paper metadata for backward compatibility with scrapers.

    This class provides additional metadata fields that may be used
    by different scrapers.
    """

    venue: Optional[str] = Field(default=None, description="Venue/conference name")
    year: Optional[int] = Field(default=None, ge=1900, le=2100, description="Publication year")
    keywords: list[str] = Field(default_factory=list, description="Paper keywords")
    awards: list[str] = Field(default_factory=list, description="Awards received")
    decision: Optional[str] = Field(default=None, description="Acceptance decision")
    track: Optional[str] = Field(default=None, description="Conference track")
    venue_tier: Optional[str] = Field(default=None, description="CCF-A|CCF-B|CCF-C|other")
    track_type: Optional[str] = Field(default=None, description="main|workshop|demo|findings")


# =============================================================================
# Paper Model
# =============================================================================


class Paper(BaseModel):
    """
    Paper model representing a single academic paper.

    Contains all metadata needed for indexing, retrieval, and citation tracking.
    """

    # Core identifiers
    id: str = Field(..., min_length=1, description="Unique paper identifier")
    title: str = Field(..., min_length=1, description="Paper title")

    # Authors
    authors: list[Author | str] = Field(
        default_factory=list,
        description="List of authors (can be Author objects or simple strings)",
    )

    # Content
    abstract: Optional[str] = Field(default=None, description="Paper abstract")
    keywords: list[str] = Field(default_factory=list, description="Paper keywords")
    tags: list[str] = Field(default_factory=list, description="Custom tags for categorization")

    # Publication info
    year: int = Field(..., ge=1900, le=2100, description="Publication year")
    conference: Conference = Field(..., description="Conference name")
    paper_type: PaperType = Field(
        default=PaperType.UNKNOWN, description="Paper type (oral, poster, etc.)"
    )
    venue_track: Optional[str] = Field(
        default=None, description="Specific track within the conference"
    )
    track_type: Optional[str] = Field(default=None, description="main|workshop|demo|findings|tutorial")
    impact_level: Optional[str] = Field(default=None, description="oral|spotlight|poster|regular")
    venue_tier: Optional[str] = Field(default=None, description="CCF-A|CCF-B|CCF-C|other")
    venue_source: Optional[str] = Field(default=None, description="proceedings source identifier")
    full_text_source: Optional[str] = Field(
        default=None, description="publisher|arxiv|oa_repo: origin of the indexed full text"
    )

    # External identifiers
    arxiv_id: Optional[str] = Field(default=None, description="arXiv identifier (e.g., 2301.00001)")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    semantic_scholar_id: Optional[str] = Field(default=None, description="Semantic Scholar paper ID")
    openreview_id: Optional[str] = Field(default=None, description="OpenReview paper ID")

    # URLs
    pdf_url: Optional[HttpUrl] = Field(default=None, description="Direct PDF download URL")
    paper_page_url: Optional[HttpUrl] = Field(default=None, description="Paper landing page URL")
    code_url: Optional[HttpUrl] = Field(default=None, description="Code repository URL")
    project_url: Optional[HttpUrl] = Field(default=None, description="Project page URL")
    video_url: Optional[HttpUrl] = Field(default=None, description="Presentation video URL")
    slides_url: Optional[HttpUrl] = Field(default=None, description="Slides URL")

    # Local processing artifacts
    pdf_path: Optional[str] = Field(default=None, description="Local path to downloaded PDF")
    markdown_path: Optional[str] = Field(
        default=None,
        description="Local path to extracted markdown content",
    )

    # Metrics
    citations: int = Field(default=0, ge=0, description="Citation count")
    influential_citations: int = Field(
        default=0, ge=0, description="Influential citation count (Semantic Scholar)"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.now, description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )
    source: Optional[str] = Field(
        default=None, description="Data source (e.g., 'openreview', 'arxiv', 'semanticscholar')"
    )
    metadata: Optional[PaperMetadata] = Field(
        default=None,
        description="Additional venue-specific metadata used during crawling and indexing",
    )

    @field_validator("authors", mode="before")
    @classmethod
    def normalize_authors(cls, v: list) -> list[Author | str]:
        """Convert author strings to Author objects if needed."""
        if not v:
            return []
        result = []
        for author in v:
            if isinstance(author, str):
                result.append(Author(name=author))
            elif isinstance(author, dict):
                result.append(Author(**author))
            else:
                result.append(author)
        return result

    @field_validator("arxiv_id", mode="before")
    @classmethod
    def normalize_arxiv_id(cls, v: Optional[str]) -> Optional[str]:
        """Extract arxiv ID from URL if full URL is provided."""
        if v is None:
            return None
        v = str(v).strip()
        # Handle full arxiv URLs
        if "arxiv.org" in v:
            # Extract ID from URLs like https://arxiv.org/abs/2301.00001
            parts = v.split("/")
            for i, part in enumerate(parts):
                if part in ("abs", "pdf"):
                    if i + 1 < len(parts):
                        return parts[i + 1].replace(".pdf", "")
        return v

    @model_validator(mode="after")
    def update_timestamp(self) -> "Paper":
        """Update the updated_at timestamp."""
        object.__setattr__(self, "updated_at", datetime.now())
        return self

    @property
    def author_names(self) -> list[str]:
        """Get list of author names as strings."""
        return [str(a) if isinstance(a, Author) else a for a in self.authors]

    @property
    def first_author(self) -> Optional[str]:
        """Get the first author's name."""
        if self.authors:
            return str(self.authors[0])
        return None

    @property
    def arxiv_url(self) -> Optional[str]:
        """Generate arXiv abstract URL from arxiv_id."""
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        return None

    @property
    def arxiv_pdf_url(self) -> Optional[str]:
        """Generate arXiv PDF URL from arxiv_id."""
        if self.arxiv_id:
            return f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"
        return None

    def matches_filter(
        self,
        conferences: Optional[list[Conference]] = None,
        years: Optional[list[int]] = None,
        paper_types: Optional[list[PaperType]] = None,
        min_citations: Optional[int] = None,
        tags: Optional[list[str]] = None,
        has_code: Optional[bool] = None,
        has_arxiv: Optional[bool] = None,
        search_query: Optional[str] = None,
    ) -> bool:
        """Check if paper matches the given filter criteria."""
        if conferences and self.conference not in conferences:
            return False
        if years and self.year not in years:
            return False
        if paper_types and self.paper_type not in paper_types:
            return False
        if min_citations is not None and self.citations < min_citations:
            return False
        if tags and not any(tag in self.tags for tag in tags):
            return False
        if has_code is True and not self.code_url:
            return False
        if has_code is False and self.code_url:
            return False
        if has_arxiv is True and not self.arxiv_id:
            return False
        if has_arxiv is False and self.arxiv_id:
            return False
        if search_query:
            query_lower = search_query.lower()
            searchable = f"{self.title} {self.abstract or ''} {' '.join(self.author_names)}".lower()
            if query_lower not in searchable:
                return False
        return True

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Paper):
            return self.id == other.id
        return False


# =============================================================================
# Paper Collection Model
# =============================================================================


class PaperFilter(BaseModel):
    """Filter criteria for paper collections."""

    conferences: Optional[list[Conference]] = Field(
        default=None, description="Filter by conferences"
    )
    years: Optional[list[int]] = Field(default=None, description="Filter by publication years")
    year_range: Optional[tuple[int, int]] = Field(
        default=None, description="Filter by year range (start, end inclusive)"
    )
    paper_types: Optional[list[PaperType]] = Field(default=None, description="Filter by paper types")
    min_citations: Optional[int] = Field(default=None, ge=0, description="Minimum citation count")
    max_citations: Optional[int] = Field(default=None, ge=0, description="Maximum citation count")
    tags: Optional[list[str]] = Field(default=None, description="Filter by tags (any match)")
    has_code: Optional[bool] = Field(default=None, description="Filter by code availability")
    has_arxiv: Optional[bool] = Field(default=None, description="Filter by arXiv availability")
    has_pdf: Optional[bool] = Field(default=None, description="Filter by PDF availability")
    search_query: Optional[str] = Field(
        default=None, description="Full-text search in title/abstract/authors"
    )
    authors: Optional[list[str]] = Field(default=None, description="Filter by author names")

    @model_validator(mode="after")
    def validate_year_range(self) -> "PaperFilter":
        """Validate year range if provided."""
        if self.year_range:
            start, end = self.year_range
            if start > end:
                raise ValueError("year_range start must be <= end")
        return self

    def to_year_list(self) -> Optional[list[int]]:
        """Convert year_range to list of years if year_range is set."""
        if self.year_range:
            return list(range(self.year_range[0], self.year_range[1] + 1))
        return self.years


class PaperSort(BaseModel):
    """Sorting configuration for paper collections."""

    field: SortField = Field(default=SortField.CITATIONS, description="Field to sort by")
    order: SortOrder = Field(default=SortOrder.DESC, description="Sort order")

    def get_key_func(self) -> Callable[[Paper], tuple]:
        """Get the sorting key function."""
        field_map = {
            SortField.CITATIONS: lambda p: (p.citations,),
            SortField.YEAR: lambda p: (p.year,),
            SortField.TITLE: lambda p: (p.title.lower(),),
            SortField.CONFERENCE: lambda p: (p.conference.value,),
            SortField.PAPER_TYPE: lambda p: (p.paper_type.value,),
            SortField.CREATED_AT: lambda p: (p.created_at,),
        }
        return field_map.get(self.field, lambda p: (p.citations,))


class PaperCollection(BaseModel):
    """
    Collection of papers with filtering and sorting capabilities.

    Provides methods for querying, filtering, and organizing papers.
    """

    papers: list[Paper] = Field(default_factory=list, description="List of papers in collection")
    name: Optional[str] = Field(default=None, description="Collection name")
    description: Optional[str] = Field(default=None, description="Collection description")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    @property
    def count(self) -> int:
        """Get total number of papers."""
        return len(self.papers)

    @property
    def conferences(self) -> set[Conference]:
        """Get unique conferences in collection."""
        return {p.conference for p in self.papers}

    @property
    def years(self) -> set[int]:
        """Get unique years in collection."""
        return {p.year for p in self.papers}

    @property
    def paper_types(self) -> set[PaperType]:
        """Get unique paper types in collection."""
        return {p.paper_type for p in self.papers}

    @property
    def total_citations(self) -> int:
        """Get total citations across all papers."""
        return sum(p.citations for p in self.papers)

    def add(self, paper: Paper) -> None:
        """Add a paper to the collection."""
        if paper not in self.papers:
            self.papers.append(paper)

    def remove(self, paper_id: str) -> bool:
        """Remove a paper by ID. Returns True if removed."""
        for i, p in enumerate(self.papers):
            if p.id == paper_id:
                self.papers.pop(i)
                return True
        return False

    def get(self, paper_id: str) -> Optional[Paper]:
        """Get a paper by ID."""
        for p in self.papers:
            if p.id == paper_id:
                return p
        return None

    def filter(self, criteria: PaperFilter) -> "PaperCollection":
        """
        Filter papers based on criteria and return a new collection.

        Args:
            criteria: PaperFilter object with filter conditions

        Returns:
            New PaperCollection with filtered papers
        """
        filtered = []
        years = criteria.to_year_list()

        for paper in self.papers:
            # Conference filter
            if criteria.conferences and paper.conference not in criteria.conferences:
                continue

            # Year filter
            if years and paper.year not in years:
                continue

            # Paper type filter
            if criteria.paper_types and paper.paper_type not in criteria.paper_types:
                continue

            # Citation filters
            if criteria.min_citations is not None and paper.citations < criteria.min_citations:
                continue
            if criteria.max_citations is not None and paper.citations > criteria.max_citations:
                continue

            # Tag filter (any match)
            if criteria.tags and not any(tag in paper.tags for tag in criteria.tags):
                continue

            # Availability filters
            if criteria.has_code is True and not paper.code_url:
                continue
            if criteria.has_code is False and paper.code_url:
                continue
            if criteria.has_arxiv is True and not paper.arxiv_id:
                continue
            if criteria.has_arxiv is False and paper.arxiv_id:
                continue
            if criteria.has_pdf is True and not paper.pdf_url:
                continue
            if criteria.has_pdf is False and paper.pdf_url:
                continue

            # Author filter
            if criteria.authors:
                author_names_lower = [name.lower() for name in paper.author_names]
                if not any(
                    author.lower() in author_names_lower for author in criteria.authors
                ):
                    continue

            # Search query
            if criteria.search_query:
                query_lower = criteria.search_query.lower()
                searchable = (
                    f"{paper.title} {paper.abstract or ''} {' '.join(paper.author_names)}"
                ).lower()
                if query_lower not in searchable:
                    continue

            filtered.append(paper)

        return PaperCollection(
            papers=filtered,
            name=f"{self.name or 'Collection'} (filtered)",
            description=f"Filtered from {self.count} to {len(filtered)} papers",
        )

    def sort(self, sort_config: PaperSort) -> "PaperCollection":
        """
        Sort papers and return a new collection.

        Args:
            sort_config: PaperSort object with sorting configuration

        Returns:
            New PaperCollection with sorted papers
        """
        key_func = sort_config.get_key_func()
        reverse = sort_config.order == SortOrder.DESC

        sorted_papers = sorted(self.papers, key=key_func, reverse=reverse)

        return PaperCollection(
            papers=sorted_papers,
            name=f"{self.name or 'Collection'} (sorted)",
            description=f"Sorted by {sort_config.field.value} {sort_config.order.value}",
        )

    def filter_and_sort(
        self,
        criteria: Optional[PaperFilter] = None,
        sort_config: Optional[PaperSort] = None,
    ) -> "PaperCollection":
        """Filter and sort papers in one operation."""
        result = self
        if criteria:
            result = result.filter(criteria)
        if sort_config:
            result = result.sort(sort_config)
        return result

    def paginate(self, page: int = 1, page_size: int = 20) -> "PaperCollection":
        """
        Get a paginated subset of papers.

        Args:
            page: Page number (1-indexed)
            page_size: Number of papers per page

        Returns:
            New PaperCollection with paginated papers
        """
        if page < 1:
            page = 1
        start = (page - 1) * page_size
        end = start + page_size
        return PaperCollection(
            papers=self.papers[start:end],
            name=f"{self.name or 'Collection'} (page {page})",
            description=f"Page {page} of {(self.count + page_size - 1) // page_size}",
        )

    def group_by_conference(self) -> dict[Conference, list[Paper]]:
        """Group papers by conference."""
        groups: dict[Conference, list[Paper]] = {}
        for paper in self.papers:
            if paper.conference not in groups:
                groups[paper.conference] = []
            groups[paper.conference].append(paper)
        return groups

    def group_by_year(self) -> dict[int, list[Paper]]:
        """Group papers by year."""
        groups: dict[int, list[Paper]] = {}
        for paper in self.papers:
            if paper.year not in groups:
                groups[paper.year] = []
            groups[paper.year].append(paper)
        return groups

    def group_by_type(self) -> dict[PaperType, list[Paper]]:
        """Group papers by paper type."""
        groups: dict[PaperType, list[Paper]] = {}
        for paper in self.papers:
            if paper.paper_type not in groups:
                groups[paper.paper_type] = []
            groups[paper.paper_type].append(paper)
        return groups

    def statistics(self) -> dict:
        """Get collection statistics."""
        if not self.papers:
            return {
                "total_papers": 0,
                "conferences": {},
                "years": {},
                "paper_types": {},
                "citations": {"total": 0, "mean": 0, "max": 0, "min": 0},
                "with_code": 0,
                "with_arxiv": 0,
            }

        citations = [p.citations for p in self.papers]
        return {
            "total_papers": self.count,
            "conferences": {c.value: len(ps) for c, ps in self.group_by_conference().items()},
            "years": {y: len(ps) for y, ps in self.group_by_year().items()},
            "paper_types": {t.value: len(ps) for t, ps in self.group_by_type().items()},
            "citations": {
                "total": sum(citations),
                "mean": sum(citations) / len(citations),
                "max": max(citations),
                "min": min(citations),
            },
            "with_code": sum(1 for p in self.papers if p.code_url),
            "with_arxiv": sum(1 for p in self.papers if p.arxiv_id),
        }

    def merge(self, other: "PaperCollection") -> "PaperCollection":
        """Merge with another collection, avoiding duplicates."""
        existing_ids = {p.id for p in self.papers}
        new_papers = [p for p in other.papers if p.id not in existing_ids]
        return PaperCollection(
            papers=self.papers + new_papers,
            name=f"Merged: {self.name or 'A'} + {other.name or 'B'}",
            description=f"Merged {self.count} + {other.count} papers ({len(new_papers)} new)",
        )

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return iter(self.papers)

    def __contains__(self, paper: Paper) -> bool:
        return paper in self.papers


# =============================================================================
# Download Status Model
# =============================================================================


class DownloadStatus(BaseModel):
    """
    Download and processing status for a paper.

    Tracks the state of PDF download, markdown conversion, and indexing.
    """

    paper_id: str = Field(..., description="Associated paper ID")

    # Download status
    pdf_downloaded: bool = Field(default=False, description="PDF has been downloaded")
    pdf_path: Optional[str] = Field(default=None, description="Local path to downloaded PDF")
    pdf_size_bytes: Optional[int] = Field(default=None, ge=0, description="PDF file size in bytes")
    pdf_download_time: Optional[datetime] = Field(default=None, description="PDF download timestamp")

    # Conversion status
    md_converted: bool = Field(default=False, description="PDF has been converted to Markdown")
    md_path: Optional[str] = Field(default=None, description="Local path to Markdown file")
    md_conversion_time: Optional[datetime] = Field(
        default=None, description="Markdown conversion timestamp"
    )

    # Indexing status
    indexed: bool = Field(default=False, description="Paper has been indexed for search")
    index_time: Optional[datetime] = Field(default=None, description="Indexing timestamp")
    chunk_count: Optional[int] = Field(
        default=None, ge=0, description="Number of chunks created for indexing"
    )
    embedding_model: Optional[str] = Field(
        default=None, description="Embedding model used for indexing"
    )

    # Error tracking
    error_message: Optional[str] = Field(default=None, description="Error message if any step failed")
    error_step: Optional[Literal["download", "convert", "index"]] = Field(
        default=None, description="Step where error occurred"
    )
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    last_attempt: Optional[datetime] = Field(default=None, description="Last processing attempt")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Status record created")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last status update")

    @property
    def is_complete(self) -> bool:
        """Check if all processing steps are complete."""
        return self.pdf_downloaded and self.md_converted and self.indexed

    @property
    def has_error(self) -> bool:
        """Check if there's an error."""
        return self.error_message is not None

    @property
    def progress_percentage(self) -> float:
        """Get processing progress as percentage."""
        steps = [self.pdf_downloaded, self.md_converted, self.indexed]
        return (sum(steps) / len(steps)) * 100

    def mark_downloaded(self, path: str, size_bytes: Optional[int] = None) -> None:
        """Mark PDF as downloaded."""
        self.pdf_downloaded = True
        self.pdf_path = path
        self.pdf_size_bytes = size_bytes
        self.pdf_download_time = datetime.now()
        self.updated_at = datetime.now()
        self._clear_error_if_step("download")

    def mark_converted(self, path: str) -> None:
        """Mark as converted to Markdown."""
        self.md_converted = True
        self.md_path = path
        self.md_conversion_time = datetime.now()
        self.updated_at = datetime.now()
        self._clear_error_if_step("convert")

    def mark_indexed(self, chunk_count: int, embedding_model: str) -> None:
        """Mark as indexed."""
        self.indexed = True
        self.chunk_count = chunk_count
        self.embedding_model = embedding_model
        self.index_time = datetime.now()
        self.updated_at = datetime.now()
        self._clear_error_if_step("index")

    def mark_error(
        self,
        step: Literal["download", "convert", "index"],
        message: str,
    ) -> None:
        """Record an error."""
        self.error_step = step
        self.error_message = message
        self.retry_count += 1
        self.last_attempt = datetime.now()
        self.updated_at = datetime.now()

    def _clear_error_if_step(self, step: str) -> None:
        """Clear error if it was for this step."""
        if self.error_step == step:
            self.error_message = None
            self.error_step = None

    def reset(self) -> None:
        """Reset all status flags."""
        self.pdf_downloaded = False
        self.pdf_path = None
        self.pdf_size_bytes = None
        self.pdf_download_time = None
        self.md_converted = False
        self.md_path = None
        self.md_conversion_time = None
        self.indexed = False
        self.index_time = None
        self.chunk_count = None
        self.embedding_model = None
        self.error_message = None
        self.error_step = None
        self.retry_count = 0
        self.last_attempt = None
        self.updated_at = datetime.now()


# =============================================================================
# Crawl Task Model
# =============================================================================


class CrawlProgress(BaseModel):
    """Progress information for a crawl task."""

    total_papers: int = Field(default=0, ge=0, description="Total papers to crawl")
    crawled_papers: int = Field(default=0, ge=0, description="Papers crawled so far")
    downloaded_pdfs: int = Field(default=0, ge=0, description="PDFs downloaded")
    converted_mds: int = Field(default=0, ge=0, description="Markdowns converted")
    indexed_papers: int = Field(default=0, ge=0, description="Papers indexed")
    failed_papers: int = Field(default=0, ge=0, description="Papers that failed processing")
    skipped_papers: int = Field(default=0, ge=0, description="Papers skipped (already processed)")

    current_conference: Optional[Conference] = Field(
        default=None, description="Currently processing conference"
    )
    current_year: Optional[int] = Field(default=None, description="Currently processing year")
    current_paper_id: Optional[str] = Field(
        default=None, description="Currently processing paper ID"
    )

    started_at: Optional[datetime] = Field(default=None, description="Crawl start time")
    last_update: datetime = Field(default_factory=datetime.now, description="Last progress update")
    estimated_completion: Optional[datetime] = Field(
        default=None, description="Estimated completion time"
    )

    @property
    def percentage(self) -> float:
        """Get progress percentage."""
        if self.total_papers == 0:
            return 0.0
        return (self.crawled_papers / self.total_papers) * 100

    @property
    def success_rate(self) -> float:
        """Get success rate percentage."""
        processed = self.crawled_papers - self.skipped_papers
        if processed == 0:
            return 100.0
        return ((processed - self.failed_papers) / processed) * 100

    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None

    @property
    def papers_per_second(self) -> float:
        """Get crawling speed."""
        elapsed = self.elapsed_time
        if elapsed and elapsed > 0:
            return self.crawled_papers / elapsed
        return 0.0


class CrawlTask(BaseModel):
    """
    Crawl task definition and state.

    Defines what to crawl and tracks progress.
    """

    # Task identification
    id: str = Field(..., description="Unique task identifier")
    name: Optional[str] = Field(default=None, description="Human-readable task name")
    description: Optional[str] = Field(default=None, description="Task description")

    # Target specification
    target_conferences: list[Conference] = Field(
        ..., min_length=1, description="Conferences to crawl"
    )
    year_start: int = Field(..., ge=2000, le=2100, description="Start year (inclusive)")
    year_end: int = Field(..., ge=2000, le=2100, description="End year (inclusive)")

    # Filter conditions
    paper_types: Optional[list[PaperType]] = Field(
        default=None, description="Paper types to include (None = all)"
    )
    min_citations: Optional[int] = Field(
        default=None, ge=0, description="Minimum citation count filter"
    )
    max_papers_per_venue: Optional[int] = Field(
        default=None, ge=1, description="Maximum papers per conference/year"
    )
    include_keywords: Optional[list[str]] = Field(
        default=None, description="Keywords that must appear in title/abstract"
    )
    exclude_keywords: Optional[list[str]] = Field(
        default=None, description="Keywords to exclude"
    )

    # Processing options
    download_pdf: bool = Field(default=True, description="Download PDF files")
    convert_to_md: bool = Field(default=True, description="Convert PDFs to Markdown")
    index_for_search: bool = Field(default=True, description="Index papers for search")
    update_existing: bool = Field(default=False, description="Re-process existing papers")
    max_concurrent: int = Field(default=5, ge=1, le=20, description="Max concurrent downloads")
    retry_failed: bool = Field(default=True, description="Retry previously failed papers")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")

    # State
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    progress: CrawlProgress = Field(
        default_factory=CrawlProgress, description="Progress information"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation time")
    started_at: Optional[datetime] = Field(default=None, description="Task start time")
    completed_at: Optional[datetime] = Field(default=None, description="Task completion time")
    paused_at: Optional[datetime] = Field(default=None, description="Task pause time")

    # Error tracking
    errors: list[str] = Field(default_factory=list, description="Error messages encountered")

    @field_validator("year_end")
    @classmethod
    def validate_year_range(cls, v: int, info) -> int:
        """Ensure year_end >= year_start."""
        if "year_start" in info.data and v < info.data["year_start"]:
            raise ValueError("year_end must be >= year_start")
        return v

    @property
    def year_range(self) -> list[int]:
        """Get list of years to crawl."""
        return list(range(self.year_start, self.year_end + 1))

    @property
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        """Check if task has finished (completed, failed, or cancelled)."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get task duration in seconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def start(self) -> None:
        """Start the task."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
        self.progress.started_at = self.started_at

    def pause(self) -> None:
        """Pause the task."""
        self.status = TaskStatus.PAUSED
        self.paused_at = datetime.now()

    def resume(self) -> None:
        """Resume the task."""
        self.status = TaskStatus.RUNNING
        self.paused_at = None

    def complete(self) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.errors.append(error)

    def cancel(self) -> None:
        """Cancel the task."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(f"[{datetime.now().isoformat()}] {error}")

    def get_filter(self) -> PaperFilter:
        """Generate a PaperFilter from task configuration."""
        return PaperFilter(
            conferences=self.target_conferences,
            year_range=(self.year_start, self.year_end),
            paper_types=self.paper_types,
            min_citations=self.min_citations,
        )

    def summary(self) -> dict:
        """Get task summary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "conferences": [c.value for c in self.target_conferences],
            "years": f"{self.year_start}-{self.year_end}",
            "progress": f"{self.progress.percentage:.1f}%",
            "crawled": f"{self.progress.crawled_papers}/{self.progress.total_papers}",
            "failed": self.progress.failed_papers,
            "duration": f"{self.duration_seconds:.1f}s" if self.duration_seconds else None,
            "errors": len(self.errors),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_paper(
    id: str,
    title: str,
    authors: list[str],
    year: int,
    conference: Conference | str,
    **kwargs,
) -> Paper:
    """Convenience function to create a Paper with minimal required fields."""
    if isinstance(conference, str):
        try:
            conference = Conference(conference)
        except ValueError:
            conference = Conference.OTHER
    return Paper(
        id=id,
        title=title,
        authors=authors,
        year=year,
        conference=conference,
        **kwargs,
    )


def create_crawl_task(
    id: str,
    conferences: list[Conference | str],
    year_start: int,
    year_end: int,
    **kwargs,
) -> CrawlTask:
    """Convenience function to create a CrawlTask."""
    normalized_conferences = []
    for conf in conferences:
        if isinstance(conf, str):
            try:
                normalized_conferences.append(Conference(conf))
            except ValueError:
                normalized_conferences.append(Conference.OTHER)
        else:
            normalized_conferences.append(conf)

    return CrawlTask(
        id=id,
        target_conferences=normalized_conferences,
        year_start=year_start,
        year_end=year_end,
        **kwargs,
    )
