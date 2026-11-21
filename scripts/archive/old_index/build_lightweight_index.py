"""
Lightweight Paper Search Index Builder

This script builds a lightweight search index for papers using:
1. Whoosh for full-text search (BM25)
2. Simple JSON index for quick keyword/author/conference lookups

No embedding models or ChromaDB required - pure text-based search.
"""

import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Whoosh imports
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, NUMERIC, KEYWORD, STORED
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer
from whoosh.scoring import BM25F

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_JSON = DATA_DIR / "papers" / "metadata" / "top_1000_papers.json"
WHOOSH_INDEX_DIR = DATA_DIR / "index" / "whoosh_papers"
SIMPLE_INDEX_PATH = DATA_DIR / "index" / "simple_search_index.json"


def load_papers(json_path: Path) -> List[Dict[str, Any]]:
    """Load papers from JSON file"""
    print(f"Loading papers from: {json_path}")

    if not json_path.exists():
        raise FileNotFoundError(f"Papers file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"  Loaded {len(papers)} papers")
    return papers


def create_whoosh_schema() -> Schema:
    """Create Whoosh schema for paper indexing"""
    return Schema(
        # Core identifiers
        id=ID(stored=True, unique=True),

        # Searchable text fields
        title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        abstract=TEXT(analyzer=StemmingAnalyzer()),
        authors=TEXT(stored=True),  # Stored for display
        keywords=KEYWORD(stored=True, commas=True, lowercase=True),

        # Metadata fields
        conference=KEYWORD(stored=True, lowercase=True),
        year=NUMERIC(stored=True, sortable=True),
        paper_type=KEYWORD(stored=True, lowercase=True),
        venue=TEXT(stored=True),

        # URLs
        url=STORED,
        pdf_url=STORED,

        # Impact metrics
        impact_score=NUMERIC(stored=True, sortable=True),
        impact_rank=NUMERIC(stored=True, sortable=True),
        impact_tier=KEYWORD(stored=True),
        is_high_impact=STORED,
    )


def build_whoosh_index(papers: List[Dict[str, Any]], index_dir: Path) -> index.Index:
    """Build Whoosh full-text index"""
    print(f"\nBuilding Whoosh index at: {index_dir}")

    # Create index directory
    os.makedirs(index_dir, exist_ok=True)

    # Create schema and index
    schema = create_whoosh_schema()

    # Create fresh index (overwrite if exists)
    if index.exists_in(str(index_dir)):
        print("  Removing existing index...")
        import shutil
        shutil.rmtree(index_dir)
        os.makedirs(index_dir, exist_ok=True)

    ix = index.create_in(str(index_dir), schema)
    writer = ix.writer()

    indexed_count = 0
    error_count = 0

    for i, paper in enumerate(papers):
        try:
            # Extract authors as string
            authors = paper.get("authors", [])
            if isinstance(authors, list):
                authors_str = ", ".join(str(a) for a in authors)
            else:
                authors_str = str(authors)

            # Extract keywords as comma-separated string
            keywords = paper.get("keywords", [])
            if isinstance(keywords, list):
                keywords_str = ", ".join(str(k) for k in keywords)
            else:
                keywords_str = str(keywords) if keywords else ""

            # Add document to index
            writer.add_document(
                id=str(paper.get("id", f"paper_{i}")),
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                authors=authors_str,
                keywords=keywords_str,
                conference=paper.get("conference", ""),
                year=paper.get("year", 0),
                paper_type=paper.get("paper_type", ""),
                venue=paper.get("venue", ""),
                url=paper.get("url", ""),
                pdf_url=paper.get("pdf_url", ""),
                impact_score=paper.get("impact_score", 0.0),
                impact_rank=paper.get("impact_rank", 0),
                impact_tier=paper.get("impact_tier", ""),
                is_high_impact=paper.get("is_high_impact", False),
            )
            indexed_count += 1

            # Progress update
            if (i + 1) % 100 == 0:
                print(f"  Indexed {i + 1}/{len(papers)} papers...")

        except Exception as e:
            error_count += 1
            print(f"  Error indexing paper {paper.get('id', i)}: {e}")

    # Commit changes
    print("  Committing index...")
    writer.commit()

    print(f"  Indexed {indexed_count} papers successfully")
    if error_count > 0:
        print(f"  {error_count} errors occurred")

    return ix


def extract_title_keywords(title: str) -> List[str]:
    """Extract meaningful keywords from title"""
    # Remove common stop words and extract significant terms
    stop_words = {
        'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'can', 'with', 'without', 'by', 'from', 'as',
        'this', 'that', 'these', 'those', 'it', 'its', 'via', 'using'
    }

    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z]{2,}\b', title.lower())
    keywords = [w for w in words if w not in stop_words]

    return keywords


def build_simple_index(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build simple JSON-based search index"""
    print("\nBuilding simple JSON index...")

    # Index structures
    title_keyword_index = defaultdict(list)  # keyword -> [paper_ids]
    author_index = defaultdict(list)  # author_name -> [paper_ids]
    conference_year_index = defaultdict(list)  # "CONF_YEAR" -> [paper_ids]
    impact_tier_index = defaultdict(list)  # tier -> [paper_ids]

    # Paper lookup table
    papers_lookup = {}  # id -> paper summary

    for paper in papers:
        paper_id = paper.get("id", "")
        if not paper_id:
            continue

        # Store paper summary for quick lookup
        papers_lookup[paper_id] = {
            "id": paper_id,
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "conference": paper.get("conference", ""),
            "year": paper.get("year", 0),
            "paper_type": paper.get("paper_type", ""),
            "impact_score": paper.get("impact_score", 0.0),
            "impact_rank": paper.get("impact_rank", 0),
            "impact_tier": paper.get("impact_tier", ""),
            "url": paper.get("url", ""),
        }

        # Index by title keywords
        title = paper.get("title", "")
        for keyword in extract_title_keywords(title):
            title_keyword_index[keyword].append(paper_id)

        # Also index explicit keywords
        for kw in paper.get("keywords", []):
            if isinstance(kw, str):
                for word in extract_title_keywords(kw):
                    title_keyword_index[word].append(paper_id)

        # Index by author
        authors = paper.get("authors", [])
        for author in authors:
            author_name = author if isinstance(author, str) else str(author)
            # Normalize author name
            author_key = author_name.lower().strip()
            author_index[author_key].append(paper_id)

            # Also index by last name
            parts = author_name.split()
            if len(parts) > 1:
                last_name = parts[-1].lower().strip()
                author_index[last_name].append(paper_id)

        # Index by conference + year
        conference = paper.get("conference", "")
        year = paper.get("year", 0)
        if conference and year:
            conf_year_key = f"{conference}_{year}"
            conference_year_index[conf_year_key].append(paper_id)
            # Also index by conference alone
            conference_year_index[conference].append(paper_id)
            # And by year alone
            conference_year_index[str(year)].append(paper_id)

        # Index by impact tier
        tier = paper.get("impact_tier", "")
        if tier:
            impact_tier_index[tier].append(paper_id)

    # Create final index structure
    simple_index = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_papers": len(papers),
            "indexed_papers": len(papers_lookup),
        },
        "indexes": {
            "by_title_keyword": dict(title_keyword_index),
            "by_author": dict(author_index),
            "by_conference_year": dict(conference_year_index),
            "by_impact_tier": dict(impact_tier_index),
        },
        "papers": papers_lookup,
    }

    # Print statistics
    print(f"  Title keywords indexed: {len(title_keyword_index)}")
    print(f"  Authors indexed: {len(author_index)}")
    print(f"  Conference/year combinations: {len(conference_year_index)}")
    print(f"  Impact tiers: {len(impact_tier_index)}")

    return simple_index


def save_simple_index(simple_index: Dict[str, Any], output_path: Path):
    """Save simple index to JSON file"""
    print(f"\nSaving simple index to: {output_path}")

    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simple_index, f, ensure_ascii=False, indent=2)

    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  Index size: {file_size:.2f} MB")


def test_whoosh_search(index_dir: Path, queries: List[str]):
    """Test Whoosh search functionality"""
    print("\n" + "=" * 60)
    print("Testing Whoosh Search")
    print("=" * 60)

    ix = index.open_dir(str(index_dir))

    # Create multi-field parser
    parser = MultifieldParser(
        ["title", "abstract", "keywords", "authors"],
        schema=ix.schema,
        group=OrGroup
    )

    with ix.searcher(weighting=BM25F()) as searcher:
        for query_str in queries:
            print(f"\nQuery: '{query_str}'")
            print("-" * 40)

            try:
                query = parser.parse(query_str)
                results = searcher.search(query, limit=5)

                print(f"Found {len(results)} results (showing top 5):")

                for i, hit in enumerate(results, 1):
                    title = hit.get("title", "N/A")[:60]
                    score = hit.score
                    year = hit.get("year", "N/A")
                    conf = hit.get("conference", "N/A")
                    impact_rank = hit.get("impact_rank", "N/A")

                    print(f"  {i}. [{conf} {year}] {title}...")
                    print(f"     Score: {score:.4f}, Impact Rank: #{impact_rank}")

            except Exception as e:
                print(f"  Error: {e}")


def test_simple_search(index_path: Path, queries: List[str]):
    """Test simple JSON index search"""
    print("\n" + "=" * 60)
    print("Testing Simple JSON Index Search")
    print("=" * 60)

    with open(index_path, "r", encoding="utf-8") as f:
        simple_index = json.load(f)

    title_idx = simple_index["indexes"]["by_title_keyword"]
    papers = simple_index["papers"]

    for query_str in queries:
        print(f"\nQuery: '{query_str}'")
        print("-" * 40)

        # Extract keywords from query
        keywords = extract_title_keywords(query_str)

        # Find papers matching any keyword
        matching_ids = set()
        for kw in keywords:
            if kw in title_idx:
                matching_ids.update(title_idx[kw])

        print(f"Found {len(matching_ids)} papers matching keywords: {keywords}")

        # Sort by impact rank and show top 5
        matching_papers = [papers[pid] for pid in matching_ids if pid in papers]
        matching_papers.sort(key=lambda p: p.get("impact_rank", 999999))

        for i, paper in enumerate(matching_papers[:5], 1):
            title = paper.get("title", "N/A")[:60]
            conf = paper.get("conference", "N/A")
            year = paper.get("year", "N/A")
            impact_rank = paper.get("impact_rank", "N/A")

            print(f"  {i}. [{conf} {year}] {title}...")
            print(f"     Impact Rank: #{impact_rank}")


def main():
    print("=" * 60)
    print("Lightweight Paper Search Index Builder")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Load papers
    try:
        papers = load_papers(PAPERS_JSON)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Step 2: Build Whoosh index
    ix = build_whoosh_index(papers, WHOOSH_INDEX_DIR)

    # Step 3: Build simple JSON index
    simple_index = build_simple_index(papers)
    save_simple_index(simple_index, SIMPLE_INDEX_PATH)

    # Step 4: Test search functionality
    test_queries = [
        "transformer",
        "attention mechanism",
    ]

    test_whoosh_search(WHOOSH_INDEX_DIR, test_queries)
    test_simple_search(SIMPLE_INDEX_PATH, test_queries)

    # Summary
    print("\n" + "=" * 60)
    print("Index Building Complete!")
    print("=" * 60)
    print(f"Whoosh index: {WHOOSH_INDEX_DIR}")
    print(f"Simple index: {SIMPLE_INDEX_PATH}")
    print(f"Total papers indexed: {len(papers)}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
