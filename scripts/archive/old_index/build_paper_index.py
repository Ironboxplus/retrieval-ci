"""
Build Paper Index - Uses the PaperIndexer to create vector and BM25 indexes

This script reads the converted papers metadata and builds the full search index
using the existing retrieval infrastructure (ChromaDB + Whoosh).
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.paper_crawler.integration.index_integration import (
    PaperIndexer,
    PaperIndexConfig,
)
from pipeline.paper_crawler.models.paper import Paper, Author, Conference, PaperType


def load_papers_from_json(json_path: Path) -> list:
    """Load papers from JSON metadata file"""
    with open(json_path, "r", encoding="utf-8") as f:
        papers_data = json.load(f)

    papers = []
    for p_data in papers_data:
        # Skip papers without markdown_path
        if not p_data.get("markdown_path"):
            continue

        md_path = Path(p_data["markdown_path"])
        if not md_path.exists():
            continue

        try:
            # Convert conference string to enum
            conference = p_data.get("conference", "OTHER")
            if isinstance(conference, str):
                try:
                    conference = Conference(conference)
                except ValueError:
                    conference = Conference.OTHER

            # Convert paper_type string to enum
            paper_type = p_data.get("paper_type", "unknown")
            if isinstance(paper_type, str):
                try:
                    paper_type = PaperType(paper_type)
                except ValueError:
                    paper_type = PaperType.UNKNOWN

            # Build authors list
            authors = []
            for author_data in p_data.get("authors", []):
                if isinstance(author_data, dict):
                    authors.append(Author(
                        name=author_data.get("name", "Unknown"),
                        email=author_data.get("email"),
                        affiliation=author_data.get("affiliation"),
                    ))
                elif isinstance(author_data, str):
                    authors.append(Author(name=author_data))

            paper = Paper(
                id=p_data.get("id", md_path.stem),
                title=p_data.get("title", "Untitled"),
                authors=authors,
                abstract=p_data.get("abstract", ""),
                keywords=p_data.get("keywords", []),
                year=p_data.get("year", 2024),
                conference=conference,
                paper_type=paper_type,
                pdf_url=p_data.get("pdf_url"),
                paper_page_url=p_data.get("paper_page_url"),
                arxiv_id=p_data.get("arxiv_id"),
                code_url=p_data.get("code_url"),
                citations=p_data.get("citations", 0),
                source=p_data.get("source", "unknown"),
            )

            papers.append((paper, md_path))

        except Exception as e:
            print(f"Error loading paper {p_data.get('id', 'unknown')}: {e}")
            continue

    return papers


def progress_callback(current: int, total: int, title: str):
    """Progress callback for indexing"""
    if current % 10 == 0 or current == total:
        print(f"  [{current}/{total}] {title[:60]}...")


async def build_index(papers_with_paths: list, config: PaperIndexConfig) -> dict:
    """Build the index for all papers"""
    indexer = PaperIndexer(config=config)

    results = {}
    total = len(papers_with_paths)

    for i, (paper, md_path) in enumerate(papers_with_paths):
        try:
            # Read markdown content
            content = md_path.read_text(encoding="utf-8")

            # Index the paper
            doc_id = await indexer.index_paper(
                paper=paper,
                content=content,
            )

            results[paper.id] = doc_id

            # Progress update
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"  [{i + 1}/{total}] Indexed: {paper.title[:50]}...")

        except Exception as e:
            print(f"  Error indexing {paper.id}: {e}")
            continue

    return results


def main():
    print("=" * 60)
    print("Paper Index Builder")
    print("=" * 60)

    # Find the metadata file
    metadata_dir = project_root / "data" / "papers" / "metadata"

    # Use the converted file with markdown_path
    metadata_file = metadata_dir / "papers_20260119_172502_with_paths_converted.json"

    if not metadata_file.exists():
        # Find the most recent converted file
        converted_files = list(metadata_dir.glob("*_converted.json"))
        if converted_files:
            metadata_file = max(converted_files, key=lambda f: f.stat().st_mtime)
        else:
            print(f"Error: No converted metadata file found in {metadata_dir}")
            sys.exit(1)

    print(f"\nMetadata file: {metadata_file}")

    # Load papers
    print("\nLoading papers...")
    papers_with_paths = load_papers_from_json(metadata_file)
    print(f"  Found {len(papers_with_paths)} papers with markdown content")

    if not papers_with_paths:
        print("No papers to index!")
        sys.exit(1)

    # Configure indexer
    config = PaperIndexConfig(
        collection_name="papers",
        include_summary=True,
        index_metadata=True,
        generate_structured_summary=True,
        max_concurrent_indexing=1,  # Sequential for stability
    )

    # Build index
    print("\nBuilding index (this may take a while)...")
    print("  - Chunking documents")
    print("  - Generating embeddings")
    print("  - Storing in ChromaDB (vector)")
    print("  - Storing in Whoosh (BM25)")
    print()

    start_time = datetime.now()

    results = asyncio.run(build_index(papers_with_paths, config))

    elapsed = datetime.now() - start_time

    print()
    print("=" * 60)
    print("Indexing Complete!")
    print("=" * 60)
    print(f"  Papers indexed: {len(results)}")
    print(f"  Time elapsed: {elapsed}")
    print(f"  Index location: {project_root / 'data' / 'indices'}")

    # Save index mapping
    mapping_file = metadata_dir / "paper_index_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Mapping saved: {mapping_file}")


if __name__ == "__main__":
    main()
