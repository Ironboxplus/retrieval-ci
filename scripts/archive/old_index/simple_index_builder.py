"""
Simple Paper Index Builder

Indexes papers using the existing IndexManager directly,
without the PaperIndexer wrapper for simplicity.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.indexing.index_manager import IndexManager
from app.models.schemas import FileType


def main():
    print("=" * 60)
    print("Simple Paper Index Builder")
    print("=" * 60)

    # Find the metadata file
    metadata_file = project_root / "data" / "papers" / "metadata" / "papers_20260119_172502_with_paths_converted.json"

    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        sys.exit(1)

    print(f"\nMetadata file: {metadata_file}")

    # Load papers data
    print("\nLoading papers...")
    with open(metadata_file, "r", encoding="utf-8") as f:
        papers_data = json.load(f)

    # Filter papers with markdown
    papers_with_md = []
    for p in papers_data:
        md_path = p.get("markdown_path")
        if md_path and Path(md_path).exists():
            papers_with_md.append(p)

    print(f"  Total papers: {len(papers_data)}")
    print(f"  Papers with markdown: {len(papers_with_md)}")

    if not papers_with_md:
        print("No papers with markdown to index!")
        sys.exit(1)

    # Initialize index manager
    print("\nInitializing IndexManager...")
    index_manager = IndexManager()

    print(f"  Current documents: {len(index_manager.documents)}")
    print(f"  Current chunks: {len(index_manager.chunks)}")

    # Index papers
    print("\nBuilding index (this may take a while)...")
    print("  - Loading embedding model...")

    start_time = datetime.now()
    indexed = 0
    errors = 0

    for i, p_data in enumerate(papers_with_md):
        try:
            paper_id = p_data.get("id", "unknown")
            title = p_data.get("title", "Untitled")
            md_path = Path(p_data["markdown_path"])

            # Read markdown content
            content = md_path.read_text(encoding="utf-8")

            # Build metadata
            metadata = {
                "paper_id": paper_id,
                "authors": [a.get("name", "") for a in p_data.get("authors", [])],
                "conference": p_data.get("conference", ""),
                "year": p_data.get("year", 0),
                "abstract": p_data.get("abstract", "")[:500],  # Truncate
                "keywords": p_data.get("keywords", []),
                "pdf_url": p_data.get("pdf_url"),
                "paper_page_url": p_data.get("paper_page_url"),
            }

            # Add document to index
            doc = index_manager.add_document(
                title=title,
                content=content,
                source=p_data.get("paper_page_url") or paper_id,
                source_type=FileType.PDF,
                metadata=metadata,
            )

            indexed += 1

            # Progress update
            if (i + 1) % 10 == 0 or (i + 1) == len(papers_with_md):
                elapsed = datetime.now() - start_time
                rate = indexed / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                remaining = (len(papers_with_md) - indexed) / rate if rate > 0 else 0
                print(f"  [{i + 1}/{len(papers_with_md)}] {title[:40]}... ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

        except Exception as e:
            errors += 1
            print(f"  Error indexing {p_data.get('id', 'unknown')}: {e}")
            continue

    elapsed = datetime.now() - start_time

    print()
    print("=" * 60)
    print("Indexing Complete!")
    print("=" * 60)
    print(f"  Papers indexed: {indexed}")
    print(f"  Errors: {errors}")
    print(f"  Time elapsed: {elapsed}")
    print(f"  Documents in index: {len(index_manager.documents)}")
    print(f"  Chunks in index: {len(index_manager.chunks)}")

    # Get index stats
    stats = index_manager.get_stats()
    print(f"\nIndex Stats:")
    print(f"  Vector index size: {stats.vector_index_size}")
    print(f"  BM25 index size: {stats.bm25_index_size}")
    print(f"  Chunk levels: {stats.chunk_levels}")


if __name__ == "__main__":
    main()
