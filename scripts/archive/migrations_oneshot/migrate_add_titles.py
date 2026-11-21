#!/usr/bin/env python
"""
Migration script to add title field to existing ChromaDB documents.

This script:
1. Gets all unique doc_ids from ChromaDB
2. Finds corresponding markdown files
3. Extracts titles from markdown files (first # or ## heading after frontmatter)
4. Updates ChromaDB metadata with title field

Usage:
    python scripts/migrate_add_titles.py [--dry-run] [--batch-size=500]

Example:
    # Dry run to see what would be changed
    python scripts/migrate_add_titles.py --dry-run

    # Apply migration
    python scripts/migrate_add_titles.py
"""
import sys
import os
import re
import argparse
from pathlib import Path
from typing import Dict, Optional, Set
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings


# Markdown directory path
MARKDOWN_DIR = Path(r"E:\JETB\pythons\sf\retrival\data\papers\markdown")


def extract_title_from_markdown(file_path: Path) -> Optional[str]:
    """
    Extract title from markdown file.

    The title is extracted as:
    1. First # or ## heading after YAML frontmatter
    2. If multiple consecutive ## headings, combine them

    Args:
        file_path: Path to markdown file

    Returns:
        Extracted title or None if not found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None

    # Skip YAML frontmatter
    if content.startswith('---'):
        # Find the closing ---
        end_idx = content.find('---', 3)
        if end_idx != -1:
            content = content[end_idx + 3:].strip()

    # Split into lines
    lines = content.split('\n')

    title_parts = []
    in_title_section = False

    for line in lines:
        line = line.strip()

        # Skip empty lines at the beginning
        if not line and not in_title_section:
            continue

        # Check for heading (# or ##)
        if line.startswith('# ') or line.startswith('## '):
            # Extract heading text
            heading_text = re.sub(r'^#+\s*', '', line).strip()

            # Skip common non-title headings
            if heading_text.lower() in ['abstract', 'introduction', 'references', 'appendix']:
                if title_parts:
                    break
                continue

            title_parts.append(heading_text)
            in_title_section = True
        elif in_title_section:
            # Stop collecting if we hit a non-heading line (except empty lines between headings)
            if line and not line.startswith('#'):
                break

    if title_parts:
        # Combine multiple heading parts
        title = ' '.join(title_parts)
        # Clean up the title
        title = re.sub(r'\s+', ' ', title).strip()
        return title

    return None


def find_markdown_file(doc_id: str) -> Optional[Path]:
    """
    Find markdown file for a given doc_id.

    Doc_id format: paper_id (e.g., zzv4Bf50RW)
    File location: MARKDOWN_DIR/{venue}/{year}/{paper_id}.md

    Args:
        doc_id: Document ID from ChromaDB

    Returns:
        Path to markdown file or None if not found
    """
    # Search recursively for the file
    pattern = f"**/{doc_id}.md"
    matches = list(MARKDOWN_DIR.glob(pattern))

    if matches:
        return matches[0]

    return None


def build_doc_id_to_title_map(doc_ids: Set[str], verbose: bool = True) -> Dict[str, str]:
    """
    Build a mapping from doc_id to extracted title.

    Args:
        doc_ids: Set of document IDs
        verbose: Whether to print progress

    Returns:
        Dictionary mapping doc_id to title
    """
    doc_id_to_title = {}
    not_found_count = 0
    no_title_count = 0

    total = len(doc_ids)

    for i, doc_id in enumerate(doc_ids):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processing doc_id {i + 1}/{total}...")

        # Find markdown file
        md_path = find_markdown_file(doc_id)
        if not md_path:
            not_found_count += 1
            continue

        # Extract title
        title = extract_title_from_markdown(md_path)
        if title:
            doc_id_to_title[doc_id] = title
        else:
            no_title_count += 1

    if verbose:
        print(f"\n  Summary:")
        print(f"    Titles extracted: {len(doc_id_to_title)}")
        print(f"    Files not found: {not_found_count}")
        print(f"    No title extracted: {no_title_count}")

    return doc_id_to_title


def migrate_chromadb_titles(dry_run: bool = True, batch_size: int = 500):
    """
    Add title field to all ChromaDB chunks.

    Args:
        dry_run: If True, only report what would be changed
        batch_size: Number of chunks to update at a time
    """
    print(f"Connecting to ChromaDB at {settings.chroma_dir}...")
    client = chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False)
    )

    collection = client.get_collection("papers")
    total_count = collection.count()
    print(f"Found {total_count} chunks in collection 'papers'")

    if total_count == 0:
        print("No documents to migrate.")
        return

    # Get all documents
    print("\nLoading all documents from ChromaDB...")
    results = collection.get(include=["metadatas"])

    print(f"Loaded {len(results['ids'])} chunks")

    # Collect unique doc_ids and count chunks per doc_id
    doc_id_chunks = defaultdict(list)
    chunks_without_title = 0
    chunks_with_empty_title = 0

    for i, chunk_id in enumerate(results["ids"]):
        metadata = results["metadatas"][i] if results["metadatas"] else {}
        doc_id = metadata.get("doc_id", "")

        if doc_id:
            doc_id_chunks[doc_id].append((chunk_id, metadata))

        # Count chunks needing title
        title = metadata.get("title")
        if title is None:
            chunks_without_title += 1
        elif title == "" or title.strip() == "":
            chunks_with_empty_title += 1

    print(f"\nStatistics:")
    print(f"  Unique doc_ids: {len(doc_id_chunks)}")
    print(f"  Chunks without title field: {chunks_without_title}")
    print(f"  Chunks with empty title: {chunks_with_empty_title}")
    print(f"  Total needing migration: {chunks_without_title + chunks_with_empty_title}")

    # Build doc_id to title mapping
    print("\nExtracting titles from markdown files...")
    doc_id_to_title = build_doc_id_to_title_map(set(doc_id_chunks.keys()))

    # Prepare updates
    ids_to_update = []
    new_metadatas = []

    for doc_id, chunks in doc_id_chunks.items():
        title = doc_id_to_title.get(doc_id)
        if not title:
            continue

        for chunk_id, metadata in chunks:
            current_title = metadata.get("title")

            # Only update if title is missing or empty
            if current_title is None or current_title.strip() == "":
                new_metadata = metadata.copy()
                new_metadata["title"] = title
                ids_to_update.append(chunk_id)
                new_metadatas.append(new_metadata)

    print(f"\nMigration plan:")
    print(f"  Chunks to update: {len(ids_to_update)}")
    print(f"  Chunks skipped (already have title): {total_count - len(ids_to_update)}")

    if len(ids_to_update) > 0:
        # Show sample titles
        print(f"\n  Sample titles to be added:")
        sample_count = min(5, len(doc_id_to_title))
        for i, (doc_id, title) in enumerate(list(doc_id_to_title.items())[:sample_count]):
            print(f"    {i+1}. [{doc_id}] {title[:80]}{'...' if len(title) > 80 else ''}")

    if dry_run:
        print(f"\n[DRY RUN] Would update {len(ids_to_update)} chunks")
        print("Run without --dry-run to apply changes.")
        return

    if len(ids_to_update) == 0:
        print("\nNo chunks need updating. Migration complete.")
        return

    # Update in batches
    print(f"\nUpdating {len(ids_to_update)} chunks in batches of {batch_size}...")

    for i in range(0, len(ids_to_update), batch_size):
        batch_ids = ids_to_update[i:i + batch_size]
        batch_metadatas = new_metadatas[i:i + batch_size]

        try:
            collection.update(
                ids=batch_ids,
                metadatas=batch_metadatas
            )
        except Exception as e:
            print(f"  Error updating batch {i//batch_size + 1}: {e}")
            continue

        progress = min(i + batch_size, len(ids_to_update))
        print(f"  Updated {progress}/{len(ids_to_update)} chunks...")

    print(f"\nMigration complete! Updated {len(ids_to_update)} chunks.")

    # Verify
    print("\nVerifying migration...")
    verify_results = collection.get(limit=10, include=["metadatas"])
    has_title_count = 0
    for meta in verify_results["metadatas"]:
        title = meta.get("title", "")
        if title and title.strip():
            has_title_count += 1
    print(f"  Verification: {has_title_count}/10 sampled chunks have non-empty title")

    # Show sample results
    print("\n  Sample results:")
    for i, meta in enumerate(verify_results["metadatas"][:5]):
        doc_id = meta.get("doc_id", "N/A")
        title = meta.get("title", "N/A")
        if title and len(title) > 60:
            title = title[:60] + "..."
        print(f"    {i+1}. doc_id={doc_id}, title={title}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ChromaDB to add title field from markdown files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be changed without making changes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for updates (default: 500)"
    )

    args = parser.parse_args()

    migrate_chromadb_titles(
        dry_run=args.dry_run,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
