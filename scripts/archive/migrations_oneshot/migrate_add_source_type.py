#!/usr/bin/env python
"""
Migration script to add source_type and data_source fields to existing ChromaDB documents.

This script:
1. Reads all existing chunks from ChromaDB
2. Adds source_type and data_source fields to metadata if missing
3. Updates the chunks back to ChromaDB

Usage:
    python scripts/migrate_add_source_type.py [--dry-run] [--default-source-type=paper] [--default-data-source=system]
"""
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings


def migrate_chromadb(dry_run: bool = True,
                     default_source_type: str = "paper",
                     default_data_source: str = "system",
                     batch_size: int = 100):
    """
    Add source_type and data_source to all documents that don't have them.

    Args:
        dry_run: If True, only report what would be changed without making changes
        default_source_type: Default value for source_type (paper|webpage|upload)
        default_data_source: Default value for data_source (system|user)
        batch_size: Number of documents to process at a time
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
    print("Loading all documents...")
    results = collection.get(include=["metadatas", "embeddings", "documents"])

    ids_to_update = []
    new_metadatas = []
    updated_count = 0
    skipped_count = 0

    for i, chunk_id in enumerate(results["ids"]):
        metadata = results["metadatas"][i] if results["metadatas"] else {}

        needs_update = False
        new_metadata = metadata.copy()

        # Check if source_type is missing
        if "source_type" not in metadata or not metadata.get("source_type"):
            new_metadata["source_type"] = default_source_type
            needs_update = True

        # Check if data_source is missing
        if "data_source" not in metadata or not metadata.get("data_source"):
            new_metadata["data_source"] = default_data_source
            needs_update = True

        if needs_update:
            ids_to_update.append(chunk_id)
            new_metadatas.append(new_metadata)
            updated_count += 1
        else:
            skipped_count += 1

    print(f"\nMigration summary:")
    print(f"  Chunks needing update: {updated_count}")
    print(f"  Chunks already migrated: {skipped_count}")

    if dry_run:
        print(f"\n[DRY RUN] Would update {updated_count} chunks with:")
        print(f"  - source_type = '{default_source_type}'")
        print(f"  - data_source = '{default_data_source}'")
        print("\nRun without --dry-run to apply changes.")
        return

    if updated_count == 0:
        print("No chunks need updating. Migration complete.")
        return

    # Update in batches
    print(f"\nUpdating {updated_count} chunks in batches of {batch_size}...")
    for i in range(0, len(ids_to_update), batch_size):
        batch_ids = ids_to_update[i:i + batch_size]
        batch_metadatas = new_metadatas[i:i + batch_size]

        collection.update(
            ids=batch_ids,
            metadatas=batch_metadatas
        )

        progress = min(i + batch_size, len(ids_to_update))
        print(f"  Updated {progress}/{len(ids_to_update)} chunks...")

    print(f"\nMigration complete! Updated {updated_count} chunks.")

    # Verify
    print("\nVerifying migration...")
    verify_results = collection.get(limit=5, include=["metadatas"])
    for i, meta in enumerate(verify_results["metadatas"][:5]):
        source_type = meta.get("source_type", "NOT FOUND")
        data_source = meta.get("data_source", "NOT FOUND")
        print(f"  Sample {i+1}: source_type={source_type}, data_source={data_source}")


def main():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB to add source_type and data_source fields")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be changed without making changes")
    parser.add_argument("--default-source-type", type=str, default="paper",
                        choices=["paper", "webpage", "upload"],
                        help="Default source_type for existing documents (default: paper)")
    parser.add_argument("--default-data-source", type=str, default="system",
                        choices=["system", "user"],
                        help="Default data_source for existing documents (default: system)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for updates (default: 100)")

    args = parser.parse_args()

    migrate_chromadb(
        dry_run=args.dry_run,
        default_source_type=args.default_source_type,
        default_data_source=args.default_data_source,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
