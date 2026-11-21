#!/usr/bin/env python
"""
Reload embeddings from .npy file into ChromaDB
"""
import json
import numpy as np
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "papers" / "embeddings"
INDEX_DIR = DATA_DIR / "index"
PAPERS_JSON = DATA_DIR / "papers" / "metadata" / "top_1000_papers.json"

def main():
    print("=" * 60)
    print("Reloading embeddings into ChromaDB")
    print("=" * 60)

    # Load embeddings
    embeddings_file = EMBEDDINGS_DIR / "embeddings.npy"
    paper_ids_file = EMBEDDINGS_DIR / "paper_ids.json"
    metadata_file = EMBEDDINGS_DIR / "metadata.json"

    print(f"\n1. Loading embeddings from: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    print(f"   Shape: {embeddings.shape}")

    print(f"\n2. Loading paper IDs from: {paper_ids_file}")
    with open(paper_ids_file, 'r') as f:
        paper_ids = json.load(f)
    print(f"   Count: {len(paper_ids)}")

    print(f"\n3. Loading paper metadata from: {PAPERS_JSON}")
    with open(PAPERS_JSON, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    print(f"   Count: {len(papers)}")

    # Create paper lookup
    paper_lookup = {p["id"]: p for p in papers}

    # Initialize ChromaDB
    chroma_dir = INDEX_DIR / "chroma"
    print(f"\n4. Initializing ChromaDB at: {chroma_dir}")

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False)
    )

    # Delete existing collection
    collection_name = "papers"
    try:
        client.delete_collection(collection_name)
        print(f"   Deleted existing collection: {collection_name}")
    except Exception:
        pass

    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"   Created collection: {collection_name}")

    # Prepare data
    print("\n5. Preparing data for insertion...")
    ids = []
    docs = []
    metadatas = []
    valid_embeddings = []

    for i, paper_id in enumerate(paper_ids):
        paper = paper_lookup.get(paper_id)
        if paper:
            ids.append(paper_id)
            docs.append(f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}")
            metadatas.append({
                "title": paper.get("title", ""),
                "authors": ", ".join(paper.get("authors", [])) if isinstance(paper.get("authors"), list) else str(paper.get("authors", "")),
                "conference": paper.get("conference", ""),
                "year": int(paper.get("year", 0)) if paper.get("year") else 0,
                "paper_type": paper.get("paper_type", ""),
                "impact_score": float(paper.get("impact_score", 0.0)),
                "impact_rank": int(paper.get("impact_rank", 0)) if paper.get("impact_rank") else 0,
                "url": paper.get("url", ""),
                "doc_id": paper_id,
                "paper_id": paper_id,
            })
            valid_embeddings.append(embeddings[i].tolist())

    print(f"   Valid papers: {len(ids)}")

    # Insert in batches
    print("\n6. Inserting into ChromaDB...")
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=valid_embeddings[i:end_idx],
            documents=docs[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        print(f"   Added {end_idx}/{len(ids)} papers...")

    # Verify
    print("\n7. Verification:")
    print(f"   Collection count: {collection.count()}")

    # Test query
    print("\n8. Test similarity search:")
    results = collection.query(
        query_embeddings=[valid_embeddings[0]],
        n_results=3
    )
    print(f"   Top 3 results for first paper:")
    for j, doc_id in enumerate(results["ids"][0]):
        title = results["metadatas"][0][j].get("title", "")[:60]
        print(f"   {j+1}. {doc_id}: {title}...")

    print("\n" + "=" * 60)
    print("Embeddings reloaded successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
