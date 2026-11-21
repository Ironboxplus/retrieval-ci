"""
GPU-Accelerated Paper Index Builder

This script builds paper indexes using GPU acceleration:
1. Recalculates impact_score with improved formula
2. Generates embeddings using BAAI/bge-small-zh-v1.5 (GPU)
3. Saves embeddings to .npy files and optionally to ChromaDB
4. Updates papers JSON with new impact_score and impact_rank

Environment: E:\JETB\pythons\.venv
Working directory: E:\JETB\pythons\sf\retrival
"""

import json
import os
import sys
import time
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Check CUDA availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_JSON = DATA_DIR / "papers" / "metadata" / "top_1000_papers.json"
OUTPUT_DIR = DATA_DIR / "papers" / "embeddings"
INDEX_DIR = DATA_DIR / "index"


@dataclass
class ImpactConfig:
    """Configuration for impact score calculation"""
    # Paper type weights
    paper_type_weights: Dict[str, float] = None
    # Year weight parameters
    current_year: int = 2025
    year_decay_rate: float = 0.1  # How much older papers are penalized
    min_year_weight: float = 0.5  # Minimum weight for old papers
    # Citation parameters
    citation_log_base: float = 10.0  # Use log scale for citations
    citation_weight: float = 0.3  # Weight of citation component

    def __post_init__(self):
        if self.paper_type_weights is None:
            self.paper_type_weights = {
                "best_paper": 100,
                "outstanding": 90,
                "oral": 70,
                "spotlight": 50,
                "poster": 30,
                "unknown": 20,
            }


def calculate_impact_score(
    paper: Dict[str, Any],
    config: ImpactConfig
) -> float:
    """
    Calculate improved impact score for a paper.

    Formula:
    impact_score = paper_type_weight * year_weight + citation_bonus

    Where:
    - paper_type_weight: Based on paper type (best_paper=100, oral=70, etc.)
    - year_weight: Newer papers get higher weight (exponential decay)
    - citation_bonus: Log-scaled citation count contribution

    Args:
        paper: Paper dictionary with type, year, and citation info
        config: Configuration for score calculation

    Returns:
        Calculated impact score
    """
    # Get paper type weight
    paper_type = paper.get("paper_type", "unknown").lower()
    type_weight = config.paper_type_weights.get(paper_type, 20)

    # Calculate year weight (newer papers score higher)
    year = paper.get("year", 2020)
    years_old = max(0, config.current_year - year)
    year_weight = max(
        config.min_year_weight,
        1.0 - (years_old * config.year_decay_rate)
    )

    # Calculate citation bonus (log scale to prevent extreme values)
    citation_count = paper.get("citation_count", 0)
    if citation_count is None:
        citation_count = 0

    if citation_count > 0:
        citation_bonus = math.log(citation_count + 1, config.citation_log_base) * 10
    else:
        citation_bonus = 0

    # Combine scores
    # Base score from type and year
    base_score = type_weight * year_weight

    # Add citation bonus (weighted)
    final_score = base_score * (1 - config.citation_weight) + citation_bonus * config.citation_weight

    # Normalize to 0-100 range with some headroom for exceptional papers
    final_score = min(100, max(0, final_score))

    return round(final_score, 2)


def determine_impact_tier(score: float) -> str:
    """Determine impact tier based on score"""
    if score >= 80:
        return "S"  # Exceptional
    elif score >= 65:
        return "A"  # High impact
    elif score >= 50:
        return "B"  # Good impact
    elif score >= 35:
        return "C"  # Moderate impact
    else:
        return "D"  # Standard


def update_impact_scores(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Update impact scores for all papers and recalculate ranks.

    Args:
        papers: List of paper dictionaries

    Returns:
        Updated list of papers with new impact scores and ranks
    """
    print("\n[Step 1] Calculating new impact scores...")

    config = ImpactConfig()

    # Calculate scores for all papers
    for paper in papers:
        paper["impact_score"] = calculate_impact_score(paper, config)
        paper["impact_tier"] = determine_impact_tier(paper["impact_score"])

    # Sort by score to determine ranks
    papers_sorted = sorted(papers, key=lambda p: p["impact_score"], reverse=True)

    # Assign ranks
    for rank, paper in enumerate(papers_sorted, 1):
        paper["impact_rank"] = rank
        paper["is_high_impact"] = paper["impact_tier"] in ["S", "A"]

    # Print statistics
    tier_counts = {}
    for paper in papers:
        tier = paper["impact_tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print(f"  Total papers: {len(papers)}")
    print(f"  Impact tier distribution:")
    for tier in ["S", "A", "B", "C", "D"]:
        count = tier_counts.get(tier, 0)
        pct = count / len(papers) * 100
        print(f"    Tier {tier}: {count} papers ({pct:.1f}%)")

    # Show top 5 papers
    print(f"\n  Top 5 papers by impact score:")
    for i, paper in enumerate(papers_sorted[:5], 1):
        print(f"    {i}. [{paper['impact_tier']}] {paper['impact_score']:.1f} - {paper['title'][:50]}...")

    return papers


class GPUEmbedder:
    """GPU-accelerated text embedder using sentence-transformers"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = None,
        batch_size: int = 32
    ):
        """
        Initialize the embedder.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"\n[Embedder] Using device: {self.device}")
        if self.device == "cuda":
            print(f"[Embedder] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[Embedder] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def load_model(self):
        """Load the embedding model"""
        if self.model is not None:
            return

        print(f"\n[Embedder] Loading model: {self.model_name}")

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

        # Get embedding dimension
        test_embedding = self.model.encode("test", convert_to_numpy=True)
        self.dimension = len(test_embedding)

        print(f"[Embedder] Model loaded successfully")
        print(f"[Embedder] Embedding dimension: {self.dimension}")

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings (N x dimension)
        """
        self.load_model()

        print(f"\n[Embedder] Encoding {len(texts)} texts...")
        print(f"[Embedder] Batch size: {self.batch_size}")

        start_time = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Important for cosine similarity
        )

        elapsed = time.time() - start_time
        texts_per_second = len(texts) / elapsed

        print(f"[Embedder] Encoding complete in {elapsed:.1f}s ({texts_per_second:.1f} texts/s)")

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        self.load_model()
        return self.dimension


def prepare_paper_texts(papers: List[Dict[str, Any]]) -> List[str]:
    """
    Prepare text representations for embedding.

    For each paper, combines title and abstract with a separator.
    This gives the embedding model both the concise title and
    detailed abstract for better semantic representation.

    Args:
        papers: List of paper dictionaries

    Returns:
        List of text strings for embedding
    """
    texts = []

    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        # Combine title and abstract
        # Use a simple format that works well with most embedding models
        if abstract:
            text = f"{title}\n\n{abstract}"
        else:
            text = title

        # Truncate if too long (most models have a max token limit)
        # BGE models typically support 512 tokens
        max_chars = 2000  # Approximate limit
        if len(text) > max_chars:
            text = text[:max_chars]

        texts.append(text)

    return texts


def save_embeddings(
    embeddings: np.ndarray,
    paper_ids: List[str],
    output_dir: Path
) -> Dict[str, str]:
    """
    Save embeddings to .npy files.

    Creates:
    - embeddings.npy: The embedding matrix (N x dimension)
    - paper_ids.json: Mapping from index to paper ID
    - metadata.json: Information about the embeddings

    Args:
        embeddings: Numpy array of embeddings
        paper_ids: List of paper IDs corresponding to each embedding
        output_dir: Directory to save files

    Returns:
        Dictionary with paths to saved files
    """
    print(f"\n[Step 3] Saving embeddings to {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings as .npy
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"  Saved embeddings: {embeddings_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {embeddings_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Save paper ID mapping
    ids_path = output_dir / "paper_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(paper_ids, f, ensure_ascii=False, indent=2)
    print(f"  Saved paper IDs: {ids_path}")

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "num_papers": len(paper_ids),
        "embedding_dimension": embeddings.shape[1],
        "model_name": "BAAI/bge-small-zh-v1.5",
        "normalized": True,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    return {
        "embeddings": str(embeddings_path),
        "paper_ids": str(ids_path),
        "metadata": str(metadata_path),
    }


def save_to_chromadb(
    embeddings: np.ndarray,
    papers: List[Dict[str, Any]],
    collection_name: str = "papers_gpu"
) -> str:
    """
    Save embeddings to ChromaDB for vector search.

    Args:
        embeddings: Numpy array of embeddings
        papers: List of paper dictionaries
        collection_name: Name for the ChromaDB collection

    Returns:
        Path to ChromaDB directory
    """
    print(f"\n[Step 4] Saving to ChromaDB collection: {collection_name}")

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("  ChromaDB not installed, skipping...")
        return None

    chroma_dir = INDEX_DIR / "chroma"
    os.makedirs(chroma_dir, exist_ok=True)

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False)
    )

    # Delete existing collection if exists
    try:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection: {collection_name}")
    except Exception:
        pass

    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare data for insertion
    ids = [paper["id"] for paper in papers]
    documents = [f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}" for paper in papers]
    metadatas = [
        {
            "title": paper.get("title", ""),
            "authors": ", ".join(paper.get("authors", [])) if isinstance(paper.get("authors"), list) else str(paper.get("authors", "")),
            "conference": paper.get("conference", ""),
            "year": paper.get("year", 0),
            "paper_type": paper.get("paper_type", ""),
            "impact_score": paper.get("impact_score", 0.0),
            "impact_rank": paper.get("impact_rank", 0),
            "impact_tier": paper.get("impact_tier", ""),
            "url": paper.get("url", ""),
        }
        for paper in papers
    ]

    # Insert in batches (ChromaDB has limits)
    batch_size = 500
    for i in range(0, len(papers), batch_size):
        end_idx = min(i + batch_size, len(papers))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx].tolist(),
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        print(f"  Added {end_idx}/{len(papers)} papers...")

    print(f"  ChromaDB collection created: {collection_name}")
    print(f"  Location: {chroma_dir}")

    return str(chroma_dir)


def update_papers_json(
    papers: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Save updated papers to JSON file.

    Args:
        papers: List of updated paper dictionaries
        output_path: Path to save the JSON file
    """
    print(f"\n[Step 5] Saving updated papers JSON...")

    # Create backup of original
    if output_path.exists():
        backup_path = output_path.with_suffix(".json.bak")
        import shutil
        shutil.copy(output_path, backup_path)
        print(f"  Created backup: {backup_path}")

    # Save updated papers
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_path}")
    print(f"  Size: {file_size:.2f} MB")


def test_similarity_search(
    embeddings: np.ndarray,
    paper_ids: List[str],
    papers: List[Dict[str, Any]],
    embedder: GPUEmbedder,
    query: str = "transformer attention mechanism",
    top_k: int = 5
) -> None:
    """
    Test the embedding search with a sample query.

    Args:
        embeddings: Paper embeddings
        paper_ids: List of paper IDs
        papers: List of paper dictionaries
        embedder: The embedder to use for query
        query: Search query
        top_k: Number of results to show
    """
    print(f"\n[Test] Similarity search for: '{query}'")
    print("-" * 60)

    # Create ID to paper mapping
    id_to_paper = {p["id"]: p for p in papers}

    # Encode query
    query_embedding = embedder.embed_texts([query], show_progress=False)[0]

    # Calculate cosine similarities (embeddings are normalized)
    similarities = np.dot(embeddings, query_embedding)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"Top {top_k} results:")
    for i, idx in enumerate(top_indices, 1):
        paper_id = paper_ids[idx]
        paper = id_to_paper.get(paper_id, {})
        similarity = similarities[idx]

        title = paper.get("title", "Unknown")[:60]
        conf = paper.get("conference", "?")
        year = paper.get("year", "?")
        impact_rank = paper.get("impact_rank", "?")

        print(f"  {i}. [{conf} {year}] {title}...")
        print(f"     Similarity: {similarity:.4f}, Impact Rank: #{impact_rank}")


def main():
    """Main entry point"""
    print("=" * 60)
    print("GPU-Accelerated Paper Index Builder")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {PROJECT_ROOT}")

    # Load papers
    print(f"\n[Loading] Reading papers from: {PAPERS_JSON}")

    if not PAPERS_JSON.exists():
        print(f"Error: Papers file not found: {PAPERS_JSON}")
        sys.exit(1)

    with open(PAPERS_JSON, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"  Loaded {len(papers)} papers")

    # Step 1: Update impact scores
    papers = update_impact_scores(papers)

    # Step 2: Generate embeddings
    print("\n[Step 2] Generating embeddings with GPU...")

    # Initialize embedder
    embedder = GPUEmbedder(
        model_name="BAAI/bge-small-zh-v1.5",
        batch_size=64 if torch.cuda.is_available() else 16
    )

    # Prepare texts for embedding
    texts = prepare_paper_texts(papers)
    paper_ids = [paper["id"] for paper in papers]

    # Generate embeddings
    embeddings = embedder.embed_texts(texts, show_progress=True)

    # Step 3: Save embeddings to .npy
    save_embeddings(embeddings, paper_ids, OUTPUT_DIR)

    # Step 4: Save to ChromaDB (optional)
    save_to_chromadb(embeddings, papers)

    # Step 5: Update papers JSON
    update_papers_json(papers, PAPERS_JSON)

    # Test search
    test_similarity_search(
        embeddings, paper_ids, papers, embedder,
        query="large language model training"
    )

    test_similarity_search(
        embeddings, paper_ids, papers, embedder,
        query="image generation diffusion"
    )

    # Summary
    print("\n" + "=" * 60)
    print("Index Building Complete!")
    print("=" * 60)
    print(f"Papers processed: {len(papers)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Embeddings saved to: {OUTPUT_DIR}")
    print(f"Papers JSON updated: {PAPERS_JSON}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
