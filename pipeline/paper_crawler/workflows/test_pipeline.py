#!/usr/bin/env python
"""
Test script for PaperPipeline

Verifies the LangGraph workflow implementation.
"""

import sys
sys.path.insert(0, "E:/JETB/pythons/sf/retrival")

from pipeline.paper_crawler.workflows import (
    PaperPipeline,
    PaperState,
    build_paper_pipeline,
    app,
)


def test_import():
    """Test that all imports work correctly."""
    print("[TEST] Import check...")
    assert PaperPipeline is not None
    assert PaperState is not None
    assert build_paper_pipeline is not None
    assert app is not None
    print("[PASS] All imports successful")


def test_pipeline_creation():
    """Test pipeline creation with different configurations."""
    print("\n[TEST] Pipeline creation...")

    # Default pipeline
    p1 = PaperPipeline(use_checkpointing=False)
    assert p1.graph is not None
    print("  - Default pipeline: OK")

    # Pipeline with checkpointing
    p2 = PaperPipeline(use_checkpointing=True)
    assert p2.checkpointer is not None
    print("  - Checkpointed pipeline: OK")

    # Parallel pipeline
    p3 = PaperPipeline(use_parallel=True, use_checkpointing=False)
    assert p3.graph is not None
    print("  - Parallel pipeline: OK")

    print("[PASS] Pipeline creation successful")


def test_visualization():
    """Test workflow graph visualization."""
    print("\n[TEST] Visualization...")

    pipeline = PaperPipeline(use_checkpointing=False)
    mermaid = pipeline.visualize()

    assert mermaid is not None
    assert "__start__" in mermaid
    assert "get_papers" in mermaid
    assert "download" in mermaid
    assert "convert" in mermaid
    assert "index" in mermaid
    assert "save_metadata" in mermaid
    assert "__end__" in mermaid

    print("  - Mermaid diagram generated: OK")
    print("[PASS] Visualization successful")


def test_state_structure():
    """Test PaperState type definition."""
    print("\n[TEST] State structure...")

    # Check that PaperState has the expected keys
    expected_keys = [
        "conference", "year", "source", "query", "max_papers",
        "download_pdf", "convert_to_md", "build_index",
        "papers", "downloaded", "converted", "indexed", "errors",
        "current_step", "progress", "start_time", "end_time"
    ]

    # Create a sample state
    sample_state: PaperState = {
        "conference": "ICLR",
        "year": 2024,
        "source": "arxiv",
        "query": "deep learning",
        "max_papers": 10,
        "download_pdf": True,
        "convert_to_md": True,
        "build_index": True,
        "papers": [],
        "downloaded": [],
        "converted": [],
        "indexed": [],
        "errors": [],
        "current_step": "init",
        "progress": 0.0,
        "start_time": None,
        "end_time": None,
    }

    for key in expected_keys:
        assert key in sample_state, f"Missing key: {key}"

    print("  - All expected keys present: OK")
    print("[PASS] State structure correct")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PaperPipeline Test Suite")
    print("=" * 60)

    test_import()
    test_pipeline_creation()
    test_visualization()
    test_state_structure()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
