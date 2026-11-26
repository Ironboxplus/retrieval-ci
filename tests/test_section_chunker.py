"""TDD tests for section-aware markdown chunker."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import Chunk

SAMPLE_PAPER_MD = """---
paper_id: cvpr-2024-test
title: Test Paper on Vision Transformers
conference: CVPR
year: 2024
---

# Test Paper on Vision Transformers

Authors: Alice, Bob, Charlie

## Abstract

Vision Transformers have shown remarkable performance in image classification.
This paper proposes a novel approach to improve efficiency while maintaining accuracy.
We achieve state-of-the-art results on ImageNet and COCO benchmarks.

## Introduction

Deep learning has revolutionized computer vision. Convolutional Neural Networks
dominated the field for years, but recently Vision Transformers (ViT) have emerged
as a competitive alternative. However, their quadratic attention complexity limits
scalability to high-resolution images.

In this work, we address this limitation by introducing a linear-complexity
attention mechanism that preserves the global receptive field of standard attention.

## Method

### Overview

Our method consists of three main components: a patch embedding module,
a linear attention block, and a classification head.

### Linear Attention

We replace the standard softmax attention with a kernel-based approximation.
Given query Q, key K, and value V matrices, standard attention computes:
Attention(Q,K,V) = softmax(QK^T / sqrt(d))V

Our linear attention instead computes:
LinearAttn(Q,K,V) = phi(Q)(phi(K)^T V) / (phi(Q) sum(phi(K)))

where phi is a feature map that approximates the softmax kernel.

### Training Details

We train all models using AdamW optimizer with cosine learning rate schedule.
Base learning rate is 1e-3 with weight decay 0.05. We use standard data
augmentation including RandAugment and Mixup.

## Experiments

### ImageNet Classification

We evaluate on ImageNet-1K with 224x224 resolution.
Our model achieves 84.2% top-1 accuracy with only 60% of the FLOPs.

### COCO Object Detection

Using Mask R-CNN framework, our backbone achieves 47.3 AP on COCO val2017,
outperforming DeiT-S by 1.5 AP while being 2x faster.

## Conclusion

We presented a linear-complexity Vision Transformer that achieves competitive
performance with significantly reduced computational cost. Future work includes
extending to video understanding and 3D point cloud processing.
"""

SHORT_PAPER_MD = """---
paper_id: short-test
title: Short Paper
conference: AAAI
year: 2025
---

# Short Paper

This is a very short paper with no sections.
"""


class TestSectionChunkerImport(unittest.TestCase):
    def test_can_import(self):
        from core.indexing.chunker import SectionAwareChunker
        self.assertTrue(callable(SectionAwareChunker))


class TestSectionChunkerBasic(unittest.TestCase):
    def setUp(self):
        from core.indexing.chunker import SectionAwareChunker
        self.chunker = SectionAwareChunker()

    def test_returns_chunks(self):
        chunks = self.chunker.chunk_document("doc1", SAMPLE_PAPER_MD)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], Chunk)

    def test_all_chunks_have_doc_id(self):
        chunks = self.chunker.chunk_document("doc1", SAMPLE_PAPER_MD)
        for c in chunks:
            self.assertEqual(c.doc_id, "doc1")

    def test_all_chunks_have_content(self):
        chunks = self.chunker.chunk_document("doc1", SAMPLE_PAPER_MD)
        for c in chunks:
            self.assertTrue(len(c.content.strip()) > 0)


class TestSectionMetadata(unittest.TestCase):
    def setUp(self):
        from core.indexing.chunker import SectionAwareChunker
        self.chunker = SectionAwareChunker()
        self.chunks = self.chunker.chunk_document("doc1", SAMPLE_PAPER_MD)

    def test_chunks_have_section_name(self):
        for c in self.chunks:
            self.assertIn("section_name", c.metadata)

    def test_known_sections_present(self):
        section_names = {c.metadata["section_name"] for c in self.chunks}
        self.assertTrue(section_names & {"Abstract", "Method", "Experiments", "Conclusion"})

    def test_subsections_preserved(self):
        section_names = {c.metadata["section_name"] for c in self.chunks}
        self.assertTrue(
            section_names & {"Linear Attention", "ImageNet Classification", "Overview"}
            or section_names & {"Method > Linear Attention", "Method > Overview"}
        )


class TestSectionChunkerStripsYamlFrontmatter(unittest.TestCase):
    def setUp(self):
        from core.indexing.chunker import SectionAwareChunker
        self.chunker = SectionAwareChunker()

    def test_no_frontmatter_in_chunks(self):
        chunks = self.chunker.chunk_document("doc1", SAMPLE_PAPER_MD)
        for c in chunks:
            self.assertNotIn("paper_id:", c.content)
            self.assertNotIn("---\n", c.content[:10])


class TestShortDocument(unittest.TestCase):
    def setUp(self):
        from core.indexing.chunker import SectionAwareChunker
        self.chunker = SectionAwareChunker()

    def test_short_doc_produces_chunks(self):
        chunks = self.chunker.chunk_document("doc1", SHORT_PAPER_MD)
        self.assertGreater(len(chunks), 0)

    def test_short_doc_no_empty_chunks(self):
        chunks = self.chunker.chunk_document("doc1", SHORT_PAPER_MD)
        for c in chunks:
            self.assertTrue(len(c.content.strip()) > 10)


class TestChunkSizeControl(unittest.TestCase):
    def setUp(self):
        from core.indexing.chunker import SectionAwareChunker
        self.chunker = SectionAwareChunker(max_chunk_chars=500)

    def test_long_sections_are_split(self):
        chunks = self.chunker.chunk_document("doc1", SAMPLE_PAPER_MD)
        for c in chunks:
            self.assertLessEqual(len(c.content), 600)

    def test_split_sections_keep_metadata(self):
        chunks = self.chunker.chunk_document("doc1", SAMPLE_PAPER_MD)
        # Subsections under Method get their own section names (Overview, Linear Attention, etc.)
        method_subsections = [c for c in chunks if c.metadata.get("section_name", "") in
                             ("Overview", "Linear Attention", "Training Details")]
        self.assertGreater(len(method_subsections), 0)


class TestBackwardCompatibility(unittest.TestCase):
    """Old TextChunker must still work."""

    def test_old_chunker_still_works(self):
        from core.indexing.chunker import TextChunker
        chunker = TextChunker()
        chunks = chunker.chunk_document("doc1", "Hello world. " * 100)
        self.assertGreater(len(chunks), 0)


if __name__ == "__main__":
    unittest.main()
