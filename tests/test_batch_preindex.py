"""Test batch preindex: GPU-saturated embedding across multiple papers."""
import asyncio
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper, build_default_corpus_root
from pipeline.paper_crawler.integration.index_integration import PaperIndexConfig, PaperIndexer
from pipeline.paper_crawler.runtime_files import load_json_with_recovery


def _find_unindexed(indexer: PaperIndexer, n: int) -> list[CorpusPaper]:
    manifest = load_json_with_recovery(
        build_default_corpus_root() / "manifests" / "paper_manifest.json",
        default_value=[],
    )
    records = [
        CorpusPaper.from_manifest_record(r)
        for r in manifest
        if r.get("markdown_path") and not r.get("extract_error")
    ]
    result = []
    for r in records:
        if indexer.get_indexed_doc_id(r.paper_id) is None and Path(r.markdown_path).exists():
            result.append(r)
        if len(result) >= n:
            break
    return result


class TestBatchPreindex(unittest.TestCase):
    def setUp(self):
        self.indexer = PaperIndexer(
            PaperIndexConfig(
                collection_name="papers",
                data_source="system",
                embedding_device="cuda",
                require_gpu=True,
            )
        )

    def test_batch_index_papers_exists(self):
        """batch_index_papers method must exist on PaperIndexer."""
        self.assertTrue(hasattr(self.indexer, "batch_index_papers"))

    def test_batch_indexes_multiple_papers(self):
        """Batch should index N papers and return correct count."""
        papers = _find_unindexed(self.indexer, 3)
        if len(papers) < 3:
            self.skipTest("Not enough unindexed papers")

        result = asyncio.run(
            self.indexer.batch_index_papers(
                papers=papers,
                skip_whoosh=True,
            )
        )
        self.assertEqual(result["indexed"], 3)
        self.assertEqual(result["failed"], 0)

    def test_batch_faster_than_sequential(self):
        """Batch embed should be faster per-paper than sequential."""
        papers = _find_unindexed(self.indexer, 6)
        if len(papers) < 6:
            self.skipTest("Not enough unindexed papers")

        t0 = time.perf_counter()
        result = asyncio.run(
            self.indexer.batch_index_papers(
                papers=papers,
                skip_whoosh=True,
            )
        )
        elapsed = time.perf_counter() - t0
        per_paper = elapsed / max(result["indexed"], 1)

        print(f"\nBatch: {result['indexed']} papers in {elapsed:.1f}s = {per_paper:.1f}s/paper")
        self.assertLess(per_paper, 6.0, "Batch should be faster than 8s/paper sequential")


if __name__ == "__main__":
    unittest.main()
