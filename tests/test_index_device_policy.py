import unittest

from pipeline.paper_crawler.integration.index_integration import PaperIndexConfig, PaperIndexer
from scripts.preindex_existing_extracted_corpus_papers import build_argument_parser
from app.services.index_service import IndexService


class DummyEmbedder:
    def __init__(self, device: str):
        self.device = device


class IndexDevicePolicyTests(unittest.TestCase):
    def test_index_service_rejects_cpu_when_gpu_required(self):
        service = IndexService(require_gpu=True)
        service._new_embedder = DummyEmbedder(device="cpu")

        with self.assertRaisesRegex(RuntimeError, "GPU indexing is required"):
            service._ensure_index_device_ready()

    def test_index_service_allows_cuda_when_gpu_required(self):
        service = IndexService(require_gpu=True)
        service._new_embedder = DummyEmbedder(device="cuda")

        service._ensure_index_device_ready()

    def test_paper_indexer_passes_explicit_embedding_policy_to_index_service(self):
        indexer = PaperIndexer(
            PaperIndexConfig(
                embedding_device="cuda",
                require_gpu=True,
            )
        )

        self.assertEqual(indexer.index_service._embedding_device, "cuda")
        self.assertTrue(indexer.index_service._require_gpu)

    def test_preindex_cli_requires_gpu_by_default(self):
        args = build_argument_parser().parse_args([])

        self.assertEqual(args.embedding_device, "cuda")
        self.assertFalse(args.allow_cpu)


if __name__ == "__main__":
    unittest.main()
