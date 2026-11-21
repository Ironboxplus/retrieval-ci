"""Code knowledge-base: AST chunking + indexing + search over source repos.

Separate corpus from the paper corpus (its own Qdrant collection), but shares the
retrieval engine. Multi-language chunking is structure-aware via chonkie's CodeChunker
(bundled tree-sitter grammars) behind the CodeChunk interface.
"""
