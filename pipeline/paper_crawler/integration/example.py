"""
Example - 论文索引集成使用示例

展示如何将论文爬取模块与检索系统集成使用。

运行方式:
    cd E:/JETB/pythons/sf/retrival
    python -m paper_crawler.integration.example
"""

import asyncio
import sys
from pathlib import Path

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def example_basic_usage():
    """基本使用示例 - 索引和搜索单篇论文"""
    print("=" * 60)
    print("示例 1: 基本使用 - 索引和搜索论文")
    print("=" * 60)

    from pipeline.paper_crawler.integration import PaperIndexer, PaperIndexConfig
    from pipeline.paper_crawler.models.paper import Paper, Author, PaperSource, PaperType, PaperMetadata

    # 创建配置
    config = PaperIndexConfig(
        collection_name="example_papers",
        include_summary=True,
        generate_structured_summary=True,
    )

    # 创建索引器
    indexer = PaperIndexer(config=config)

    # 创建示例论文
    paper = Paper(
        id="example_paper_001",
        title="Attention Is All You Need",
        abstract=(
            "The dominant sequence transduction models are based on complex recurrent or "
            "convolutional neural networks that include an encoder and a decoder. "
            "The best performing models also connect the encoder and decoder through an "
            "attention mechanism. We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
        ),
        authors=[
            Author(name="Ashish Vaswani", affiliation="Google Brain"),
            Author(name="Noam Shazeer", affiliation="Google Brain"),
            Author(name="Niki Parmar", affiliation="Google Research"),
        ],
        source=PaperSource.ARXIV,
        url="https://arxiv.org/abs/1706.03762",
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        paper_type=PaperType.LONG_PAPER,
        metadata=PaperMetadata(
            venue="NeurIPS",
            year=2017,
            keywords=["transformer", "attention", "neural networks", "NLP"],
            arxiv_id="1706.03762",
            citations=50000,
        ),
    )

    # 使用论文内容直接索引 (而不是从文件读取)
    content = f"""
# {paper.title}

## Abstract

{paper.abstract}

## Introduction

Recurrent neural networks, long short-term memory and gated recurrent neural networks
in particular, have been firmly established as state of the art approaches in sequence
modeling and transduction problems such as language modeling and machine translation.

The Transformer allows for significantly more parallelization and can reach a new
state of the art in translation quality after being trained for as little as twelve
hours on eight P100 GPUs.

## The Transformer Architecture

The Transformer follows an encoder-decoder structure using stacked self-attention
and point-wise, fully connected layers for both the encoder and decoder.

### Encoder

The encoder is composed of a stack of N = 6 identical layers. Each layer has two
sub-layers: multi-head self-attention mechanism and position-wise fully connected
feed-forward network.

### Decoder

The decoder is also composed of a stack of N = 6 identical layers. In addition to
the two sub-layers in each encoder layer, the decoder inserts a third sub-layer,
which performs multi-head attention over the output of the encoder stack.

## Attention Mechanism

An attention function can be described as mapping a query and a set of key-value
pairs to an output, where the query, keys, values, and output are all vectors.

### Scaled Dot-Product Attention

We compute the dot products of the query with all keys, divide each by sqrt(d_k),
and apply a softmax function to obtain the weights on the values.

### Multi-Head Attention

Instead of performing a single attention function with d_model-dimensional keys,
values and queries, we found it beneficial to linearly project the queries, keys
and values h times with different, learned linear projections.

## Conclusion

In this work, we presented the Transformer, the first sequence transduction model
based entirely on attention, replacing the recurrent layers most commonly used in
encoder-decoder architectures with multi-headed self-attention.
"""

    # 索引论文
    print(f"\n索引论文: {paper.title}")
    doc_id = await indexer.index_paper(paper=paper, content=content)
    print(f"创建的文档 ID: {doc_id}")

    # 搜索论文
    print("\n搜索查询: 'attention mechanism transformer'")
    results = await indexer.search_papers(
        query="attention mechanism transformer",
        top_k=5,
        method="hybrid",
    )

    print(f"\n找到 {len(results)} 篇相关论文:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r.title}")
        print(f"   得分: {r.score:.4f}")
        print(f"   命中 chunk 数: {r.hit_count}")
        print(f"   作者: {', '.join(r.authors[:3])}")
        if r.venue:
            print(f"   会议: {r.venue} {r.year}")

    # 获取统计信息
    stats = indexer.get_stats()
    print(f"\n索引统计:")
    print(f"  - 总论文数: {stats['total_papers']}")
    print(f"  - 总文档数: {stats['total_documents']}")
    print(f"  - 总分块数: {stats['total_chunks']}")

    return indexer


async def example_batch_indexing():
    """批量索引示例"""
    print("\n" + "=" * 60)
    print("示例 2: 批量索引多篇论文")
    print("=" * 60)

    from pipeline.paper_crawler.integration import PaperIndexer, PaperIndexConfig
    from pipeline.paper_crawler.models.paper import Paper, Author, PaperSource, PaperType, PaperMetadata

    # 创建带进度回调的配置
    def progress_callback(current: int, total: int, title: str):
        print(f"  [{current}/{total}] 正在索引: {title[:50]}...")

    config = PaperIndexConfig(
        collection_name="batch_example",
        progress_callback=progress_callback,
        max_concurrent_indexing=3,
    )

    indexer = PaperIndexer(config=config)

    # 创建多篇示例论文
    papers = [
        Paper(
            id="bert_2019",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            abstract="We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers.",
            authors=[Author(name="Jacob Devlin"), Author(name="Ming-Wei Chang")],
            source=PaperSource.ARXIV,
            url="https://arxiv.org/abs/1810.04805",
            paper_type=PaperType.LONG_PAPER,
            metadata=PaperMetadata(venue="NAACL", year=2019, keywords=["BERT", "NLP"]),
        ),
        Paper(
            id="gpt3_2020",
            title="Language Models are Few-Shot Learners",
            abstract="We train GPT-3, an autoregressive language model with 175 billion parameters.",
            authors=[Author(name="Tom Brown"), Author(name="Benjamin Mann")],
            source=PaperSource.ARXIV,
            url="https://arxiv.org/abs/2005.14165",
            paper_type=PaperType.LONG_PAPER,
            metadata=PaperMetadata(venue="NeurIPS", year=2020, keywords=["GPT-3", "LLM"]),
        ),
        Paper(
            id="vit_2021",
            title="An Image is Worth 16x16 Words: Transformers for Image Recognition",
            abstract="While the Transformer architecture has become the de-facto standard for NLP tasks, its applications to computer vision remain limited.",
            authors=[Author(name="Alexey Dosovitskiy"), Author(name="Lucas Beyer")],
            source=PaperSource.ARXIV,
            url="https://arxiv.org/abs/2010.11929",
            paper_type=PaperType.LONG_PAPER,
            metadata=PaperMetadata(venue="ICLR", year=2021, keywords=["ViT", "Vision"]),
        ),
    ]

    # 准备内容映射
    contents = {
        "bert_2019": "# BERT\n\nBERT is a method of pre-training language representations...",
        "gpt3_2020": "# GPT-3\n\nGPT-3 achieves strong performance on many NLP tasks...",
        "vit_2021": "# Vision Transformer\n\nViT applies Transformer to image patches...",
    }

    # 批量索引
    print("\n开始批量索引:")
    results = await indexer.index_batch(papers=papers, contents=contents)

    print(f"\n索引完成: {len(results)} 篇论文")
    for paper_id, doc_id in results.items():
        status = "成功" if doc_id else "失败"
        print(f"  - {paper_id}: {status}")

    return indexer


async def example_with_pipeline():
    """与 PaperPipeline 集成示例"""
    print("\n" + "=" * 60)
    print("示例 3: 与 PaperPipeline 集成")
    print("=" * 60)

    print("""
此示例展示如何将 PaperPipeline 的输出直接索引。
实际使用时的代码:

    from pipeline.paper_crawler.workflows.paper_pipeline import PaperPipeline
    from pipeline.paper_crawler.integration import index_papers_from_pipeline
    from pipeline.paper_crawler.config import config

    # 运行爬取流程
    pipeline = PaperPipeline()
    state = pipeline.run(
        query="large language models",
        source="arxiv",
        limit=10,
        download=True,
        convert=True,
    )

    # 索引爬取的论文
    paper_doc_mapping = await index_papers_from_pipeline(
        papers=state["papers"],
        markdown_dir=config.markdown_dir,
    )

    print(f"成功索引 {len(paper_doc_mapping)} 篇论文")
""")


async def example_fastapi_integration():
    """FastAPI 集成示例"""
    print("\n" + "=" * 60)
    print("示例 4: FastAPI 集成")
    print("=" * 60)

    print("""
将论文搜索 API 集成到现有 FastAPI 应用:

    # main.py
    from fastapi import FastAPI
    from pipeline.paper_crawler.integration import mount_paper_routes, PaperIndexer

    app = FastAPI(title="My Retrieval System")

    # 可选: 创建共享的 indexer 实例
    indexer = PaperIndexer()

    # 挂载论文路由 (添加 /api/papers/* 端点)
    mount_paper_routes(app, indexer=indexer)

    # 现在可以访问以下端点:
    # POST /api/papers/search     - 搜索论文
    # GET  /api/papers/search     - 搜索论文 (GET 方式)
    # POST /api/papers/index      - 索引单篇论文
    # POST /api/papers/index/batch - 批量索引
    # GET  /api/papers/{paper_id} - 获取论文详情
    # DELETE /api/papers/{paper_id} - 删除论文索引
    # GET  /api/papers/           - 列出所有已索引论文
    # GET  /api/papers/stats      - 获取统计信息

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

搜索 API 调用示例:

    curl -X POST "http://localhost:8000/api/papers/search" \\
         -H "Content-Type: application/json" \\
         -d '{
           "query": "transformer attention",
           "top_k": 10,
           "method": "hybrid",
           "filters": {"year": 2023}
         }'
""")


async def example_search_with_filters():
    """带过滤条件的搜索示例"""
    print("\n" + "=" * 60)
    print("示例 5: 带过滤条件的搜索")
    print("=" * 60)

    from pipeline.paper_crawler.integration import PaperIndexer

    indexer = PaperIndexer()

    print("""
搜索时可以使用多种过滤条件:

    # 按年份过滤
    results = await indexer.search_papers(
        query="neural networks",
        filters={"year": 2023},
    )

    # 按来源过滤
    results = await indexer.search_papers(
        query="neural networks",
        filters={"source": "arxiv"},
    )

    # 按会议过滤
    results = await indexer.search_papers(
        query="neural networks",
        filters={"venue": "NeurIPS"},
    )

    # 组合过滤
    results = await indexer.search_papers(
        query="transformer",
        filters={
            "year": [2022, 2023],  # 列表表示 OR
            "source": "arxiv",
        },
    )

    # 选择不同的检索方法
    bm25_results = await indexer.search_papers(query="...", method="bm25")
    vector_results = await indexer.search_papers(query="...", method="vector")
    hybrid_results = await indexer.search_papers(query="...", method="hybrid")
""")


async def main():
    """运行所有示例"""
    print("\n论文索引集成示例")
    print("=" * 60)

    # 运行基本示例
    try:
        await example_basic_usage()
    except Exception as e:
        print(f"基本示例出错 (可能缺少依赖): {e}")

    # 运行批量索引示例
    try:
        await example_batch_indexing()
    except Exception as e:
        print(f"批量索引示例出错: {e}")

    # 展示其他集成方式
    await example_with_pipeline()
    await example_fastapi_integration()
    await example_search_with_filters()

    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
