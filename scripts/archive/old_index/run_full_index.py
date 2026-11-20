"""
全量索引脚本 - 带进度日志和内存监控
"""
import sys
sys.path.insert(0, '.')

import gc
import time
import psutil
import torch
from pathlib import Path
from datetime import datetime

from core.indexing.chunker import TextChunker
from core.indexing.embedder import Embedder
from core.storage.chroma_store import ChromaStore

# 日志文件
LOG_FILE = Path('data/index_progress.log')

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{timestamp}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def main():
    log('=== Starting Full Index ===')

    # 初始化
    chunker = TextChunker()
    embedder = Embedder()
    embedder.load_model()
    store = ChromaStore('papers')
    store.clear()

    device_info = embedder.get_device_info()
    log(f'Device: {device_info["device"]}, Model: {device_info["model_name"]}')

    # 获取所有论文
    papers = list(Path('data/papers/markdown').glob('*.md'))
    total = len(papers)
    log(f'Total papers: {total}')

    total_chunks = 0
    start_time = time.time()

    for i, paper in enumerate(papers):
        try:
            # 读取
            content = paper.read_text(encoding='utf-8')

            # 分块
            chunks = chunker.chunk_document(paper.stem, content)

            # 嵌入
            embedded = embedder.embed_chunks(chunks)

            # 存储
            store.add_chunks(embedded)

            total_chunks += len(chunks)

            # 清理
            del content, chunks, embedded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 进度日志（每10篇）
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                mem = get_memory_mb()
                rate = (i + 1) / elapsed * 60  # papers/min
                eta = (total - i - 1) / rate if rate > 0 else 0
                log(f'Progress: {i+1}/{total} | Chunks: {total_chunks} | Mem: {mem:.0f}MB | Rate: {rate:.1f}/min | ETA: {eta:.0f}min')

                # 内存警告
                if mem > 6000:
                    log(f'WARNING: Memory high ({mem:.0f}MB)')
                    gc.collect()

        except Exception as e:
            log(f'ERROR on {paper.name}: {e}')

    elapsed = time.time() - start_time
    log(f'=== Completed ===')
    log(f'Total papers: {total}')
    log(f'Total chunks: {total_chunks}')
    log(f'Total time: {elapsed/60:.1f} min')
    log(f'Final DB count: {store.count()}')

if __name__ == '__main__':
    main()
