"""功能3 substrate 升级 —— L2（正文分块）按 paper 均值池化，产出 per-paper 向量。

为什么：L1（title+abstract 摘要）模板按 ingest 批次不一致（NeurIPS/CVPR 无 abstract
+ 样板词），带 venue 批次效应；且无-abstract 批次话题表达薄。改用每篇所有 L2 正文
分块 dense 向量的均值当 paper 向量 —— 正文模板一致、内容厚、免 L1 样板/连字噪声。

**必须在 4090 盒本机跑**（intra-box gRPC ~3600 pts/s；从 Arc 走 overlay 要 5h）。
对线上 Qdrant 只读 scroll（会吃约 51GB 页缓存压力，read-only 不爆 commit）。

输出：data/landscape/l2pool.npz {paper_ids[str], vecs[N,1024] f32(L2-normalized)}。
用法（4090 本机）：/data/arc/order/.venv/bin/python scripts/landscape/pool_l2.py --out data/landscape/l2pool.npz
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

COLLECTION = "papers"


def _client(host, port, grpc_port):
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(v, None)
    os.environ["NO_PROXY"] = "*"
    from qdrant_client import QdrantClient

    return QdrantClient(host=host, port=port, grpc_port=grpc_port,
                        prefer_grpc=True, timeout=300)


def pool_l2(client, *, batch=2048, max_papers=200000, log_every=1_000_000):
    """Stream level=2 点，按 paper_id 累加 dense 向量均值。返回 (paper_ids, vecs)。"""
    from qdrant_client import models as qm

    flt = qm.Filter(must=[qm.FieldCondition(key="level", match=qm.MatchValue(value=2))])
    idx_of: dict[str, int] = {}
    sums = np.zeros((max_papers, 1024), dtype=np.float32)
    counts = np.zeros(max_papers, dtype=np.int64)
    n_pts = 0
    offset = None
    t0 = time.time()
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, scroll_filter=flt, limit=batch,
            with_payload=["paper_id"], with_vectors=["dense"], offset=offset)
        if not points:
            break
        rows = []
        vecs = []
        for p in points:
            pid = (p.payload or {}).get("paper_id") or ""
            if not pid:
                continue
            vec = p.vector.get("dense") if isinstance(p.vector, dict) else p.vector
            if vec is None:
                continue
            r = idx_of.get(pid)
            if r is None:
                r = len(idx_of)
                if r >= max_papers:
                    raise SystemExit(f"max_papers={max_papers} exceeded; bump it")
                idx_of[pid] = r
            rows.append(r)
            vecs.append(vec)
        if rows:
            ridx = np.asarray(rows, dtype=np.int64)
            vmat = np.asarray(vecs, dtype=np.float32)
            np.add.at(sums, ridx, vmat)
            np.add.at(counts, ridx, 1)
        n_pts += len(points)
        if n_pts % log_every < batch:
            dt = time.time() - t0
            print(f"  pooled {n_pts:,} pts / {len(idx_of):,} papers (+{dt:.0f}s, "
                  f"{n_pts/max(dt,1):.0f} pts/s)", flush=True)
        if offset is None:
            break
    n = len(idx_of)
    print(f"pool_l2: {n_pts:,} L2 pts -> {n:,} papers in {time.time()-t0:.0f}s", flush=True)
    paper_ids = np.empty(n, dtype=object)
    for pid, r in idx_of.items():
        paper_ids[r] = pid
    sums = sums[:n]
    counts = counts[:n].reshape(-1, 1)
    mean = sums / np.maximum(counts, 1)
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return paper_ids.astype(str), (mean / norms).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--grpc-port", type=int, default=6334)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out", default="data/landscape/l2pool.npz")
    a = ap.parse_args()

    client = _client(a.host, a.port, a.grpc_port)
    paper_ids, vecs = pool_l2(client, batch=a.batch)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, paper_ids=paper_ids, vecs=vecs)
    print(f"wrote {out} ({out.stat().st_size/1e6:.0f} MB, {vecs.shape})", flush=True)


if __name__ == "__main__":
    main()
