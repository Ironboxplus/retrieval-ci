"""功能3 轴1 —— 语料库主题 landscape 离线预计算管线（UMAP + HDBSCAN）。

读 4090 Qdrant 的 L1（level=1 = 每篇论文 title+abstract 摘要）dense 向量，降维聚类
打标，产出一个前端可直接喂 WebGL 散点图的 JSON。

  Qdrant(level=1, 1024d) --scroll(只读)--> X[N,1024]
     --UMAP--> xy[N,2] (可视化) + emb50[N,50] (聚类)
     --HDBSCAN--> cluster_id[N]
     --TF-IDF(标题)--> 每簇 top 关键词 = 簇标签
     --> data/landscape/landscape.json  ({meta, clusters, points})

独立模块：纯 qdrant_client + sklearn/umap/hdbscan，不 import app/*，对线上检索零侵入。
对 Qdrant 只读 scroll。重计算全程在本机离线跑，不碰 4090 后端进程。

用法（先 smoke 再放量，符合 smoke-first 铁律）：
    python scripts/landscape/build_landscape.py --sample 8000        # smoke：抽样验路线
    python scripts/landscape/build_landscape.py --sample 0           # 全量 94K
    python scripts/landscape/build_landscape.py --sample 8000 --smoke-report  # 打印簇质量
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

COLLECTION = "papers"
DEFAULT_HOST = "10.100.100.111"
DEFAULT_PORT = 6333
PAYLOAD_FIELDS = ["doc_id", "content", "venue", "year", "paper_id"]

# 领域高频虚词：聚类标签里没区分度，从 TF-IDF 标签剔掉。
DOMAIN_STOPWORDS = {
    "learning", "model", "models", "network", "networks", "neural", "deep",
    "via", "using", "based", "approach", "method", "methods", "towards",
    "framework", "novel", "efficient", "robust", "performance", "data",
    "task", "tasks", "system", "systems", "analysis", "study", "training",
    "learn", "new", "use", "results", "paper", "problem",
    # 出版/会议/年份样板词：簇按 venue/year 倾斜时会污染标签，剔掉。
    "published", "authors", "title", "conference", "proceedings", "preprint",
    "arxiv", "abstract", "supplementary", "appendix",
    "neurips", "icml", "iclr", "cvpr", "iccv", "eccv", "acl", "emnlp", "naacl",
    "aaai", "ijcai", "kdd", "sigir", "sigmod", "vldb", "www", "wsdm", "ccs",
    "2020", "2021", "2022", "2023", "2024", "2025", "2026",
}


def _client(host: str, port: int):
    # NO_PROXY：4090 在 lab overlay，走系统代理会挂（W 系列约定）。
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "*"
    from qdrant_client import QdrantClient

    return QdrantClient(host=host, port=port, timeout=120, prefer_grpc=False)


def export_l1(client, sample: int, batch: int = 512):
    """Scroll level=1 点（带向量），返回 (X[N,1024] float32, metas[list[dict]])。

    sample<=0 取全量；否则到 sample 即停。只读。
    """
    from qdrant_client import models as qm

    flt = qm.Filter(must=[qm.FieldCondition(key="level", match=qm.MatchValue(value=1))])
    vecs: list[np.ndarray] = []
    metas: list[dict] = []
    offset = None
    t0 = time.time()
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=flt,
            limit=batch,
            with_payload=PAYLOAD_FIELDS,
            with_vectors=["dense"],
            offset=offset,
        )
        for p in points:
            vec = p.vector.get("dense") if isinstance(p.vector, dict) else p.vector
            if vec is None:
                continue
            pl = p.payload or {}
            content = pl.get("content") or ""
            metas.append({
                "doc_id": pl.get("doc_id"),
                "paper_id": pl.get("paper_id") or "",
                "title": _title_of(content),
                "venue": pl.get("venue") or "",
                "year": pl.get("year"),
            })
            vecs.append(np.asarray(vec, dtype=np.float32))
        if sample and len(vecs) >= sample:
            vecs, metas = vecs[:sample], metas[:sample]
            break
        if offset is None:
            break
        if len(vecs) % 5120 == 0:
            print(f"  scrolled {len(vecs)} (+{time.time()-t0:.0f}s)", flush=True)
    X = np.vstack(vecs) if vecs else np.zeros((0, 1024), np.float32)
    print(f"export_l1: {X.shape[0]} L1 vectors in {time.time()-t0:.0f}s", flush=True)
    return X, metas


def _title_of(content: str) -> str:
    import unicodedata

    for line in (content or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # 部分 venue/ingest 批次的 L1 content 带字面 "Title:" 前缀（不一致），剥掉。
        low = line.lower()
        if low.startswith("title:"):
            line = line[len("title:"):].strip()
        if line:
            # NFKC：PDF 抽取的标题含连字 ﬁ/ﬂ/ﬀ（U+FB01…）→ 标签出现 "effcient/unifed"
            # 之类伪词；归一化映射回 "fi/fl/ffi"。
            return unicodedata.normalize("NFKC", line)[:300]
    return ""


def center_by_venue(X, metas):
    """按 venue 减质心去除 ingest-template 批次效应（provenance 偏移），再 L2 归一化。

    根因（W 待编号）：L1 content 模板按采集批次不一致 —— NeurIPS=`Title:…Published:
    NeurIPS ---`（无 abstract）、CVPR/ICML 多一行 `Authors:`、ICLR/EMNLP/AAAI=纯标题+
    真 abstract。BGE-M3 把这些样板词嵌成强 venue 方向 → UMAP 撕成 3 个 venue 岛。
    每 venue 减自身均值即移除该加性偏移，保留 venue 内的话题离散（话题方向跨 venue 共享）。
    """
    X = X.astype(np.float32, copy=True)
    venues = np.array([(m.get("venue") or "") for m in metas])
    for v in np.unique(venues):
        idx = np.where(venues == v)[0]
        if len(idx) >= 2:
            X[idx] -= X[idx].mean(0, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def reduce(X, *, n_neighbors=15, min_dist=0.1, cluster_dims=5, seed=42,
           cluster_seed=42):
    """UMAP → 2D(viz) + 低维(cluster)。返回 (xy[N,2], emb_c[N,cdims])。

    关键：聚类在 ~5D（不是 50D）—— 高维受维度灾难。BERTopic 同款配方。
    seed 控 2D viz（--no-seed 放开多核提速，layout 非确定无所谓）；cluster_seed 恒定
    42 锁死 5D 聚类空间 → **聚类/标签每次跑都一样**（修早期 --no-seed 每跑簇都变的坑）。
    """
    import umap

    t0 = time.time()
    # init="pca"：spectral 默认 init 在大图上会裂成多个不连通"岛"（94K 实测 3 块）；
    # PCA init 给确定性、全局连通的起点，UMAP 再局部细化 → 单块连通流形。
    xy = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                   metric="cosine", init="pca", random_state=seed).fit_transform(X)
    print(f"  UMAP 2D done (+{time.time()-t0:.0f}s)", flush=True)
    cdims = min(cluster_dims, X.shape[1])
    emb_c = umap.UMAP(n_components=cdims, n_neighbors=n_neighbors, min_dist=0.0,
                      metric="cosine", random_state=cluster_seed).fit_transform(X)
    print(f"  UMAP {cdims}D done (+{time.time()-t0:.0f}s)", flush=True)
    return np.asarray(xy), np.asarray(emb_c)


def scroll_l1_metas(client):
    """scroll level=1 建 {paper_id: {doc_id,title,venue,year}}（给 l2pool 配标题元数据）。"""
    from qdrant_client import models as qm

    flt = qm.Filter(must=[qm.FieldCondition(key="level", match=qm.MatchValue(value=1))])
    out: dict[str, dict] = {}
    offset = None
    t0 = time.time()
    while True:
        pts, offset = client.scroll(
            collection_name=COLLECTION, scroll_filter=flt, limit=512,
            with_payload=PAYLOAD_FIELDS, with_vectors=False, offset=offset)
        for p in pts:
            pl = p.payload or {}
            pid = pl.get("paper_id") or ""
            if pid:
                out[pid] = {"doc_id": pl.get("doc_id"),
                            "title": _title_of(pl.get("content") or ""),
                            "venue": pl.get("venue") or "", "year": pl.get("year")}
        if offset is None:
            break
    print(f"scroll_l1_metas: {len(out)} papers in {time.time()-t0:.0f}s", flush=True)
    return out


def load_l2pool(pool_file, client):
    """加载 pool_l2.py 产的 per-paper L2 均值向量，配 L1 标题元数据。返回 (X, metas)。"""
    z = np.load(pool_file, allow_pickle=True)
    pids = z["paper_ids"].astype(str)
    vecs = z["vecs"].astype(np.float32)
    meta_by_pid = scroll_l1_metas(client)
    X, metas = [], []
    miss = 0
    for pid, v in zip(pids, vecs):
        m = meta_by_pid.get(pid)
        if m is None:
            miss += 1
            continue
        X.append(v)
        metas.append({**m, "paper_id": pid})
    print(f"load_l2pool: {len(X)} paired (l2pool {len(pids)}, L1 {len(meta_by_pid)}, "
          f"miss {miss})", flush=True)
    return np.vstack(X) if X else np.zeros((0, 1024), np.float32), metas


def representative_titles(metas, emb_c, labels, *, per_cluster=18):
    """每簇取 emb_c 质心最近的若干篇标题，喂 LLM 起名。返回 {cid: [titles]}。"""
    out: dict[int, list] = {}
    for c in sorted(set(int(x) for x in labels) - {-1}):
        idx = np.where(labels == c)[0]
        cen = emb_c[idx].mean(0)
        d = ((emb_c[idx] - cen) ** 2).sum(1)
        nearest = idx[np.argsort(d)[:per_cluster]]
        out[c] = [metas[i].get("title") or "" for i in nearest]
    return out


def cluster(emb_c, *, min_cluster_size=50, min_samples=None, method="eom",
            epsilon=0.0):
    """HDBSCAN 聚类（便宜，调参主战场）。返回 labels[N]（-1=噪声）。"""
    import hdbscan

    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        metric="euclidean", cluster_selection_method=method,
        cluster_selection_epsilon=epsilon,
    ).fit_predict(emb_c)
    return np.asarray(labels)


def cluster_kmeans(emb_c, *, k=50, seed=42):
    """KMeans：每个点都有簇、零噪声，适合"全图都要上色"的 landscape。"""
    from sklearn.cluster import KMeans

    return np.asarray(KMeans(n_clusters=k, random_state=seed, n_init=10)
                      .fit_predict(emb_c))


def assign_noise_nearest(emb_c, labels):
    """把 HDBSCAN 噪声点(-1)分给最近簇质心，全覆盖。返回 (labels, is_core[bool])。

    保留 is_core 标记：core=密度峰真成员（高置信），assigned=贴边补进来的（灰一点）。
    （= BERTopic reduce_outliers 思路）
    """
    labels = np.asarray(labels).copy()
    is_core = labels != -1
    uniq = sorted(set(int(c) for c in labels) - {-1})
    if not uniq:
        return labels, is_core
    cents = np.vstack([emb_c[labels == c].mean(0) for c in uniq])
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx):
        d = ((emb_c[noise_idx][:, None, :] - cents[None, :, :]) ** 2).sum(-1)
        nearest = d.argmin(1)
        labels[noise_idx] = np.array([uniq[i] for i in nearest])
    return labels, is_core


def _cluster_stats(labels) -> str:
    sizes: dict[int, int] = {}
    for c in labels:
        sizes[int(c)] = sizes.get(int(c), 0) + 1
    n = len(labels)
    noise = sizes.get(-1, 0)
    real = sorted([s for c, s in sizes.items() if c != -1], reverse=True)
    return (f"clusters={len(real)} noise={noise} ({100*noise/max(n,1):.0f}%) "
            f"sizes[top5]={real[:5]} median={real[len(real)//2] if real else 0}")


def label_clusters(texts, labels, *, top_k=10):
    """每簇用 c-TF-IDF（每簇拼接所有标题当一篇文档）top 关键词命名。

    代表性优先：每簇先取自己 c-TF-IDF 最高的 top-3（不做全局词抑制 —— 全局抑制
    会让大簇抢光 canonical 词、小同族簇被逼用 "pde solvers"/"disabled vector"
    之类尾词，反而不具代表性）。仅当两簇的 top-3 标签字符串完全相撞时，给较小的
    那个追加一个区分词（"language / large language" → "… / agents"），最小消歧。
    （标题比 abstract 干净：abstract 含 "Published at NeurIPS 2023" 等样板词会污染。）
    返回 {cluster_id: {label, keywords}}。
    """
    from collections import defaultdict
    from sklearn.feature_extraction.text import TfidfVectorizer

    uniq = sorted(set(int(c) for c in labels) - {-1})
    out: dict[int, dict] = {}
    if not uniq:
        return out
    docs = {c: [] for c in uniq}
    sizes = {c: 0 for c in uniq}
    for t, c in zip(texts, labels):
        c = int(c)
        if c != -1:
            docs[c].append(t or "")
            sizes[c] += 1
    corpus = [" ".join(docs[c]) for c in uniq]
    stop = list(_english_stop() | DOMAIN_STOPWORDS)
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2),
                          max_features=8000, min_df=1, sublinear_tf=True)
    M = vec.fit_transform(corpus)
    terms = np.array(vec.get_feature_names_out())
    ranked: dict[int, list[str]] = {}
    for i, c in enumerate(uniq):
        row = M[i].toarray().ravel()
        order = row.argsort()[::-1]
        ranked[c] = [terms[j] for j in order if row[j] > 0][:top_k * 3]
    # 1) 每簇自然 top-3（仅避子串重复："large" vs "large language"）。
    base = {c: _top_terms(ranked[c], n=3) for c in uniq}
    # 2) 仅对完全相撞的 label 做最小消歧：大簇保原标签，小簇追加首个区分词。
    groups: dict[str, list[int]] = defaultdict(list)
    for c in uniq:
        groups[" / ".join(base[c])].append(c)
    final = {c: list(base[c]) for c in uniq}
    for members in groups.values():
        if len(members) < 2:
            continue
        for c in sorted(members, key=lambda c: sizes[c], reverse=True)[1:]:
            extra = next((t for t in ranked[c]
                          if not any(t in p or p in t for p in base[c])), None)
            if extra:
                final[c] = base[c] + [extra]
    for c in uniq:
        out[c] = {"label": " / ".join(final[c]),
                  "keywords": (ranked[c] or ["misc"])[:top_k]}
    return out


def _top_terms(cands, n=3):
    """取前 n 个候选词组标签，仅跳过子串自重复（"large" vs "large language"）。"""
    picked: list[str] = []
    for t in cands:
        if any(t in p or p in t for p in picked):
            continue
        picked.append(t)
        if len(picked) >= n:
            break
    return picked or list(cands[:n]) or ["misc"]


def _english_stop() -> set:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return set(ENGLISH_STOP_WORDS)


def build_payload(metas, xy, labels, cluster_meta, is_core=None):
    clusters = []
    sizes: dict[int, int] = {}
    for c in labels:
        sizes[int(c)] = sizes.get(int(c), 0) + 1
    for c, info in sorted(cluster_meta.items()):
        clusters.append({
            "cluster_id": c, "label": info["label"],
            "keywords": info["keywords"], "size": sizes.get(c, 0),
        })
    points = []
    for i, (m, (x, y), c) in enumerate(zip(metas, xy, labels)):
        pt = {
            "doc_id": m["doc_id"], "paper_id": m["paper_id"], "title": m["title"],
            "venue": m["venue"], "year": m["year"],
            "x": round(float(x), 4), "y": round(float(y), 4), "cluster": int(c),
        }
        if is_core is not None:
            pt["core"] = bool(is_core[i])
        points.append(pt)
    return {
        "meta": {
            "n": len(points), "n_clusters": len(clusters),
            "noise": sizes.get(-1, 0),
        },
        "clusters": clusters,
        "points": points,
    }


def smoke_report(payload, metas, labels):
    m = payload["meta"]
    print(f"\n=== SMOKE: n={m['n']} clusters={m['n_clusters']} "
          f"noise={m['noise']} ({100*m['noise']/max(m['n'],1):.1f}%) ===")
    by = {}
    for meta, c in zip(metas, labels):
        by.setdefault(int(c), []).append(meta["title"])
    top = sorted(payload["clusters"], key=lambda c: c["size"], reverse=True)[:12]
    for cl in top:
        cid = cl["cluster_id"]
        samples = by.get(cid, [])[:3]
        print(f"\n[c{cid} n={cl['size']}] {cl['label']}")
        for t in samples:
            print(f"   - {t[:90]}")


def _load_or_reduce(a):
    """缓存 export+UMAP（贵），让聚类调参秒级迭代。返回 (metas, xy, emb_c)。"""
    cv = "_cv" if a.center_venue else ""
    cache = Path("data/landscape") / (
        f"cache_{a.substrate}_s{a.sample}_nn{a.n_neighbors}_d{a.cluster_dims}{cv}.npz")
    mcache = cache.with_suffix(".metas.json")
    if cache.exists() and mcache.exists() and not a.no_cache:
        print(f"cache hit: {cache.name}", flush=True)
        z = np.load(cache)
        metas = json.loads(mcache.read_text(encoding="utf-8"))
        return metas, z["xy"], z["emb_c"]

    client = _client(a.host, a.port)
    if a.substrate == "l2pool":
        X, metas = load_l2pool(a.pool_file, client)  # 每篇 L2 正文均值，免 L1 模板噪声
    else:
        X, metas = export_l1(client, a.sample)
    if X.shape[0] < 10:
        raise SystemExit(f"too few vectors: {X.shape[0]}")
    if a.center_venue:
        X = center_by_venue(X, metas)
        print(f"  centered by venue ({len(set(m.get('venue') for m in metas))} venues)",
              flush=True)
    # 全量 94K：seed 强制单线程 UMAP 太慢 → --no-seed 放开多核（牺牲 2D layout 可复现）。
    # cluster_seed 恒 42：聚类空间锁死，簇/标签每跑一致（不受 --no-seed 影响）。
    seed = None if a.no_seed else 42
    xy, emb_c = reduce(X, n_neighbors=a.n_neighbors, cluster_dims=a.cluster_dims,
                       seed=seed, cluster_seed=42)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, xy=xy, emb_c=emb_c)
    mcache.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    return metas, xy, emb_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=8000, help="抽样数；0=全量")
    ap.add_argument("--min-cluster-size", type=int, default=50)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--cluster-dims", type=int, default=5, help="聚类用 UMAP 维度（~5，不要 50）")
    ap.add_argument("--cluster-method", default="leaf", choices=["eom", "leaf"])
    ap.add_argument("--clusterer", default="hdbscan", choices=["hdbscan", "kmeans"])
    ap.add_argument("--k", type=int, default=50, help="kmeans 簇数")
    ap.add_argument("--assign-noise", action="store_true",
                    help="hdbscan 噪声点贴最近簇质心，全覆盖上色")
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--out", default="data/landscape/landscape.json")
    ap.add_argument("--smoke-report", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--center-venue", action="store_true",
                    help="按 venue 减质心去 ingest-template 批次效应（修 3 个 venue 岛）")
    ap.add_argument("--substrate", default="l1", choices=["l1", "l2pool"],
                    help="底座向量：l1=title+abstract 摘要；l2pool=每篇 L2 正文均值（更厚）")
    ap.add_argument("--pool-file", default="data/landscape/l2pool.npz",
                    help="l2pool 向量文件（pool_l2.py 产出）")
    ap.add_argument("--llm-label", action="store_true",
                    help="用本地 Qwen 给每簇起干净唯一话题名（替代 c-TF-IDF 关键词）")
    ap.add_argument("--llm-model", default=None, help="LLM 路径（默认 llm_label.DEFAULT_MODEL）")
    ap.add_argument("--no-seed", action="store_true",
                    help="放开 2D UMAP 多核（不固定 random_state）提速；聚类空间仍恒定")
    ap.add_argument("--sweep", action="store_true", help="在缓存 UMAP 上扫多组 HDBSCAN 配置")
    a = ap.parse_args()

    metas, xy, emb_c = _load_or_reduce(a)

    if a.sweep:
        print("\n=== HDBSCAN sweep (cached UMAP) ===")
        for method in ("eom", "leaf"):
            for mcs in (15, 25, 50, 80):
                lbl = cluster(emb_c, min_cluster_size=mcs, method=method)
                print(f"  method={method:4} mcs={mcs:3}: {_cluster_stats(lbl)}")
        return

    is_core = None
    if a.clusterer == "kmeans":
        labels = cluster_kmeans(emb_c, k=a.k)
        print(f"  KMeans k={a.k}: {_cluster_stats(labels)}", flush=True)
    else:
        labels = cluster(emb_c, min_cluster_size=a.min_cluster_size,
                         min_samples=a.min_samples, method=a.cluster_method,
                         epsilon=a.epsilon)
        print(f"  HDBSCAN: {_cluster_stats(labels)}", flush=True)
        if a.assign_noise:
            labels, is_core = assign_noise_nearest(emb_c, labels)
            print(f"  +assign-noise: {_cluster_stats(labels)} "
                  f"core={int(is_core.sum())}", flush=True)
    if a.llm_label:
        from llm_label import DEFAULT_MODEL, llm_label_clusters

        sizes = {int(c): int((labels == c).sum()) for c in set(int(x) for x in labels)}
        rep = representative_titles(metas, emb_c, labels)
        print(f"  LLM labeling {len(rep)} clusters via "
              f"{a.llm_model or DEFAULT_MODEL} ...", flush=True)
        llm = llm_label_clusters(rep, model_path=a.llm_model or DEFAULT_MODEL, sizes=sizes)
        # 仍保留 c-TF-IDF keywords 当悬停补充信息。
        kw = label_clusters([m.get("title") or "" for m in metas], labels)
        cluster_meta = {c: {"label": llm[c]["label"],
                            "keywords": kw.get(c, {}).get("keywords", [])}
                        for c in llm}
    else:
        cluster_meta = label_clusters([m.get("title") or "" for m in metas], labels)
    payload = build_payload(metas, xy, labels, cluster_meta, is_core=is_core)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out} ({out.stat().st_size/1e6:.1f} MB)")

    if a.smoke_report or a.sample:
        smoke_report(payload, metas, labels)


if __name__ == "__main__":
    main()
