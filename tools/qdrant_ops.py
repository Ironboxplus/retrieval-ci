"""Qdrant 运维抓手 — 统一 CLI 收口所有常用 ops。

固化来自 2026-05-16 W8/W10 排查的命令模式：
  - 重启代价大（128 segment-level payload index 补建，30-60 min 不可用）
  - search latency 实测必须 strip 系统 HTTP_PROXY（10808 污染）
  - HNSW dense build 实际单核工作（max_indexing_threads 是 segment 并行 cap，不是单 segment 内 thread）
  - in-flight indexing 阻塞 GET /collections/{name} → 走 /cluster + /collections list 探活

子命令：
  health      一次性诊断：cluster + collections list + collection meta + search latency
  segments    按 size 列 segment + payload_index/vector_index 子目录
  progress    采样 N 次（默认 60s × 2）看 indexing 进度增长率 + 粗估 ETA
  search      探 search 延迟（dense 单向量，可指定 limit）
  stop        优雅停 Qdrant（PowerShell Stop-Process -Name qdrant -Force）
  start       启 Qdrant 后台 + log 到 /tmp/qdrant.log

用法：
  python tools/qdrant_ops.py health
  python tools/qdrant_ops.py progress --samples 3 --interval 60
  python tools/qdrant_ops.py search --limit 10 --timeout 15
  python tools/qdrant_ops.py stop && python tools/qdrant_ops.py start

约定：
  - 不依赖项目 venv 的额外 pkg（只用 stdlib + requests）
  - NO_PROXY=localhost,127.0.0.1,::1 强制注入，避免代理污染
  - URL 默认 http://localhost:6333，可 --url 覆盖
  - Windows 友好：stop/start 走 powershell，路径用相对
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 stdout on Windows so emoji markers don't crash GBK terminals.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Strip proxy env BEFORE importing requests (otherwise httpx/requests caches it)
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import requests  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "http://localhost:6333"
DEFAULT_COLL = "papers"
QDRANT_EXE = ROOT / "tools" / "qdrant" / "qdrant.exe"
STORAGE_PATH = ROOT / "data" / "indices" / "qdrant_server"
LOG_PATH = Path("/tmp/qdrant.log")
SEGMENTS_DIR = STORAGE_PATH / "collections" / DEFAULT_COLL / "0" / "segments"


def _get(url: str, timeout: float = 5.0) -> tuple[int, dict | None, float]:
    """Return (http_code, json_body_or_none, latency_sec)."""
    t0 = time.time()
    try:
        r = requests.get(url, timeout=timeout)
        dt = time.time() - t0
        try:
            return r.status_code, r.json(), dt
        except Exception:
            return r.status_code, None, dt
    except requests.RequestException:
        return -1, None, time.time() - t0


def _post(url: str, body: dict, timeout: float = 15.0) -> tuple[int, dict | None, float]:
    t0 = time.time()
    try:
        r = requests.post(url, json=body, timeout=timeout)
        dt = time.time() - t0
        try:
            return r.status_code, r.json(), dt
        except Exception:
            return r.status_code, None, dt
    except requests.RequestException:
        return -1, None, time.time() - t0


def _du_mb(path: Path) -> int:
    """Recursive size in MB. Returns 0 if path missing."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total // (1024 * 1024)


def cmd_health(args) -> int:
    """One-shot health probe: ALL endpoints + search latency."""
    print(f"# Qdrant health probe @ {args.url}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    code, body, dt = _get(f"{args.url}/readyz", timeout=3)
    print(f"  readyz                  : http={code} dt={dt:.3f}s body={body!r}")

    code, body, dt = _get(f"{args.url}/cluster", timeout=3)
    print(f"  /cluster                : http={code} dt={dt:.3f}s")

    code, body, dt = _get(f"{args.url}/collections", timeout=3)
    if code == 200 and body:
        cnt = len(body["result"]["collections"])
        print(f"  /collections (list)     : http={code} dt={dt:.3f}s ({cnt} collections)")
    else:
        print(f"  /collections (list)     : http={code} dt={dt:.3f}s")

    print()
    print(f"  /collections/{args.coll} (meta, may be slow if indexing):")
    code, body, dt = _get(f"{args.url}/collections/{args.coll}", timeout=args.timeout)
    if code != 200 or not body:
        print(f"    [WARN] FAILED http={code} dt={dt:.2f}s (likely W8: in-flight indexing lock)")
    else:
        d = body["result"]
        print(f"    status               = {d['status']}")
        print(f"    points_count         = {d['points_count']:,}")
        print(f"    indexed_vectors      = {d['indexed_vectors_count']:,}")
        print(f"    segments_count       = {d['segments_count']}")
        print(f"    update_queue.length  = {d['update_queue']['length']}")
        ps = d.get("payload_schema", {}) or {}
        print(f"    payload_schema       = {list(ps.keys()) or 'EMPTY (W9 坑)'}")
        cfg = d.get("config", {})
        # Per-vector hnsw_config overrides top-level. Production schema declares
        # on_disk / max_indexing_threads on vectors.dense, so prefer that — top-level
        # only carries defaults and was the source of a false "schema drift" alarm
        # on 2026-05-16.
        dense_cfg = (cfg.get("params", {}).get("vectors", {}) or {}).get("dense", {}) or {}
        hc_top = cfg.get("hnsw_config", {}) or {}
        hc = {**hc_top, **(dense_cfg.get("hnsw_config") or {})}
        dense_on_disk = dense_cfg.get("on_disk", hc_top.get("on_disk"))
        oc = cfg.get("optimizer_config", {})
        print(f"    hnsw m={hc.get('m')} ef_construct={hc.get('ef_construct')} "
              f"max_idx_threads={hc.get('max_indexing_threads')} hnsw_on_disk={hc.get('on_disk')} "
              f"dense_on_disk={dense_on_disk}")
        print(f"    optimizer max_threads={oc.get('max_optimization_threads')} "
              f"indexing_threshold={oc.get('indexing_threshold')} memmap_threshold={oc.get('memmap_threshold')}")
        print(f"    quantization         = {cfg.get('quantization_config')}")

    print()
    print(f"  search probe (dense top_k=1, timeout={args.search_timeout}s):")
    body = {"vector": {"name": "dense", "vector": [0.001] * 1024},
            "limit": 1, "with_payload": False}
    code, _, dt = _post(f"{args.url}/collections/{args.coll}/points/search",
                        body, timeout=args.search_timeout)
    if code == 200:
        marker = "✅" if dt < 1.0 else ("[WARN]" if dt < 5.0 else "❌")
        print(f"    {marker} http={code} dt={dt:.3f}s")
    else:
        print(f"    ❌ http={code} dt={dt:.2f}s (likely W8 lock)")
    return 0


def cmd_segments(args) -> int:
    """List segment dirs by size + payload/vector index breakdown."""
    if not SEGMENTS_DIR.exists():
        print(f"❌ {SEGMENTS_DIR} not found")
        return 1
    rows = []
    for seg in SEGMENTS_DIR.iterdir():
        if not seg.is_dir():
            continue
        rows.append((
            seg.name[:12],
            _du_mb(seg),
            _du_mb(seg / "payload_index"),
            _du_mb(seg / "vector_index-dense"),
            _du_mb(seg / "vector_index-sparse"),
        ))
    rows.sort(key=lambda r: -r[1])
    print(f"# {len(rows)} segments under {SEGMENTS_DIR}")
    print(f"{'segment':14}{'total':>8}{'payload_idx':>14}{'vec-dense':>12}{'vec-sparse':>12}")
    for name, total, pi, vd, vs in rows:
        print(f"{name:14}{total:>8}{pi:>14}{vd:>12}{vs:>12}")
    s_total = sum(r[1] for r in rows)
    s_pi = sum(r[2] for r in rows)
    s_vd = sum(r[3] for r in rows)
    s_vs = sum(r[4] for r in rows)
    print(f"{'TOTAL':14}{s_total:>8}{s_pi:>14}{s_vd:>12}{s_vs:>12}  (MB)")
    return 0


def cmd_progress(args) -> int:
    """Sample dense/sparse/payload index size over time → infer ETA."""
    print(f"# indexing progress sampler — {args.samples} samples × {args.interval}s")
    print()
    samples = []
    for i in range(args.samples):
        t = time.time()
        pi = sum(_du_mb(s / "payload_index") for s in SEGMENTS_DIR.iterdir() if s.is_dir())
        vd = sum(_du_mb(s / "vector_index-dense") for s in SEGMENTS_DIR.iterdir() if s.is_dir())
        vs = sum(_du_mb(s / "vector_index-sparse") for s in SEGMENTS_DIR.iterdir() if s.is_dir())
        samples.append((t, pi, vd, vs))
        print(f"  T+{int(t - samples[0][0]):4d}s  "
              f"payload_idx={pi:>5}MB  vec-dense={vd:>5}MB  vec-sparse={vs:>5}MB")
        if i < args.samples - 1:
            time.sleep(args.interval)
    if len(samples) >= 2:
        t0, pi0, vd0, vs0 = samples[0]
        tN, piN, vdN, vsN = samples[-1]
        dt = tN - t0
        print()
        print(f"  growth over {int(dt)}s:")
        print(f"    payload_idx  : +{piN - pi0:>5}MB  ({(piN - pi0) / dt:.2f} MB/s)")
        print(f"    vec-dense    : +{vdN - vd0:>5}MB  ({(vdN - vd0) / dt:.2f} MB/s)")
        print(f"    vec-sparse   : +{vsN - vs0:>5}MB  ({(vsN - vs0) / dt:.2f} MB/s)")
        if piN == pi0 and vdN == vd0 and vsN == vs0:
            print("  [WARN] 0 增长 — 已建完 OR 单核 HNSW build 不写盘 (内存 graph 阶段)")
    return 0


def cmd_search(args) -> int:
    """Single search latency probe."""
    body = {"vector": {"name": "dense", "vector": [0.001] * 1024},
            "limit": args.limit, "with_payload": False}
    code, resp, dt = _post(f"{args.url}/collections/{args.coll}/points/search",
                           body, timeout=args.timeout)
    marker = "✅" if code == 200 and dt < 1.0 else "[WARN]" if code == 200 else "❌"
    print(f"{marker} search top_k={args.limit}  http={code}  dt={dt:.3f}s")
    if code == 200 and resp:
        hits = resp["result"]
        if hits:
            print(f"  top score = {hits[0]['score']:.4f}  ({len(hits)} hits)")
    return 0 if code == 200 else 1


def cmd_stop(args) -> int:
    """Stop Qdrant via PowerShell Stop-Process."""
    ps = "Stop-Process -Name qdrant -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 3; (Get-Process qdrant -ErrorAction SilentlyContinue | Measure-Object).Count"
    r = subprocess.run(["powershell", "-Command", ps], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode == 0 and "0" in r.stdout.strip().splitlines()[-1:]:
        print("✅ Qdrant stopped")
        return 0
    print(f"[WARN] Stop ambiguous: stdout={r.stdout!r} stderr={r.stderr!r}")
    return 1


def cmd_start(args) -> int:
    """Start Qdrant in background with proper env + log redirect."""
    if not QDRANT_EXE.exists():
        print(f"❌ {QDRANT_EXE} not found")
        return 1
    env = {**os.environ, "QDRANT__STORAGE__STORAGE_PATH": str(STORAGE_PATH)}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("ab")
    p = subprocess.Popen(
        [str(QDRANT_EXE), "--disable-telemetry"],
        env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT),
    )
    print(f"started qdrant pid={p.pid} log={LOG_PATH}")
    print("[WARN] 重启代价提醒 (W10)：")
    print("   - 每 segment 加载时补建缺失 payload index → 128 task × 单核串行")
    print("   - WAL replay (queue 内 entries)")
    print("   - HNSW dense build single-thread per segment")
    print("   - search 不可用 30-60 min（取决于 collection 大小）")
    print("   监控：python tools/qdrant_ops.py health")
    print("        python tools/qdrant_ops.py progress --samples 3 --interval 60")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="qdrant_ops", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--coll", default=DEFAULT_COLL)
    sub = p.add_subparsers(dest="cmd", required=True)

    sh = sub.add_parser("health", help="one-shot full health probe")
    sh.add_argument("--timeout", type=float, default=10.0,
                    help="collection meta probe timeout (s)")
    sh.add_argument("--search-timeout", type=float, default=10.0)
    sh.set_defaults(func=cmd_health)

    sub.add_parser("segments", help="list segment dirs by size").set_defaults(func=cmd_segments)

    pr = sub.add_parser("progress", help="sample indexing growth → ETA")
    pr.add_argument("--samples", type=int, default=2)
    pr.add_argument("--interval", type=int, default=60, help="seconds between samples")
    pr.set_defaults(func=cmd_progress)

    se = sub.add_parser("search", help="single search latency probe")
    se.add_argument("--limit", type=int, default=10)
    se.add_argument("--timeout", type=float, default=15.0)
    se.set_defaults(func=cmd_search)

    sub.add_parser("stop", help="kill qdrant via powershell").set_defaults(func=cmd_stop)
    sub.add_parser("start", help="launch qdrant in background").set_defaults(func=cmd_start)

    args = p.parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
