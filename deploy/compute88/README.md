# Retrieval compute node `.88`

This directory contains the isolated user-service configuration for moving the
Retrieval compute workload to `10.100.100.88`. The host already runs unrelated
services owned by other users. These units must not reuse, stop, inspect, or
modify those services.

## Isolation contract

- Qdrant listens only on `127.0.0.1:16333` (HTTP) and
  `127.0.0.1:16334` (gRPC).
- The Retrieval backend listens on `0.0.0.0:18080` (trusted/internal door) and
  `127.0.0.1:18181` (public door). Port `18081` belongs to PeachObj's Pion
  runtime on this shared host and is outside Retrieval's deployment scope.
- Existing host ports `6333`, `6334`, and `8000` remain outside this deployment.
- No credentials or public gateway secrets belong in these unit files.
- GPU0 is explicitly authorized for Retrieval on `.88`. This is a host-specific
  exception; it does not change the GPU0 prohibition on other compute hosts.
- Medical PDF execution is enabled only in this unit via
  `MEDICAL_DOWNLOAD_HOST=10.100.100.88`. `MEDICAL_DATA_ROOT` is
  `/mnt/lsb/retrival/medical`, never `/home/arc` or a Windows path. The unit
  creates its `manifest/` and `pdf/` directories only after the mount is ready;
  it keeps a 20 GiB free-space floor (`MEDICAL_MIN_FREE_BYTES=21474836480`).

## Memory contract

The host has about 15 GiB RAM and will not receive a physical memory upgrade.
Deployed on 2026-10-15 at 17:45: production runs without `LOW_MEMORY_MODE`, with
`MemoryHigh=12G` / `MemoryMax=13G` / `MemorySwapMax=0`. Runtime telemetry confirms
`low_memory_mode=disabled`; papers' quantized data is loaded into process memory
instead of file mappings. The swap prohibition protects Qdrant's anonymous
quantization pages; file-backed indexes remain reclaimable. At 17:49 Qdrant used
about 10.32 GiB of cgroup memory and zero swap, with zero memory.high/max/OOM
events. The restart and a 180-second workload guard passed. Backend memory is
not protected by this Qdrant-only setting, so host contention can still swap
out model/tokenizer pages and cause request timeouts.
The prerequisite collection conversions completed at 02:01 on 2026-10-15.
`code_chunks`/`public_code_chunks` have dense vectors, dense HNSW, sparse
indexes and payload indexes on disk. `papers` has its payload indexes on disk,
dense, HNSW and sparse on disk. Its TurboQuant 2-bit quantization is configured
with `always_ram` (about 5.8 GiB). `journal_chunks` (0.6 GiB) keeps its existing
in-memory configuration.
Any new collection or payload index must be created on disk, or it will load
into RAM at startup. Smoke after the earlier 14:37 restart passed; its first-query
latency ranged from 3.8 to 14.5 seconds, while repeats took 0.6–0.9 seconds. Cold-search
performance remains an open task; this deployment does not establish a latency SLA.
Native fault-address tracing on 2026-10-15 identified the default `source_type`
and `venue_tier` filters as another major cold path: 192 of 202 sampled faults
in one request hit their point-to-values files (about 1.13 GiB combined). Sparse
posting lists also caused substantial reads. A temporary read-only shared-mmap
locking probe eliminated paper-filter faults, but is not installed as a service;
one-pass warming did not establish durable residency or end-to-end improvement.
At 18:19 the backend's shared `embed_texts` wrapper was updated and the backend
fully restarted. Background callers now release the shared model scheduler after
at most eight texts, including Code KB and Journal ingestion batches of 128 or
more. Interactive calls retain their existing behavior. This bounds scheduling
granularity, not GPU execution time per long text; background throughput may fall.
Deployment receipt and the previous module are in
`/home/arc/retrival-releases/search-latency-debug/`.
The 18:21 smoke returned HTTP 200 for two simultaneous searches, but retrieval
still took about 24 seconds each; embedding took 47/88 ms. Configuration polling
remained responsive at 1–5 ms. Functional health does not imply cold-latency acceptance.
The old 7/8 GiB boundary accumulated 329,392 `memory.high` events while the
host still had about 10 GiB `MemAvailable`, repeatedly evicting the paper index
working set.  Qdrant uses four search threads and the v1.18 `async_scorer`; the
backend unit currently has no systemd memory cap. Collection conversion to
on-disk dense vectors, dense HNSW, sparse indexes, and payload storage is
performed separately with `scripts/configure_qdrant_low_memory.py`; startup must
not hide that rebuild.

GPU0 is available to both online Retrieval inference and local pipeline workers.
The unit sets no free-memory floor and no idle ratio for the local stages (the old
`PIPELINE_LOCAL_MIN_FREE_MB`, `PIPELINE_LOCAL_MIN_FREE_RATIO` and `PIPELINE_LOCAL_EMBED_MIN_FREE_MB`
are gone, and the code reads none of them):

- The local embed stage is a client of the vLLM embedding service and puts nothing on the card.
  Its preflight only checks that the service answers (`check_service`), and it holds the
  `embed-service` resource, not `gpu-local`, so an embed and an extract run at the same time.
- The local extract's floor is one figure, `extract_min_free_mb` in `local_extractor.json`:
  6,657 MiB, the least `scripts/remote_pvl/pvl_worker.py` can start with on the 12,282 MiB card
  (`LAYOUT_RESERVE_MIB` 3,700 + `MEMORY_MARGIN_MIB` 500 + `MIN_SERVER_FRACTION` 0.20 x 12,282).
  The card always carries the two resident vLLM services (4,434 MiB, about 7,850 MiB free), so
  there is no idle-ratio gate. `tests/test_gpu_gates_after_vllm.py` ties the figure to the worker's
  constants; change them together.
- `PIPELINE_LOCAL_GPU_IDS=0` stays: it is the allowlist, not a floor.

GPU authorization is not proof that every PDF batch fits; the worker sizes its own server
from what is free when it starts.

Since 2026-11-14 it also sets no `RERANKER__BATCH_SIZE`: the reranker now reads whole passages (`max_length` 2,560 = chunk budget 2,048 + query 512, P0 truncation fix), and at that length batch 128 peaks at 6.10 GiB on fp16 vs 2.45 GiB at batch 32 (local RTX 3090 measurement), so the code default 32 in `RerankerSettings` is the single source.

Since 2026-09-21 the backend unit sets no `RERANKER__CUDA_MEMORY_FRACTION` (operator
decision: every OOM under the cap still reported 7.6-8.5 GiB free), and since 2026-10-07
it sets `LimitNOFILE=65536` (the default 1024 descriptors were exhausted by silent
forwarded connections). History of the removed cap:

The backend unit used to pin `RERANKER__CUDA_MEMORY_FRACTION=0.35`,
`RERANKER__BATCH_SIZE=128`, and `RERANKER__IDLE_CACHE_LIMIT_MIB=0`. The first
value limits the process-wide PyTorch allocator before BGE-M3 and the reranker
load; it was raised from 0.25 because the exact legacy batch-128 match path
otherwise failed with a propagated CUDA OOM.  A real three-chunk production
match used at most 4,122 MiB reserved during reranking and returned to 2,198 MiB
after every call, with no OOM.  The third setting returns idle allocator blocks
after GPU work. These values are not generic defaults: they are the `.88` release contract and must move in
lockstep with `scripts/verify_reranker_cuda_canary.py` and its real-result
regressions.

## Installation and activation

The private Code KB path-glob selector uses SQL file inventory, not a Qdrant
owner-wide path facet. Install its covering index on the actual runtime database:

```bash
python3 scripts/ensure_code_search_path_index.py \
  --database /home/arc/retrival-4090-migration/retrival/data/db/retrieval.db
```

The operation is idempotent and keeps catalog rows unchanged. It takes a bounded
SQLite write lock while constructing the index. On 2026-10-15 at 21:13 it completed
in about four seconds. Incident deployment receipts and the installer used on the
host are in `/home/arc/retrival-releases/search-latency-debug/`.

The artifact mount serves the completed local project copy at
`/home/arc/retrieval-storage/retrival`. Install it only after the NAS-to-local
transfer and verification finish. The bind mount preserves the existing
`/mnt/lsb/retrival` paths used by the artifact resolver and medical service.
This storage is on `.88`'s ext4 SATA SSD; it does not require NAS connectivity.
Install the artifact mount as system units and enable only the automount:

```bash
sudo install -m 0644 mnt-lsb.mount mnt-lsb.automount /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/systemd/system/mnt-lsb.mount /etc/systemd/system/mnt-lsb.automount
sudo systemctl enable --now mnt-lsb.automount
ls /mnt/lsb/retrival/papers/markdown >/dev/null
findmnt -T /mnt/lsb
```

The original NAS can be mounted separately at `/mnt/retrieval_nas` for rollback
or manual comparison with `mnt-retrieval_nas.mount` and its automount. Its
root-only credential file remains `/etc/smbcredentials224`; do not copy the
contents into this repository. This optional mount is not a backend startup
dependency. Changing the `.88` artifact mount does not require changing any
other machine's mounts.

`retrieval-smb.conf` provides an authenticated SMB3 share of the local storage
at `\\10.100.100.88\lsb`. Samba credentials are provisioned at runtime; the
configuration contains no password. The `retrival` child directory must become
visible only after the completed staging tree is renamed into place.

Copy the user-service files into `~/.config/systemd/user/`, then run:

```bash
systemctl --user daemon-reload
systemd-analyze --user verify ~/.config/systemd/user/arc-retrival-qdrant.service
systemd-analyze --user verify ~/.config/systemd/user/arc-retrival-backend.service
```

Before any backend restart, run `wait_idle.sh` and stop the service right
after it exits 0. It reads `pending` and `running` from the tasks API; never
gate a restart on quiet logs, because a long code index logs nothing per batch.

Do not enable or start the backend until the source SQLite backup and Qdrant
snapshot have passed their integrity checks and the artifact automount has been
verified. Persistent user services also require an administrator to run
`loginctl enable-linger arc` once on `.88`.

For a Qdrant-only canary on the disposable pre-cutover copy, first verify that
ports `16333` and `16334` are free. Start only
`arc-retrival-qdrant.service`, collect its memory/status evidence, and stop it
again before the final synchronization.

## Model services (vLLM)

Every model runs as a vLLM user service; the backend holds no model and is their client
(`core/indexing/embedders/VllmBgeM3Embedder`, `core/retrieval/reranker.VllmReranker`).

| Unit | Model | Port | Keys |
|---|---|---|---|
| `arc-vllm-embed.service` | BGE-M3, dense + sparse (`bge_m3_sparse_plugin`) | `127.0.0.1:18200` | `vllm_bge_m3.yaml` |
| `arc-vllm-rerank.service` | bge-reranker-v2-m3 | `127.0.0.1:18201` | `vllm_reranker.yaml` |

- Both run from `/home/arc/.venvs/retrival-vllm` (vLLM 0.33.0 and the sparse plugin from
  `scripts/remote_embed/bge_m3_sparse_plugin`, pinned to that vLLM), never from the backend's torch
  environment. The model directories are the modelscope snapshots the backend unit's `BGE_M3_PATH` /
  `BGE_RERANKER_PATH` name (the backend keeps `BGE_M3_PATH` for the tokenizer).
- **The embedding model directory carries the revision in its name.** `.88` keeps the BGE-M3 snapshot at
  `.../modelscope/BAAI/bge-m3`, a path without a revision, and `vllm_client.check_service` accepts only a
  directory whose path names the profile's revision (`5617a9f61b028005a4858fdac845db406aefb181`; a wrong
  snapshot would change every vector without changing the fingerprint stamped on it). The unit therefore serves
  `.../modelscope/BAAI/bge-m3-5617a9f61b028005a4858fdac845db406aefb181`, a symlink to the original directory.
  vllm-tune compared the contents with arc99's: `pytorch_model.bin` and `tokenizer.json` have the same md5.
- **The embedding unit runs with cuBLAS fp16 accumulation** (user's decision, 2026-11-14): its `PYTHONPATH`
  holds `scripts/remote_embed/fp16_accumulation`, whose `sitecustomize.py` both vLLM processes import.
  123 -> 171 chunks/s on arc99's 3090; gate: dense cos mean >= 0.9999 and min >= 0.998 against FlagEmbedding
  fp16. The reranking unit does not (only embedding was verified). Do not start it through
  `bge_m3_service.sh`: that script keeps a pidfile for WSL and does not fit `Type=simple`.
- The backend unit sets `EMBEDDING__SERVICE_URL` / `RERANKER__SERVICE_URL` to these ports and orders itself
  `After=`/`Wants=` them. `Wants`, not `Requires`: the backend still starts while a service restarts.
- At start-up the backend calls `require_embedding_service()` (`main.py`, `lifespan`): it waits up to
  `EMBEDDING__SERVICE_WAIT_S` (default 180 s) for the service to answer, **refuses to start** if the service
  serves another revision or another `max_model_len`, and, if the service stays silent, logs an error and runs
  (embedding calls raise until it is up).
- Install like the other user units (copy to `~/.config/systemd/user/`, `systemctl --user daemon-reload`,
  `enable --now`), embedding service first. Start-up takes 50-75 s (load, compile, graph capture).

The venv (`install_vllm_venv.sh`, step 0 below): `/home/arc/.venvs/retrival-vllm`, an exact copy of the
reference environment's 200 pins (`vllm_venv_requirements.txt`; key packages: vllm 0.33.0, torch 2.14.0 with CUDA 13.0
wheels, transformers 5.20.2, flashinfer-python 0.7.2.post1, triton 3.8.0, tokenizers 0.23.3, numpy 2.3.5, fastapi
0.136.3, plus `bge-m3-sparse-plugin` 0.1 built from `scripts/remote_embed/bge_m3_sparse_plugin`). Needs Python 3.12, an
NVIDIA driver >= 580 and about 12 GB free under `/home` (the reference venv is 9.1 GB installed, measured). Time:
the download volume has **not been measured** (estimate 4-6 GB of wheels, most of it the CUDA 13 libraries and
torch), so the time is that volume over the proxy's speed: roughly 15-25 min at 5 MB/s, plus 2-3 min unpacking; the
script prints what it installs, and a re-run after an interruption re-uses pip's cache. It touches nothing else on the
host, in particular not `retrival-pvl-server` (PaddleOCR-VL's vLLM 0.30 environment). The proxy has no default: `env.md`
only records the Arc Windows one (`http://127.0.0.1:10808`, that machine's loopback), so pass the one that reaches PyPI
from `.88` (or tunnel that port with `ssh -R`).

Deployment steps for the embedding unit (in this order):

```bash
# 0. the venv, once (re-runnable; --check needs no network): exact pins of the reference environment, every download
#    through the given http proxy, the run stops if the proxy cannot reach PyPI; never touches retrival-pvl-server
bash deploy/compute88/install_vllm_venv.sh --proxy http://HOST:PORT
bash deploy/compute88/install_vllm_venv.sh --check
# 1. the revision-named directory (a symlink; nothing is copied) and a content check against the profile
cd /home/arc/retrival-4090-migration/.cache/modelscope/BAAI
ln -s bge-m3 bge-m3-5617a9f61b028005a4858fdac845db406aefb181
md5sum bge-m3-5617a9f61b028005a4858fdac845db406aefb181/pytorch_model.bin        bge-m3-5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json     # must equal arc99's
# 2. install both units, start the embedding one, wait for /health (50-75 s)
cp /home/arc/retrival-4090-migration/retrival/deploy/compute88/arc-vllm-{embed,rerank}.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now arc-vllm-embed.service
# 3. acceptance: fp16 line from the API and the engine process, the service serves the profile, the numeric gate
python3 deploy/compute88/verify_embed_service.py        # prints RESULT fp16=ok service=ok numbers=ok
# 4. only then the reranking unit and the backend restart (wait_idle.sh first)
```

`verify_embed_service.py` runs `tests/test_vllm_client.py::test_vectors_match_the_stored_flagembedding_fp16_vectors`
against the service (48 stored production vectors, `tests/fixtures/bge_m3_flagembedding_reference.json.gz`); a
skipped run is a failure.
- `--gpu-memory-utilization 0.2` is each unit's share of the card; a pooling model has no KV cache, so the real
  footprint is weights + CUDA graphs + one engine step.

Measured on the RTX 3090 of arc99 (RESEARCH-2026-11-14-vllm-serving.md, PLAN-2026-11-14-vllm-embed-host.md):
about 1.7 GiB idle and 1.9-2.5 GiB at the largest step for each service.
**Not yet measured on `.88`**: the GPU memory and the resident host RAM of the two service processes on the
4070 SUPER. Record both after the first start (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`,
`ps -o pid,rss,cmd -C vllm`, and `free -m` before and after) and abort if available memory drops below 1 GiB;
available memory on this host is tight.
