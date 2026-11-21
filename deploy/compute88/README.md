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
Qdrant keeps `LOW_MEMORY_MODE=no_populate` because all four collections cannot
be prefaulted together, but its service boundary is `MemoryHigh=11G` and
`MemoryMax=13G`.  The old 7/8 GiB boundary accumulated 329,392 `memory.high`
events while the host still had about 10 GiB `MemAvailable`, repeatedly evicting
the paper index working set.  Qdrant uses four search threads and the v1.18
`async_scorer`; the backend unit currently has no systemd memory cap.
Collection conversion to on-disk dense vectors, dense HNSW, sparse indexes, and
payload storage is performed separately with
`scripts/configure_qdrant_low_memory.py`; startup must not hide that rebuild.

GPU0 is available to both online Retrieval inference and local pipeline workers.
`PIPELINE_LOCAL_MIN_FREE_MB=8704` requires 8.5 GiB to be free before a pipeline
worker is admitted. This replaces the old 12 GiB-free gate, which no 12,282 MiB
card could ever pass, while retaining headroom for the resident online models.
The embedding and extraction stages remain serialized by the pipeline resource
gate. A representative Marker extraction must still pass separately before a
large PDF conversion is scheduled; GPU authorization is not proof that every
batch size fits.

The backend unit also pins `RERANKER__CUDA_MEMORY_FRACTION=0.35`,
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

The host must already contain the root-only CIFS credentials file
`/etc/smbcredentials224`. Do not copy its contents into this repository. Install
the artifact mount as system units and enable only the automount:

```bash
sudo install -m 0644 mnt-lsb.mount mnt-lsb.automount /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/systemd/system/mnt-lsb.mount /etc/systemd/system/mnt-lsb.automount
sudo systemctl enable --now mnt-lsb.automount
ls /mnt/lsb/retrival/papers/markdown >/dev/null
findmnt -T /mnt/lsb
```

The `gid=arc` mapping is deliberate: `.88` does not share `.111`'s local
`diskusers` group database. It keeps the artifact tree writable by the
Retrieval service account without changing the CIFS server or storing papers on
the `.88` system filesystem.

Copy the user-service files into `~/.config/systemd/user/`, then run:

```bash
systemctl --user daemon-reload
systemd-analyze --user verify ~/.config/systemd/user/arc-retrival-qdrant.service
systemd-analyze --user verify ~/.config/systemd/user/arc-retrival-backend.service
```

Do not enable or start the backend until the source SQLite backup and Qdrant
snapshot have passed their integrity checks and the artifact automount has been
verified. Persistent user services also require an administrator to run
`loginctl enable-linger arc` once on `.88`.

For a Qdrant-only canary on the disposable pre-cutover copy, first verify that
ports `16333` and `16334` are free. Start only
`arc-retrival-qdrant.service`, collect its memory/status evidence, and stop it
again before the final synchronization.
