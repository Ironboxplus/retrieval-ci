from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT_DIR = ROOT / "deploy" / "compute88"


def _unit(name: str) -> str:
    return (UNIT_DIR / name).read_text(encoding="utf-8")


def test_compute88_qdrant_unit_is_isolated_and_memory_bounded():
    unit = _unit("arc-retrival-qdrant.service")
    assert "QDRANT__SERVICE__HOST=127.0.0.1" in unit
    assert "QDRANT__SERVICE__HTTP_PORT=16333" in unit
    assert "QDRANT__SERVICE__GRPC_PORT=16334" in unit
    assert "QDRANT__STORAGE__LOW_MEMORY_MODE=no_populate" in unit
    assert "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=4" in unit
    assert "QDRANT__STORAGE__PERFORMANCE__ASYNC_SCORER=true" in unit
    assert "QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=1" in unit
    assert "QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET=1" in unit
    assert "MemoryHigh=11G" in unit
    assert "MemoryMax=13G" in unit
    assert "/home/arc/retrival-4090-migration" in unit
    assert ":6333" not in unit
    assert ":6334" not in unit


def test_compute88_backend_unit_keeps_trust_doors_gpu_and_qdrant_isolated():
    unit = _unit("arc-retrival-backend.service")
    assert "Requires=arc-retrival-qdrant.service" in unit
    assert "INTERNAL_API_PORT=18080" in unit
    # PeachObj owns 18081 on this shared host; Retrieval must keep its public
    # loopback door isolated instead of racing the unrelated Pion runtime.
    assert "PUBLIC_API_PORT=18181" in unit
    assert "QDRANT_URL=http://127.0.0.1:16333" in unit
    assert "QDRANT_GRPC_PORT=16334" in unit
    assert "QDRANT_COLLECTION_ON_DISK=true" in unit
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in unit
    assert "CUDA_VISIBLE_DEVICES=0" in unit
    assert "RERANKER__CUDA_MEMORY_FRACTION=0.35" in unit
    assert "RERANKER__BATCH_SIZE=128" in unit
    assert "RERANKER__IDLE_CACHE_LIMIT_MIB=0" in unit
    assert "PIPELINE_LOCAL_GPU_IDS=0" in unit
    # GPU0 is explicitly available on .88. The real ACL canary started with
    # 9,012 MiB free and bottomed at 6,127 MiB while batch-64 BGE-M3 completed;
    # 8.5 GiB admits that measured-safe state without changing model/batch.
    assert "PIPELINE_LOCAL_MIN_FREE_MB=8704" in unit
    assert "MemoryHigh=" not in unit
    assert "MemoryMax=" not in unit
    assert "MemorySwapMax=" not in unit
    assert ":8000" not in unit
    assert ":8001" not in unit


def test_compute88_units_do_not_embed_secrets():
    combined = "\n".join(
        _unit(name)
        for name in ("arc-retrival-qdrant.service", "arc-retrival-backend.service")
    )
    for forbidden in ("API_KEY=", "TOKEN=", "PASSWORD=", "COOKIE=", "SECRET="):
        assert forbidden not in combined


def test_compute88_artifact_mount_uses_the_shared_store_without_inline_secrets():
    unit = _unit("mnt-lsb.mount")
    assert "What=//10.100.100.9/lsb" in unit
    assert "Where=/mnt/lsb" in unit
    assert "Type=cifs" in unit
    assert "credentials=/etc/smbcredentials224" in unit
    assert "gid=arc" in unit
    assert "dir_mode=02775" in unit
    assert "file_mode=0664" in unit
    assert "password=" not in unit.lower()


def test_compute88_artifact_mount_is_persisted_as_an_automount():
    unit = _unit("mnt-lsb.automount")
    assert "Where=/mnt/lsb" in unit
    assert "TimeoutIdleSec=0" in unit
    assert "WantedBy=multi-user.target" in unit
