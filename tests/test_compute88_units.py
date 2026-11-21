from pathlib import Path

from core.indexing.embedding_profile import get_profile


ROOT = Path(__file__).resolve().parents[1]
UNIT_DIR = ROOT / "deploy" / "compute88"


def _unit(name: str) -> str:
    return (UNIT_DIR / name).read_text(encoding="utf-8")


def test_compute88_qdrant_unit_is_isolated_and_memory_bounded():
    unit = _unit("arc-retrival-qdrant.service")
    assert "QDRANT__SERVICE__HOST=127.0.0.1" in unit
    assert "QDRANT__SERVICE__HTTP_PORT=16333" in unit
    assert "QDRANT__SERVICE__GRPC_PORT=16334" in unit
    # Dropped so papers' always_ram TurboQuant is actually resident; the collections'
    # persisted configs carry the on-disk placement instead.
    assert "LOW_MEMORY_MODE=" not in unit
    assert "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=4" in unit
    assert "QDRANT__STORAGE__PERFORMANCE__ASYNC_SCORER=true" in unit
    assert "QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=1" in unit
    assert "QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET=2" in unit
    assert "MemoryHigh=12G" in unit
    assert "MemoryMax=13G" in unit
    # always_ram is heap placement, not mlock. Quantized scoring must not fault
    # its hot vectors back from swap on the latency-sensitive Qdrant service.
    assert "MemorySwapMax=0" in unit
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
    assert "QDRANT_PAPERS_DENSE_OVERSAMPLING=2.0" in unit
    assert "16335" not in unit and "COMPACT_PAPERS" not in unit
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in unit
    assert "CUDA_VISIBLE_DEVICES=0" in unit
    # The CUDA allocator cap was removed on 2026-09-21 by operator decision.
    assert not any(line.startswith("Environment=RERANKER__CUDA_MEMORY_FRACTION")
                   for line in unit.splitlines())
    assert "LimitNOFILE=65536" in unit
    # RERANKER__BATCH_SIZE was removed on purpose (deployed; deploy/compute88/README.md): the reranker
    # reads full-length passages now, where batch 128 peaks at 6.10 GiB and the default 32 at 2.45 GiB.
    assert "RERANKER__BATCH_SIZE" not in unit
    assert "RERANKER__IDLE_CACHE_LIMIT_MIB=0" in unit
    assert "PIPELINE_LOCAL_GPU_IDS=0" in unit
    # The local stages' GPU gates are code, not unit settings: the embed stage is a client of the vLLM
    # service (no GPU query), and the extract floor is the registered extractor's own figure
    # (local_extractor.json, tied to pvl_worker in tests/test_gpu_gates_after_vllm.py).
    assert "PIPELINE_LOCAL_MIN_FREE" not in unit and "PIPELINE_LOCAL_EMBED_MIN_FREE_MB" not in unit
    assert "PIPELINE_LOCAL_EXTRACT_MIN_FREE_MB" not in unit
    # The local extract stage runs PaddleOCR-VL through the registered extractor file.
    assert ("PIPELINE_LOCAL_EXTRACTOR=/home/arc/retrival-4090-migration/retrival/deploy/compute88/"
            "local_extractor.json") in unit
    assert "MemoryHigh=" not in unit
    assert "MemoryMax=" not in unit
    assert "MemorySwapMax=" not in unit
    assert ":8000" not in unit
    assert ":8001" not in unit


def test_compute88_backend_unit_signals_only_the_backend_on_stop():
    """The backend stops its stage workers itself and leaves their tasks resumable
    (orchestrator.request_shutdown); a SIGTERM straight to a worker made the extract stage
    gate its partial Markdown (task 2577, 2026-11-21)."""
    lines = _unit("arc-retrival-backend.service").splitlines()
    assert [line for line in lines if line.startswith("KillMode=")] == ["KillMode=mixed"]


def test_compute88_backend_unit_is_a_client_of_the_two_vllm_services():
    unit = _unit("arc-retrival-backend.service")
    assert "EMBEDDING__SERVICE_URL=http://127.0.0.1:18200" in unit
    assert "RERANKER__SERVICE_URL=http://127.0.0.1:18201" in unit
    after = next(line for line in unit.splitlines() if line.startswith("After="))
    wants = next(line for line in unit.splitlines() if line.startswith("Wants="))
    for service in ("arc-vllm-embed.service", "arc-vllm-rerank.service"):
        assert service in after and service in wants
        assert f"Requires={service}" not in unit              # the backend must start while a service restarts


def test_compute88_vllm_service_units_use_the_pinned_venv_the_shared_config_and_their_own_port():
    for name, port, model, config in (
        # a directory name with the revision: vllm_client.check_service reads it back from /v1/models
        ("arc-vllm-embed.service", "18200", f"BAAI/bge-m3-{get_profile().revision}", "vllm_bge_m3.yaml"),
        ("arc-vllm-rerank.service", "18201", "BAAI/bge-reranker-v2-m3", "vllm_reranker.yaml"),
    ):
        unit = _unit(name)
        exec_line = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
        assert exec_line.startswith("ExecStart=/home/arc/.venvs/retrival-vllm/bin/vllm serve ")
        assert f"/.cache/modelscope/{model} " in exec_line
        assert f"--config /home/arc/retrival-4090-migration/retrival/deploy/compute88/{config} " in exec_line
        assert f"--port {port} " in exec_line and "--gpu-memory-utilization" in exec_line
        assert "Restart=on-failure" in unit and "WantedBy=default.target" in unit
        assert "CUDA_VISIBLE_DEVICES=0" in unit and "HF_HUB_OFFLINE=1" in unit
        assert ":8000" not in unit and ":8001" not in unit and f":{port}" not in unit.replace(f"--port {port}", "")
    assert (UNIT_DIR / "vllm_bge_m3.yaml").is_file()


def test_compute88_embedding_service_runs_with_fp16_accumulation_and_the_reranker_does_not():
    """The user chose fp16 accumulation for the embedding service on 2026-11-14. ``vllm serve`` never loads
    scripts/remote_embed/fp16_accumulation/sitecustomize.py by itself: the unit has to put it on PYTHONPATH,
    which both the API process and the engine process it spawns inherit."""
    directory = ROOT / "scripts" / "remote_embed" / "fp16_accumulation"
    assert (directory / "sitecustomize.py").is_file()
    path = "Environment=PYTHONPATH=/home/arc/retrival-4090-migration/retrival/scripts/remote_embed/fp16_accumulation"
    embed = _unit("arc-vllm-embed.service")
    assert path in embed.splitlines()
    assert "bge_m3_service.sh" not in chr(10).join(line for line in embed.splitlines() if not line.startswith("#"))
    assert "PYTHONPATH" not in _unit("arc-vllm-rerank.service")      # only the embedding numbers were verified
    for fact in ("2026-11-14", "123", "171", "0.9999", "0.998"):
        assert fact in chr(10).join(line for line in embed.splitlines() if line.startswith("#")), fact


def test_compute88_units_do_not_embed_secrets():
    combined = "\n".join(
        _unit(name)
        for name in ("arc-retrival-qdrant.service", "arc-retrival-backend.service",
                     "arc-vllm-embed.service", "arc-vllm-rerank.service")
    )
    for forbidden in ("API_KEY=", "TOKEN=", "PASSWORD=", "COOKIE=", "SECRET="):
        assert forbidden not in combined


def test_compute88_optional_nas_mount_has_no_inline_secrets():
    unit = _unit("mnt-retrieval_nas.mount")
    assert "What=//10.100.100.9/lsb" in unit
    assert "Where=/mnt/retrieval_nas" in unit
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


def test_compute88_local_extractor_registry_loads_through_the_stage_loader(monkeypatch):
    from app.services.pipeline import stages

    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(UNIT_DIR / "local_extractor.json"))
    marker = stages.load_local_extractor()
    assert marker.python == "/home/arc/.venvs/retrival-pvl/bin/python"
    assert marker.worker == "scripts/remote_pvl/pvl_worker.py" and (ROOT / marker.worker).is_file()
    # one worker per card: it runs an inference server of its own that needs the whole card
    assert marker.ranks_per_gpu == 1
    # the least the worker can start with beside the resident vLLM services on the 4070 SUPER:
    # 3,700 layout reserve + 500 margin + 0.20 x 12,282 MiB of server (pvl_worker.server_memory_fraction)
    assert marker.extract_min_free_mb == 6657
    assert marker.env["PADDLE_PDX_MODEL_SOURCE"] == "modelscope"
    config = marker.args[marker.args.index("--server-config") + 1]
    assert (ROOT / config).is_file()
    assert "--server-prefix" in marker.args and "MemoryHigh=" in marker.args[marker.args.index("--server-prefix") + 1]
    # the two model servers run in the server venv (vLLM + torch), the worker in the pipeline venv (paddle): two
    # environments because paddlepaddle-gpu and torch pin different CUDA wheels
    assert marker.python.endswith("/retrival-pvl/bin/python")
    for flag, logits_processor in (("--server-command", False), ("--fallback-command", True)):
        command = marker.args[marker.args.index(flag) + 1]
        assert command.startswith("/home/arc/.venvs/retrival-pvl-server/bin/vllm serve /home/arc/")
        assert "--config {config}" in command and "--port {port}" in command
        assert ("--logits-processors mineru_vl_utils:MinerULogitsProcessor" in command) is logits_processor
    template = marker.args[marker.args.index("--server-command") + 1].split("--chat-template ")[1].split()[0]
    assert (ROOT / template.replace("{root}/", "")).is_file()
    fallback_config = marker.args[marker.args.index("--fallback-config") + 1]
    assert (ROOT / fallback_config).is_file() and marker.args[marker.args.index("--fallback-model") + 1] == "MinerU2.5-Pro-2605-1.2B"
    assert "PIPELINE_LOCAL_EXTRACTOR" in _unit("arc-retrival-backend.service")
