"""Shared real fixtures for the re-embed pipeline tests.

Real Qdrant on loopback (scratch collections with unique names, dropped afterwards), a real
SQLite task store, the real BGE-M3 tokenizer, and a transport shim that lets the embed host
stage's ``ssh``/``scp`` calls reach this machine itself (the registered host ``arc99`` is
this machine, but its ssh alias is only defined on other machines' configs).
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qm  # noqa: E402

from app.services.pipeline.store import TaskStore  # noqa: E402
from tests.sqlite_task_db import make_session_factory  # noqa: E402

_STORE_DIR = tempfile.TemporaryDirectory(prefix="reembed_tests_", ignore_cleanup_errors=True)

# Loopback only and hardcoded, never from settings/QDRANT_URL.
LOCAL_URL = "http://127.0.0.1:6333"
DIM = 1024


def new_store() -> tuple[TaskStore, Callable]:
    """A task store over a real SQLite file of its own.

    A file, not one shared in-memory connection: the orchestrator's heartbeat thread and the stage
    thread read the same store at once, and a single connection tore those reads (an intermittent
    ``Invalid isoformat string`` / ``StaleDataError`` in about one orchestrator run in eight).
    """
    factory = make_session_factory(Path(_STORE_DIR.name) / f"{uuid.uuid4().hex}.db")
    return TaskStore(factory), factory


def qdrant_client() -> QdrantClient:
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=60)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    return client


def create_scratch_collection(client: QdrantClient, prefix: str, *, keyword_indexes: tuple[str, ...]) -> str:
    name = f"{prefix}_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": qm.VectorParams(size=DIM, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"sparse": qm.SparseVectorParams()},
    )
    for field in keyword_indexes:
        client.create_payload_index(name, field, qm.PayloadSchemaType.KEYWORD, wait=True)
    return name


def drop_collection(client: QdrantClient, name: str) -> None:
    for attempt in range(4):
        try:
            client.delete_collection(name)
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.4 * (attempt + 1))


def unit_vector(seed: int) -> list[float]:
    vec = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


def stored_point(point_id: int, payload: dict, *, seed: int | None = None) -> qm.PointStruct:
    """A point as an old (stale) writer left it: some dense vector, a tiny sparse one."""
    seed = point_id if seed is None else seed
    return qm.PointStruct(
        id=point_id,
        vector={"dense": unit_vector(seed), "sparse": qm.SparseVector(indices=[seed % 1000], values=[1.0])},
        payload=payload,
    )


def write_chunk_shard(chunks_dir: Path, index: int, records: list[dict]) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(chunks_dir / f"chunks_{index:04d}.jsonl.gz", "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_chunk_shard(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def install_local_host(monkeypatch, stages_module, log: list) -> None:
    """Route the host stage's ssh and scp to this machine; ``log`` records every command.

    ``run_remote`` runs the command line in a local shell (the host registry's own python and
    worker paths are local paths here) and ``run_local`` of ``scp`` copies the file between
    the local directory and the "host" directory. Anything else (the SQL dump) is recorded
    and run for real, so a test sees whether it happened.
    """
    real_run_local = stages_module.run_local

    def run_remote(target, cmd, sink, *, env_prefix="", abort=None, popen_factory=None):
        log.append(("ssh", target, cmd))
        done = subprocess.run(f"{env_prefix}{cmd}", shell=True, capture_output=True, text=True)
        for line in done.stdout.splitlines():
            if line.strip():
                sink(line)
        if done.returncode != 0:
            raise RuntimeError(f"command failed (exit {done.returncode}): {cmd}\n{done.stderr[-600:]}")
        return 0

    def run_local(cmd, cwd, sink, **kw):
        if cmd and cmd[0] == "scp":
            log.append(("scp", *cmd[2:]))
            src, dst = str(cmd[-2]), str(cmd[-1])
            if src.startswith("shim-host:"):
                shutil.copy2(src.split(":", 1)[1], dst)
            else:
                target_dir = dst.split(":", 1)[1]
                Path(target_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, Path(target_dir) / Path(src).name)
            return 0
        log.append(("local", *map(str, cmd)))
        return real_run_local(cmd, cwd, sink, **kw)

    monkeypatch.setattr(stages_module, "run_remote", run_remote)
    monkeypatch.setattr(stages_module, "run_local", run_local)


def write_host_registry(path: Path, *, workdir: Path, python: str, worker: Path, model_dir: str,
                        shard_size: int = 25_000) -> None:
    path.write_text(json.dumps({"shim-host": {
        "ssh_target": "shim-host",
        "python": python.replace("\\", "/"),
        "worker": str(worker).replace("\\", "/"),
        "model_dir": model_dir.replace("\\", "/"),
        "workdir": str(workdir).replace("\\", "/"),
        "batch_size": 64,
        "shard_size": shard_size,
    }}), encoding="utf-8")
