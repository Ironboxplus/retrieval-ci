"""Run a real Qdrant smoke test for ACE Journal vector storage.

This starts a temporary Qdrant server on local ports and points it at a temporary
storage directory under the OS temp folder. It never touches
`data/indices/qdrant_server` and never uses the production `papers` collection.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_proxy_key, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

QDRANT_EXE = REPO_ROOT / "tools" / "qdrant" / "qdrant.exe"
DEFAULT_COLLECTION = "journal_smoke"


def wait_qdrant_ready(url: str) -> None:
    deadline = time.time() + 15
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{url}/readyz", timeout=0.5)
            if response.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001 - keep probing until timeout.
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"temporary Qdrant did not become ready; last_error={last_error!r}")


def start_temporary_qdrant(http_port: int, grpc_port: int, storage_path: Path) -> subprocess.Popen:
    if not QDRANT_EXE.exists():
        raise FileNotFoundError(f"qdrant.exe not found: {QDRANT_EXE}")
    env = os.environ.copy()
    env["QDRANT__SERVICE__HTTP_PORT"] = str(http_port)
    env["QDRANT__SERVICE__GRPC_PORT"] = str(grpc_port)
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(storage_path)
    env["QDRANT_URL"] = f"http://127.0.0.1:{http_port}"
    # QdrantStore prefers gRPC.  Keep the client-side port in lockstep with the
    # isolated server instead of silently falling back to a production/default 6334.
    env["QDRANT_GRPC_PORT"] = str(grpc_port)
    return subprocess.Popen(
        [str(QDRANT_EXE), "--disable-telemetry"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )


def reset_shared_qdrant_client() -> None:
    from core.storage import qdrant_store

    if qdrant_store._shared_client is not None:
        try:
            qdrant_store._shared_client.close()
        except Exception:
            pass
    qdrant_store._shared_client = None
    qdrant_store._shared_client_key = None
    qdrant_store._shared_client_validated_at = 0.0


def run_vector_smoke(qdrant_url: str, collection: str) -> dict:
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["QDRANT_GRPC_PORT"] = str(int(qdrant_url.rsplit(":", 1)[1]) + 1)

    from app.services.journal.chunker import JournalTextChunk
    from app.services.journal.vector_store import JournalVectorStore

    reset_shared_qdrant_client()
    store = JournalVectorStore(collection_name=collection)
    chunks = [
        JournalTextChunk(
            chunk_id="journal_chunk_smoke_alpha",
            entry_id="journal_entry_smoke_alpha",
            chunk_index=0,
            level=2,
            heading="body",
            content="alpha owner isolation qdrant journal smoke",
            token_count=6,
        ),
        JournalTextChunk(
            chunk_id="journal_chunk_smoke_beta",
            entry_id="journal_entry_smoke_beta",
            chunk_index=0,
            level=2,
            heading="body",
            content="beta other owner qdrant journal smoke",
            token_count=6,
        ),
    ]
    dense_vectors = [
        [1.0] + [0.0] * 1023,
        [0.0, 1.0] + [0.0] * 1022,
    ]
    sparse_vectors = [
        {"1": 1.0, "2": 0.5},
        {"3": 1.0, "2": 0.5},
    ]

    inserted_alpha = store.upsert_chunks(
        [chunks[0]],
        project_id=1,
        project_slug="retrival-smoke",
        entry_id="journal_entry_smoke_alpha",
        owner_user_id="user-one",
        client="qdrant-smoke",
        source_kind="session_summary",
        tags=["smoke"],
        dense_vectors=[dense_vectors[0]],
        sparse_vectors=[sparse_vectors[0]],
    )
    inserted_beta = store.upsert_chunks(
        [chunks[1]],
        project_id=2,
        project_slug="retrival-smoke",
        entry_id="journal_entry_smoke_beta",
        owner_user_id="user-two",
        client="qdrant-smoke",
        source_kind="session_summary",
        tags=["smoke"],
        dense_vectors=[dense_vectors[1]],
        sparse_vectors=[sparse_vectors[1]],
    )

    alpha_hits = store.search(
        dense_vector=dense_vectors[0],
        sparse_vector=sparse_vectors[0],
        mode="hybrid",
        top_k=5,
        filters={"project_slug": "retrival-smoke", "owner_user_id": "user-one"},
    )
    beta_view_alpha = store.search(
        dense_vector=dense_vectors[0],
        sparse_vector=sparse_vectors[0],
        mode="hybrid",
        top_k=5,
        filters={"project_slug": "retrival-smoke", "owner_user_id": "user-two"},
    )
    deleted = store.delete_by_entry_id("journal_entry_smoke_alpha")
    after_delete = store.search(
        dense_vector=dense_vectors[0],
        sparse_vector=sparse_vectors[0],
        mode="hybrid",
        top_k=5,
        filters={"project_slug": "retrival-smoke", "owner_user_id": "user-one"},
    )

    summary = {
        "qdrant_url": qdrant_url,
        "collection": collection,
        "inserted_alpha": inserted_alpha,
        "inserted_beta": inserted_beta,
        "alpha_hits": [hit["metadata"]["entry_id"] for hit in alpha_hits],
        "beta_view_alpha": [hit["metadata"]["entry_id"] for hit in beta_view_alpha],
        "deleted_alpha_vectors": deleted,
        "after_delete_alpha_hits": [hit["metadata"]["entry_id"] for hit in after_delete],
    }
    summary["assertions"] = {
        "upserted_two_points": inserted_alpha == 1 and inserted_beta == 1,
        "alpha_owner_filter_hits_alpha": "journal_entry_smoke_alpha" in summary["alpha_hits"],
        "alpha_owner_filter_excludes_beta": "journal_entry_smoke_beta" not in summary["alpha_hits"],
        "beta_owner_filter_excludes_alpha": "journal_entry_smoke_alpha" not in summary["beta_view_alpha"],
        "delete_by_entry_removed_alpha": deleted == 1 and summary["after_delete_alpha_hits"] == [],
    }
    summary["failed"] = [name for name, ok in summary["assertions"].items() if not ok]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-port", type=int, default=18337)
    parser.add_argument("--grpc-port", type=int, default=18338)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="qdrant_journal_smoke_", ignore_cleanup_errors=True) as tmp:
        storage_path = Path(tmp) / "storage"
        qdrant_url = f"http://127.0.0.1:{args.http_port}"
        proc = start_temporary_qdrant(args.http_port, args.grpc_port, storage_path)
        try:
            wait_qdrant_ready(qdrant_url)
            summary = run_vector_smoke(qdrant_url, args.collection)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 1 if summary["failed"] else 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            reset_shared_qdrant_client()


if __name__ == "__main__":
    raise SystemExit(main())
