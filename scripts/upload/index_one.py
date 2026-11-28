"""Local upload-index worker: chunk + embed + Qdrant-upsert one document.

Invoked by the upload pipeline's `index` stage via run_local (cwd=project root):

    python scripts/upload/index_one.py --doc-id N

Reads the markdown the extract stage wrote (resolved from the doc's file_path via
the shared upload_md_path convention — so this stays self-contained on doc_id
alone, which is what backend-only-polls + resume need), then calls the proven
IndexService.index_document. That one call chunks, embeds (BGE-M3 dense+sparse),
upserts to Qdrant (8-field payload whitelist, INT doc_id per W24), writes the SQL
Chunk rows, and flips the document to `indexed` — atomically. Reusing it means we
do NOT hand-roll the Qdrant payload / point-id scheme (the W9/W10/W24 landmines).

Runs in its own process (off the event loop) and is serialized on the shared
`qdrant` resource lock by the orchestrator, so it never races a venue ingest for
Qdrant's tight commit-charge headroom (W58/W60). Exit 0 on success.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _preflight_qdrant() -> str | None:
    """Ping Qdrant BEFORE the embed. Returns an error message, or None if reachable.

    Order matters: embedding is the expensive half (a 2.5M-char book measured 281s
    / 5,997 chunks on one 4090) and the upsert is the first thing that touches
    Qdrant, so a bad endpoint is not discovered until all of that work is already
    spent. And it fails *quietly* — `_get_shared_client` logs and returns a
    degraded client rather than raising, so there is no exception to trip over.

    This bit for real: run without the backend's environment, the module defaults
    apply (localhost:6333 / grpc 6334) while this deployment's Qdrant listens on
    QDRANT_URL=...:16333 with QDRANT_GRPC_PORT=16334. Upload task 329 embedded for
    276s and then died on `Connection refused`; task 328 burned 175s reaching for
    huggingface.co for the same class of reason (BGE_M3_PATH / *_OFFLINE unset).
    Under the pipeline the env is inherited from the backend and this check is a
    no-op costing one round trip.
    """
    from core.storage import qdrant_store as qs

    url = getattr(qs.settings, "qdrant_url", None) or qs.QDRANT_URL
    where = f"{url} grpc={qs.QDRANT_GRPC_PORT} prefer_grpc={qs.QDRANT_PREFER_GRPC}"
    client = None
    try:
        client = qs._build_server_client(url)
        client.get_collections()
    except Exception as exc:
        return (
            f"Qdrant unreachable at {where}: {exc}\n"
            f"       Set QDRANT_URL / QDRANT_GRPC_PORT / QDRANT_PREFER_GRPC to match the\n"
            f"       running backend (read them from /proc/<backend-pid>/environ) before\n"
            f"       indexing — otherwise the embed runs to completion and is thrown away."
        )
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    _log(f"[index] qdrant reachable: {where}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc-id", type=int, required=True)
    args = ap.parse_args()

    from core.db import Document, get_db_session
    from app.services.pipeline.upload_context import upload_md_path

    # Pull the doc's metadata (title/source/metadata + file_path) up front; the
    # session is released before the heavy embed so we don't hold a connection.
    with get_db_session() as db:
        doc = db.query(Document).filter(Document.id == args.doc_id).first()
        if doc is None:
            _log(f"[index] ERROR doc {args.doc_id} not in DB")
            return 3
        title = doc.title
        source = doc.source
        metadata = dict(doc.metadata_json or {})
        file_path = doc.file_path

    md = upload_md_path(file_path)
    _log(f"[index] doc={args.doc_id} reading md={md}")
    if not md.is_file() or md.stat().st_size == 0:
        _log(f"[index] ERROR md artifact missing/empty: {md}")
        return 4
    content = md.read_text(encoding="utf-8", errors="replace")

    # Fail on an unreachable Qdrant now, not after the embed. See _preflight_qdrant.
    problem = _preflight_qdrant()
    if problem:
        _log(f"[index] ERROR {problem}")
        return 6

    _log(f"[index] {len(content)} chars -> chunk + embed (BGE-M3) + Qdrant upsert")

    from app.services.index_service import get_index_service
    with get_db_session() as db:
        result = get_index_service().index_document(
            doc_id=args.doc_id,
            content=content,
            title=title,
            source=source,
            metadata=metadata,
            db=db,
            source_type="upload",
            data_source="user",
        )
    if not result.get("success"):
        _log(f"[index] ERROR index_document failed: {result.get('error')}")
        return 5
    _log(f"[index] OK chunks={result.get('chunk_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
