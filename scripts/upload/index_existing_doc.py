"""Run the upload pipeline for an ALREADY-CREATED document whose Markdown exists.

The normal path is POST /api/documents/upload with auto_index=true, which extracts
with marker and then indexes. That is wrong for a document whose Markdown was
produced out-of-band, and re-running marker on a 1,369-page book costs ~2 GPU
hours and, with the pipeline's hardcoded `disable_ocr=True`, produces unusable
output for line-numbered editions (see marker_extract_book.py).

So: upload with auto_index=false, drop the good Markdown at the path
`upload_md_path()` dictates, and call this. The extract stage is
skip_when_satisfied and its probe is `md_exists()`, so it records the artifact as
already present and the run goes straight to indexing.

This refuses to start when the Markdown is missing — otherwise the extract stage
would quietly run marker with the wrong config, which is the exact outcome the
whole exercise exists to avoid.

Run it with the BACKEND's environment, not a bare shell. The indexing subprocess
inherits this process's env, and the module defaults point at localhost:6333 /
grpc 6334 while this deployment's Qdrant is on 16333/16334 — plus BGE_M3_PATH and
the OFFLINE flags, without which the embedder reaches for huggingface.co. The
cheapest correct way to get all of them is to lift them off the running backend:

    tr '\\0' '\\n' < /proc/$(pgrep -f 'uvicorn.*main:app' | head -1)/environ \\
      | grep -E '^(QDRANT_|BGE_|HF_|TRANSFORMERS_)' > /tmp/backend_env.sh
    set -a && . /tmp/backend_env.sh && set +a

index_one.py preflights Qdrant before embedding, so a half-configured environment
fails in seconds rather than after the GPU work.

Usage:
    python scripts/upload/index_existing_doc.py --doc-id 114853
    python scripts/upload/index_existing_doc.py --doc-id 114853 --wait
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc-id", type=int, required=True)
    ap.add_argument("--wait", action="store_true",
                    help="drive the pipeline in this process instead of a daemon thread")
    args = ap.parse_args()

    from app.services.pipeline.upload_context import upload_md_path
    from app.services.pipeline.upload_service import get_upload_runner
    from core.db import get_db_session
    from core.db.models import Document

    with get_db_session() as db:
        doc = db.query(Document).filter(Document.id == args.doc_id).one_or_none()
        if doc is None:
            print(f"[fail] no document with id {args.doc_id}")
            return 2
        file_path = doc.file_path
        title = doc.title
        source_type = doc.source_type.value if doc.source_type else None

    md = upload_md_path(file_path)
    if not (md.is_file() and md.stat().st_size > 0):
        print(
            f"[fail] markdown missing for doc {args.doc_id}: {md}\n"
            f"       place the extracted Markdown there first — starting without it "
            f"would run marker with the pipeline's default config."
        )
        return 3

    print(f"[doc]  {args.doc_id} type={source_type} title={title!r}")
    print(f"[pdf]  {file_path}")
    print(f"[md]   {md} ({md.stat().st_size:,} bytes)")

    result = get_upload_runner().start_upload(args.doc_id, file_path, wait=args.wait)
    print(f"[run]  {result}")
    return 0 if result.get("status") == "started" else 1


if __name__ == "__main__":
    raise SystemExit(main())
