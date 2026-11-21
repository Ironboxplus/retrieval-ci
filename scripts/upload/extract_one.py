"""Local upload-extract worker: marker PDF→Markdown + SMB persistence.

Invoked by the upload pipeline's `extract` stage via run_local (cwd=project root):

    python scripts/upload/extract_one.py --doc-id N --input FILE --output MD

What it does — and crucially, it runs in its OWN process, OFF the FastAPI event
loop. A 275MB / 1000-page PDF used to be parsed synchronously inside the upload
request handler (pypdf, in the event loop) and froze the whole app; that is the
bug this worker exists to kill.

  1. Persist the source PDF to the SMB corpus share (papers/pdf/uploads/), so an
     uploaded doc lives alongside the 64K-paper corpus on the 11T NAS instead of
     only on Arc's local disk. SMB writes go via robocopy — git-bash '>' can't
     write the CIFS mount (env.md / W21).
  2. PDF → Markdown via the SAME MarkerExtractor the venue pipeline wraps
     (marker-pdf 2.0.0 + surya 2), with a PyMuPDF fallback so a marker crash on a
     weird PDF still yields text rather than hard-failing the upload. NOT pypdf
     (poor text). Non-PDF uploads (.txt/.md/.html) reuse the existing parser
     dispatch — no wheel reinvented.
  3. Write the markdown locally (--output: the stage's md_exists() probe artifact
     and the index stage's input) AND persist it to SMB (papers/markdown/uploads/),
     mirroring the paper markdown layout.

SMB persistence is best-effort (logged loudly on failure): a transient CIFS hiccup
must not throw away an otherwise-good extraction — the local md still drives
indexing. Exit 0 on success; non-zero makes run_local raise → stage → task failed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# W70: lock CUDA enumeration to PCI-bus order BEFORE any torch/marker/surya import
# so the GPU marker grabs matches nvidia-smi. Defensive — Arc is single-box today,
# but the rule is free and prevents a surprise if a second card ever appears.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Silence marker/surya tqdm progress bars. run_local streams this worker's stdout
# (and its marker subprocesses', which inherit this env) line-by-line into the
# task event log — a 1300-page book emitted 5400+ tqdm lines and bloated the UI.
# Our progress signal is the extract probe (md_exists 0/1), not per-page bars.
# tqdm reads TQDM_* env vars as defaults for its params, so this disables every bar.
os.environ.setdefault("TQDM_DISABLE", "1")

# SMB corpus share (same NAS the papers live on; env-overridable). Mirrors
# StageContext.arc_smb_unc. Uploads land under papers/{pdf,markdown}/uploads/,
# parallel to papers/{pdf,markdown}/{conf}/{year}/.
_SMB_UNC = os.environ.get("RETRIVAL_SMB_UNC", "//10.100.100.9/lsb/retrival")
_SMB_PDF_DIR = f"{_SMB_UNC}/papers/pdf/uploads"
_SMB_MD_DIR = f"{_SMB_UNC}/papers/markdown/uploads"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _robocopy_to_smb(src_file: Path, dst_dir: str) -> bool:
    """robocopy ONE file into an SMB dir (native SMB write; W21/env.md). robocopy
    exit codes 0-7 are success (files copied / nothing to do), >=8 is failure."""
    # MSYS2_ARG_CONV_EXCL='*' stops msys path-munging //host/share → /host/share.
    env = dict(os.environ, MSYS2_ARG_CONV_EXCL="*")
    cmd = [
        "robocopy", str(src_file.parent), dst_dir, src_file.name,
        "/R:2", "/W:2", "/NFL", "/NDL", "/NJH", "/NJS", "/NP",
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    except Exception as exc:  # never let an SMB hiccup kill the extraction
        _log(f"[extract] SMB persist FAILED (exec) {src_file.name} -> {dst_dir}: {exc}")
        return False
    if proc.returncode >= 8:
        tail = (proc.stdout or proc.stderr or "").strip()[-300:]
        _log(f"[extract] SMB persist FAILED rc={proc.returncode} {src_file.name} -> {dst_dir}: {tail}")
        return False
    _log(f"[extract] SMB persisted {src_file.name} -> {dst_dir} (rc={proc.returncode})")
    return True


def _extract_markdown(src: Path, doc_id: int) -> str:
    """Source file → markdown text. PDF via marker (pymupdf fallback); plain text
    passed through; everything else (html) via the existing parser dispatch."""
    ext = src.suffix.lower()
    if ext == ".pdf":
        from pipeline.paper_crawler.converters.extractors import (
            FallbackExtractor, MarkerExtractor, PyMuPDFExtractor,
        )
        # marker primary (high-quality layout/markdown) → pymupdf fallback so a
        # marker failure degrades to plain text instead of failing the upload.
        # disable_ocr=True: academic PDFs / textbooks are born-digital (reliable
        # text layer), so skip the ~60% of marker wall-time spent on OCR text
        # recognition — same throughput choice the venue worker made. Measured on
        # Arc: 27s vs 69s for a 10-page paper, identical structure (27 headers).
        # Layout/structure extraction still runs → far better than pypdf.
        chain = FallbackExtractor(
            primary=MarkerExtractor(extra_config={"disable_ocr": True}),
            fallback=PyMuPDFExtractor(),
        )
        _log(f"[extract] marker extracting {src.name} (this can take minutes for a large PDF) ...")
        result = chain.extract(src)
        if not result.succeeded or not (result.markdown or "").strip():
            raise RuntimeError(
                f"marker+pymupdf both failed: strategy={result.strategy} err={result.error}"
            )
        _log(f"[extract] extraction OK strategy={result.strategy} chars={len(result.markdown)}")
        return result.markdown
    if ext in (".txt", ".md"):
        _log(f"[extract] plain-text passthrough ({ext})")
        return src.read_text(encoding="utf-8", errors="replace")
    # html/htm and anything else: reuse DocumentService's parser dispatch (DRY).
    from core.db import Document, get_db_session
    from app.services.document_service import DocumentService
    _log(f"[extract] parser-dispatch extraction ({ext})")
    with get_db_session() as db:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        res = DocumentService(db=db).process_uploaded_file(
            file_path=str(src), filename=src.name, db=db, existing_doc=doc,
        )
    if not res.get("success"):
        raise RuntimeError(f"parser dispatch failed for {ext}: {res.get('error')}")
    return res.get("content") or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc-id", type=int, required=True)
    ap.add_argument("--input", required=True, help="uploaded source file path (data/uploads/...)")
    ap.add_argument("--output", required=True, help="markdown artifact path to write")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    _log(f"[extract] doc={args.doc_id} src={src.name} -> {out.name}")

    if not src.is_file():
        _log(f"[extract] ERROR input not found: {src}")
        return 2

    # Idempotent resume (extract is skip_when_satisfied): a non-empty md present
    # means the expensive marker pass already ran — don't redo it.
    if out.is_file() and out.stat().st_size > 0:
        _log(f"[extract] md already present ({out.stat().st_size} bytes) — skip extraction")
    else:
        # 1. persist source PDF to SMB (best-effort) — uploads live with papers.
        _robocopy_to_smb(src, _SMB_PDF_DIR)
        # 2. extract → markdown.
        try:
            markdown = _extract_markdown(src, args.doc_id)
        except Exception as exc:
            _log(f"[extract] ERROR extraction failed: {exc}")
            return 4
        if not markdown.strip():
            _log("[extract] ERROR empty markdown produced")
            return 5
        # 3a. local md artifact (stage probe + index input).
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        _log(f"[extract] wrote {len(markdown)} chars -> {out}")

    # 3b. persist md to SMB (best-effort), parallel to papers/markdown/.
    if out.is_file() and out.stat().st_size > 0:
        _robocopy_to_smb(out, _SMB_MD_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
