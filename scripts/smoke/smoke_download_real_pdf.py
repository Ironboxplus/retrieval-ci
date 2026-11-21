"""Real PDF download smoke using an existing artifact as the HTTP origin.

This deliberately does not manufacture PDF bytes. It serves a caller-supplied
paper through a real origin and forwarding proxy, runs the production corpus
downloader with concurrent workers, validates byte identity, and removes the
temporary downloads before exit.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import http.server
import importlib.util
import json
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from pypdf import PdfReader


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_downloader(args: argparse.Namespace):
    if args.repo_root:
        sys.path.insert(0, str(args.repo_root.resolve()))
    if args.download_control_file:
        import pipeline  # noqa: F401 - establish the real parent package

        _load_module("pipeline.download_control", args.download_control_file.resolve())
    if args.crawler_module_file:
        import pipeline.paper_crawler  # noqa: F401 - establish parent package

        module = _load_module(
            "pipeline.paper_crawler.ccf_a_main_corpus",
            args.crawler_module_file.resolve(),
        )
        return module.CorpusPaper, module.MainConferenceCorpusBuilder

    from pipeline.paper_crawler.ccf_a_main_corpus import (
        CorpusPaper,
        MainConferenceCorpusBuilder,
    )

    return CorpusPaper, MainConferenceCorpusBuilder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--download-control-file", type=Path)
    parser.add_argument("--crawler-module-file", type=Path)
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--max-bytes-per-second", type=int, default=2_000_000)
    args = parser.parse_args()

    source_pdf = args.source_pdf.resolve()
    if not source_pdf.is_file():
        parser.error(f"source PDF does not exist: {source_pdf}")
    if args.copies < 2:
        parser.error("--copies must be at least 2 to exercise concurrency")
    if args.max_bytes_per_second <= 0:
        parser.error("--max-bytes-per-second must be positive")
    PdfReader(str(source_pdf))

    CorpusPaper, MainConferenceCorpusBuilder = _load_downloader(args)
    proxy_receipts: list[dict[str, int | str]] = []
    receipt_lock = threading.Lock()
    direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    class Origin(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            if self.path != "/paper.pdf":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(source_pdf.stat().st_size))
            self.end_headers()
            with source_pdf.open("rb") as source:
                for chunk in iter(lambda: source.read(64 * 1024), b""):
                    self.wfile.write(chunk)

        def log_message(self, *_args) -> None:
            return

    class ForwardProxy(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            target = self.path
            if not target.startswith("http://"):
                self.send_error(400, "absolute-form HTTP target required")
                return
            transferred = 0
            with direct_opener.open(target, timeout=30) as upstream:
                self.send_response(upstream.status)
                self.send_header("Content-Type", upstream.headers.get_content_type())
                self.send_header("Content-Length", upstream.headers.get("Content-Length", ""))
                self.end_headers()
                while True:
                    chunk = upstream.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    transferred += len(chunk)
            with receipt_lock:
                proxy_receipts.append({"target": target, "bytes": transferred})

        def log_message(self, *_args) -> None:
            return

    origin = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Origin)
    proxy = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ForwardProxy)
    origin.daemon_threads = True
    proxy.daemon_threads = True
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    threading.Thread(target=proxy.serve_forever, daemon=True).start()

    source_hash = _sha256(source_pdf)
    source_bytes = source_pdf.stat().st_size
    try:
        with tempfile.TemporaryDirectory(prefix="retrival-real-pdf-smoke-") as temp_dir:
            corpus_root = Path(temp_dir) / "corpus"
            source_url = f"http://127.0.0.1:{origin.server_address[1]}/paper.pdf"
            papers = [
                CorpusPaper(
                    paper_id=f"real-pdf-smoke-{index}",
                    title=f"Real PDF smoke copy {index}",
                    authors=[],
                    conference="ACL",
                    year=2026,
                    pdf_url=source_url,
                    paper_page_url=source_url,
                    source_name="existing-artifact-smoke",
                )
                for index in range(args.copies)
            ]
            builder = MainConferenceCorpusBuilder(
                corpus_root=corpus_root,
                proxy_url=f"http://127.0.0.1:{proxy.server_address[1]}",
                years=(2026,),
                conferences=(),
            )
            started = time.monotonic()
            completed = asyncio.run(builder.download_papers(
                papers,
                max_concurrent=args.copies,
                max_bytes_per_second=args.max_bytes_per_second,
            ))
            elapsed = time.monotonic() - started
            paths = [Path(paper.pdf_path or "") for paper in completed]
            if not all(path.is_file() for path in paths):
                raise AssertionError("not every concurrent download produced an artifact")
            if not all(_sha256(path) == source_hash for path in paths):
                raise AssertionError("downloaded PDF differs from the source artifact")
            minimum = source_bytes * args.copies / args.max_bytes_per_second
            if elapsed < minimum * 0.85:
                raise AssertionError(
                    f"aggregate limit violated: elapsed={elapsed:.3f}s minimum={minimum:.3f}s"
                )
            if len(proxy_receipts) != args.copies:
                raise AssertionError(
                    f"expected {args.copies} proxy receipts, got {len(proxy_receipts)}"
                )
            print(json.dumps({
                "status": "passed",
                "source_pdf": str(source_pdf),
                "source_bytes": source_bytes,
                "source_sha256": source_hash,
                "copies": args.copies,
                "aggregate_bytes": source_bytes * args.copies,
                "max_bytes_per_second": args.max_bytes_per_second,
                "elapsed_seconds": round(elapsed, 3),
                "proxy_receipts": proxy_receipts,
                "temporary_artifacts_retained": False,
            }, ensure_ascii=False))
    finally:
        proxy.shutdown()
        origin.shutdown()
        proxy.server_close()
        origin.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
