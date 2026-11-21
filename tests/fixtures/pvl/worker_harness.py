"""Runs the real ``main()`` of scripts/remote_pvl/pvl_worker.py with the two parts that need a GPU
replaced by scripted stand-ins, as an actual subprocess.

What is real: the worker's argument handling, shard and checkpoint helpers, batching, the
restart and flush decisions, the report and Markdown writing, the summary file, the exit code
and the watchdog thread. What is scripted (``PVL_HARNESS`` is a JSON object): the model server
(``VllmServer``, which needs nvidia-smi and a vLLM binary) and the extractor (which needs
paddle); both honour the interfaces ``pvl_worker`` uses and record what was asked of them in
``trace`` (one JSON object per line).

    broken_calls   convert() calls (1-based, in order) whose papers all come back failed and retryable
    die_calls      convert() calls that raise, as a broken checker does
    start_fails    server.start() calls that raise, as a card without room does
    healthy        what the server answers when asked whether it is alive
    unresolved     every paper keeps two failed pages until the second model has been through them
    pages          pages per paper
    refused        paper stems the checker refuses (no Markdown, a failed report, its pages handed to keep_pages)
"""
import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pipeline.paper_crawler.converters.extractors.base import ExtractionResult, ExtractionStatus  # noqa: E402
from pipeline.paper_crawler.converters.extractors.paddleocr_vl import Batch  # noqa: E402

CONFIG = json.loads(os.environ["PVL_HARNESS"])
TRACE = Path(CONFIG["trace"])
PAGES = CONFIG.get("pages", 5)


def note(**event) -> None:
    with TRACE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def written_papers(output: Path) -> int:
    return sum(1 for p in output.rglob("*.md"))


class Report:
    def __init__(self, failed):
        self.failed_pages = list(failed)
        self.error = None

    def to_json(self) -> dict:
        return {"pages": [{"page": n} for n in range(PAGES)]}


class Server:
    starts = 0

    def __init__(self, **kwargs):
        self.process = None
        self.effective_config = {}
        self.port = 1

    @property
    def url(self) -> str:
        return "http://127.0.0.1:1/v1"

    def start(self) -> None:
        Server.starts += 1
        note(event="server.start", n=Server.starts)
        if Server.starts in CONFIG.get("start_fails", []):
            raise RuntimeError("only 1900 MiB of 12282 MiB are free: 0.01 of the card is left for the model server")
        self.process = types.SimpleNamespace(pid=os.getpid())

    def healthy(self) -> bool:
        return self.process is not None and CONFIG.get("healthy", True)

    def stop(self, grace_s: float = 30.0) -> None:
        note(event="server.stop")
        self.process = None

    def kill(self) -> None:
        pass


class Extractor:
    calls = 0

    def __init__(self, output: Path):
        self.output = output
        self.server_url = ""
        self._pipeline = object()

    def release_pipeline(self, server_url=None) -> None:
        note(event="release_pipeline")
        self._pipeline = None
        if server_url:
            self.server_url = server_url

    def convert(self, pdfs, kept=None):
        Extractor.calls += 1
        n = Extractor.calls
        note(event="convert", call=n, papers=[p.stem for p in pdfs], md_written=written_papers(self.output),
             kept={p.stem: v for p, v in (kept or {}).items()})
        if n in CONFIG.get("die_calls", []):
            raise RuntimeError("scripted checker failure")
        batch = Batch(pdfs=list(pdfs), t0=0.0)
        batch.runnable = list(pdfs)
        for pdf in pdfs:
            batch.expected[pdf] = PAGES
            if n in CONFIG.get("broken_calls", []):
                batch.reports[pdf] = Report([])
                batch.pages[pdf], batch.markdown_pages[pdf] = {}, {}
                batch.results[pdf] = ExtractionResult(
                    pdf_path=pdf, status=ExtractionStatus.FAILED, strategy="scripted", error="pipeline stream failed",
                    extra={"report": batch.reports[pdf].to_json(), "retryable": True})
                continue
            batch.reports[pdf] = Report([1, 2] if CONFIG.get("unresolved") else [])
            batch.pages[pdf] = {i: {} for i in range(PAGES)}
            batch.markdown_pages[pdf] = {i: "" for i in range(PAGES)}
        batch.todo = [pdf for pdf in pdfs if pdf not in batch.results]
        return batch

    def assemble(self, batch, skip=(), only=None):
        for pdf in batch.todo:
            if pdf in skip or (only is not None and pdf not in only) or pdf in batch.results:
                continue
            if pdf.stem in CONFIG.get("refused", []):  # the checker's verdict: no Markdown, an empty report for the worker to fill
                if getattr(self, "keep_pages", None) is not None:  # the real extractor hands the refused paper's pages over
                    self.keep_pages(pdf, {"schema": 1, "paper": pdf.stem, "page_count": PAGES})
                batch.results[pdf] = ExtractionResult(
                    pdf_path=pdf, status=ExtractionStatus.FAILED, strategy="scripted",
                    error="paper failed verification: page 3: omission")
                continue
            batch.results[pdf] = ExtractionResult(
                pdf_path=pdf, status=ExtractionStatus.SUCCESS, strategy="scripted", markdown=f"# {pdf.stem}\n",
                extra={"report": batch.reports[pdf].to_json()})
        return {pdf: batch.results[pdf] for pdf in batch.pdfs
                if pdf in batch.results and pdf not in skip and (only is None or pdf in only)}


def main() -> int:
    worker_file = Path(os.environ.get("PVL_WORKER_FILE") or REPO / "scripts" / "remote_pvl" / "pvl_worker.py")
    spec = importlib.util.spec_from_file_location("pvl_worker", worker_file)
    worker = importlib.util.module_from_spec(spec)
    sys.modules["pvl_worker"] = worker
    spec.loader.exec_module(worker)

    def build_extractor(args, server, watchdog):
        note(event="extractor", omp=os.environ.get("OMP_NUM_THREADS"), allocator=os.environ.get("FLAGS_allocator_strategy"),
             loggers={n: logging.getLogger(n).level for n in ("httpx", "httpcore", "openai", "mineru_vl_utils")})
        return Extractor(args.output)

    def fallback_phase(args, extractor, held, gpu, quality, live):
        note(event="fallback", papers=sum(len(w) for _, w in held), md_written=written_papers(args.output))
        for batch, _ in held:
            for pdf in batch.todo:
                batch.reports[pdf].failed_pages = []

    worker.VllmServer = Server
    worker._build_extractor = build_extractor
    worker._fallback_phase = fallback_phase
    return worker.main()


if __name__ == "__main__":
    sys.exit(main())
