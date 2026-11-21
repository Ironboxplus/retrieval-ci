"""Single-GPU PaddleOCR-VL batch worker, a drop-in for ``marker_worker.py``.

Same contract: ``--input DIR --output DIR --gpu N [--rank R --world-size W --recursive]``
writes ``<stem>.md`` per PDF, an append-only ``<output>/done.rank{R}.txt`` checkpoint,
skips what already exists and exits non-zero when a paper failed. The PDF discovery,
sharding, mirroring and checkpoint helpers are marker_worker's own, imported as they are.

What differs is the engine. The worker owns one vLLM server per rank (the model server of
PaddleOCR-VL, started with ``paddlex_genai_server``), runs the full PaddleOCR-VL pipeline
against it through ``PaddleOCRVLExtractor`` and writes a paper's Markdown only when every
page stood up to the PDF's text layer. A refused paper gets no ``.md`` and is not
checkpointed (the next run tries it again); either way its report lands in
``<output>/_quality/<stem>.json``.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import gzip
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "remote_marker"))

from marker_worker import _gather_pdfs, _load_done, _mirror_output, _shard  # noqa: E402

PR_SET_PDEATHSIG = 1
MODEL_NAME = "PaddleOCR-VL-1.6-0.9B"
# vLLM may use what is free on the card minus what the pipeline process takes: the layout
# model, PP-OCR and the CUDA context. Measured on the RTX 4070 SUPER on 2026-11-13: that
# process peaked at 3,518 MiB on 1224x1584 pages (a smaller reserve ran Paddle out of memory).
# It is the process's peak: what the process already holds when a server is (re)started counts toward it
# (task 1306, 2026-11-15: a restart charged the CUDA context twice and the server had no KV cache left).
LAYOUT_RESERVE_MIB = 3700
MEMORY_MARGIN_MIB = 500
MIN_SERVER_FRACTION = 0.20
FALLBACK_RESERVE_MIB = 0  # the pipeline process has taken its memory by then: it is not free any more
HEALTH_TIMEOUT_S = 600
RESTART_STRIKES = 3  # batches in a row whose server could not be brought back before the worker gives up
DEFER_FLUSH_PAPERS = 60  # papers held for the second model before the card is handed over to it and they are written

log = logging.getLogger("pvl_worker")


def _die_with_parent() -> None:
    """In the child, before exec: SIGTERM when this worker dies, even by kill -9."""
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def free_gpu_memory_mib(gpu: int) -> tuple[int, int]:
    """(free, total) MiB of physical GPU ``gpu`` from nvidia-smi (no CUDA context here)."""
    out = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
        text=True)
    free, total = (int(v) for v in out.strip().split(","))
    return free, total


def _parse_compute_apps(text: str, pid: int) -> int:
    """MiB held by ``pid`` in the output of ``nvidia-smi --query-compute-apps=pid,used_memory
    --format=csv,noheader,nounits`` (one ``pid, MiB`` line per process); 0 when it is not listed."""
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                return int(fields[1])
            except ValueError:  # "[N/A]" where the driver does not report per-process memory
                return 0
    return 0


def own_gpu_memory_mib(gpu: int, pid: Optional[int] = None) -> int:
    """MiB that process ``pid`` (default: this one) holds on physical GPU ``gpu``. 0 when it has no CUDA
    context there yet, and 0 when nvidia-smi fails: the full reserve then applies."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True)
    except (OSError, subprocess.SubprocessError):
        log.debug("nvidia-smi compute-apps query failed", exc_info=True)
        return 0
    return _parse_compute_apps(out, os.getpid() if pid is None else pid)


def server_memory_fraction(free_mib: int, total_mib: int, reserve_mib: int = LAYOUT_RESERVE_MIB,
                           held_mib: int = 0) -> float:
    """``gpu-memory-utilization`` for a server that must leave room for the pipeline. The reserve is the
    pipeline process's peak; ``held_mib`` of it is already taken (and no longer free) when the server is
    restarted, so only ``max(reserve_mib - held_mib, 0)`` is set aside."""
    effective_reserve = max(reserve_mib - held_mib, 0)
    fraction = (free_mib - effective_reserve - MEMORY_MARGIN_MIB) / total_mib
    if fraction < MIN_SERVER_FRACTION:
        raise RuntimeError(
            f"only {free_mib} MiB of {total_mib} MiB are free: {fraction:.2f} of the card is left for the "
            f"model server after reserving {effective_reserve + MEMORY_MARGIN_MIB} MiB "
            f"({reserve_mib} reserve - {held_mib} already held, + {MEMORY_MARGIN_MIB} margin) (< {MIN_SERVER_FRACTION})")
    return round(min(fraction, 0.90), 3)


def server_environment() -> dict:
    """The environment of a model server this worker starts. vLLM closes an idle keep-alive connection after
    ``VLLM_HTTP_TIMEOUT_KEEP_ALIVE`` seconds (5 by default); the pipeline idles longer than that between
    bursts (layout runs, the model waits), so the client's next request met a closed connection and openai
    retried it, 16 requests at a time (the pool), 160+ "Retrying request" events in task 1302 and 0.4 s each."""
    return {**os.environ, "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "3600"}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def server_answers(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def release_gpu_cache() -> None:
    """Hand the blocks Paddle caches but no longer uses back to the driver. Its allocator keeps
    everything it ever grew to, which on a card shared with a vLLM server and the backend is memory
    the next batch, or the next server's sizing, needs (task 1302 died of exactly that)."""
    gc.collect()
    paddle = sys.modules.get("paddle")  # not imported yet = nothing cached
    if paddle is None:
        return
    try:
        paddle.device.cuda.empty_cache()
    except Exception:
        log.debug("empty_cache failed", exc_info=True)


class VllmServer:
    """One vLLM server process group owned by this worker. ``command`` is the server's command
    line with ``{port}`` and ``{config}`` where the worker's free port and the YAML it writes
    (the one it was given, with ``gpu-memory-utilization`` set from the card's free memory)
    go, and ``{root}`` for the repository; the server runs in its own environment, so the
    command names its binary."""

    def __init__(self, *, gpu: int, config: Path, prefix: list[str], port: int, log_path: Path, command: list[str],
                 reserve_mib: int = LAYOUT_RESERVE_MIB):
        self.gpu, self.config, self.prefix, self.command = gpu, config, prefix, command
        self.reserve_mib = reserve_mib
        self.port = port or free_port()
        self.log_path = log_path
        self.process: Optional[subprocess.Popen] = None
        self.effective_config: dict = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> None:
        import yaml

        free, total = free_gpu_memory_mib(self.gpu)
        settings = yaml.safe_load(self.config.read_text(encoding="utf-8")) or {}
        # a figure in the YAML is a ceiling: a model that needs little must not take what is free
        held = own_gpu_memory_mib(self.gpu) if self.reserve_mib > 0 else 0
        fraction = server_memory_fraction(free, total, self.reserve_mib, held_mib=held)
        settings["gpu-memory-utilization"] = min(fraction, settings.get("gpu-memory-utilization", fraction))
        self.effective_config = settings
        effective = self.log_path.with_suffix(".yaml")
        effective.parent.mkdir(parents=True, exist_ok=True)
        effective.write_text(yaml.safe_dump(settings), encoding="utf-8")
        command = [*self.prefix, *(token.format(port=self.port, config=effective, root=ROOT) for token in self.command)]
        log.info("server.start port=%d gpu=%d free=%dMiB held=%dMiB utilization=%s cmd=%s",
                 self.port, self.gpu, free, held, settings["gpu-memory-utilization"], " ".join(command))
        logfile = self.log_path.open("ab")
        self.process = subprocess.Popen(
            command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=_die_with_parent,
            env=server_environment())
        self._wait_healthy()

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + HEALTH_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"model server exited with {self.process.returncode}; see {self.log_path}")
            if server_answers(self.url):
                log.info("server.ready after %.0fs", HEALTH_TIMEOUT_S - (deadline - time.monotonic()))
                return
            time.sleep(3)
        self.stop()
        raise RuntimeError(f"model server not healthy after {HEALTH_TIMEOUT_S}s; see {self.log_path}")

    def healthy(self) -> bool:
        return self.process is not None and self.process.poll() is None and server_answers(self.url)

    def stop(self, grace_s: float = 30.0) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def kill(self) -> None:
        """Hard stop, used by the hang watchdog: pending requests then fail at once."""
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class ExternalServer:
    """A model server somebody else runs (``--server-url``): never started, stopped or killed here."""

    process = None
    effective_config: dict = {}

    def __init__(self, url: str):
        self.url = url

    def start(self) -> None:
        pass

    def healthy(self) -> bool:
        return server_answers(self.url)

    def stop(self, grace_s: float = 30.0) -> None:
        pass

    def kill(self) -> None:
        pass


class Peaks:
    """Peak GPU memory used and peak resident memory of this worker and its server, sampled."""

    def __init__(self, gpu: int, server_pid):
        self.gpu, self.server_pid = gpu, server_pid
        self.vram_mib = self.worker_rss_mib = self.server_rss_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="peak-sampler")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        return {"vram_mib": self.vram_mib, "worker_rss_mib": self.worker_rss_mib, "server_rss_mib": self.server_rss_mib}

    def _run(self) -> None:
        import psutil

        me = psutil.Process()
        while not self._stop.wait(2.0):
            try:
                used = subprocess.check_output(
                    ["nvidia-smi", "-i", str(self.gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True)
                self.vram_mib = max(self.vram_mib, int(used.strip()))
                self.worker_rss_mib = max(self.worker_rss_mib, me.memory_info().rss // 2**20)
                pid = self.server_pid()
                if pid:
                    server = psutil.Process(pid)
                    rss = server.memory_info().rss + sum(c.memory_info().rss for c in server.children(recursive=True))
                    self.server_rss_mib = max(self.server_rss_mib, rss // 2**20)
            except Exception:  # a sampler must never take the worker down
                continue


class Watchdog:
    """Kills the model server when no page has come back for ``stall_s`` seconds.

    A hung GPU kernel (PaddleOCR issue 17693) leaves requests waiting for the library's
    600 s timeout; killing the server makes them fail at once so the batch can restart."""

    def __init__(self, server: VllmServer, stall_s: float):
        self.server, self.stall_s = server, stall_s
        self.last = time.monotonic()
        self.fired = threading.Event()
        self._busy = threading.Event()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="stall-watchdog").start()

    def beat(self) -> None:
        self.last = time.monotonic()

    def watching(self, busy: bool) -> None:
        self.last = time.monotonic()
        if busy:
            self._busy.set()
        else:
            self._busy.clear()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(5.0):
            if self._busy.is_set() and time.monotonic() - self.last > self.stall_s:
                log.error("watchdog: no page for %.0fs, killing the model server", self.stall_s)
                self.fired.set()
                self.server.kill()
                self.last = time.monotonic()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--server-config", required=True, type=Path, help="vLLM YAML (deploy/compute88/pvl_vllm.yaml)")
    parser.add_argument("--server-command", default=None, help="command line of the PaddleOCR-VL vLLM server, with {port} and {config} "
                        "(the server's own environment's vllm binary); required unless --server-url is given")
    parser.add_argument("--fallback-command", default=None, help="command line of the MinerU vLLM server that re-reads the pages "
                        "PaddleOCR-VL fails twice, same placeholders; without it a failed page fails its paper")
    parser.add_argument("--fallback-config", type=Path, default=None, help="vLLM YAML of the fallback server")
    parser.add_argument("--fallback-model", default=None, help="model name the fallback server answers to")
    parser.add_argument("--fallback-reserve-mib", type=int, default=FALLBACK_RESERVE_MIB,
                        help="GPU memory the pipeline process keeps while the fallback server runs")
    parser.add_argument("--server-prefix", default="", help="command prefix for the server, e.g. 'systemd-run --user --scope -p MemoryHigh=4G --'")
    parser.add_argument("--port", type=int, default=0, help="0 = a free port")
    parser.add_argument("--server-url", default=None, help="use this OpenAI-compatible server (http://host:port/v1) instead of starting one")
    parser.add_argument("--max-concurrency", type=int, default=16, help="concurrent VLM requests of the pipeline")
    parser.add_argument("--batch-pdfs", type=int, default=4, help="PDFs streamed through the pipeline together")
    parser.add_argument("--stall-seconds", type=float, default=300.0, help="no page this long = the server hung")
    parser.add_argument("--defer-flush-papers", type=int, default=DEFER_FLUSH_PAPERS,
                        help="papers waiting for the formula reader or the second model that make the worker hand the card "
                        "over to them, write them and bring the PaddleOCR-VL server back (a crash loses at most this many)")
    parser.add_argument("--no-ocr", action="store_true", help="do not use PP-OCRv5 as the reference of pages without a text layer")
    parser.add_argument("--no-formula-witness", action="store_true", help="do not ask PP-FormulaNet to read a display formula "
                        "whose symbols the text layer contradicts")
    parser.add_argument("--formula-device", default="gpu:0", help="where PP-FormulaNet runs, once the PaddleOCR-VL server is stopped "
                        "(the card has no room for it next to the server; on the host's cores a reading takes ~20 s)")
    parser.add_argument("--pipeline-config", default=None, help="JSON (or a JSON file) of PaddleX pipeline overrides, e.g. "
                        '{"layout_prep_cpu_workers": 3, "SubModules": {"LayoutDetection": {"batch_size": 16}}}')
    parser.add_argument("--dump-pages", type=Path, default=None, help="keep the model's raw page results here (to calibrate the checks offline)")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    # Before any paddle / torch import: the physical card, and where models come from.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
    # Paddle's own memory policy, stated rather than inherited: grow on demand (never reserve a fraction of
    # the card up front: the vLLM server and the backend share it), and one OpenMP thread, which is what
    # Paddle asks for ("OMP_NUM_THREADS set to 2, not 1") whatever the parent's environment says.
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    os.environ["OMP_NUM_THREADS"] = "1"
    local = "127.0.0.1,localhost"
    os.environ["NO_PROXY"] = f"{os.environ['NO_PROXY']},{local}" if os.environ.get("NO_PROXY") else local

    logging.basicConfig(level=args.log_level, format=f"%(asctime)s %(levelname)s rank{args.rank} %(name)s %(message)s")
    # one INFO line per VLM request (httpx), a DEBUG "Layout raw output" per page (mineru_vl_utils) and a "Retrying
    # request" per retry (openai) made task 1302 4,000 events, in a task view people read
    for name in ("httpx", "httpx2", "httpcore", "openai", "mineru_vl_utils"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if not args.input.exists():
        log.error("input dir not found: %s", args.input)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    shard = _shard(_gather_pdfs(args.input, args.recursive), args.rank, args.world_size)
    if args.limit:
        shard = shard[: args.limit]
    done_path = args.output / f"done.rank{args.rank}.txt"
    done = _load_done(done_path)
    todo = [pdf for pdf in shard if str(pdf) not in done and not _mirror_output(pdf, args.input, args.output).exists()]
    skipped = len(shard) - len(todo)
    log.info("worker.start input=%s output=%s rank=%d/%d shard=%d todo=%d skipped=%d",
             args.input, args.output, args.rank, args.world_size, len(shard), len(todo), skipped)
    if not todo:
        log.info("worker.done ok=0 fail=0 skipped=%d", skipped)
        return 0

    gpu = args.gpu if args.gpu is not None else int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    quality = args.output / "_quality"
    if not args.server_url and not args.server_command:
        parser.error("--server-command (or --server-url) is required")
    server = ExternalServer(args.server_url) if args.server_url else VllmServer(
        gpu=gpu, config=args.server_config, prefix=shlex.split(args.server_prefix), command=shlex.split(args.server_command),
        port=args.port, log_path=quality / f"_server.rank{args.rank}.log")
    peaks = Peaks(gpu, lambda: server.process.pid if server.process else None)
    watchdog: Optional[Watchdog] = None
    ok = fail = pages = 0
    t_global = t_work = perf_counter()

    live = [server]  # every server this worker has started

    def release(*_):  # SIGTERM from the stage's cancel: take the servers down with us
        for each in live:
            each.stop(grace_s=10.0)
        os._exit(143)

    signal.signal(signal.SIGTERM, release)
    try:
        server.start()
        peaks.start()
        watchdog = Watchdog(server, args.stall_seconds)
        extractor = _build_extractor(args, server, watchdog)
        # a refused paper's pages are kept, and the next run judges them again instead of converting the paper again
        extractor.keep_pages = lambda pdf, pages: _write_kept_pages(_kept_pages_path(pdf, args.input, args.output), pages)
        t_work = perf_counter()  # the rate is the model's, not the server's start-up
        with done_path.open("a", encoding="utf-8") as done_fh:
            def write(pdf, result) -> None:
                nonlocal ok, fail
                report = result.extra.get("report", {})
                _write_report(args.output, pdf, args.input, report, result.error)
                if result.succeeded and result.markdown:
                    md_path = _mirror_output(pdf, args.input, args.output)
                    md_path.parent.mkdir(parents=True, exist_ok=True)
                    part = md_path.with_name(md_path.name + ".part")
                    part.write_text(result.markdown, encoding="utf-8")
                    os.replace(part, md_path)
                    done_fh.write(str(pdf) + chr(10))
                    done_fh.flush()
                    _kept_pages_path(pdf, args.input, args.output).unlink(missing_ok=True)
                    ok += 1
                else:
                    fail += 1
                    log.warning("extract.fail pdf=%s err=%s", pdf.name, (result.error or "")[:300])

            held = []  # batches with papers waiting for the formula reader or the second model
            deferred = bool(args.fallback_command) or not args.no_formula_witness
            strikes = 0
            for start in range(0, len(todo), args.batch_pdfs):
                batch = todo[start:start + args.batch_pdfs]
                kept = _load_kept_pages(batch, args.input, args.output)
                converted, revived = _convert(batch, extractor, server, watchdog, kept)
                release_gpu_cache()
                # pages the model converted, whatever the verdict and whether or not the paper is written yet (a
                # paper judged from its kept pages was not converted)
                pages += sum(max(converted.expected.get(pdf, 0), len(converted.pages.get(pdf, {})))
                             for pdf in converted.runnable if pdf not in kept)
                waiting = {pdf for pdf, _ in converted.unresolved} if deferred else set()
                for pdf, result in extractor.assemble(converted, skip=waiting).items():
                    write(pdf, result)
                if waiting:
                    held.append((converted, waiting))
                finished = start + len(batch)
                if finished % args.progress_every < len(batch) or finished == len(todo):
                    elapsed = max(perf_counter() - t_work, 1e-6)
                    log.info("progress %d/%d ok=%d skip=%d fail=%d held=%d pages=%d rate=%.2f pages/s",
                             finished, len(todo), ok, skipped, fail, sum(len(w) for _, w in held), pages, pages / elapsed)
                strikes = 0 if revived else strikes + 1
                if strikes >= RESTART_STRIKES:
                    log.error("giving up: the model server could not be brought back for %d batches in a row; "
                              "%d paper(s) were not tried", strikes, len(todo) - finished)
                    break
                if sum(len(w) for _, w in held) >= args.defer_flush_papers and finished < len(todo):
                    resumed = _deferred_phase(args, extractor, server, held, gpu, quality, write, live, resume=True)
                    held = []
                    if not resumed:
                        log.error("giving up: the PaddleOCR-VL server did not come back after the second model; "
                                  "%d paper(s) were not tried", len(todo) - finished)
                        break
            if held:
                _deferred_phase(args, extractor, server, held, gpu, quality, write, live, resume=False)
    finally:
        # no `return` in here: it would swallow the exception that is on its way out (task 1302 exited 0 that way)
        if watchdog is not None:
            watchdog.stop()
        summary = peaks.stop()
        for each in live:
            each.stop()
        elapsed = perf_counter() - t_global
        unwritten = len(todo) - ok - fail  # papers that got neither a Markdown nor a failure report in this run
        summary.update(ok=ok, fail=fail, unwritten=unwritten, skipped=skipped, pages=pages, seconds=round(elapsed, 1),
                       pages_per_second=round(pages / max(perf_counter() - t_work, 1e-6), 3),
                       server_config=server.effective_config)
        (quality).mkdir(parents=True, exist_ok=True)
        (quality / f"_run.rank{args.rank}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        log.info("worker.done ok=%d fail=%d unwritten=%d skipped=%d elapsed=%.1fs rate=%.2f pages/s peaks=%s",
                 ok, fail, unwritten, skipped, elapsed, summary["pages_per_second"],
                 {k: summary[k] for k in ("vram_mib", "worker_rss_mib", "server_rss_mib")})
    return 0 if fail == 0 and unwritten == 0 else 1


def _load_pipeline_config(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    return json.loads(Path(raw).read_text(encoding="utf-8") if Path(raw).is_file() else raw)


def _build_extractor(args, server: VllmServer, watchdog: Watchdog):
    from pipeline.paper_crawler.converters.extractors import create_extractor

    ocr = None
    if not args.no_ocr:
        from pipeline.paper_crawler.converters.verify.ocr import OcrReference

        ocr = OcrReference()
    return create_extractor(
        "paddleocr_vl", server_url=server.url, max_concurrency=args.max_concurrency, ocr=ocr,
        on_progress=watchdog.beat, page_dump=args.dump_pages, pipeline_config=_load_pipeline_config(args.pipeline_config),
        engine={"model": MODEL_NAME, "backend": "vllm", "server": server.effective_config,
                "vllm_attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", "")},
    )


def _revive(extractor, server) -> None:
    """Make another try possible after the pipeline broke a batch. The pipeline is rebuilt first and
    Paddle's cache emptied (the server sizes itself from what the card has free, and a pipeline that
    died on an out-of-memory error still holds the memory it died with); the model server is
    restarted only when it no longer answers."""
    extractor.release_pipeline()
    release_gpu_cache()
    if not server.healthy():
        server.stop()
        server.start()
    extractor.server_url = server.url


def _retryable(converted) -> list:
    return [pdf for pdf, r in converted.results.items() if not r.succeeded and r.extra.get("retryable")]


def _kept_pages_path(pdf: Path, root: Path, output: Path) -> Path:
    return _mirror_output(pdf, root, output).parent / "_pages" / f"{pdf.stem}.json.gz"


def _write_kept_pages(path: Path, pages: dict) -> None:
    """Atomically, gzipped (a page's result is a few kB of JSON)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with gzip.open(part, "wt", encoding="utf-8") as fh:
        json.dump(pages, fh, ensure_ascii=False)
    os.replace(part, path)


def _load_kept_pages(pdfs, root: Path, output: Path) -> dict:
    """{pdf: kept pages} of the papers of ``pdfs`` refused before with their pages kept. A file that cannot be read
    is left out: that paper is converted as if nothing had been kept."""
    kept = {}
    for pdf in pdfs:
        path = _kept_pages_path(pdf, root, output)
        if not path.exists():
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                kept[pdf] = json.load(fh)
        except (OSError, ValueError, EOFError):
            log.warning("kept_pages.unreadable pdf=%s path=%s: converting it again", pdf.name, path, exc_info=True)
    return kept


def _convert(pdfs, extractor, server: VllmServer, watchdog: Watchdog, kept: Optional[dict] = None):
    """One batch through the pipeline (``extractor.convert``); papers it broke on get one more try
    once the pipeline (and the server, if it is gone) is back. Returns the batch and whether it came
    through with a model server that answers: a revival that raises leaves the broken papers failed
    (they are not checkpointed, the next run offers them again) and the worker goes on. A paper in ``kept``
    is judged from its kept pages instead of being converted (``PaddleOCRVLExtractor.convert``)."""
    kept = kept or {}
    watchdog.watching(True)
    try:
        converted = extractor.convert(pdfs, kept)
    finally:
        watchdog.watching(False)
    broken = _retryable(converted)
    if not broken:
        return converted, True
    log.warning("batch: %d paper(s) broken by a server or pipeline failure, rebuilding the pipeline", len(broken))
    try:
        _revive(extractor, server)
    except Exception:
        log.exception("batch: the server could not be brought back; %d paper(s) stay failed", len(broken))
        return converted, False
    watchdog.watching(True)
    try:
        again = extractor.convert(broken, {pdf: kept[pdf] for pdf in broken if pdf in kept})
    finally:
        watchdog.watching(False)
    converted.absorb(again, broken)
    return converted, server.healthy()  # a paper that breaks the pipeline twice is that paper's problem, not the server's


def _failed_pages(held) -> int:
    return sum(len(batch.reports[pdf].failed_pages) for batch, waiting in held for pdf in waiting)


def _deferred_phase(args, extractor, pvl_server, held, gpu: int, quality: Path, write, live: list, *,
                    resume: bool) -> bool:
    """What waits for the card, once the PaddleOCR-VL server is stopped (it leaves no room for either):
    PP-FormulaNet reads the display formulas the text layer contradicts on the pages that failed, then
    the pages still failing go to the second model (MinerU) and through the same checks; then the held
    papers are assembled and written (or refused). With ``resume`` the PaddleOCR-VL server is brought
    back afterwards for the papers still to come; False when it could not be.

    A phase that fails leaves its pages failing, which refuses their papers with a report: it never
    takes the held papers with it."""
    log.info("deferred.start %d paper(s), %d failed page(s): stopping the PaddleOCR-VL server",
             sum(len(w) for _, w in held), _failed_pages(held))
    pvl_server.stop()
    release_gpu_cache()  # what the layout model cached while the server was up: the next models need it
    if not args.no_formula_witness:
        try:
            _formula_witness(args, extractor, held)
        except Exception:
            log.exception("deferred: the formula reader failed; its pages stay failed")
    if args.fallback_command and _failed_pages(held):
        try:
            _fallback_phase(args, extractor, held, gpu, quality, live)
        except Exception:
            log.exception("deferred: the second model failed; its pages stay failed")
    for batch, waiting in held:
        for pdf, result in extractor.assemble(batch, only=waiting).items():
            write(pdf, result)
    if not resume:
        return True
    try:
        extractor.release_pipeline()
        release_gpu_cache()
        pvl_server.start()
        extractor.server_url = pvl_server.url
    except Exception:
        log.exception("deferred: the PaddleOCR-VL server could not be restarted")
        return False
    return True


def _formula_witness(args, extractor, held) -> None:
    from pipeline.paper_crawler.converters.verify.formulanet import FormulaReader

    before = _failed_pages(held)
    extractor.formula_reader = reader = FormulaReader(device=args.formula_device)
    try:
        extractor.recheck_failed_pages([batch for batch, _ in held])
    finally:
        extractor.formula_reader = None
        reader.release()
    log.info("formula_witness.done failed pages %d -> %d", before, _failed_pages(held))


def _fallback_phase(args, extractor, held, gpu: int, quality: Path, live: list) -> None:
    """The pages PaddleOCR-VL failed twice, of every held paper, go to the second model (MinerU)."""
    from pipeline.paper_crawler.converters.extractors.mineru_pages import MineruReader

    log.info("fallback.start %d page(s)", _failed_pages(held))
    server = VllmServer(gpu=gpu, config=args.fallback_config, prefix=shlex.split(args.server_prefix),
                        command=shlex.split(args.fallback_command), port=0, log_path=quality / f"_fallback.rank{args.rank}.log",
                        reserve_mib=args.fallback_reserve_mib)
    live.append(server)
    try:
        server.start()
        watchdog = Watchdog(server, args.stall_seconds)
        reader = MineruReader(server.url, model_name=args.fallback_model, max_concurrency=args.max_concurrency,
                              on_progress=watchdog.beat)
        watchdog.watching(True)
        try:
            extractor.redo_failed_pages([batch for batch, _ in held], reader)
        finally:
            watchdog.watching(False)
            watchdog.stop()
    finally:
        server.stop()  # a restarted PaddleOCR-VL server needs the card back, and a half-started one holds it
        live.remove(server)


def _write_report(output: Path, pdf: Path, root: Path, report: dict, error: Optional[str]) -> None:
    if not report:  # refused before any page ran (unreadable PDF)
        report = {"schema": 1, "paper": pdf.stem, "pdf": str(pdf), "status": "failed",
                  "failure_reasons": [error or "no result"], "pages": []}
    target = _mirror_output(pdf, root, output).parent / "_quality" / f"{pdf.stem}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(part, target)


if __name__ == "__main__":
    sys.exit(main())
