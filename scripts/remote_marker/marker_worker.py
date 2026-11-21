"""Single-GPU sharded marker batch worker.

Same shape as ``scripts/remote_mineru/mineru_worker.py`` but wraps
:class:`MarkerExtractor` instead of the MinerU CLI. Marker is much faster
on academic born-digital PDFs (~5-10s/PDF on a 3090) but uses the same
modulo sharding so the multi-GPU launcher (``run_marker_kss.sh``) can
divide the workload across N ranks identically.

Per-rank append-only checkpoint at ``<output>/done.rank{N}.txt`` survives
restart and skips finished PDFs cheaply.
"""
from __future__ import annotations

import argparse
import inspect
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

# Allow execution as `python scripts/remote_marker/marker_worker.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_marker_config(*, mode: str, disable_ocr: bool) -> dict:
    """Build the explicit Marker 2 conversion contract for this worker.

    Marker 2 chooses a device-dependent mode when ``mode`` is omitted, while
    ``disable_ocr`` now disables every VLM call, including equation recovery.
    Keep paper conversion reproducible by always selecting the mode and only
    enabling the lossy no-VLM path when the operator asks for it explicitly.
    """
    if mode not in {"balanced", "fast"}:
        raise ValueError("mode must be 'balanced' or 'fast'")

    config = {"mode": mode}
    if disable_ocr:
        config["disable_ocr"] = True
    return config


def install_windows_surya_process_cleanup_guard() -> bool:
    """Replace Surya's unsafe Windows process probe with ``psutil``.

    Surya 0.22.1 terminates its local inference server and then polls it with
    ``os.kill(pid, 0)``.  Signal zero is a harmless existence probe on POSIX,
    but on Windows zero is ``CTRL_C_EVENT``.  The resulting console interrupt
    can escape the llama-server process group and terminate the Codex host
    running this worker.  ``psutil`` uses native Windows process handles and
    also guards its destructive methods against PID reuse.

    Return ``True`` only when the affected upstream implementation was
    detected and replaced.  A future Surya release that removes the unsafe
    probe is left untouched.
    """
    if os.name != "nt":
        return False

    try:
        from surya.inference.backends import spawn as surya_spawn
    except ModuleNotFoundError:
        # Marker 1 / Surya 0.17 has no local llama-server backend.  The same
        # worker remains the rollback entrypoint, so absence means unaffected,
        # not a startup failure.
        return False

    import psutil

    current = surya_spawn._stop_process
    if getattr(current, "_retrival_windows_safe_cleanup", False):
        return True

    try:
        current_source = inspect.getsource(current)
    except (OSError, TypeError):
        current_source = ""
    if "os.kill(pid, 0)" not in current_source:
        return False

    def _stop_process_with_windows_handle(pid: int, name: str) -> None:
        try:
            process = psutil.Process(pid)
            process.terminate()
            try:
                process.wait(timeout=10)
                surya_spawn.logger.info(f"Stopped {name} (pid {pid})")
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                surya_spawn.logger.warning(f"Force-killed {name} (pid {pid})")
        except psutil.NoSuchProcess:
            return
        except (psutil.AccessDenied, OSError) as exc:
            surya_spawn.logger.warning(f"Failed to stop {name} (pid {pid}): {exc}")

    setattr(
        _stop_process_with_windows_handle,
        "_retrival_windows_safe_cleanup",
        True,
    )
    surya_spawn._stop_process = _stop_process_with_windows_handle
    return True


def _gather_pdfs(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(p for p in root.rglob("*.pdf") if p.is_file())
    return sorted(p for p in root.glob("*.pdf") if p.is_file())


def _shard(items: list[Path], rank: int, world_size: int) -> list[Path]:
    if world_size <= 1:
        return items
    return [p for i, p in enumerate(items) if i % world_size == rank]


def _mirror_output(pdf: Path, input_root: Path, output_root: Path) -> Path:
    """Mirror the input dir layout under output_root, swapping suffix to .md."""
    try:
        rel = pdf.relative_to(input_root)
    except ValueError:
        rel = Path(pdf.name)
    return (output_root / rel).with_suffix(".md")


def _load_done(done_path: Path) -> set[str]:
    if not done_path.exists():
        return set()
    return {line.strip() for line in done_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _retired() -> None:
    """RETIRED: Marker loads torch / surya models into this process; PDF->Markdown is the registered PaddleOCR-VL worker now."""
    from core.inference_retired import EXTRACTOR_INSTEAD, RetiredInferencePath

    raise RetiredInferencePath("marker_worker.py (in-process torch / surya Marker)", EXTRACTOR_INSTEAD)


def main() -> int:
    _retired()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--gpu", type=int, default=None,
        help=(
            "Physical GPU index to bind this worker to. Sets "
            "CUDA_VISIBLE_DEVICES in-process BEFORE torch/marker import so "
            "any spawned subprocess / model loader inherits it. Required when "
            "running on a shared box where shell-inline "
            "`CUDA_VISIBLE_DEVICES=N python ...` is not respected by every "
            "downstream lib (surya, accelerate)."
        ),
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("balanced", "fast"),
        default="balanced",
        help=(
            "Marker 2 conversion mode. Paper conversion defaults to balanced "
            "so inline math and low-confidence regions retain VLM recovery."
        ),
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help=(
            "Explicitly disable every Marker 2 VLM call. This skips equation "
            "and scanned-page recovery and is not the paper-corpus default."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument(
        "--log-level", default=os.environ.get("LOG_LEVEL", "INFO"),
    )
    args = parser.parse_args()

    # Lock CUDA visibility BEFORE any torch / marker / surya import. Doing it
    # via os.environ here (rather than relying on `CUDA_VISIBLE_DEVICES=N
    # python` from the launcher shell) survives library code that re-reads
    # env from spawned subprocesses or that ignores CVD when it's set after
    # cuda init. After this line a process-wide cuda:0 == physical args.gpu.
    #
    # CUDA_DEVICE_ORDER=PCI_BUS_ID is MANDATORY for the mapping to hold: without
    # it CUDA enumerates GPUs FASTEST_FIRST, so on a mixed-arch box (the 4090 alt
    # host: 2× sm_120 5090 + 2× sm_89 4090) CVD=2 does NOT select physical GPU 2 —
    # marker landed on physical GPU 0 (another user's training card) despite CVD=2
    # (witnessed 2026-05-31). PCI_BUS_ID forces CUDA index == nvidia-smi index.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logging.basicConfig(
        level=args.log_level,
        format=f"%(asctime)s %(levelname)s rank{args.rank} %(name)s %(message)s",
    )
    log = logging.getLogger("marker_worker")
    log.info(
        "worker.cuda_visible_devices=%s (forced via --gpu=%s)",
        os.environ.get("CUDA_VISIBLE_DEVICES"), args.gpu,
    )

    if install_windows_surya_process_cleanup_guard():
        log.info("worker.surya_windows_cleanup_guard=installed")

    if not args.input.exists():
        log.error("input dir not found: %s", args.input)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    from pipeline.paper_crawler.converters.extractors import MarkerExtractor

    marker_config = build_marker_config(
        mode=args.mode,
        disable_ocr=args.disable_ocr,
    )
    extractor = MarkerExtractor(extra_config=marker_config)

    all_pdfs = _gather_pdfs(args.input, args.recursive)
    shard = _shard(all_pdfs, args.rank, args.world_size)
    if args.limit:
        shard = shard[: args.limit]

    log.info(
        "worker.start input=%s output=%s rank=%d/%d total=%d shard=%d recursive=%s "
        "mode=%s disable_ocr=%s",
        args.input, args.output, args.rank, args.world_size,
        len(all_pdfs), len(shard), args.recursive, args.mode, args.disable_ocr,
    )

    done_path = args.output / f"done.rank{args.rank}.txt"
    done = _load_done(done_path)
    log.info("worker.resume done_prev=%d", len(done))

    ok = fail = skipped = 0
    t_global = perf_counter()

    with done_path.open("a", encoding="utf-8") as done_fh:
        for i, pdf in enumerate(shard, start=1):
            md_path = _mirror_output(pdf, args.input, args.output)

            key = str(pdf)
            if key in done or md_path.exists():
                skipped += 1
                if key not in done:
                    done_fh.write(key + "\n")
                    done_fh.flush()
                    done.add(key)
                continue

            md_path.parent.mkdir(parents=True, exist_ok=True)
            t0 = perf_counter()
            result = extractor.extract(pdf)
            wall = perf_counter() - t0

            if result.succeeded and result.markdown:
                md_path.write_text(result.markdown, encoding="utf-8")
                done_fh.write(key + "\n")
                done_fh.flush()
                done.add(key)
                ok += 1
                if i % args.progress_every == 0 or i == len(shard):
                    rate = (ok + skipped) / max(perf_counter() - t_global, 1e-6)
                    log.info(
                        "progress %d/%d ok=%d skip=%d fail=%d wall=%.1fs rate=%.2f pdf/s last=%s",
                        i, len(shard), ok, skipped, fail, wall, rate, pdf.name,
                    )
            else:
                fail += 1
                log.warning(
                    "extract.fail pdf=%s strategy=%s err=%s wall=%.1fs",
                    pdf.name, result.strategy, result.error, wall,
                )

    elapsed = perf_counter() - t_global
    log.info(
        "worker.done ok=%d fail=%d skipped=%d elapsed=%.1fs rate=%.2f pdf/s",
        ok, fail, skipped, elapsed, (ok + skipped) / max(elapsed, 1e-6),
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
