"""MinerU PDF→Markdown worker (single GPU, sharded by rank).

Designed to run on kss inside a venv with MinerU installed. Each
launched process owns exactly one GPU (via ``CUDA_VISIBLE_DEVICES``
exported by the launcher) and a slice of the input PDF list (modulo
``rank``/``world_size``).

The worker shells out to the ``mineru`` CLI per-PDF instead of importing
the library directly — MinerU's Python API has shifted between releases,
but the CLI flags ``-p <pdf> -o <dir>`` have been stable since 1.0. This
keeps the worker resilient to upstream churn.

Per [[feedback-library-first]]: we delegate parsing to mineru, retry
shape-handling to subprocess, progress to tqdm, atomic done.txt to stdlib
``open(a)``. No wheel-reinvention.

Per [[feedback-observability]]:
  - per-PDF log line on start (rank, pdf_name, idx/total)
  - per-PDF log line on finish (status, elapsed_ms, md_size or err)
  - done.txt append-only checkpoint so reruns skip completed PDFs
  - ETA printed every 10 PDFs

Launch:
    python mineru_worker.py \\
        --input  /mnt/lsb/retrival/papers/pdf/iclr  \\
        --output /mnt/lsb/retrival/papers/markdown/iclr \\
        --rank 0 --world-size 8 --recursive
"""
from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    from tqdm import tqdm
except ImportError:  # tqdm is small; allow worker to run without it
    def tqdm(iterable, *_, **__):
        return iterable

_logger = logging.getLogger("mineru_worker")


@dataclass
class MinerUConfig:
    cmd: str = "mineru"  # override via env MINERU_CMD if your install names it differently
    extra_args: list[str] = None  # passed verbatim to the mineru CLI

    @classmethod
    def from_env(cls) -> "MinerUConfig":
        return cls(
            cmd=os.environ.get("MINERU_CMD", "mineru"),
            extra_args=shlex.split(os.environ.get("MINERU_EXTRA_ARGS", "")),
        )


def _gather_pdfs(root: Path, recursive: bool) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"input root not found: {root}")
    if not root.is_dir():
        return [root] if root.suffix.lower() == ".pdf" else []
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(root.glob(pattern))


def _shard(items: list[Path], rank: int, world_size: int) -> list[Path]:
    return [item for i, item in enumerate(items) if i % world_size == rank]


def _expected_md_dir(pdf: Path, input_root: Path, output_root: Path) -> Path:
    """Per-PDF output root under ``output_root``. Mineru pipeline backend
    writes to ``<root>/<stem>/auto/<stem>.md``; vlm/hybrid backends use a
    different subfolder, so we hand the whole ``<root>/<stem>`` to mineru
    and discover the .md file by rglob after the subprocess finishes."""
    rel = pdf.relative_to(input_root) if pdf.is_relative_to(input_root) else Path(pdf.name)
    stem = rel.with_suffix("")
    return output_root / stem


def _find_md(stem_dir: Path, pdf_stem: str) -> Optional[Path]:
    """Pick the markdown for ``pdf_stem`` under ``stem_dir`` (recursive)."""
    if not stem_dir.exists():
        return None
    # Prefer the one matching the pdf stem; fall back to any .md
    candidates = sorted(stem_dir.rglob("*.md"))
    for md in candidates:
        if md.stem == pdf_stem:
            return md
    return candidates[0] if candidates else None


def _load_done(done_file: Path) -> set[str]:
    if not done_file.exists():
        return set()
    return {line.strip() for line in done_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def _mark_done(done_file: Path, key: str) -> None:
    done_file.parent.mkdir(parents=True, exist_ok=True)
    with done_file.open("a", encoding="utf-8") as fh:
        fh.write(key + "\n")


def _run_mineru(pdf: Path, out_dir: Path, cfg: MinerUConfig, timeout: int) -> tuple[bool, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [cfg.cmd, "-p", str(pdf), "-o", str(out_dir)]
    if cfg.extra_args:
        argv.extend(cfg.extra_args)
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    if proc.returncode != 0:
        tail = (proc.stdout.decode("utf-8", "replace") if proc.stdout else "")[-500:]
        return False, f"exit={proc.returncode}; tail={tail!r}"
    return True, ""


def _process_one(
    pdf: Path,
    idx: int,
    total: int,
    rank: int,
    input_root: Path,
    output_root: Path,
    done: set[str],
    done_file: Path,
    cfg: MinerUConfig,
    timeout: int,
) -> str:
    key = str(pdf.relative_to(input_root)) if pdf.is_relative_to(input_root) else pdf.name
    if key in done:
        return "skipped"
    stem_dir = _expected_md_dir(pdf, input_root, output_root)
    if _find_md(stem_dir, pdf.stem) is not None:
        _mark_done(done_file, key)
        return "skipped-exists"

    stem_dir.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    ok, err = _run_mineru(pdf, stem_dir.parent, cfg, timeout)
    elapsed_ms = (time.monotonic() - t0) * 1000

    if not ok:
        _logger.error(
            "mineru.fail rank=%d idx=%d/%d pdf=%s elapsed_ms=%.0f err=%s",
            rank, idx + 1, total, pdf.name, elapsed_ms, err,
        )
        return "failed"

    md_path = _find_md(stem_dir, pdf.stem)
    if md_path is None:
        _logger.error(
            "mineru.no_md rank=%d idx=%d/%d pdf=%s elapsed_ms=%.0f stem_dir=%s",
            rank, idx + 1, total, pdf.name, elapsed_ms, stem_dir,
        )
        return "failed"
    md_size = md_path.stat().st_size
    _logger.info(
        "mineru.ok rank=%d idx=%d/%d pdf=%s elapsed_ms=%.0f md_bytes=%d md=%s",
        rank, idx + 1, total, pdf.name, elapsed_ms, md_size, md_path.name,
    )
    _mark_done(done_file, key)
    return "ok"


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    cfg = MinerUConfig.from_env()
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    pdfs = _gather_pdfs(input_root, recursive=args.recursive)
    my_pdfs = _shard(pdfs, rank=args.rank, world_size=args.world_size)
    total = len(my_pdfs)
    _logger.info(
        "mineru.start rank=%d/%d input=%s output=%s total_pdfs=%d my_shard=%d cmd=%s",
        args.rank, args.world_size, input_root, output_root, len(pdfs), total, cfg.cmd,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    done_file = output_root / f"done.rank{args.rank}.txt"
    done = _load_done(done_file)
    _logger.info("mineru.checkpoint rank=%d done_loaded=%d", args.rank, len(done))

    counts = {"ok": 0, "failed": 0, "skipped": 0, "skipped-exists": 0}
    t_start = time.monotonic()

    for idx, pdf in enumerate(tqdm(my_pdfs, desc=f"rank{args.rank}")):
        status = _process_one(
            pdf=pdf,
            idx=idx,
            total=total,
            rank=args.rank,
            input_root=input_root,
            output_root=output_root,
            done=done,
            done_file=done_file,
            cfg=cfg,
            timeout=args.timeout,
        )
        counts[status] = counts.get(status, 0) + 1

        if (idx + 1) % args.progress_every == 0 or (idx + 1) == total:
            elapsed = time.monotonic() - t_start
            rate = (idx + 1) / max(elapsed, 1e-3)
            eta_min = (total - idx - 1) / max(rate, 1e-3) / 60
            _logger.info(
                "mineru.progress rank=%d done=%d/%d ok=%d failed=%d skipped=%d "
                "rate=%.2f pdf/s eta=%.1fmin",
                args.rank, idx + 1, total,
                counts["ok"], counts["failed"],
                counts["skipped"] + counts["skipped-exists"],
                rate, eta_min,
            )

    elapsed = time.monotonic() - t_start
    _logger.info(
        "mineru.done rank=%d total=%d ok=%d failed=%d skipped=%d elapsed_s=%.1f",
        args.rank, total, counts["ok"], counts["failed"],
        counts["skipped"] + counts["skipped-exists"], elapsed,
    )
    return 0 if counts["failed"] == 0 else 1


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MinerU PDF→MD worker (single GPU, sharded)")
    p.add_argument("--input", required=True, help="root dir of PDFs (recurses if --recursive)")
    p.add_argument("--output", required=True, help="root dir for MD output (mirrors input layout)")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--no-recursive", dest="recursive", action="store_false")
    p.add_argument("--timeout", type=int, default=1800, help="per-PDF MinerU timeout in seconds")
    p.add_argument("--progress-every", type=int, default=10, help="log progress every N PDFs")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
