"""Thin execution helpers — subprocess + ssh, streamed into a stage sink.

These are deliberately dependency-light and minimally tested: the stage BUILDERS
(pure cmd/probe/gate functions in stages.py) are the TDD target. Each helper just
launches a process and forwards its stdout line-by-line into the sink so the
orchestrator's event stream stays live during a long-running stage.

A `popen_factory` is injectable so a test can assert run_local launches the built
command without spawning a real process.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

from loguru import logger

EventSink = Callable[..., None]
# (args, cwd, env) -> process with .stdout (iterable of str lines) and .wait()
PopenFactory = Callable[..., subprocess.Popen]


def _default_popen(args: Sequence[str], cwd: Optional[str], env: Optional[dict]) -> subprocess.Popen:
    return subprocess.Popen(
        list(args),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )


def _has_gpu_memory_rows(output: str) -> bool:
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and all(part.isdigit() for part in parts[:3]):
            return True
    return False


def _probe_local_gpu_memory_with_torch() -> str:
    """Return nvidia-smi-compatible rows from the installed PyTorch runtime.

    Some Linux driver updates leave the installed NVML userspace library out of
    sync with the loaded kernel module. CUDA can remain fully operational in
    that state while ``nvidia-smi`` exits 18. Numeric
    ``CUDA_VISIBLE_DEVICES`` is required so logical devices can be mapped back
    to the physical allowlist; ambiguous UUID/MIG mappings fail closed.
    """
    try:
        import torch

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        count = torch.cuda.device_count()
        if visible:
            tokens = [token.strip() for token in visible.split(",")]
            if any(not token.isdigit() for token in tokens):
                logger.warning(
                    "runner.gpu_health.torch_fallback.ambiguous_visible_devices"
                )
                return ""
            physical_ids = [int(token) for token in tokens]
        else:
            physical_ids = list(range(count))
        rows = []
        for logical_id in range(min(count, len(physical_ids))):
            free_bytes, total_bytes = torch.cuda.mem_get_info(logical_id)
            rows.append(
                f"{physical_ids[logical_id]},"
                f"{total_bytes // 1048576},{free_bytes // 1048576}"
            )
        out = "\n".join(rows)
    except Exception as exc:
        logger.warning(f"runner.gpu_health.torch_fallback.error: {exc}")
        return ""
    if not _has_gpu_memory_rows(out):
        logger.warning(
            f"runner.gpu_health.torch_fallback.failed output={out.strip()[-500:]!r}"
        )
        return ""
    logger.warning("runner.gpu_health.nvml_unavailable using=torch_cuda_mem_get_info")
    return out


def run_local(
    cmd: list[str],
    cwd: Path,
    sink: EventSink,
    *,
    env: Optional[dict] = None,
    popen_factory: PopenFactory = _default_popen,
) -> int:
    """Run a local command, streaming stdout into sink. Raise on non-zero exit."""
    logger.info(f"runner.local.start cwd={cwd} cmd={' '.join(map(str, cmd))}")
    proc = popen_factory(cmd, str(cwd), env)
    if proc.stdout is not None:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                sink(line)
    code = proc.wait()
    logger.info(f"runner.local.exit code={code} cmd0={cmd[0] if cmd else '-'}")
    if code != 0:
        raise RuntimeError(f"command failed (exit {code}): {' '.join(map(str, cmd))}")
    return code


def run_remote(
    ssh_target: str,
    remote_cmd: str,
    sink: EventSink,
    *,
    env_prefix: str = "",
    popen_factory: PopenFactory = _default_popen,
) -> int:
    """Run a command on a remote host via ssh, streaming stdout into sink."""
    full = f"{env_prefix}{remote_cmd}"
    logger.info(f"runner.remote.start target={ssh_target} cmd={full}")
    return run_local(["ssh", ssh_target, full], Path("."), sink, popen_factory=popen_factory)


def probe_remote_count(
    ssh_target: str,
    remote_glob: str,
    sink: Optional[EventSink] = None,
    *,
    popen_factory: PopenFactory = _default_popen,
) -> int:
    """`ssh target "ls <glob> 2>/dev/null | wc -l"` → int. Fallback probe; prefer
    Arc-side SMB glob (ctx.md_count) when available. Returns 0 on any failure."""
    cmd = ["ssh", ssh_target, f"ls {remote_glob} 2>/dev/null | wc -l"]
    try:
        proc = popen_factory(cmd, None, None)
        out = proc.stdout.read() if proc.stdout is not None else ""
        proc.wait()
        for tok in out.split():
            if tok.strip().isdigit():
                return int(tok.strip())
        return 0
    except Exception as exc:  # best-effort remote probe — never crash the stage
        logger.warning(f"runner.remote.count.error target={ssh_target} glob={remote_glob}: {exc}")
        if sink is not None:
            sink(f"remote count failed: {exc}", level="warning")
        return 0


def gpu_health(
    ssh_target: str, *,
    min_free_mb: int = 0,
    min_free_ratio: float = 0.0,
    popen_factory: PopenFactory = _default_popen,
) -> list[int]:
    """Return USABLE GPU indices on a remote host via nvidia-smi. [] if ssh fails.

    Two safety filters (both default OFF; either may be set):

    * `min_free_mb` — absolute floor on memory.free (MiB). Useful when you know
      the workload's footprint.
    * `min_free_ratio` — fraction of total memory that must be free
      (e.g. 0.85 = "≥85% idle"). The right knob for shared boxes with mixed
      card sizes (24GB + 32GB), where MB thresholds can let a 13GB-used 32GB
      card slip through (18GB free passes a 16GB MB floor but the card is
      clearly someone else's).

    BOTH checks apply when both are set (AND-combined). A real near-miss led
    to this: an unfiltered probe handed back GPU 0 on a box where another
    user was training on it; an MB-only filter still passed because the card
    was 32GB total. The ratio gate would reject any GPU that's <85% idle
    regardless of absolute card size.
    """
    # Local mode (W106): None / "localhost" / "127.0.0.1" → run nvidia-smi
    # directly (no ssh) so a self-hosted orchestrator on the GPU box probes its
    # own cards. Same parsing + thresholds apply.
    is_local = ssh_target in (None, "", "localhost", "127.0.0.1")
    if is_local:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    else:
        cmd = [
            "ssh", ssh_target,
            "nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits",
        ]
    try:
        proc = popen_factory(cmd, None, None)
        out = proc.stdout.read() if proc.stdout is not None else ""
        code = proc.wait()
    except Exception as exc:
        logger.warning(f"runner.gpu_health.error target={ssh_target}: {exc}")
        return []
    if (
        is_local
        and popen_factory is _default_popen
        and (code != 0 or not _has_gpu_memory_rows(out))
    ):
        out = _probe_local_gpu_memory_with_torch()
    alive: list[int] = []
    skipped: list[tuple[int, int, int, float]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        idx = int(parts[0])
        total_mb = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
        free_mb = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
        ratio = (free_mb / total_mb) if total_mb > 0 else 0.0
        if (min_free_mb > 0 and free_mb < min_free_mb) or (
            min_free_ratio > 0.0 and ratio < min_free_ratio
        ):
            skipped.append((idx, free_mb, total_mb, ratio))
            continue
        alive.append(idx)
    if skipped:
        logger.warning(
            f"runner.gpu_health.skip_busy target={ssh_target} "
            f"min_free_mb={min_free_mb} min_free_ratio={min_free_ratio} "
            f"skipped={skipped}"
        )
    logger.info(
        f"runner.gpu_health target={ssh_target} alive={alive} "
        f"(min_free_mb={min_free_mb} min_free_ratio={min_free_ratio})"
    )
    return alive
