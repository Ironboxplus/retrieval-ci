"""
Runtime file helpers for long-running corpus rebuilds and indexing jobs.

These helpers centralize atomic JSON/text writes plus lightweight artifact
validation so resume logic can stay consistent across the paper rebuild stack.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from pypdf import PdfReader


LOGGER = logging.getLogger(__name__)


def build_last_good_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.last_good{path.suffix}")


def build_write_in_progress_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.write_in_progress{path.suffix}")


def remove_file_if_present(path: Path, *, attempts: int = 5) -> bool:
    """Delete ``path`` best-effort, tolerating transient Windows locks.

    A file Windows just finished writing is routinely still held for a few
    hundred milliseconds by the real-time virus scanner, so ``unlink()`` raises
    ``PermissionError`` (WinError 32) on a path that is about to become
    deletable. Failing to remove a scratch artifact is never worth losing a
    long download run over, so retry briefly and then leave the leftover for
    the next pass — every writer here opens with ``"wb"``, which truncates.

    Returns whether the path is gone.
    """
    for attempt in range(1, attempts + 1):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt == attempts:
                LOGGER.warning(
                    "Could not remove %s after %s attempts; leaving it in place",
                    path,
                    attempts,
                )
                return False
            time.sleep(0.2 * attempt)
    return False


def sanitize_utf8_text(content: str) -> str:
    sanitized_characters: list[str] = []
    for character in content:
        if 0xD800 <= ord(character) <= 0xDFFF:
            sanitized_characters.append("\uFFFD")
            continue
        sanitized_characters.append(character)
    return "".join(sanitized_characters)


def write_text_atomic(path: Path, content: str, *, durable: bool = False) -> None:
    """Write text via temp file + rename, retrying transient Windows locks.

    When the rename keeps failing the last ``OSError`` is raised and ``path``
    keeps its previous content: callers use this for manifests and resume
    state, where an in-place overwrite would leave a torn file that still
    reads as a successful write. ``durable`` also fsyncs the temp file before
    the rename, for cursors that must survive a power loss.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_utf8_text(content)
    temporary_path = build_write_in_progress_path(path)

    for attempt in range(1, 6):
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                handle.write(sanitized)
                if durable:
                    handle.flush()
                    os.fsync(handle.fileno())
            temporary_path.replace(path)
            return
        except OSError:
            if temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass
            if attempt == 5:
                raise
            time.sleep(0.2 * attempt)


def write_json_atomic(path: Path, payload: Any) -> None:
    if path.exists():
        shutil.copyfile(path, build_last_good_path(path))
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2))


def load_json_with_recovery(path: Path, default_value: Any) -> Any:
    if not path.exists():
        return default_value

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        last_good_path = build_last_good_path(path)
        if last_good_path.exists():
            LOGGER.warning(
                "Recovering %s from last-good backup after JSON decode error: %s",
                path,
                exc,
            )
            return json.loads(last_good_path.read_text(encoding="utf-8"))
        raise


def is_valid_pdf_artifact(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        reader = PdfReader(str(path))
        return len(reader.pages) > 0
    except Exception:
        return False


def is_valid_markdown_artifact(path: Path, paper_id: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return False

    return f"paper_id: {paper_id}" in content and "# " in content
