"""Process-boundary tests for native code chunking."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from core.code_kb.isolated_chunker import IsolatedChunkerError, IsolatedCodeChunker
from core.code_kb.models import CodeChunk


def _controlled_worker(
    path: str,
    source: str,
    language: str | None,
    max_chunk_size: int,
) -> list[CodeChunk]:
    if path == "native-crash.c":
        os._exit(139)
    if path == "native-hang.c":
        time.sleep(30)
    return [
        CodeChunk(
            path=path,
            start_line=1,
            end_line=1,
            content=f"# path: {path}\n{source}",
            code=source,
            label=str(os.getpid()),
        )
    ]


def test_isolated_chunker_runs_work_outside_the_calling_process():
    chunker = IsolatedCodeChunker(
        timeout_seconds=10,
        worker_function=_controlled_worker,
    )
    try:
        chunks = chunker("module.c", "int value = 1;", language="c")
    finally:
        chunker.close()

    assert chunks[0].path == "module.c"
    assert int(chunks[0].label) != os.getpid()


def test_isolated_chunker_recovers_after_a_native_worker_crash():
    chunker = IsolatedCodeChunker(
        timeout_seconds=2,
        worker_function=_controlled_worker,
    )
    try:
        with pytest.raises(IsolatedChunkerError, match="worker crashed or timed out"):
            chunker("native-crash.c", "int crash;", language="c")

        chunks = chunker("recovered.c", "int recovered;", language="c")
    finally:
        chunker.close()

    assert chunks[0].path == "recovered.c"


def test_isolated_chunker_terminates_a_timed_out_worker_and_recovers():
    chunker = IsolatedCodeChunker(
        timeout_seconds=2,
        worker_function=_controlled_worker,
    )
    try:
        with pytest.raises(IsolatedChunkerError, match="worker crashed or timed out"):
            chunker("native-hang.c", "int hang;", language="c")

        chunks = chunker("after-timeout.c", "int recovered;", language="c")
    finally:
        chunker.close()

    assert chunks[0].path == "after-timeout.c"


def test_isolated_chunker_uses_the_shared_ast_chunk_contract():
    chunker = IsolatedCodeChunker(timeout_seconds=10)
    try:
        chunks = chunker(
            "example.py",
            "def answer():\n    return 42\n",
            language="python",
        )
    finally:
        chunker.close()

    assert chunks
    assert chunks[0].path == "example.py"
    assert "def answer" in chunks[0].code


def test_public_code_service_import_does_not_load_native_chunker():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import app.services.public_code; "
                "assert 'chonkie' not in sys.modules; "
                "assert 'core.code_kb.chunker' not in sys.modules"
            ),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
