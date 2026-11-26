"""Crash-isolated adapter for the shared AST code chunker."""

from __future__ import annotations

import atexit
from functools import lru_cache
import math
import multiprocessing
from multiprocessing.pool import Pool
import threading
from typing import Callable

from core.code_kb.models import CodeChunk
from core.code_kb.source_policy import DEFAULT_MAX_CHUNK_SIZE


class IsolatedChunkerError(RuntimeError):
    """Raised when an isolated native chunk worker cannot return a result."""


ChunkWorker = Callable[[str, str, str | None, int], list[CodeChunk]]
_WORKER_THREAD_EXECUTOR = None


def _initialize_chunk_worker() -> None:
    """Give native parser calls a large-stack thread inside the child process."""

    global _WORKER_THREAD_EXECUTOR
    from core.code_kb.chunker import create_chunk_executor

    _WORKER_THREAD_EXECUTOR = create_chunk_executor(1)


def _chunk_in_worker(
    path: str,
    source: str,
    language: str | None,
    max_chunk_size: int,
) -> list[CodeChunk]:
    if _WORKER_THREAD_EXECUTOR is None:
        _initialize_chunk_worker()
    from core.code_kb.chunker import chunk_source

    future = _WORKER_THREAD_EXECUTOR.submit(
        chunk_source,
        path,
        source,
        language=language,
        max_chunk_size=max_chunk_size,
    )
    return future.result()


class IsolatedCodeChunker:
    """Run native AST chunking outside the API/model process.

    One persistent spawned process owns its parser instances. Each parser call runs on
    the existing large-stack chunk thread. A native abort, segmentation fault, or hard
    timeout destroys only this pool; the next call starts a clean worker while the
    backend and its shared embedding model remain alive.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        worker_function: ChunkWorker = _chunk_in_worker,
        start_method: str = "spawn",
    ):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self._timeout_seconds = timeout_seconds
        self._worker_function = worker_function
        self._context = multiprocessing.get_context(start_method)
        self._pool: Pool | None = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    def __call__(
        self,
        path: str,
        source: str,
        *,
        language: str | None,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> list[CodeChunk]:
        with self._lock:
            pool = self._get_pool_locked()
            result = pool.apply_async(
                self._worker_function,
                (path, source, language, max_chunk_size),
            )
            try:
                return result.get(timeout=self._timeout_seconds)
            except multiprocessing.TimeoutError as exc:
                self._discard_pool_locked()
                raise IsolatedChunkerError(
                    f"native chunk worker crashed or timed out for {path}"
                ) from exc
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._discard_pool_locked()
                raise IsolatedChunkerError(
                    f"native chunk worker crashed or timed out for {path}"
                ) from exc
            except Exception as exc:
                raise IsolatedChunkerError(f"code chunking failed for {path}: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            self._discard_pool_locked()

    def _get_pool_locked(self) -> Pool:
        if self._pool is None:
            self._pool = self._context.Pool(
                processes=1,
                initializer=_initialize_chunk_worker,
            )
        return self._pool

    def _discard_pool_locked(self) -> None:
        if self._pool is None:
            return
        pool = self._pool
        self._pool = None
        pool.terminate()
        pool.join()


@lru_cache(maxsize=1)
def get_isolated_code_chunker(timeout_seconds: float = 60.0) -> IsolatedCodeChunker:
    """Return the process-wide parser-isolation boundary."""

    return IsolatedCodeChunker(timeout_seconds=timeout_seconds)


__all__ = [
    "IsolatedChunkerError",
    "IsolatedCodeChunker",
    "get_isolated_code_chunker",
]
