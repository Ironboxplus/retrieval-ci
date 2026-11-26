"""Partial-download cleanup must never be fatal.

The IJCAI 2026 pool download died at 194/972 with

    PermissionError: [WinError 32] ...: 'ijcai-2026-0163.write_in_progress.pdf'

raised from a cleanup ``unlink()`` in the downloader's own failure handler.
Windows holds a just-written file open for a short while (real-time virus
scanning, indexing), so deleting a scratch artifact is *expected* to fail
transiently. Because the exception escaped ``download_single`` through
``worker()`` into ``asyncio.gather``, it aborted every concurrent worker and
discarded all progress since the last 200-paper checkpoint.

These tests hold a genuine open handle — CPython opens files without
``FILE_SHARE_DELETE``, so ``unlink()`` really does raise WinError 32 — rather
than patching ``Path.unlink`` to raise.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

from pipeline.download_control import ChainedProxyDownloadError, ChainedProxyPool
from pipeline.paper_crawler.runtime_files import remove_file_if_present


windows_only = pytest.mark.skipif(
    sys.platform != "win32",
    reason="an open handle only blocks unlink() on Windows",
)


def test_missing_file_is_reported_gone(tmp_path: Path) -> None:
    assert remove_file_if_present(tmp_path / "absent.pdf") is True


def test_unlocked_file_is_removed(tmp_path: Path) -> None:
    target = tmp_path / "ijcai-2026-0163.write_in_progress.pdf"
    target.write_bytes(b"%PDF-1.4 partial")

    assert remove_file_if_present(target) is True
    assert not target.exists()


@windows_only
def test_locked_file_does_not_raise(tmp_path: Path) -> None:
    """The exact crash: a held handle must degrade to a warning, not an exception."""
    target = tmp_path / "ijcai-2026-0163.write_in_progress.pdf"
    target.write_bytes(b"%PDF-1.4 partial")

    with target.open("rb"):
        assert remove_file_if_present(target, attempts=2) is False
        assert target.exists()

    # Still removable once the scanner-equivalent handle is gone.
    assert remove_file_if_present(target) is True


@windows_only
def test_lock_released_mid_retry_is_absorbed(tmp_path: Path) -> None:
    """A lock that clears within the retry window removes the file after all."""
    target = tmp_path / "ijcai-2026-0163.write_in_progress.pdf"
    target.write_bytes(b"%PDF-1.4 partial")

    handle = target.open("rb")
    released = threading.Event()

    def release_after_first_attempt() -> None:
        time.sleep(0.25)
        handle.close()
        released.set()

    releaser = threading.Thread(target=release_after_first_attempt)
    releaser.start()
    try:
        assert remove_file_if_present(target, attempts=5) is True
        assert released.is_set(), "should have retried past the initial lock"
        assert not target.exists()
    finally:
        releaser.join()
        handle.close()


@windows_only
def test_pool_download_failure_surfaces_as_download_error(tmp_path: Path) -> None:
    """A locked destination must not turn an exit failure into a PermissionError.

    ``download_to_path`` cleans up ``destination`` in its own ``except`` block.
    Before the fix that cleanup raised WinError 32 straight out of the handler,
    replacing the real, retryable "this exit failed" signal with an OS error
    that the caller's retry ladder does not understand.

    No network: the first hop is a closed loopback port, so every exit attempt
    fails at connect time.
    """
    destination = tmp_path / "ijcai-2026-0163.write_in_progress.pdf"
    destination.write_bytes(b"%PDF-1.4 leftover from the previous attempt")

    pool = ChainedProxyPool(
        "socks5://127.0.0.1:1",
        ["socks5://user:pass@127.0.0.1:2", "socks5://user:pass@127.0.0.1:3"],
    )

    with destination.open("rb"):
        with pytest.raises(ChainedProxyDownloadError):
            asyncio.run(
                pool.download_to_path(
                    "https://example.invalid/paper.pdf",
                    destination,
                    max_attempts=2,
                    validator=lambda path: False,
                    timeout_seconds=5.0,
                )
            )
        assert destination.exists(), "locked leftover stays; it is not the caller's problem"
