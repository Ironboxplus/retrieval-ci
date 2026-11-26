r"""Locate a paper's on-disk artifact (markdown / pdf) on the CURRENT host.

Why this exists (2026-06-18): the full stack moved Arc/Windows -> 4090/Linux, but
``papers.txt_path`` / ``papers.pdf_path`` still hold the Windows paths captured at
ingest time:

  * UNC    ``\\10.100.100.9\lsb\retrival\papers\markdown\<venue>\<year>\<base>``
           (the SMB share, mounted on 4090 at ``/mnt/lsb/...``)
  * drive  ``C:\\Users\\Arc\\...\\markdown\<venue>\<year>\<base>``
           (Arc-local; the file may have been synced to a Linux root)

A bare ``open(stored_path)`` on Linux fails, so ``/api/papers/{id}/fulltext`` 404s.
Rather than rewrite 95k rows (lossy, irreversible, breaks rollback to Arc), this
resolver maps a stored path to a readable one at request time: prefer the stored
path if it opens here, else probe a list of host roots using the uniform
``<venue>/<year>/<basename>`` layout that every ingest batch follows.

Pure + injectable (``exists``) so it is unit-tested without a filesystem; the
route passes ``settings.paper_markdown_roots`` / ``paper_pdf_roots``.
"""
from __future__ import annotations

import logging
import os
import posixpath
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)


def _basename(path: str) -> str:
    """Final path component, tolerant of both ``\\`` (Windows) and ``/`` separators."""
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def resolve_paper_artifact(
    stored_path: Optional[str],
    venue: Optional[str],
    year,
    roots: Sequence[str],
    *,
    fallback_basename: Optional[str] = None,
    exists: Callable[[str], bool] = os.path.exists,
) -> Optional[str]:
    """Return a host-readable path for a paper artifact, or ``None``.

    Resolution order:
    1. ``stored_path`` itself, if it opens on this host (native Linux rows, or
       running back on Arc where the original Windows paths are valid).
    2. ``<root>/<venue-lower>/<year>/<basename>`` for each root, then the venue
       spelling recorded in SQLite when it differs.  The lower-case form is the
       canonical SMB layout; the second probe preserves access to older local
       Linux trees that stored conference directories as ``ICML``/``CVPR``.

    ``fallback_basename`` covers records whose artifact was successfully written
    to the canonical conference/year directory but whose path was never
    persisted to SQLite. It must be a filename, not an arbitrary path.
    """
    if stored_path and exists(stored_path):
        return stored_path

    base = _basename(stored_path) if stored_path else _basename(fallback_basename or "")
    if not base:
        return None
    venue_raw = str(venue).strip() if venue else ""
    # Keep the normalized SMB layout first, but do not turn a historical
    # case-sensitive local directory into a false 404 after a host migration.
    venue_segments = tuple(dict.fromkeys((venue_raw.lower(), venue_raw))) or ("",)
    year_seg = str(year) if year is not None else ""
    for root in roots:
        # roots are POSIX paths (4090 SMB mount / local /data); never os.path.join
        # here or a Windows dev host would build backslash candidates.
        for venue_seg in venue_segments:
            candidate = posixpath.join(root, venue_seg, year_seg, base)
            if exists(candidate):
                return candidate

    logger.warning(
        "paper artifact unresolved: stored=%s venue=%s year=%s base=%s "
        "(tried %d candidate paths)",
        stored_path, venue, year, base, len(roots) * len(venue_segments),
    )
    return None
