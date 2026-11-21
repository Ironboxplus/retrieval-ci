"""The shared cache and TTLs behind the WebUI statistics endpoints.

Production measurement 2026-11-09: `SELECT level, COUNT(*) FROM chunks GROUP BY level`
takes 19 s, papers by venue 2 s, documents by data_source 4.3 s. One instance keeps the
dashboard, the papers page and `/api/stats` from each scanning the same tables.
"""
from __future__ import annotations

from core.cache.stale_while_revalidate import StaleWhileRevalidateCache

# GROUP BY distributions, tag lists, per-table row counts: change only when an ingest lands.
DISTRIBUTION_TTL_SECONDS = 600.0
# Headline totals shown next to live indexing progress.
TOTAL_COUNT_TTL_SECONDS = 60.0

stats_cache = StaleWhileRevalidateCache("stats-cache", ttl_seconds=DISTRIBUTION_TTL_SECONDS)
