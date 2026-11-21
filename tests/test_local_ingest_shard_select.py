"""W81: local_ingest must select parquet shards by NUMBER, not list position.

The bug: `shards = sorted(glob('vectors_*.parquet'))[shard_start:shard_end]`
slices the sorted list BY POSITION. When concurrent pipelines share one
vectors/ dir, a shard < shard_end may not be on disk yet (its producing task
hasn't ship-down'd it), so positions shift and ingest grabs the WRONG files
(e.g. another venue's vectors_0130) while missing its own (vectors_0127).

Fix: select by parsing the NUMBER out of vectors_NNNN and filtering
[shard_start, shard_end). Immune to gaps and to other venues' shards.
"""
from __future__ import annotations

from scripts.remote_embed.local_ingest import select_shards


def _paths(tmp_path, nums):
    return [tmp_path / f"vectors_{n:04d}.parquet" for n in nums]


def test_select_shards_filters_by_number_not_position(tmp_path):
    # A gap at 127 (not shipped yet) + an out-of-range 130 from another venue.
    paths = _paths(tmp_path, [125, 126, 128, 130])
    got = [p.name for p in select_shards(paths, 125, 129)]
    # NUMBER-based: 125,126,128 are in [125,129); 130 is excluded. The missing
    # 127 must NOT cause a position shift that pulls in 130.
    assert got == ["vectors_0125.parquet", "vectors_0126.parquet", "vectors_0128.parquet"]


def test_select_shards_excludes_other_venue_above_end(tmp_path):
    paths = _paths(tmp_path, [120, 121, 122, 140, 141])
    got = [p.name for p in select_shards(paths, 120, 123)]
    assert got == ["vectors_0120.parquet", "vectors_0121.parquet", "vectors_0122.parquet"]


def test_select_shards_no_upper_bound(tmp_path):
    paths = _paths(tmp_path, [125, 126, 130])
    got = [p.name for p in select_shards(paths, 126, 0)]  # 0 = open upper bound
    assert got == ["vectors_0126.parquet", "vectors_0130.parquet"]


def test_select_shards_backward_compat_contiguous_from_zero(tmp_path):
    # Historical single-pipeline runs: contiguous 0..4, so NUMBER == position.
    # Number-based filtering must give the same result as the old [2:4] slice.
    paths = _paths(tmp_path, range(5))
    got = [p.name for p in select_shards(paths, 2, 4)]
    assert got == ["vectors_0002.parquet", "vectors_0003.parquet"]


def test_select_shards_lower_bound_only_when_end_zero(tmp_path):
    paths = _paths(tmp_path, [0, 1, 2, 3, 4])
    got = [p.name for p in select_shards(paths, 3, 0)]
    assert got == ["vectors_0003.parquet", "vectors_0004.parquet"]
