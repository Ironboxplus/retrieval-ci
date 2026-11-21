r"""TDD for core.storage.artifact_resolver.resolve_paper_artifact.

Context (W-bug, 2026-06-18): the full-stack moved Arc/Windows -> 4090/Linux, but
papers.txt_path / pdf_path still hold Windows paths:
  * UNC   \\10.100.100.9\lsb\retrival\papers\markdown\<venue>\<year>\<base>
  * drive C:\Users\Arc\Prd\retrival\data\papers\...\markdown\<venue>\<year>\<base>
Neither opens on Linux, so /api/papers/{id}/fulltext 404s ("Markdown file not
found on disk"). The files actually live under host-reachable roots
(/mnt/lsb/retrival/papers/markdown SMB mount, or local /data tree) in a uniform
<venue>/<year>/<basename> layout. The resolver maps a stored path -> a readable
one without mutating the DB.
"""
from core.storage.artifact_resolver import resolve_paper_artifact


def _exists_in(present):
    """Build an exists() that returns True only for paths in `present`."""
    s = set(present)
    return lambda p: p in s


def test_native_path_that_exists_is_returned_asis():
    p = "/data/arc/retrival/data/papers/.../markdown/ndss/2026/ndss-2026-f797.md"
    assert resolve_paper_artifact(p, "NDSS", 2026, ["/mnt/lsb/x"],
                                  exists=_exists_in([p])) == p


def test_unc_path_resolves_under_mount_root():
    stored = r"\\10.100.100.9\lsb\retrival\papers\markdown\iclr\2022\iclr-2022-abc.md"
    root = "/mnt/lsb/retrival/papers/markdown"
    expected = root + "/iclr/2022/iclr-2022-abc.md"
    assert resolve_paper_artifact(stored, "ICLR", 2022, [root],
                                  exists=_exists_in([expected])) == expected


def test_windows_drive_path_resolves_under_second_root():
    stored = r"C:\Users\Arc\Prd\retrival\data\papers\ccf\markdown\aaai\2025\aaai-2025-31974.md"
    r1 = "/mnt/lsb/retrival/papers/markdown"          # miss
    r2 = "/data/arc/retrival/data/papers/ccf/markdown"  # hit
    expected = r2 + "/aaai/2025/aaai-2025-31974.md"
    assert resolve_paper_artifact(stored, "AAAI", 2025, [r1, r2],
                                  exists=_exists_in([expected])) == expected


def test_venue_is_lowercased_in_candidate():
    stored = r"\\10.100.100.9\lsb\x\cvpr\2024\cvpr-2024-1.md"
    root = "/mnt/lsb/m"
    expected = root + "/cvpr/2024/cvpr-2024-1.md"
    # only the lowercased candidate exists; an upper one must NOT be how we find it
    assert resolve_paper_artifact(stored, "CVPR", 2024, [root],
                                  exists=_exists_in([expected])) == expected


def test_preserves_venue_case_as_a_fallback_for_legacy_linux_artifacts():
    """Historical local trees used ``ICML`` while the SMB corpus uses ``icml``.

    Linux is case-sensitive, so resolving only the normalized lower-case layout
    makes a readable local artifact look missing after migration.
    """
    stored = "/data/arc/retrival/data/papers/markdown/ICML/2026/icml-2026-a.md"
    root = "/data/retrival/papers/markdown"
    expected = root + "/ICML/2026/icml-2026-a.md"

    assert resolve_paper_artifact(stored, "ICML", 2026, [root],
                                  exists=_exists_in([expected])) == expected


def test_prefers_canonical_lowercase_venue_when_both_layouts_exist():
    """SMB's canonical lower-case directory wins over a legacy case variant."""
    stored = "/data/arc/retrival/data/papers/markdown/ICML/2026/icml-2026-a.md"
    root = "/data/retrival/papers/markdown"
    canonical = root + "/icml/2026/icml-2026-a.md"
    legacy = root + "/ICML/2026/icml-2026-a.md"

    assert resolve_paper_artifact(stored, "ICML", 2026, [root],
                                  exists=_exists_in([canonical, legacy])) == canonical


def test_unresolved_everywhere_returns_none():
    stored = r"C:\x\markdown\kdd\2023\kdd-2023-9.md"
    assert resolve_paper_artifact(stored, "KDD", 2023, ["/mnt/lsb/m", "/data/m"],
                                  exists=_exists_in([])) is None


def test_empty_or_none_stored_path_returns_none():
    assert resolve_paper_artifact(None, "KDD", 2023, ["/m"], exists=_exists_in([])) is None
    assert resolve_paper_artifact("", "KDD", 2023, ["/m"], exists=_exists_in([])) is None


def test_missing_stored_path_uses_canonical_fallback_basename():
    root = "/mnt/lsb/retrival/papers/pdf"
    expected = root + "/aaai/2026/aaai-2026-37659.pdf"

    assert resolve_paper_artifact(
        None,
        "AAAI",
        2026,
        [root],
        fallback_basename="aaai-2026-37659.pdf",
        exists=_exists_in([expected]),
    ) == expected


def test_native_preferred_over_root_probe():
    """If the stored path itself is readable, use it (don't waste mount stats)."""
    stored = "/data/local/markdown/icml/2025/icml-2025-x.md"
    root = "/mnt/lsb/m"
    root_cand = root + "/icml/2025/icml-2025-x.md"
    # both exist; native (stored) must win
    out = resolve_paper_artifact(stored, "ICML", 2025, [root],
                                 exists=_exists_in([stored, root_cand]))
    assert out == stored


def test_basename_extracted_from_backslash_path():
    stored = r"\\h\s\markdown\sigir\2021\sigir-2021-77.md"
    root = "/mnt/lsb/m"
    expected = root + "/sigir/2021/sigir-2021-77.md"
    assert resolve_paper_artifact(stored, "SIGIR", 2021, [root],
                                  exists=_exists_in([expected])) == expected
