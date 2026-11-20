"""TDD: payload canonicalize contract.

Legacy Qdrant payload schema:
  conference (str)            -- pre-canonical alias for venue
  author_names (pipe-sep str) -- pre-canonical pipe-separated authors

Canonical schema:
  venue (str)
  authors (list[str])

The same canonicalize logic is used by two scripts:
  scripts/migrate_qdrant_payload.canonicalize_patch  -- in-place migration via set/delete payload
  scripts/remote_embed/local_ingest.canonicalize_payload  -- at-ingest cleanup

Both must agree on the contract. These tests lock the contract for both.
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "remote_embed"))


class TestMigratePatch(unittest.TestCase):
    """scripts/migrate_qdrant_payload.canonicalize_patch — returns (add, drop)."""

    def setUp(self):
        from migrate_qdrant_payload import canonicalize_patch
        self.fn = canonicalize_patch

    def test_legacy_conference_promotes_to_venue(self):
        payload = {"conference": "ICML", "doc_id": 1}
        add, drop = self.fn(payload)
        self.assertEqual(add.get("venue"), "ICML")
        self.assertIn("conference", drop)

    def test_existing_venue_not_overwritten(self):
        payload = {"conference": "ICML", "venue": "ICML", "doc_id": 1}
        add, drop = self.fn(payload)
        self.assertNotIn("venue", add)
        self.assertIn("conference", drop)

    def test_author_names_string_splits_to_list(self):
        payload = {"author_names": "Alice | Bob | Carol", "doc_id": 1}
        add, drop = self.fn(payload)
        self.assertEqual(add.get("authors"), ["Alice", "Bob", "Carol"])
        self.assertIn("author_names", drop)

    def test_existing_authors_list_preserved(self):
        payload = {"author_names": "Should | Be | Ignored",
                   "authors": ["Original", "Author"], "doc_id": 1}
        add, drop = self.fn(payload)
        self.assertNotIn("authors", add)
        self.assertIn("author_names", drop)

    def test_canonical_only_payload_no_migration(self):
        payload = {"venue": "ICML", "authors": ["Alice"], "doc_id": 1}
        add, drop = self.fn(payload)
        self.assertEqual(add, {})
        self.assertEqual(drop, [])

    def test_empty_string_author_names_does_not_create_empty_authors(self):
        payload = {"author_names": "   |  |   ", "doc_id": 1}
        add, drop = self.fn(payload)
        # All-empty tokens should not produce an authors entry
        self.assertNotIn("authors", add)
        self.assertIn("author_names", drop)

    def test_both_legacy_keys_handled(self):
        payload = {"conference": "ACL", "author_names": "A | B", "doc_id": 1}
        add, drop = self.fn(payload)
        self.assertEqual(add["venue"], "ACL")
        self.assertEqual(add["authors"], ["A", "B"])
        self.assertIn("conference", drop)
        self.assertIn("author_names", drop)


class TestLocalIngestCanonicalize(unittest.TestCase):
    """scripts/remote_embed/local_ingest.canonicalize_payload — whitelist projection.

    Architecture: Qdrant payload is a minimal whitelist (13 keys: identity +
    filter pushdown + content). Display fields (title/abstract/authors/...) live
    in SQL Paper table — Qdrant payload MUST NOT duplicate them, because that
    leads to double-bookkeeping and drift. SQL is single source of truth for
    display data; Qdrant payload is retrieve/rerank input only.
    """

    def setUp(self):
        from local_ingest import canonicalize_payload, PAYLOAD_KEEP_KEYS
        self.fn = canonicalize_payload
        self.KEEP = PAYLOAD_KEEP_KEYS

    def test_drops_legacy_conference_after_promotion(self):
        out = self.fn({"conference": "NeurIPS", "doc_id": 5})
        self.assertEqual(out.get("venue"), "NeurIPS")
        self.assertNotIn("conference", out)

    def test_preserves_canonical_venue_when_both_present(self):
        out = self.fn({"conference": "OldName", "venue": "CanonName", "doc_id": 5})
        self.assertEqual(out["venue"], "CanonName")
        self.assertNotIn("conference", out)

    def test_author_names_string_dropped(self):
        """author_names is legacy schema; authors comes from SQL JOIN, not payload."""
        out = self.fn({"author_names": "Alice | Bob", "doc_id": 5})
        self.assertNotIn("author_names", out)
        self.assertNotIn("authors", out)

    def test_authors_list_dropped(self):
        """authors is display data; SQL Paper.authors_json is single source."""
        out = self.fn({"authors": ["Alice", "Bob"], "doc_id": 5})
        self.assertNotIn("authors", out)

    def test_clean_payload_projects_to_whitelist(self):
        """authors is dropped (not in whitelist); other allowed keys preserved.

        doc_id is normalized to str (D2, papers_payload_contract_plan.md 3.1)
        and source_type defaults to "paper" — local_ingest.py only ever
        ingests the papers batch/remote-embed pipeline (paper + text-only
        review), never upload/webpage, so legacy shards dumped before
        source_type tagging existed are still the paper profile.
        """
        out = self.fn({"venue": "CVPR", "authors": ["A", "B"], "doc_id": 5, "level": 2})
        self.assertEqual(out, {"venue": "CVPR", "doc_id": "5", "level": 2, "source_type": "paper"})

    def test_paper_id_preserved(self):
        """paper_id is identity key — must always pass through."""
        out = self.fn({"paper_id": "aaai-2022-20281", "doc_id": 5})
        self.assertEqual(out.get("paper_id"), "aaai-2022-20281")

    def test_drops_all_display_fields(self):
        """Title/abstract/pdf_url/citation_count etc. are SQL-owned — never in payload."""
        out = self.fn({
            "conference": "Old", "venue": "New",
            "author_names": "X | Y", "authors": ["A"],
            "title": "Foo", "abstract": "Bar",
            "pdf_url": "http://x", "paper_page_url": "http://y",
            "citation_count": 42, "citations": 42,
            "keyword_terms": "k1,k2", "paper_type": "main",
            "markdown_path": "/p/x.md", "track_type": "main",
            "doc_id": 5,
        })
        for k in ("conference", "author_names", "authors", "title", "abstract",
                  "pdf_url", "paper_page_url", "citation_count", "citations",
                  "keyword_terms", "paper_type", "markdown_path", "track_type"):
            self.assertNotIn(k, out, f"display field {k!r} leaked into payload")
        self.assertEqual(out["venue"], "New")
        self.assertEqual(out["doc_id"], "5")  # D2: normalized to str

    def test_whitelist_size_locked(self):
        """Regression: PAYLOAD_KEEP_KEYS contract is fixed at 8. Removed in
        post-D5 cleanup (2026-05-16): chunk_id (display-only), start_idx/end_idx
        /chunk_size (never filtered). W22 (2026-05-17) further dropped
        `data_source` (dump path never populated it → 100% null → useless
        filter; JOIN documents.data_source via doc_id when needed). 8 fields is
        the CLAUDE.md rule #1 whitelist. Adding fields re-introduces the
        double-bookkeeping bug — change deliberately."""
        self.assertEqual(len(self.KEEP), 8,
                         f"PAYLOAD_KEEP_KEYS changed: {sorted(self.KEEP)}")
        self.assertEqual(self.KEEP, frozenset({
            "doc_id", "paper_id",
            "source_type", "level",
            "venue", "year", "venue_tier",
            "content",
        }))

    def test_whitelist_is_the_projector_profile_not_a_copy(self):
        """papers_payload_contract_plan.md 3.2: local_ingest must delegate to
        core.storage.papers_payload instead of redefining the whitelist —
        that duplication is exactly how api_routes.py's payload comment
        drifted to a wrong "9-field" claim for months (plan.md 1.4)."""
        from core.storage.papers_payload import PAPER_PROFILE_KEYS
        self.assertIs(self.KEEP, PAPER_PROFILE_KEYS)


class TestCanonicalizeAcrossScripts(unittest.TestCase):
    """The two scripts have DIFFERENT contracts:

    - migrate_qdrant_payload.canonicalize_patch: legacy→canonical schema migration
      (preserves valuable fields, returns add/drop diff for in-place update).
    - local_ingest.canonicalize_payload: whitelist projection at ingest time
      (drops everything not in 13-key whitelist, including authors/title/etc).

    The only overlap is venue promotion — both promote `conference→venue`.
    """

    def test_both_promote_conference_to_venue_consistently(self):
        from migrate_qdrant_payload import canonicalize_patch
        from local_ingest import canonicalize_payload

        legacy = {"conference": "ICCV", "doc_id": 1}
        add, _ = canonicalize_patch(dict(legacy))
        cleaned = canonicalize_payload(dict(legacy))
        self.assertEqual(add["venue"], cleaned["venue"])

    def test_local_ingest_drops_authors_migrate_preserves(self):
        """Intentional asymmetry: ingest is whitelist; migrate is schema rewrite."""
        from migrate_qdrant_payload import canonicalize_patch
        from local_ingest import canonicalize_payload

        legacy = {"author_names": "Alice | Bob | Carol", "doc_id": 1}
        add, _ = canonicalize_patch(dict(legacy))
        cleaned = canonicalize_payload(dict(legacy))
        # migrate produces authors (preserved for legacy display compatibility)
        self.assertEqual(add.get("authors"), ["Alice", "Bob", "Carol"])
        # local_ingest drops authors entirely (whitelist projection)
        self.assertNotIn("authors", cleaned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
