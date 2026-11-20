"""TDD tests for YAML-driven conference registry."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONF_YAML = Path(__file__).resolve().parent.parent / "conf" / "conferences.yaml"


class TestConferenceYamlExists(unittest.TestCase):
    def test_yaml_file_exists(self):
        self.assertTrue(CONF_YAML.exists())

    def test_yaml_is_valid(self):
        import yaml
        data = yaml.safe_load(CONF_YAML.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)
        self.assertIn("conferences", data)
        self.assertIn("categories", data)
        self.assertGreater(len(data["conferences"]), 0)


class TestConferenceYamlSchema(unittest.TestCase):
    def setUp(self):
        import yaml
        raw = yaml.safe_load(CONF_YAML.read_text(encoding="utf-8"))
        self.entries = raw["conferences"]
        self.categories = raw["categories"]

    def test_all_entries_have_required_fields(self):
        for entry in self.entries:
            with self.subTest(title=entry.get("title")):
                self.assertIn("title", entry)
                self.assertIn("description", entry)
                self.assertIn("sub", entry)
                self.assertIn("rank", entry)
                self.assertIn("ccf", entry["rank"])
                self.assertIn("dblp", entry)
                self.assertIn("source_class", entry)
                self.assertIn("years", entry)
                self.assertIsInstance(entry["years"], list)

    def test_all_known_venues_registered(self):
        titles = {e["title"] for e in self.entries}
        expected = {
            # AI
            "AAAI", "ACL", "CVPR", "EMNLP", "ICCV", "ICLR", "ICML", "IJCAI", "NeurIPS",
            # DB
            "KDD", "SIGIR", "SIGMOD", "VLDB",
            # SC
            "CCS", "USENIX Security", "NDSS",
            # CG
            "ACM MM",
            # MX (interdisciplinary): WWW = The Web Conference, CSUR = ACM Computing Surveys (journal)
            "WWW", "CSUR",
            # DS (Computer Architecture / Parallel / Storage) — open-access full-text only.
            # IEEE-paywalled MICRO / HPCA / ISCA<=2024 / SC-even-years deferred (W89).
            "ASPLOS", "ISCA", "PPoPP", "EuroSys", "DAC", "FAST", "USENIX ATC",
        }
        self.assertEqual(titles, expected)

    def test_ccf_tiers_correct(self):
        tier_map = {e["title"]: e["rank"]["ccf"] for e in self.entries}
        for name in ("AAAI", "ACL", "CVPR", "ICCV", "ICLR", "ICML", "NeurIPS",
                     "KDD", "SIGIR", "SIGMOD", "VLDB",
                     "CCS", "USENIX Security", "NDSS", "ACM MM",
                     "WWW", "CSUR",
                     "ASPLOS", "ISCA", "PPoPP", "EuroSys", "DAC", "FAST", "USENIX ATC"):
            self.assertEqual(tier_map[name], "A", f"{name} should be CCF-A")
        self.assertEqual(tier_map["EMNLP"], "B", "EMNLP should be CCF-B")
        self.assertEqual(tier_map["IJCAI"], "B", "IJCAI should be CCF-B")

    def test_sub_categories_valid(self):
        for entry in self.entries:
            self.assertIn(entry["sub"], self.categories,
                          f"{entry['title']} sub={entry['sub']} not in categories")

    def test_categories_have_names(self):
        for key, cat in self.categories.items():
            self.assertIn("name", cat)
            self.assertIn("name_en", cat)

    def test_subcategory_coverage(self):
        by_sub = {}
        for e in self.entries:
            by_sub.setdefault(e["sub"], []).append(e["title"])
        self.assertTrue(set(by_sub["AI"]).issuperset({"AAAI", "NeurIPS", "ICLR", "ICML", "CVPR", "ICCV", "ACL"}))
        self.assertTrue(set(by_sub["DB"]).issuperset({"KDD", "SIGIR", "SIGMOD", "VLDB"}))
        self.assertTrue(set(by_sub["SC"]).issuperset({"CCS", "USENIX Security", "NDSS"}))
        self.assertTrue(set(by_sub["CG"]).issuperset({"ACM MM"}))
        self.assertTrue(set(by_sub["MX"]).issuperset({"WWW", "CSUR"}))
        self.assertTrue(set(by_sub["DS"]).issuperset(
            {"ASPLOS", "ISCA", "PPoPP", "EuroSys", "DAC", "FAST", "USENIX ATC"}))


class TestLoadFromYaml(unittest.TestCase):
    def test_load_conference_registry(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import load_conference_registry
        registry = load_conference_registry()
        self.assertIsInstance(registry, list)
        self.assertEqual(len(registry), 26)

    def test_registry_entries_have_source(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import load_conference_registry
        registry = load_conference_registry()
        for entry in registry:
            self.assertIn("name", entry)
            self.assertIn("source", entry)
            self.assertTrue(hasattr(entry["source"], "collect_papers"),
                            f"{entry['name']} source has no collect_papers")

    def test_build_sources_covers_all_entries(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import build_sources
        sources = build_sources()
        self.assertEqual(len(sources), 26)
        for required in ("ICLR", "CCS", "VLDB", "ACM MM", "USENIX Security", "NDSS", "WWW", "CSUR",
                         "ASPLOS", "ISCA", "PPoPP", "EuroSys", "DAC", "FAST", "USENIX ATC"):
            self.assertIn(required, sources)

    def test_registry_derives_venue_tier(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import load_conference_registry
        registry = load_conference_registry()
        for entry in registry:
            self.assertIn("venue_tier", entry)
            self.assertTrue(entry["venue_tier"].startswith("CCF-"))

    def test_registry_enriches_sub_name(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import load_conference_registry
        registry = load_conference_registry()
        ai_entries = [e for e in registry if e.get("sub") == "AI"]
        self.assertTrue(len(ai_entries) > 0)
        self.assertEqual(ai_entries[0]["sub_name"], "人工智能")

    def test_load_categories(self):
        from pipeline.paper_crawler.ccf_a_main_corpus import load_categories
        cats = load_categories()
        self.assertIn("AI", cats)
        self.assertIn("DB", cats)


if __name__ == "__main__":
    unittest.main()
