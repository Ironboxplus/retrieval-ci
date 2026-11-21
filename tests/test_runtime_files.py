from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.paper_crawler.runtime_files import sanitize_utf8_text, write_text_atomic


class RuntimeFileTests(unittest.TestCase):
    def test_sanitize_utf8_text_replaces_surrogate_code_points(self) -> None:
        original_text = "prefix " + chr(0xD835) + " suffix"
        sanitized_text = sanitize_utf8_text(original_text)
        self.assertEqual(sanitized_text, "prefix \uFFFD suffix")
        self.assertEqual(sanitized_text.encode("utf-8"), b"prefix \xef\xbf\xbd suffix")

    def test_write_text_atomic_handles_invalid_surrogates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="retrival_runtime_files_") as temp_dir:
            output_path = Path(temp_dir) / "sample.md"
            write_text_atomic(output_path, "ok " + chr(0xD835) + " done")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "ok \uFFFD done")


if __name__ == "__main__":
    unittest.main()
