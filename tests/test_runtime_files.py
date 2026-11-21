from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from pipeline.paper_crawler.runtime_files import (
    sanitize_utf8_text,
    write_json_atomic,
    write_text_atomic,
)


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

    # An open() handle on Windows is opened without FILE_SHARE_DELETE, so the
    # rename-over is refused while a plain overwrite of the same file succeeds.
    # That is exactly the case where a direct-write fallback would hand the
    # caller a torn file and a success.
    @unittest.skipUnless(os.name == "nt", "needs Windows delete-sharing semantics")
    def test_write_text_atomic_raises_instead_of_writing_in_place(self) -> None:
        with tempfile.TemporaryDirectory(prefix="retrival_runtime_files_") as temp_dir:
            output_path = Path(temp_dir) / "state.json"
            output_path.write_text("old", encoding="utf-8")
            with open(output_path, "rb"):
                with self.assertRaises(OSError):
                    write_text_atomic(output_path, "new")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "old")
            self.assertEqual([item.name for item in Path(temp_dir).iterdir()], ["state.json"])

    @unittest.skipUnless(os.name == "nt", "needs Windows delete-sharing semantics")
    def test_write_json_atomic_raises_and_keeps_previous_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="retrival_runtime_files_") as temp_dir:
            output_path = Path(temp_dir) / "state.json"
            output_path.write_text('{"version": 1}', encoding="utf-8")
            with open(output_path, "rb"):
                with self.assertRaises(OSError):
                    write_json_atomic(output_path, {"version": 2})
            self.assertEqual(output_path.read_text(encoding="utf-8"), '{"version": 1}')


if __name__ == "__main__":
    unittest.main()
