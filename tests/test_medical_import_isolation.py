"""Medical download imports must not require the optional browser crawler stack."""
from __future__ import annotations

import subprocess
import sys


def test_medical_pdf_fetch_import_does_not_load_optional_browser_use() -> None:
    """The real subprocess import mirrors the minimal .88 backend environment."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pipeline.medical_corpus.pdf_fetch; "
            "assert 'browser_use' not in sys.modules, sorted(sys.modules)",
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
