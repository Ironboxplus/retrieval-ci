"""
CLI wrapper for rebuilding the clean CCF-A main-conference corpus.

The actual crawler, downloader, extractor, and indexer logic lives in the
paper_crawler package so that the workflow stays reusable beyond this entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.paper_crawler.ccf_a_main_corpus import main


if __name__ == "__main__":
    raise SystemExit(main())
