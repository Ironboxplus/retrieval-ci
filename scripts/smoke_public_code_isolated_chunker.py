"""Run one committed Git blob through the crash-isolated code chunker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from core.code_kb.isolated_chunker import IsolatedCodeChunker
from core.code_kb.source_policy import resolve_language
from core.code_kb.source_text import decode_source_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    arguments = parser.parse_args()

    source_bytes = subprocess.check_output(
        [
            "git",
            "-C",
            str(arguments.mirror),
            "show",
            f"{arguments.revision}:{arguments.path}",
        ]
    )
    source = decode_source_bytes(source_bytes)
    language = resolve_language(arguments.path)
    chunker = IsolatedCodeChunker(timeout_seconds=arguments.timeout_seconds)
    try:
        chunks = chunker(arguments.path, source, language=language)
    finally:
        chunker.close()

    print(json.dumps({
        "path": arguments.path,
        "language": language,
        "source_bytes": len(source_bytes),
        "chunks": len(chunks),
    }))


if __name__ == "__main__":
    main()
