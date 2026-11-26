from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path


def test_default_pytest_collection_is_limited_to_test_suite():
    parser = ConfigParser()
    parser.read(Path("pytest.ini"), encoding="utf-8")

    configured_paths = parser.get("pytest", "testpaths", fallback="").split()

    assert configured_paths == ["tests"]
