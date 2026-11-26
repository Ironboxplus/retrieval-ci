"""MarkerExtractor is retired (user, 2026-11-14: "Marker 报错，Paddle 保留").

The behaviour tests of the old wrapper (converter cache, empty markdown, missing marker-pdf) went with
it. What remains here is the factory contract around the retired name; every other way into Marker is
in tests/test_retired_marker.py.
"""
from __future__ import annotations

import pytest

from core.inference_retired import RetiredInferencePath
from pipeline.paper_crawler.converters.extractors import MarkerExtractor, create_extractor


def test_marker_is_a_known_strategy_that_raises_instead_of_an_unknown_one():
    """"marker" stays registered: an old caller learns that Marker is retired, not that it mistyped a name."""
    with pytest.raises(RetiredInferencePath, match="MarkerExtractor"):
        create_extractor("marker")
    with pytest.raises(RetiredInferencePath):
        create_extractor("marker", paginate_output=True, use_llm=False)


def test_an_unknown_strategy_is_still_a_value_error():
    with pytest.raises(ValueError):
        create_extractor("no-such-extractor")


def test_the_retired_class_keeps_its_name_and_strategy_label():
    assert MarkerExtractor.strategy_name == "marker"
