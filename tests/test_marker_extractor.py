"""TDD for MarkerExtractor — Strategy-pattern marker-pdf wrapper.

These tests do NOT load real marker models (slow + GPU-dependent). Real
end-to-end conversion is exercised by the smoke script
``scripts/smoke_marker.py``. Here we verify only:

  * file-not-found / wrong suffix path → FAILED uniformly
  * marker absent → FAILED with helpful hint, not a crash
  * happy path produces SUCCESS with markdown, elapsed_ms, page_count
  * empty markdown is treated as FAILED, not a phantom success
  * class-level converter cache is reused across calls
  * factory ``create_extractor("marker")`` returns a MarkerExtractor
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.paper_crawler.converters.extractors import (
    ExtractionStatus,
    MarkerExtractor,
    create_extractor,
)


@pytest.fixture(autouse=True)
def _reset_marker_cache():
    MarkerExtractor.reset_cache()
    yield
    MarkerExtractor.reset_cache()


def test_factory_creates_marker_extractor():
    extractor = create_extractor("marker")
    assert isinstance(extractor, MarkerExtractor)
    assert extractor.strategy_name == "marker"


def test_factory_creates_marker_with_kwargs():
    extractor = create_extractor("marker", paginate_output=True, use_llm=False)
    assert isinstance(extractor, MarkerExtractor)
    assert extractor.paginate_output is True
    assert extractor.use_llm is False


def test_missing_pdf_returns_failed(tmp_path):
    extractor = MarkerExtractor()
    result = extractor.extract(tmp_path / "ghost.pdf")
    assert result.status == ExtractionStatus.FAILED
    assert "not found" in (result.error or "").lower()
    assert result.strategy == "marker"


def test_non_pdf_suffix_returns_failed(tmp_path):
    fake = tmp_path / "notes.txt"
    fake.write_text("hello", encoding="utf-8")
    result = MarkerExtractor().extract(fake)
    assert result.status == ExtractionStatus.FAILED
    assert "not a pdf" in (result.error or "").lower()


def test_marker_not_installed_returns_failed_with_hint(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    extractor = MarkerExtractor()
    with patch.object(
        MarkerExtractor, "_get_converter",
        side_effect=ImportError("No module named 'marker'"),
    ):
        result = extractor.extract(pdf)

    assert result.status == ExtractionStatus.FAILED
    assert "marker-pdf" in (result.error or "")
    assert "pip install" in (result.error or "")


def test_successful_extraction_returns_markdown(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    fake_md = "# Title\n\n## Section 1\n\nBody text here.\n"
    fake_metadata = {"page_count": 7}
    fake_images: dict = {}

    class _FakeConverter:
        def __call__(self, path):
            return ("fake_rendered_obj",)

    fake_text_from_rendered = lambda _: (fake_md, fake_metadata, fake_images)

    fake_module = type(sys)("marker.output")
    fake_module.text_from_rendered = fake_text_from_rendered

    extractor = MarkerExtractor()
    with patch.object(MarkerExtractor, "_get_converter", return_value=_FakeConverter()):
        with patch.dict(sys.modules, {"marker.output": fake_module}):
            result = extractor.extract(pdf)

    assert result.status == ExtractionStatus.SUCCESS
    assert result.markdown == fake_md
    assert result.page_count == 7
    assert result.elapsed_ms > 0
    assert result.strategy == "marker"
    assert result.has_images is False


def test_successful_extraction_with_images(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    fake_text_from_rendered = lambda _: (
        "# OK\n\nbody\n",
        {"page_count": 3},
        {"img_0.png": b"PNG", "img_1.png": b"PNG"},
    )
    fake_module = type(sys)("marker.output")
    fake_module.text_from_rendered = fake_text_from_rendered

    class _FakeConverter:
        def __call__(self, path):
            return object()

    with patch.object(MarkerExtractor, "_get_converter", return_value=_FakeConverter()):
        with patch.dict(sys.modules, {"marker.output": fake_module}):
            result = MarkerExtractor().extract(pdf)

    assert result.status == ExtractionStatus.SUCCESS
    assert result.has_images is True
    assert result.extra.get("image_count") == 2


def test_empty_markdown_returns_failed(tmp_path):
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    fake_module = type(sys)("marker.output")
    fake_module.text_from_rendered = lambda _: ("   \n\n  ", {}, {})

    class _FakeConverter:
        def __call__(self, path):
            return object()

    with patch.object(MarkerExtractor, "_get_converter", return_value=_FakeConverter()):
        with patch.dict(sys.modules, {"marker.output": fake_module}):
            result = MarkerExtractor().extract(pdf)

    assert result.status == ExtractionStatus.FAILED
    assert "empty" in (result.error or "").lower()


def test_conversion_exception_returns_failed_not_crash(tmp_path):
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    class _BoomConverter:
        def __call__(self, path):
            raise RuntimeError("CUDA out of memory")

    with patch.object(MarkerExtractor, "_get_converter", return_value=_BoomConverter()):
        result = MarkerExtractor().extract(pdf)

    assert result.status == ExtractionStatus.FAILED
    assert "CUDA out of memory" in (result.error or "")


def test_converter_cache_is_reused_across_calls(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    fake_module = type(sys)("marker.output")
    fake_module.text_from_rendered = lambda _: ("# H\n\nbody\n", {}, {})

    call_count = {"n": 0}

    class _FakeConverter:
        def __call__(self, path):
            call_count["n"] += 1
            return object()

    fake_converter = _FakeConverter()
    init_count = {"n": 0}

    def fake_get_converter(self):
        if MarkerExtractor._converter is None:
            init_count["n"] += 1
            MarkerExtractor._converter = fake_converter
        return MarkerExtractor._converter

    with patch.object(MarkerExtractor, "_get_converter", new=fake_get_converter):
        with patch.dict(sys.modules, {"marker.output": fake_module}):
            ext = MarkerExtractor()
            r1 = ext.extract(pdf)
            r2 = ext.extract(pdf)
            r3 = MarkerExtractor().extract(pdf)

    assert r1.status == r2.status == r3.status == ExtractionStatus.SUCCESS
    assert init_count["n"] == 1, "converter should be cached at class scope, not re-init per call"
    assert call_count["n"] == 3


def test_build_config_includes_use_llm_only_when_true():
    cfg_off = MarkerExtractor(use_llm=False)._build_config()
    cfg_on = MarkerExtractor(use_llm=True)._build_config()
    assert "use_llm" not in cfg_off
    assert cfg_on["use_llm"] is True


def test_extra_config_overrides_defaults():
    ext = MarkerExtractor(extra_config={"output_format": "json", "custom_key": 42})
    cfg = ext._build_config()
    assert cfg["output_format"] == "json"
    assert cfg["custom_key"] == 42
