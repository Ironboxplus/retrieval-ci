"""JATS full text (PMC OA bucket) to Markdown with pandoc: formulas stay LaTeX.

The fixture is a real PLOS Medicine article (PMC12463208, CC BY 4.0) with 5
display formulas, 17 inline formulas, 1 table (with row and column spans) and
69 references; pandoc is the real bundled
binary, nothing is stubbed.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import pytest

from pipeline.medical_corpus.jats_markdown import jats_to_markdown

FIXTURE = Path(__file__).parent / "fixtures" / "jats" / "PMC12463208.1.xml.gz"


@pytest.fixture()
def article(tmp_path: Path) -> Path:
    xml = tmp_path / "med-pmc12463208.xml"
    xml.write_bytes(gzip.decompress(FIXTURE.read_bytes()))
    return xml


def test_display_and_inline_formulas_become_latex(article: Path, tmp_path: Path) -> None:
    md = jats_to_markdown(article, tmp_path / "out.md").read_text(encoding="utf-8")
    display = re.findall(r"^\$\$(.+?)\$\$$", md, flags=re.M)
    assert len(display) == 5
    assert r"Z = \frac{\beta_{1} - \beta_{2}}{\sqrt{{SE}_{1}^{2} + {SE}_{2}^{2}}}" in display
    inline = re.findall(r"(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)", md)
    assert r"\beta_{1}" in inline and len(inline) >= 17
    assert "<mml:" not in md and "<math" not in md


def test_title_abstract_sections_and_tables_survive(article: Path, tmp_path: Path) -> None:
    md = jats_to_markdown(article, tmp_path / "out.md").read_text(encoding="utf-8")
    first = md.splitlines()[0]
    assert first.startswith("# ") and "cyclone" in first.lower()
    assert "## Abstract" in md and "Methods and findings" in md
    assert re.search(r"^## Method and materials$", md, flags=re.M)
    assert re.search(r"^### Background$", md, flags=re.M)  # abstract section, under ## Abstract
    assert not re.search(r"^#{7,}", md, flags=re.M) and "{#" not in md and ":::" not in md
    refs = md.split("\n## References\n", 1)[1].split("\n[^", 1)[0]
    items = re.findall(r"^\d+\. .+$", refs, flags=re.M)
    assert len(items) == 69  # every <ref>, numbered, as published (no citeproc guessing)
    assert "n.d." not in refs and all(len(i) > 30 for i in items)
    assert len(re.findall(r"^\|[-: |]+\|$", md, flags=re.M)) == 1  # Table 1, spans flattened into a pipe table
    assert "<table" not in md and "<div" not in md  # no raw HTML left over
    assert "--" not in re.sub(r"^\|[-: |]+\|$", "", md, flags=re.M)  # dashes stay as typed (no smart)


def test_written_atomically_and_refuses_a_non_article(article: Path, tmp_path: Path) -> None:
    out = jats_to_markdown(article, tmp_path / "deep" / "a.md")
    assert out.exists() and not list(out.parent.glob("*.write_in_progress*"))
    bad = tmp_path / "bad.xml"
    bad.write_text("<html><body>Just a moment...</body></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="JATS"):
        jats_to_markdown(bad, tmp_path / "bad.md")
    assert not (tmp_path / "bad.md").exists()
