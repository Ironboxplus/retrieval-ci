"""JATS full text (PMC OA bucket, ``<PMCID>.<ver>.xml``) to Markdown with pandoc.

The XML carries every formula as MathML, so the Markdown keeps it as LaTeX
(``$...$`` inline, ``$$...$$`` display) without any recognition step; tables
become pipe tables. pandoc is the binary bundled with ``pypandoc_binary``.

Layout matches the converted papers: ``# Title``, ``## Abstract`` with its own
sections a level below, then the body with its top sections at ``##``, and a
numbered reference list with each ``<ref>`` as published (pandoc's citeproc reads
unstructured PMC citations as "n.d.", so the list is not left to it).
"""
from __future__ import annotations

import re
from pathlib import Path

import pypandoc
from lxml import etree

from pipeline.paper_crawler.runtime_files import build_write_in_progress_path

_HERE = Path(__file__).resolve().parent
_TEMPLATE = _HERE / "jats_markdown.template.md"
_FILTER = _HERE / "jats_markdown.lua"
# commonmark_x for pipe tables and footnotes; dollars for the math. Off: raw HTML
# (what pandoc can't express is dropped, not left as markup), attributes and fenced
# divs (JATS ids such as {#sec001} and ::: wrappers mean nothing to a reader), and
# smart, so "2000–2005" stays as typed instead of becoming "2000--2005".
_TO = "commonmark_x+tex_math_dollars-raw_html-smart-attributes-fenced_divs"


def jats_to_markdown(xml_path: Path, md_path: Path) -> Path:
    """Convert one JATS article; ``md_path`` appears only once complete."""
    head = xml_path.read_bytes()[:65536]
    if b"<article" not in head:
        raise ValueError(f"{xml_path}: not a JATS article document")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = build_write_in_progress_path(md_path)
    try:
        pypandoc.convert_file(
            str(xml_path), _TO, format="jats", outputfile=str(tmp),
            extra_args=["--wrap=none", "--standalone", f"--template={_TEMPLATE}",
                        f"--lua-filter={_FILTER}"],
        )
        body = tmp.read_text(encoding="utf-8")
        tmp.write_text(_with_references(body, _references(xml_path)), encoding="utf-8")
        tmp.replace(md_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return md_path


def _references(xml_path: Path) -> list[str]:
    """Each ``<ref>``'s citation as one line: the text of its parts, space-joined
    (``<surname>`` and ``<given-names>`` carry no space between them)."""
    refs = []
    for ref in etree.parse(str(xml_path)).iterfind(".//ref-list/ref"):
        cite = next(iter(ref.xpath("mixed-citation|element-citation|citation")), None)
        if cite is None:
            continue
        text = " ".join(t.strip() for t in cite.itertext() if t.strip())
        text = re.sub(r"\(\s+", "(", text)
        refs.append(re.sub(r"\s+([.,;:)])", r"\1", text))
    return refs


def _with_references(md: str, refs: list[str]) -> str:
    """Replace pandoc's (empty) References section with the numbered list; the
    footnotes pandoc writes at the end stay after it."""
    head, notes = md, ""
    m = re.search(r"^\[\^1\]:", md, flags=re.M)
    if m:
        head, notes = md[:m.start()], md[m.start():]
    head = re.sub(r"\n## References\n.*\Z", "\n", head, flags=re.S).rstrip() + "\n"
    if refs:
        head += "\n## References\n\n" + "".join(f"{i}. {r}\n" for i, r in enumerate(refs, 1))
    return head + ("\n" + notes if notes else "")
