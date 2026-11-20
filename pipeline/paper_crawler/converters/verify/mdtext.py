"""Split one PaddleOCR-VL page result into what the checks compare: prose, table cells, formulas.

The input is the page's ``res.json["res"]`` (``parsing_res_list`` blocks: label, content and
a bounding box in rendered-image pixels). Blocks are used rather than the finished page
Markdown because they carry the layout box of every piece, and the Markdown is a pure
function of them (PaddleOCR's MarkdownConverter plus ``restructure_pages``).

Layout merging (``merge_layout_blocks``) puts several adjacent blocks' text into the first
block of a ``group_id`` and leaves the others empty; a group therefore counts as one
segment whose box is the union of its blocks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Union

from lxml import html as lxml_html

from .textlayer import BBox

# Regions that hold no text the Markdown is expected to reproduce, whatever the pipeline
# was told to ignore: pictures and charts (use_chart_recognition is off) and formula numbers.
NO_TEXT_LABELS = frozenset({"image", "chart", "seal", "header_image", "footer_image", "formula_number"})

_FORMULA = re.compile(
    r"\$\$(?P<dd>.+?)\$\$|\\\[(?P<br>.+?)\\\]|\\\((?P<pa>.+?)\\\)|\$(?P<d>.+?)\$", re.DOTALL)
# A price ("$1,000", "$10K") or a unit ("($/month)") is no formula delimiter: the dollar sign of
# one is followed by a number that is not a formula's own ("$2^{19}$"), or sits in "($/". A number
# can also open a formula ("$1 - \delta$"): a dollar whose closing one holds LaTeX is a delimiter.
_CURRENCY = re.compile(r"\$(?=\d[\d,]*(?:\.\d+)?[KMBG]?(?:\s|[^\w^_{\\$]|$))|(?<=\()\$(?=/)")
_WORD = re.compile(r"[A-Za-z]{2,}")
_LATEX_MARK = re.compile(r"[\\^_{}=]")
_FULLWIDTH_DOLLAR = "\uff04"  # what split_formulas writes for them: not counted as a stray delimiter
_HTML_TAG = re.compile(
    r"</?(?:b|i|u|sup|sub|br|div|span|p|a|em|strong|img|table|thead|tbody|tr|td|th)\b[^>]*>", re.IGNORECASE)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_PREFIX = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+|[-*+][ \t]+|>[ \t]*)", re.MULTILINE)
_LATEX_COMMAND = re.compile(r"\\([A-Za-z]+)")
GREEK_NAMES = frozenset(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma tau upsilon phi chi psi omega".split())


def _latex_command(match: re.Match) -> str:
    name = match.group(1)
    bare = name[3:] if name.startswith("var") and name[3:].lower() in GREEK_NAMES else name
    return f" {bare} " if bare.lower() in GREEK_NAMES else " "


@dataclass(frozen=True)
class FormulaSpan:
    tex: str
    display: bool
    bbox: BBox  # the segment's box: inline formulas have no box of their own


Part = Union[str, FormulaSpan]


@dataclass
class Segment:
    label: str
    parts: list[Part]
    bbox: BBox

    @property
    def text(self) -> str:
        """The prose of the segment, formulas left out."""
        return "".join(p for p in self.parts if isinstance(p, str))


@dataclass
class TableSpan:
    cells: list[list[Part]]
    bbox: BBox


@dataclass
class PageMarkdown:
    page_no: int
    prose: list[Segment] = field(default_factory=list)
    tables: list[TableSpan] = field(default_factory=list)
    formulas: list[FormulaSpan] = field(default_factory=list)
    skip_boxes: list[BBox] = field(default_factory=list)
    stray_dollars: int = 0  # `$` left over after pairing: a formula that never closed


def _closes_a_formula(content: str, after: int) -> bool:
    """The next ``$`` after position ``after`` closes a span that is mathematics: it has LaTeX
    in it, or no word (``$2m+O(n)$``; the span between two prices is prose)."""
    close = content.find("$", after)
    return close != -1 and (_LATEX_MARK.search(content, after, close) is not None
                            or _WORD.search(content, after, close) is None)


def split_formulas(content: str, bbox: BBox, *, display_hint: bool = False) -> list[Part]:
    """``content`` as alternating text and formula parts (``$..$``, ``$$..$$``, ``\\(..\\)``, ``\\[..\\]``)."""
    content = content.replace("\\$", _FULLWIDTH_DOLLAR)  # an escaped dollar sign (MinerU's, a price or a SQL quote) is text
    content = _CURRENCY.sub(lambda m: m.group() if _closes_a_formula(content, m.end()) else _FULLWIDTH_DOLLAR, content)
    parts: list[Part] = []
    pos = 0
    for match in _FORMULA.finditer(content):
        if match.start() > pos:
            parts.append(content[pos:match.start()])
        tex = next(group for group in match.groups() if group is not None).strip()
        parts.append(FormulaSpan(tex, display_hint or match.lastgroup in ("dd", "br"), bbox))
        pos = match.end()
    if pos < len(content):
        parts.append(content[pos:])
    return parts


def prose_text(raw: str) -> str:
    """Markdown/HTML syntax removed from a prose string (images, links, tags, list markers)."""
    raw = _MD_IMAGE.sub(" ", raw)
    raw = _MD_LINK.sub(r"\1", raw)
    raw = _MD_PREFIX.sub("", raw)
    return _HTML_TAG.sub(" ", raw)


def _without_latex_commands(parts: list[Part]) -> list[Part]:
    """``parts`` with the LaTeX command names of its text removed. A pseudo-code block often
    comes back as LaTeX lines with no ``$`` around them (``\\text{foreach } t \\in R_i``): the
    command names are no text of the page, so they go, except a Greek letter's, which the
    page spells out."""
    return [_LATEX_COMMAND.sub(_latex_command, p) if isinstance(p, str) else p for p in parts]


def parse_page(page: dict, scale: float, page_no: Optional[int] = None) -> PageMarkdown:
    """Split ``page`` (``res.json["res"]``); ``scale`` converts its pixels to PDF points."""
    settings = page.get("model_settings") or {}
    ignored = set(settings.get("markdown_ignore_labels") or ()) | NO_TEXT_LABELS
    out = PageMarkdown(page_no if page_no is not None else int(page.get("page_index") or 0))

    def box(raw) -> BBox:
        return (raw[0] * scale, raw[1] * scale, raw[2] * scale, raw[3] * scale)

    blocks = page.get("parsing_res_list") or []
    union: dict[int, BBox] = {}
    for block in blocks:
        b = box(block["block_bbox"])
        key = block.get("group_id", block["block_id"])
        old = union.get(key)
        union[key] = b if old is None else (min(old[0], b[0]), min(old[1], b[1]), max(old[2], b[2]), max(old[3], b[3]))

    for block in blocks:
        label, content = block["block_label"], block.get("block_content") or ""
        if label in ignored:
            out.skip_boxes.append(box(block["block_bbox"]))
            continue
        if not content.strip():
            continue
        where = union[block.get("group_id", block["block_id"])]
        if label == "table" and content.lstrip().startswith("<"):
            out.tables.append(_parse_table(content, where))
        elif "formula" in label:
            parts = split_formulas(content, where, display_hint=True)
            if not any(isinstance(p, FormulaSpan) for p in parts):  # bare LaTeX without delimiters
                parts = [FormulaSpan(content.strip().strip("$").strip(), True, where)]
            out.prose.append(Segment(label, parts, where))
        else:
            out.prose.append(Segment(label, _without_latex_commands(split_formulas(prose_text(content), where)), where))

    for segment in out.prose:
        out.formulas += [p for p in segment.parts if isinstance(p, FormulaSpan)]
    for table in out.tables:
        out.formulas += [p for cell in table.cells for p in cell if isinstance(p, FormulaSpan)]
    for segment in out.prose:
        out.stray_dollars += sum(p.count("$") for p in segment.parts if isinstance(p, str))
    for table in out.tables:
        out.stray_dollars += sum(p.count("$") for cell in table.cells for p in cell if isinstance(p, str))
    return out


# A table cell the model wrote with a line break comes back with the two characters backslash and n in it
# ("0.058\n(0.001)"; PaddleOCR-VL does it for a cell of two lines), and a plus-minus as a bare LaTeX command
# ("84.7\pm1.0"). Neither is text of the page: the first is a line break (a LaTeX command that starts with n is not),
# the second the sign the page prints. Inside mathematics both are LaTeX and stay.
_LITERAL_NEWLINE = re.compile(
    r"(?<!\\)\\n(?!(?:u|abla|e|eq|eg|ot|otin|i|ewline|earrow|warrow|mid|leq|geq|less|gtr|subseteq|supseteq|rightarrow|"
    r"leftarrow|parallel|vdash|cong|exists)(?![A-Za-z]))")
_BARE_PLUS_MINUS = re.compile(r"(?<!\\)\\pm(?![A-Za-z])")


def clean_table_html(html: str, line_break: str = "<br>") -> str:
    """``html`` of a table with a literal backslash-n in a cell turned into ``line_break`` and a bare ``\\pm`` into ±,
    mathematics (``$..$``, ``\\(..\\)`` ...) left as it is. The extractor writes the result; the checks read it."""
    out, pos = [], 0
    for match in _FORMULA.finditer(html):
        out.append(_table_text(html[pos:match.start()], line_break))
        out.append(match.group())
        pos = match.end()
    out.append(_table_text(html[pos:], line_break))
    return "".join(out)


def _table_text(text: str, line_break: str) -> str:
    return _BARE_PLUS_MINUS.sub("±", _LITERAL_NEWLINE.sub(line_break, text))


def _parse_table(content: str, where: BBox) -> TableSpan:
    root = lxml_html.fromstring(f"<div>{clean_table_html(content, ' ')}</div>")
    cells = []
    for cell in root.iter("td", "th"):
        for line_break in cell.iter("br"):  # a line break is a space between the words it separates
            line_break.tail = " " + (line_break.tail or "")
        text = " ".join(cell.text_content().split())
        if text:
            cells.append(split_formulas(text, where))
    return TableSpan(cells, where)
