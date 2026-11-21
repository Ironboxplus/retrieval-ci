"""arXiv LaTeXML HTML -> structured document JSON -> Markdown.

The JSON is the lossless-enough canonical form kept on disk: section tree,
typed blocks carrying LaTeXML anchors (``S3.SS2.p1``) so a hit can deep-link
to ``arxiv.org/html/<id>#<anchor>``, and the bibliography kept apart from the
body. The Markdown is derived from it for the existing preindex path
(``SectionAwareChunker`` splits on headings, so heading text becomes each
chunk's ``section_name``); the bibliography and page chrome are left out so
they never reach the index.

Math keeps its LaTeX source (``alttext``): ``$...$`` inline, ``$$...$$`` for
display equations.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

SCHEMA = "arxiv-html-doc/1"
FLOATS_SECTION = "_floats"

_SECTION_CLASSES = {
    "ltx_section": 1, "ltx_appendix": 1, "ltx_chapter": 1, "ltx_part": 1,
    "ltx_subsection": 2, "ltx_subsubsection": 3, "ltx_paragraph": 4, "ltx_subparagraph": 5,
}
_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)\]])")


@dataclass
class Section:
    id: str
    level: int
    number: str
    title: str
    appendix: bool = False


@dataclass
class Block:
    id: str
    section: Optional[str]
    kind: str  # paragraph | equation | figure | table | list | theorem | proof | code | footnote
    text: str
    caption: Optional[str] = None
    label: Optional[str] = None


@dataclass
class Reference:
    id: str
    tag: str
    text: str


@dataclass
class ArxivHtmlDoc:
    arxiv_id: str
    version: Optional[str]
    title: str
    abstract: str
    keywords: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    schema: str = SCHEMA

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "ArxivHtmlDoc":
        return cls(
            arxiv_id=data["arxiv_id"], version=data.get("version"), title=data["title"],
            abstract=data.get("abstract", ""), keywords=list(data.get("keywords") or []),
            sections=[Section(**s) for s in data.get("sections", [])],
            blocks=[Block(**b) for b in data.get("blocks", [])],
            references=[Reference(**r) for r in data.get("references", [])],
            schema=data.get("schema", SCHEMA),
        )


class NotLatexmlDocument(ValueError):
    """The page has no LaTeXML article (error page or conversion failure)."""


def _norm(text: str) -> str:
    text = _WS.sub(" ", text.replace("\xa0", " ")).strip()
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


def _classes(node: Tag) -> set[str]:
    return set(node.get("class") or [])


# MathML is most of a LaTeXML page's bytes and nearly all of its parse time,
# yet only the ``alttext`` LaTeX is kept. Each <math> is swapped for a tiny
# placeholder before parsing; <math> never nests, so a lazy match is exact.
_MATH_RE = re.compile(r"<math\b([^>]*)>.*?</math>", re.DOTALL)
_ALTTEXT_RE = re.compile(r'\balttext="([^"]*)"')
_DISPLAY_RE = re.compile(r'\bdisplay="block"')


def _shrink_math(html: str) -> str:
    def repl(match: re.Match) -> str:
        attrs = match.group(1)
        alt = _ALTTEXT_RE.search(attrs)
        if alt is None:
            return match.group(0)
        display = "block" if _DISPLAY_RE.search(attrs) else "inline"
        return f'<span class="ltx_mathtex" data-display="{display}" data-tex="{alt.group(1)}"></span>'
    return _MATH_RE.sub(repl, html)


def _is_math(node: Tag) -> bool:
    return node.name == "math" or "ltx_mathtex" in _classes(node)


def _math_tex(node: Tag) -> str:
    if node.name != "math":
        tex, block = node.get("data-tex", ""), node.get("data-display") == "block"
    else:
        tex = node.get("alttext")
        if tex is None:
            ann = node.find("annotation", attrs={"encoding": "application/x-tex"})
            tex = ann.get_text() if ann else node.get_text(" ")
        block = node.get("display") == "block"
    tex = _norm(tex)
    return f"$${tex}$$" if block else f"${tex}$"


def _foreign_objects(svg: Tag) -> Iterable[Tag]:
    """Outermost foreignObject elements under an svg (lxml lowercases the name)."""
    for child in svg.children:
        if isinstance(child, Tag):
            if child.name in {"foreignobject", "foreignObject"}:
                yield child
            else:
                yield from _foreign_objects(child)


def _inline(node: Tag | NavigableString, footnotes: list[str] | None = None) -> str:
    """Flatten inline content to text; math -> LaTeX, footnotes lifted out."""
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    name = node.name
    cls = _classes(node)
    if _is_math(node):
        return f" {_math_tex(node)} "
    if name == "svg":
        # tcolorbox prompt boxes and TikZ labels: the text sits in foreignObject.
        return " ".join(_inline(fo, footnotes) for fo in _foreign_objects(node))
    if name in {"annotation", "annotation-xml", "script", "style", "img"}:
        return ""
    if "ltx_note" in cls:
        content = node.find(class_="ltx_note_content")
        if footnotes is not None and content is not None:
            for mark in content.find_all(class_="ltx_note_mark"):
                mark.decompose()
            text = _norm(_inline(content))
            if text:
                footnotes.append(text)
        return ""
    if "ltx_note_mark" in cls or "ltx_tag_note" in cls:
        return ""
    # ``ltx_break`` also marks framed boxes that start on a new line; only an
    # empty one is a bare line break.
    if name == "br" or ("ltx_break" in cls and not node.get_text(strip=True) and node.find(_is_math) is None):
        return " "
    parts = [_inline(child, footnotes) for child in node.children]
    text = "".join(parts)
    if name in {"p", "div", "li", "td", "th", "tr", "figcaption", "table"}:
        return f" {text} "
    return text


def _table_markdown(table: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue  # nested tabular: flattened into its parent cell's text
        cells = [
            _norm(_inline(td)).replace("|", "\\|")
            for td in tr.find_all(["td", "th"], recursive=False)
        ]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _caption(node: Tag) -> tuple[Optional[str], Optional[str]]:
    """The figure's own caption, removed from the tree so walking the figure's
    content does not read it again. A subfigure's caption stays with the
    subfigure (a recursive find would return the first subfigure's)."""
    cap = node.find("figcaption", recursive=False)
    if cap is None:
        cap = next((c for c in node.find_all("figcaption") if c.find_parent("figure") is node), None)
    if cap is None:
        return None, None
    cap.extract()
    tag = cap.find(class_="ltx_tag")
    label = _norm(tag.get_text(" ")).rstrip(":") if tag else None
    if tag:
        tag.extract()
    text = _norm(_inline(cap))
    return label, text or None


def _owner(node: Tag) -> Optional[Tag]:
    """Nearest enclosing figure or section: the float a tabular belongs to."""
    return node.find_parent(["figure", "section"])


def _heading(sec: Tag) -> tuple[str, str]:
    head = sec.find(["h1", "h2", "h3", "h4", "h5", "h6"], class_="ltx_title", recursive=False)
    if head is None:
        return "", ""
    tag = head.find(class_="ltx_tag")
    number = _norm(tag.get_text(" ")) if tag else ""
    if tag:
        tag.extract()
    return number, _norm(_inline(head))


class _Walker:
    def __init__(self) -> None:
        self.sections: list[Section] = []
        self.blocks: list[Block] = []
        self._auto = 0
        self._trailing_floats = False

    def _id(self, node: Tag, prefix: str) -> str:
        if node.get("id"):
            return node["id"]
        self._auto += 1
        return f"{prefix}{self._auto}"

    def _add(self, node: Tag, section: Optional[str], kind: str, text: str,
             caption: Optional[str] = None, label: Optional[str] = None) -> None:
        text = text.strip()
        if text or caption:
            self.blocks.append(Block(self._id(node, "b"), self._home(section), kind, text, caption, label))

    def _home(self, section: Optional[str]) -> Optional[str]:
        # LaTeXML parks some floats at the end of the article, outside every
        # section; after the body they belong to a trailing floats section,
        # not in front of the introduction.
        if section is None and self.sections:
            self._trailing_floats = True
            return FLOATS_SECTION
        return section

    def finish(self) -> None:
        if self._trailing_floats:
            self.sections.append(Section(FLOATS_SECTION, 1, "", "Figures and tables"))

    def _paragraph(self, node: Tag, section: Optional[str]) -> None:
        footnotes: list[str] = []
        # Display equations and lists nested in a paragraph become their own blocks
        # so the equation keeps its $$ form and list items keep their bullets.
        text_parts: list[str] = []
        for child in node.children:
            if isinstance(child, Tag) and ({"ltx_equation", "ltx_equationgroup"} & _classes(child)):
                self._flush(node, section, text_parts, footnotes)
                self._equation(child, section)
            elif isinstance(child, Tag) and child.name in {"ul", "ol"}:
                self._flush(node, section, text_parts, footnotes)
                self._list(child, section)
            else:
                text_parts.append(_inline(child, footnotes))
        self._flush(node, section, text_parts, footnotes)

    def _flush(self, node: Tag, section: Optional[str], parts: list[str], footnotes: list[str]) -> None:
        text = _norm("".join(parts))
        parts.clear()
        if text:
            self._add(node, section, "paragraph", text)
        for note in footnotes:
            self.blocks.append(Block(self._id(node, "fn") + f".fn{len(self.blocks)}", self._home(section), "footnote", note))
        footnotes.clear()

    def _equation(self, node: Tag, section: Optional[str]) -> None:
        tags = [_norm(_inline(t)) for t in node.find_all(class_="ltx_tag_equation")]
        for t in node.find_all(class_="ltx_tag_equation"):
            t.decompose()  # the label is rendered once, after the equation
        maths = [_math_tex(m) for m in node.find_all(_is_math)]
        body = " ".join(m if m.startswith("$$") else f"$${m.strip('$')}$$" for m in maths)
        self._add(node, section, "equation", body, label=tags[0] if tags else None)

    def _list(self, node: Tag, section: Optional[str]) -> None:
        items = []
        for li in node.find_all("li", recursive=False):
            tag = li.find(class_="ltx_tag", recursive=False)
            if tag:
                tag.extract()
            item = _norm(_inline(li))
            if item:
                items.append(f"- {item}")
        self._add(node, section, "list", "\n".join(items))

    def walk(self, node: Tag, section: Optional[str], level: int) -> None:
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            cls = _classes(child)
            sec_cls = cls & _SECTION_CLASSES.keys()
            if child.name == "section" and "ltx_bibliography" in cls:
                continue
            if child.name == "section" and "ltx_glossary" in cls:
                _, title = _heading(child)
                sid = self._id(child, "sec")
                self.sections.append(Section(sid, 1, "", title or "Glossary"))
                self.walk(child, sid, 1)
                continue
            if sec_cls and child.name in {"section", "div"}:
                number, title = _heading(child)
                sec_level = min(_SECTION_CLASSES[c] for c in sec_cls)
                sid = self._id(child, "sec")
                self.sections.append(Section(sid, sec_level, number, title, "ltx_appendix" in cls))
                self.walk(child, sid, sec_level)
            elif "ltx_para" in cls:
                self._paragraph(child, section)
            elif {"ltx_equation", "ltx_equationgroup"} & cls:
                self._equation(child, section)
            elif child.name == "figure":
                # The float's own tabulars form its body; everything else in it
                # (text boxes, listings, subfigures, and whole sections when an
                # unclosed LaTeX float swallowed the rest of the paper) is walked.
                label, caption = _caption(child)
                tables = [t for t in child.find_all("table", class_="ltx_tabular")
                          if _owner(t) is child and t.find_parent("table", class_="ltx_tabular") is None]
                body = "\n\n".join(filter(None, (_table_markdown(t) for t in tables)))
                self._add(child, section, "table" if "ltx_table" in cls else "figure", body, caption, label)
                self.walk(child, section, level)
            elif child.name == "table" and "ltx_tabular" in cls:
                if _owner(child) is None or _owner(child).name == "section":
                    self._add(child, section, "table", _table_markdown(child))
            elif child.name == "p" and "ltx_p" in cls:
                self._paragraph(child, section)
            elif child.name == "svg":
                # A prompt box drawn as a picture outside any paragraph; bare
                # diagram labels (a few words) stay out.
                text = _norm(_inline(child))
                if len(text.split()) >= 8:
                    self._add(child, section, "paragraph", text)
            elif {"ltx_theorem", "ltx_proof"} & cls:
                kind = "proof" if "ltx_proof" in cls else "theorem"
                head = child.find(class_="ltx_title", recursive=False)
                heading = _norm(_inline(head)) if head else ""
                if head:
                    head.extract()
                start = len(self.blocks)
                self.walk(child, section, level)
                if heading and len(self.blocks) > start and self.blocks[start].kind == "paragraph":
                    self.blocks[start].kind = kind
                    self.blocks[start].text = f"{heading} {self.blocks[start].text}"
                elif heading:
                    self._add(child, section, kind, heading)
            elif "ltx_listing" in cls:
                lines = [_norm(_inline(line)) for line in child.find_all(class_="ltx_listingline")]
                self._add(child, section, "code", "\n".join(lines))
            elif child.name in {"ul", "ol"}:
                self._list(child, section)
            elif child.name == "dl":
                items, term = [], ""
                for node in child.find_all(["dt", "dd"], recursive=False):
                    text = _norm(_inline(node))
                    if node.name == "dt":
                        term = text
                    elif text:
                        items.append(f"- {term}: {text}" if term else f"- {text}")
                        term = ""
                self._add(child, section, "list", "\n".join(items))
            elif child.name in {"h1", "h2", "h3", "h4", "h5", "h6"} or cls & {"ltx_authors", "ltx_abstract",
                                                                              "ltx_keywords", "ltx_pagination",
                                                                              "ltx_classification", "ltx_dates"}:
                continue
            elif child.name in {"div", "span", "p"}:
                self.walk(child, section, level)


def parse_arxiv_html(html: str | bytes, arxiv_id: str) -> ArxivHtmlDoc:
    text = html if isinstance(html, str) else html.decode("utf-8", "replace")
    soup = BeautifulSoup(_shrink_math(text), "lxml")
    article = soup.select_one("article.ltx_document")
    if article is None:
        raise NotLatexmlDocument(f"{arxiv_id}: no LaTeXML article element")

    title_node = article.find(class_="ltx_title_document")
    title = ""
    if title_node is not None:
        for note in title_node.find_all(class_="ltx_note"):
            note.decompose()
        title = _norm(_inline(title_node))

    abstract_parts = []
    for ab in article.find_all(class_="ltx_abstract"):
        head = ab.find(class_="ltx_title_abstract")
        if head:
            head.extract()
        abstract_parts.append(_norm(_inline(ab)))
    keywords_node = article.find(class_="ltx_keywords")
    keywords = []
    if keywords_node is not None:
        head = keywords_node.find(class_="ltx_title")
        if head:
            head.extract()
        keywords = [k.strip() for k in re.split(r"[;,·]", _norm(_inline(keywords_node))) if k.strip()]

    references = []
    for item in article.select("section.ltx_bibliography li.ltx_bibitem"):
        tag = item.find(class_="ltx_tag")
        tag_text = _norm(tag.get_text(" ")) if tag else ""
        if tag:
            tag.extract()
        references.append(Reference(item.get("id", ""), tag_text, _norm(_inline(item))))

    walker = _Walker()
    walker.walk(article, None, 0)
    walker.finish()

    version = None
    match = re.search(re.escape(arxiv_id) + r"v(\d+)", text)
    if match:
        version = f"v{match.group(1)}"
    return ArxivHtmlDoc(
        arxiv_id=arxiv_id, version=version, title=title, abstract=" ".join(p for p in abstract_parts if p),
        keywords=keywords, sections=walker.sections, blocks=walker.blocks, references=references,
    )


def _section_heading(sec: Section) -> str:
    hashes = "#" * min(sec.level + 1, 6)
    label = f"{sec.number} {sec.title}".strip() if sec.number else sec.title
    if sec.appendix and sec.number and not label.lower().startswith("appendix"):
        label = f"Appendix {label}"
    return f"{hashes} {label or 'Untitled section'}"


def _render_block(block: Block) -> str:
    if block.kind in {"figure", "table"}:
        name = block.label or ("Figure" if block.kind == "figure" else "Table")
        head = f"**{name}:** {block.caption}" if block.caption else f"**{name}**"
        return f"{head}\n\n{block.text}" if block.text else head
    if block.kind == "equation":
        return f"{block.text} {block.label}".strip() if block.label else block.text
    if block.kind == "code":
        return f"```\n{block.text}\n```"
    if block.kind == "footnote":
        return f"> Footnote: {block.text}"
    return block.text


def to_markdown(doc: ArxivHtmlDoc) -> str:
    """Markdown for the existing heading-aware chunker; bibliography excluded."""
    out: list[str] = [f"# {doc.title or doc.arxiv_id}"]
    if doc.abstract:
        out += ["## Abstract", doc.abstract]
    if doc.keywords:
        out.append("Keywords: " + ", ".join(doc.keywords))
    by_section: dict[Optional[str], list[Block]] = {}
    for block in doc.blocks:
        by_section.setdefault(block.section, []).append(block)
    for block in by_section.get(None, []):
        out.append(_render_block(block))
    for sec in doc.sections:
        out.append(_section_heading(sec))
        out += [_render_block(b) for b in by_section.get(sec.id, [])]
    return "\n\n".join(part for part in out if part).strip() + "\n"


def iter_stats(docs: Iterable[ArxivHtmlDoc]) -> dict[str, int]:
    stats = {"docs": 0, "sections": 0, "blocks": 0, "references": 0, "equations": 0,
             "tables": 0, "figures": 0, "empty_body": 0}
    for doc in docs:
        stats["docs"] += 1
        stats["sections"] += len(doc.sections)
        stats["blocks"] += len(doc.blocks)
        stats["references"] += len(doc.references)
        stats["equations"] += sum(b.kind == "equation" for b in doc.blocks)
        stats["tables"] += sum(b.kind == "table" for b in doc.blocks)
        stats["figures"] += sum(b.kind == "figure" for b in doc.blocks)
        stats["empty_body"] += not any(b.kind == "paragraph" for b in doc.blocks)
    return stats
