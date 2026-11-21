"""Text normalisation shared by every comparison between a Markdown page and its reference.

``normalize_text`` is copied from olmOCR's benchmark (the same normalisation its
``TextPresenceTest`` applies before ``rapidfuzz.fuzz.partial_ratio``):
https://github.com/allenai/olmocr/blob/ab294c6ab4471f452665484082e01841ca77f263/olmocr/bench/tests.py
(Apache-2.0, ``normalize_text``, commit ab294c6 of 2025-11-19). ``comparable`` stacks NFKC,
case folding and whitespace removal on top so ligatures, mathematical alphanumerics
(``𝑘`` -> ``k``), line breaks and tokenisation differences cannot count as a mismatch.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

_WHITESPACE = re.compile(r"\s+")


def normalize_text(md_content: Optional[str]) -> Optional[str]:
    if md_content is None:
        return None

    # Normalize <br> and <br/> to newlines
    md_content = re.sub(r"<br/?>", " ", md_content)

    # Remove markdown bold formatting (** or __ for bold)
    md_content = re.sub(r"\*\*(.*?)\*\*", r"\1", md_content)
    md_content = re.sub(r"__(.*?)__", r"\1", md_content)
    md_content = re.sub(r"</?b>", "", md_content)  # Remove <b> tags if they exist
    md_content = re.sub(r"</?i>", "", md_content)  # Remove <i> tags if they exist

    # Remove markdown italics formatting (* or _ for italics)
    # Logic: The dot (.) in regex matches any character EXCEPT a newline.
    # This automatically prevents matching **start \n\n end**.
    # We use group \1 to ensure we match matching pairs (**...** or __...__).
    md_content = re.sub(r"(\*\*|__)(.*?)\1", r"\2", md_content)  # Bold
    md_content = re.sub(r"(\*|_)(.*?)\1", r"\2", md_content)  # Italics

    # Normalize whitespace in the md_content
    md_content = re.sub(r"\s+", " ", md_content)

    # Convert down to a consistent unicode form, so é == e + accent, unicode forms
    md_content = unicodedata.normalize("NFC", md_content)

    # Dictionary of characters to replace: keys are fancy characters, values are ASCII equivalents, unicode micro with greek mu comes up often enough too
    replacements = {"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "＿": "_", "–": "-", "—": "-", "‑": "-", "‒": "-", "−": "-", "µ": "μ"}

    # Apply all replacements from the dictionary
    for fancy_char, ascii_char in replacements.items():
        md_content = md_content.replace(fancy_char, ascii_char)

    return md_content


_SPELLED = {"lamda": "lambda"}


def greek_name(char: str) -> Optional[str]:
    """``gamma`` for ``γ`` or ``Γ``, None for anything that is not a Greek letter."""
    name = unicodedata.name(char, "")
    if not name.startswith("GREEK") or " LETTER " not in name:
        return None
    last = name.rsplit(" ", 1)[1].lower()
    return _SPELLED.get(last, last)


def spell_greek(text: str) -> str:
    """``text`` with every Greek letter written out by name. Some PDF fonts reach the text
    layer as the glyph's name (``Exp(init, gamma)``) and others as the character, so only the
    spelled form of both sides agrees. Expects NFKC, case-folded text."""
    return "".join(greek_name(c) or c for c in text) if any("\u0370" <= c <= "\u03ff" for c in text) else text


def repair_math(text: str) -> str:
    """``text`` with the mathematical alphanumerics some text layers lose the top bit of put
    back: U+1D456 (math italic i) arrives as U+D456, a Hangul syllable that no paper in
    English uses. The result is meant for NFKC, which turns them into plain letters."""
    if not any("\ud400" <= c <= "\ud7ff" for c in text):
        return text
    return "".join(chr(ord(c) + 0x10000) if "\ud400" <= c <= "\ud7ff" else c for c in text)


_FOLDED = str.maketrans({
    "\u2206": "\u03b4",  # INCREMENT: the Delta of Computer Modern's maths fonts, which KaTeX writes as U+0394
    "\u0131": "i",       # dotless i and j: the letters under a hat or a tilde
    "\u0237": "j",
})


def fold_letters(text: str) -> str:
    """``text`` with the differences between a layer's spelling of a letter and the Markdown's taken out: a letter
    with an accent is the letter (the layer writes "ö" as o + a diaeresis or as ö, the Markdown whichever the model
    wrote), a dotless i or j is i or j, U+2206 is the Greek Delta, and a spacing accent glyph (ˆ ˇ ˜ ´ ` ¨ ˙, which a
    layer sets as a character of its own) is dropped. A combining mark on anything but a letter stays: the dot over
    the second 3 of 0.3̇3 is part of that figure. Expects NFKC, case-folded text."""
    text = text.translate(_FOLDED)
    if text.isascii():
        return text
    kept: list[str] = []
    for char in unicodedata.normalize("NFKD", text):
        category = unicodedata.category(char)
        if category in ("Lm", "Sk") or (category == "Mn" and kept and kept[-1].isalpha()):
            continue
        kept.append(char)
    return unicodedata.normalize("NFC", "".join(kept))


def comparable(text: str) -> str:
    """The form two texts are compared in: normalised, NFKC, case-folded, accents folded, Greek spelled out,
    no whitespace."""
    return spell_greek(fold_letters(_WHITESPACE.sub("", unicodedata.normalize("NFKC", repair_math(normalize_text(text)))).casefold()))


def words_of(text: str) -> list[str]:
    """Whitespace-separated words of ``text`` in comparable form (empty ones dropped)."""
    return [w for w in (comparable(token) for token in text.split()) if w]
