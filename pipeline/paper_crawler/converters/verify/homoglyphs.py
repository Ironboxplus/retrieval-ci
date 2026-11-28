"""Symbols a reader cannot tell apart: what the model wrote versus what the PDF's text layer holds.

A maths font reaches the text layer as whatever code point the font's author mapped the glyph to:
a Greek nu as a Latin ``v``, a Delta as the white triangle, an epsilon as the open e ``ɛ``. A formula
symbol that is a look-alike of the layer's is the same symbol to a reader, so it is accepted.

The table is Unicode's own confusables data (the ``confusable-homoglyphs`` package) plus the pairs of
mathematics it lacks (``MATH_PAIRS``). Symbols are spelled the way ``checks._symbols`` spells them: a
Latin letter as itself, a Greek letter as its name (``nu``); the look-alikes come back in the form
``normalize.comparable`` gives a text layer: lower case, NFKC, Greek spelled out.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache

from confusable_homoglyphs import confusables

from .normalize import spell_greek

# Pairs Unicode's table does not have and mathematics needs (either direction). Case-insensitive.
MATH_PAIRS = (
    ("τ", "t"), ("ω", "w"), ("κ", "k"), ("χ", "x"), ("μ", "u"),
    ("ε", "e"), ("ε", "ɛ"), ("ε", "∈"), ("ϵ", "e"),
    ("∅", "0"), ("∅", "ø"),
    ("l", "1"), ("l", "|"), ("1", "|"),
)


def _form(text: str) -> str:
    """``text`` as a text layer is compared: NFKC, case-folded, Greek spelled out."""
    return spell_greek(unicodedata.normalize("NFKC", text).casefold())


@lru_cache(maxsize=None)
def _by_char() -> dict[str, frozenset[str]]:
    table: dict[str, set[str]] = {}
    for a, b in MATH_PAIRS:
        table.setdefault(a, set()).add(b)
        table.setdefault(b, set()).add(a)
    return {c: frozenset(v) for c, v in table.items()}


# Greek capitals drawn exactly like a Latin capital: a Beta is a B on the page, so only the small letter stands
# for the name (a beta is not a b); the capitals that are not a Latin letter (Delta, Sigma...) are looked up too
_LATIN_CAPITALS = frozenset("ΑΒΕΖΗΙΚΜΝΟΡΤΥΧ")


def _characters(symbol: str) -> set[str]:
    """The code points a symbol is written with: a Latin letter in both cases, a Greek letter by name."""
    if len(symbol) == 1:
        return {symbol, symbol.upper()}
    out = set()
    for case in ("SMALL", "CAPITAL"):
        for name in (symbol, "lamda"):  # Unicode spells lambda without the b
            try:
                char = unicodedata.lookup(f"GREEK {case} LETTER {name.upper()}")
            except KeyError:
                continue
            if char not in _LATIN_CAPITALS:
                out.add(char)
    return out


@lru_cache(maxsize=None)
def lookalikes(symbol: str) -> frozenset[str]:
    """The other symbols (as text-layer forms) a reader takes ``symbol`` for."""
    found: set[str] = set()
    extra = _by_char()
    for char in _characters(symbol):
        for entry in confusables.is_confusable(char, greedy=True) or ():
            found.update(h["c"] for h in entry["homoglyphs"])
        found.update(extra.get(char, ()))
        found.update(extra.get(char.casefold(), ()))
    forms = {_form(h) for h in found}
    forms.discard(symbol)
    forms.discard("")
    return frozenset(forms)


def present(symbol: str, have: set[str], text: str) -> bool:
    """``symbol`` or a look-alike of it is in ``text`` (``have`` is its set of characters)."""
    return any((f in have) if len(f) == 1 else (f in text) for f in lookalikes(symbol))
