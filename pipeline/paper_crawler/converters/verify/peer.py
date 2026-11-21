"""A second engine's reading of a display formula, as the witness ``check_page`` asks for.

Where the text layer cannot spell a maths font (a lambda arrives as "ø", a Greek letter as an arrow), a display
formula is judged by a second reader of its box. PP-FormulaNet is one (verify/formulanet.py); the other engine's own
page is another: PaddleOCR-VL and MinerU read the same picture with nothing else in common, so two readings that
agree on the symbols are evidence against the layer, exactly as one PP-FormulaNet reading is.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

PEER_MIN_OVERLAP = 0.4  # intersection over union of the two engines' boxes for one formula

_OPENING = re.compile(r"^(?:\$\$|\\\[)\s*")
_CLOSING = re.compile(r"\s*(?:\$\$|\\\])$")


def _overlap(a: tuple, b: tuple) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def peer_witness(peer: dict, page_width: float) -> Callable[[tuple], Optional[str]]:
    """``witness(box)``: the LaTeX the other engine wrote for the display formula of its page that sits on ``box``
    (points), or None. ``peer`` is that engine's page in the pipeline's shape (``parsing_res_list`` of block
    dicts, boxes in the page image's pixels, ``width`` its pixel width); ``page_width`` the PDF page's, in points."""
    scale = page_width / peer["width"]
    formulas = []
    for block in peer["parsing_res_list"]:
        if "formula" not in block["block_label"]:
            continue
        tex = _CLOSING.sub("", _OPENING.sub("", block["block_content"].strip())).strip()
        formulas.append((tuple(v * scale for v in block["block_bbox"]), tex))

    def read(box: tuple) -> Optional[str]:
        best = max(formulas, key=lambda f: _overlap(box, f[0]), default=None)
        return best[1] if best is not None and _overlap(box, best[0]) >= PEER_MIN_OVERLAP else None

    return read
