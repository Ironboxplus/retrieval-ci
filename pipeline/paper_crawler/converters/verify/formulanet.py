"""A second reader for display formulas: PP-FormulaNet (PaddleX), not the model that wrote the Markdown.

The text layer is the witness for a formula's symbols, but a maths font without a usable
ToUnicode map reaches it as stray characters (a Greek alpha as "u", a bar as "~", a sum as "o").
For such a page the check asks this model to read the formula's box and accepts the Markdown's
formula when the two readings have the same symbols (``checks._witness_agrees``). The model
is loaded on first use, so importing this module needs neither paddle nor a GPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .ocr import render_page_bgr
from .textlayer import BBox

MODEL_NAME = "PP-FormulaNet_plus-L"
RENDER_DPI = 200
PAD_POINTS = 3.0  # a margin around the box: the layout box is tight to the glyphs


class FormulaReader:
    def __init__(self, model_name: str = MODEL_NAME, dpi: int = RENDER_DPI, **model_options):
        self.model_name = model_name
        self.dpi = dpi
        # On the CPU a reading takes ~20 s of the host's cores (PP-FormulaNet_plus-L on a Haswell i5), a
        # second when it has the card; but next to the layout model and the model server there is no room
        # on the card for it (the layout step failed after its first batch), so a worker that serves a
        # model gives it the card only after the server is gone.
        self._options = {"device": "cpu", "cpu_threads": 2, **model_options}  # ``device="gpu:0"`` for the card
        self._model: Optional[object] = None
        self._read: dict[tuple, Optional[str]] = {}  # a page is judged again after a retry: the box is read once

    def release(self) -> None:
        """Give the card's memory back (the model is loaded again by the next reading)."""
        self._model = None
        import gc

        gc.collect()
        try:
            import paddle

            paddle.device.cuda.empty_cache()
        except Exception:  # a CPU-only build: nothing was held on a card
            pass

    def latex(self, pdf: Path, page_no: int, box: BBox) -> Optional[str]:
        """The LaTeX the model reads in ``box`` (PDF points) of page ``page_no`` (0-based) of ``pdf``."""
        key = (str(pdf), page_no, tuple(round(v) for v in box))
        if key not in self._read:
            self._read[key] = self._latex(pdf, page_no, box)
        return self._read[key]

    def _latex(self, pdf: Path, page_no: int, box: BBox) -> Optional[str]:
        image, (width, height) = render_page_bgr(pdf, page_no, self.dpi)
        scale = self.dpi / 72.0
        x0, y0, x1, y1 = box
        left, top = max(int((x0 - PAD_POINTS) * scale), 0), max(int((y0 - PAD_POINTS) * scale), 0)
        right, bottom = min(int((x1 + PAD_POINTS) * scale), image.shape[1]), min(int((y1 + PAD_POINTS) * scale), image.shape[0])
        if right - left < 8 or bottom - top < 8:
            return None
        if self._model is None:
            from paddlex import create_model

            self._model = create_model(self.model_name, **self._options)
        result = next(iter(self._model.predict(image[top:bottom, left:right])))
        formula = result["rec_formula"] if "rec_formula" in result else None
        return formula or None
