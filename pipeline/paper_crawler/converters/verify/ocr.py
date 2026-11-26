"""A text reference for pages that have no text layer: PP-OCRv5 (PaddleOCR), not a generative model.

Detection plus recognition can drop or misread a word but does not invent text, which is
what makes it usable as a witness for a page the VLM read from pixels. The model is
loaded on first use, so importing this module needs neither paddle nor a GPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .textlayer import PageRef

RENDER_DPI = 200
# On the host's cores, not the card: the card is shared with the model server and the layout model, the
# reference is only asked for pages with no text layer, and a lazily loaded GPU model took the last
# 400 MiB of it in the middle of a run (task 1302).
OCR_DEVICE = {"device": "cpu", "cpu_threads": 4}


def render_page_bgr(pdf: Path, page_no: int, dpi: int = RENDER_DPI):
    """Page ``page_no`` (0-based) of ``pdf`` as a BGR uint8 array, and its size in PDF points."""
    import numpy as np
    import pymupdf

    with pymupdf.open(pdf) as document:
        page = document[page_no]
        pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        size = (page.rect.width, page.rect.height)
    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
    return rgb[:, :, ::-1].copy(), size


class OcrReference:
    def __init__(self, dpi: int = RENDER_DPI, **ocr_options):
        self.dpi = dpi
        self._options = {"use_doc_orientation_classify": False, "use_doc_unwarping": False,
                         "use_textline_orientation": False, **OCR_DEVICE, **ocr_options}
        self._ocr: Optional[object] = None

    def page_reference(self, pdf: Path, page_no: int) -> PageRef:
        image, (width, height) = render_page_bgr(pdf, page_no, self.dpi)
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(**self._options)
        result = self._ocr.predict(image)[0]
        to_points = 72.0 / self.dpi
        lines = [(text, (float(x0) * to_points, float(y0) * to_points, float(x1) * to_points, float(y1) * to_points))
                 for text, (x0, y0, x1, y1) in zip(result["rec_texts"], result["rec_boxes"])]
        return PageRef.from_lines(page_no, width, height, lines, source="ocr")
