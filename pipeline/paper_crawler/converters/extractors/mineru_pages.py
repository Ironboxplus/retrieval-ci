"""MinerU2.5-Pro as the second reader of the pages PaddleOCR-VL got wrong.

A page the first model failed twice is rendered and sent to a MinerU vLLM server through
``mineru-vl-utils`` (``MinerUClient(backend="http-client").two_step_extract``: a layout pass,
then one request per block). The blocks come back with a type, a bounding box (fractions of
the page, ``[x0, y0, x1, y1]``) and text, LaTeX or an HTML table. They are written out in the
shape of a PaddleOCR-VL page result (``parsing_res_list`` of labelled blocks, boxes in pixels
of the rendered page), which is what the checks and ``restructure_pages`` read, so a page of
either model is verified and merged by the same code and no second checker exists.

Coordinates: the page is rendered at ``dpi``; a block's pixel box is its fraction times the
image size, and ``verify.mdtext.parse_page`` converts pixels to PDF points with
``page width in points / image width`` exactly as for PaddleOCR-VL pages.

``mineru_vl_utils`` is imported on first use: it is a client of the server, which runs in its
own environment (it needs a newer vLLM than PaddleOCR-VL's).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

_logger = logging.getLogger(__name__)

RENDER_DPI = 200
MODEL_NAME = "OpenDataLab/MinerU2.5-Pro-2605-1.2B"

# MinerU block type -> the label PaddleOCR-VL gives the same kind of block
LABELS = {
    "text": "text", "title": "paragraph_title", "doc_title": "doc_title", "paragraph_title": "paragraph_title",
    "table": "table", "equation": "display_formula", "equation_block": "display_formula",
    "formula_number": "formula_number", "code": "algorithm", "algorithm": "algorithm",
    "aside_text": "aside_text", "ref_text": "reference_content", "index": "text", "phonetic": "text",
    "list_item": "text", "list": "text",
    "table_caption": "figure_title", "image_caption": "figure_title", "code_caption": "figure_title",
    "caption": "figure_title",
    "table_footnote": "vision_footnote", "image_footnote": "vision_footnote", "footnote": "vision_footnote",
    "header": "header", "footer": "footer", "page_number": "number", "page_footnote": "footnote",
    "image": "image", "chart": "image", "image_block": "image", "unknown": "text",
}
# the model_settings of a PaddleOCR-VL page: the pages of both models are merged together, and the merge
# takes the flags of the last page for all of them
MODEL_SETTINGS = {
    "use_doc_preprocessor": False, "use_layout_detection": True, "use_chart_recognition": False,
    "use_seal_recognition": False, "use_ocr_for_image_block": False, "format_block_content": False,
    "merge_layout_blocks": True, "return_layout_polygon_points": True,
}
# what the PaddleOCR-VL pipeline leaves out of the Markdown (markdown_ignore_labels of its config)
IGNORED = ("number", "footnote", "header", "header_image", "footer", "footer_image", "aside_text")


def blocks_to_page(blocks: list, width: int, height: int, page_index: int) -> tuple[dict, str]:
    """A page result in PaddleOCR-VL's shape, and the page's Markdown text (blocks that the
    pipeline would not write left out), from MinerU's content blocks for an image of
    ``width`` x ``height`` pixels."""
    parsing = []
    for order, block in enumerate(blocks):
        x0, y0, x1, y1 = block["bbox"]
        label = LABELS.get(block["type"], "text")
        content = block.get("content") or ""
        if label in ("image",) and not content:
            content = ""
        parsing.append({
            "block_id": order, "group_id": order, "block_label": label,
            "block_bbox": [int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)],
            "block_content": content,
        })
    data = {
        "page_index": page_index, "width": width, "height": height,
        "model_settings": {**MODEL_SETTINGS, "markdown_ignore_labels": list(IGNORED)},
        "parsing_res_list": parsing,
        # restructure_pages reads it of every page (and of nothing else a first model's pipeline adds)
        "layout_det_res": {"input_path": "", "page_index": page_index,
                           "boxes": [{"label": b["block_label"], "coordinate": b["block_bbox"]} for b in parsing]},
    }
    markdown = "\n\n".join(b["block_content"] for b in parsing if b["block_label"] not in IGNORED and b["block_content"])
    return data, markdown


def render_page(pdf: Path, page_no: int, dpi: int = RENDER_DPI):
    """Page ``page_no`` (0-based) of ``pdf`` as an RGB PIL image."""
    import pymupdf
    from PIL import Image

    with pymupdf.open(pdf) as document:
        pixmap = document[page_no].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


class MineruReader:
    """``redo(pdf, page numbers)`` for ``PaddleOCRVLExtractor.redo_failed_pages``."""

    name = "mineru"

    def __init__(self, server_url: str, *, model_name: Optional[str] = None, dpi: int = RENDER_DPI,
                 max_concurrency: int = 16, on_progress=None):
        self.server_url, self.model_name, self.dpi = server_url, model_name, dpi
        self.max_concurrency = max_concurrency
        self.on_progress = on_progress or (lambda: None)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from mineru_vl_utils import MinerUClient

            self._client = MinerUClient(backend="http-client", server_url=self.server_url,
                                        model_name=self.model_name, max_concurrency=self.max_concurrency,
                                        use_tqdm=False)
        return self._client

    def redo(self, pdf: Path, numbers: List[int]) -> Dict[int, tuple[dict, str]]:
        images = [render_page(pdf, n, self.dpi) for n in numbers]
        client = self._get_client()
        results = client.concurrent_two_step_extract(images)
        out: Dict[int, tuple[dict, str]] = {}
        for number, image, blocks in zip(numbers, images, results):
            out[number] = blocks_to_page(list(blocks), image.width, image.height, number)
            self.on_progress()
        return out

    def close(self) -> None:
        self._client = None
