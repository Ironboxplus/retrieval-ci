"""The quality report of one converted paper: every page's verdict, retries and engine.

It is written next to the Markdown (``_quality/<paper>.json``), also when the paper
failed, so a gate or a person can see which pages were refused and why without
re-running anything. ``status`` and ``failure_reasons`` are what the local extract gate
reads (``_quality_failure_summary`` in app/services/pipeline/stages.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .checks import PageVerdict

SCHEMA = 1


@dataclass
class PaperReport:
    paper: str
    pdf: str
    engine: dict = field(default_factory=dict)
    attempts: dict[int, list[dict]] = field(default_factory=dict)  # page_no -> verdicts, oldest first
    error: Optional[str] = None  # a failure that is not about one page (server hang, no pages)
    seconds: float = 0.0

    def record(self, verdict: PageVerdict, params: Optional[dict] = None, blocks: Optional[list] = None,
               engine: str = "paddleocr_vl") -> None:
        """``blocks`` is what the model wrote for a refused page (label, box, content), kept so
        a person can see the defect without converting the page again. ``engine`` is the model
        that wrote this attempt."""
        entry = {"engine": engine, "status": verdict.status, "reasons": verdict.reasons, "soft": verdict.soft,
                 "metrics": verdict.metrics, "sampling": params or {}}
        if blocks is not None:
            entry["blocks"] = blocks
        self.attempts.setdefault(verdict.page_no, []).append(entry)

    def final(self, page_no: int) -> dict:
        return self.attempts[page_no][-1]

    @property
    def failed_pages(self) -> list[int]:
        return [n for n in sorted(self.attempts) if self.final(n)["status"] == "fail"]

    @property
    def soft_issues(self) -> int:
        """Issues recorded on the pages as they finally stand (a refused page's are its hard reasons)."""
        return sum(len(self.final(n).get("soft", [])) for n in self.attempts)

    @property
    def status(self) -> str:
        """``failed``: a page failed a hard check (no Markdown). ``pass_with_issues``: every page stands but
        some carry soft issues, listed per page, so they can be looked at or converted again later."""
        if self.error or self.failed_pages or not self.attempts:
            return "failed"
        return "pass_with_issues" if self.soft_issues else "passed"

    def failure_reasons(self) -> list[str]:
        reasons = [f"page {n + 1}: {self.final(n)['reasons'][0]}" for n in self.failed_pages]
        if self.error:
            reasons.insert(0, self.error)
        if not self.attempts and not self.error:
            reasons.append("no page was converted")
        return reasons

    def to_json(self) -> dict:
        pages = [{"page": n + 1, "retried": len(history) > 1, "attempts": history, **history[-1]}
                 for n, history in sorted(self.attempts.items())]
        counts = {s: sum(1 for p in pages if p["status"] == s) for s in ("pass", "fail", "unverifiable")}
        return {
            "schema": SCHEMA, "paper": self.paper, "pdf": self.pdf, "status": self.status,
            "failure_reasons": self.failure_reasons(), "soft_issues": self.soft_issues, "page_counts": counts,
            "seconds": round(self.seconds, 1), "engine": self.engine, "pages": pages,
        }

    def write(self, path: Path) -> None:
        """Atomically (a reader never sees half a report)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(part, path)
