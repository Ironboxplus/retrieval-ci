"""Formula legality and symbols from KaTeX, the renderer the frontend displays Markdown with.

One long-lived ``node tools/katex_check/check.mjs`` process answers JSON lines. KaTeX
0.18.9 (pinned in ``tools/katex_check/package.json``, the frontend's version) either
renders a formula or raises, and its MathML leaves are the characters a reader sees.
A dead process is restarted once; a second failure raises, so a broken toolchain can
never read as "every formula is fine".
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Optional

DEFAULT_CHECK_DIR = Path(__file__).resolve().parents[4] / "tools" / "katex_check"


class KatexUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FormulaRender:
    ok: bool
    error: str = ""
    chars: str = ""  # NFKC text of the MathML leaves, "" when the formula did not render
    # the same with a base's superscript before its subscript, the order a TeX-made PDF writes them to its text layer
    # ("kp ij" for k_{ij}^{p}); KaTeX's leaves are base, subscript, superscript whichever way the source writes them.
    # "" when the formula has no base with both, or when it renders the same.
    layer_chars: str = ""


_COMMAND = re.compile(r"\\(?:[A-Za-z]+|.)", re.DOTALL)
_HAS_BOTH_SCRIPTS = re.compile(r"_[^^]*\^|\^[^_]*_", re.DOTALL)


def _argument(tex: str, at: int):
    """The argument of a script operator that ends just before ``at``: (its text without braces, where it ends), or
    (None, at) when there is none."""
    while at < len(tex) and tex[at] == " ":
        at += 1
    if at >= len(tex):
        return None, at
    if tex[at] == "{":
        depth, i = 0, at
        while i < len(tex):
            if tex[i] == "\\":
                i += 2
                continue
            depth += (tex[i] == "{") - (tex[i] == "}")
            i += 1
            if depth == 0:
                return tex[at + 1:i - 1], i
        return None, at
    token = _COMMAND.match(tex, at)
    return (token.group(), token.end()) if token else (tex[at], at + 1)


def layer_script_order(tex: str) -> str:
    """``tex`` with every base that has both scripts written superscript first: ``x_{a}^{b}`` and ``x^{b}_{a}`` both
    become ``x^{b}{}_{a}`` (the empty group takes the subscript, so KaTeX keeps the superscript's leaves first)."""
    out: list[str] = []
    i = 0
    while i < len(tex):
        char = tex[i]
        if char == "\\":
            token = _COMMAND.match(tex, i)
            out.append(token.group() if token else char)
            i = token.end() if token else i + 1
        elif char in "_^":
            first, after = _argument(tex, i + 1)
            at = after
            while at < len(tex) and tex[at] == " ":
                at += 1
            if first is not None and at < len(tex) and tex[at] in "_^" and tex[at] != char:
                second, end = _argument(tex, at + 1)
                if second is not None:
                    sub, sup = (first, second) if char == "_" else (second, first)
                    out.append("^{" + layer_script_order(sup) + "}{}_{" + layer_script_order(sub) + "}")
                    i = end
                    continue
            out.append(char)
            i += 1  # the argument is scanned in its turn, so a pair inside it is found too
        else:
            out.append(char)
            i += 1
    return "".join(out)


class KatexRenderer:
    def __init__(self, check_dir: Optional[Path] = None, node: Optional[str] = None):
        self.check_dir = Path(os.environ.get("KATEX_CHECK_DIR") or check_dir or DEFAULT_CHECK_DIR)
        self.node = node or os.environ.get("KATEX_NODE") or shutil.which("node")
        self._process: Optional[subprocess.Popen] = None
        self._next_id = 0
        self._cache: dict[tuple[str, bool], FormulaRender] = {}

    def render(self, tex: str, display: bool = False) -> FormulaRender:
        key = (tex, display)
        if key not in self._cache:
            render = self._ask(tex, display)
            if render.ok and _HAS_BOTH_SCRIPTS.search(tex):
                other = self._ask(layer_script_order(tex), display)
                if other.ok and other.chars != render.chars:
                    render = replace(render, layer_chars=other.chars)
            self._cache[key] = render
        return self._cache[key]

    def render_many(self, items: list[tuple[str, bool]]) -> list[FormulaRender]:
        return [self.render(tex, display) for tex, display in items]

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def __enter__(self) -> "KatexRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----------------------------------------------------------------- process
    def _start(self) -> subprocess.Popen:
        if not self.node:
            raise KatexUnavailable("node is not on PATH (set KATEX_NODE)")
        if not (self.check_dir / "node_modules" / "katex").is_dir():
            raise KatexUnavailable(f"katex is not installed: run `npm ci` in {self.check_dir}")
        return subprocess.Popen(
            [self.node, "check.mjs"], cwd=self.check_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
        )

    def _ask(self, tex: str, display: bool) -> FormulaRender:
        last_error = ""
        for _ in range(2):  # the process may have died since the last call: one restart
            if self._process is None or self._process.poll() is not None:
                self._process = self._start()
            self._next_id += 1
            request = json.dumps({"id": self._next_id, "tex": tex, "display": display}, ensure_ascii=True)
            try:
                self._process.stdin.write(request + "\n")
                self._process.stdin.flush()
                line = self._process.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                line, last_error = "", str(exc)
            if not line:
                self._process = None
                continue
            reply = json.loads(line)
            if reply.get("id") != self._next_id:
                raise KatexUnavailable(f"katex replied to request {reply.get('id')} instead of {self._next_id}")
            if reply["ok"]:
                return FormulaRender(True, chars=unicodedata.normalize("NFKC", reply["chars"]))
            return FormulaRender(False, error=reply["error"])
        raise KatexUnavailable(f"katex process died twice in a row {last_error}".strip())


@lru_cache(maxsize=1)
def shared_renderer() -> KatexRenderer:
    return KatexRenderer()
