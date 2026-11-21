"""split_to_budget: spans of at most N real tokens that partition the text exactly.

Real BGE-M3 tokenizer (no model, CPU only) on the inputs that break token-estimating
chunkers: one-line minified JS, base64 blobs, CJK without spaces, a 30,000-word paragraph.
"""
from __future__ import annotations

import base64
import random
from pathlib import Path

import pytest

from tests.embedding_model_support import bge_m3_snapshot

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _snapshot():
    path = bge_m3_snapshot()
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", path)
    mp.setenv("HF_HUB_OFFLINE", "1")
    yield
    mp.undo()


def _count(text: str) -> int:
    from core.indexing.embedding_profile import count_tokens, get_profile
    return count_tokens([text], get_profile())[0]


def _minified_js() -> str:
    return "".join(f"function f{i}(a,b){{return a*{i}+b[{i % 7}];}}var v{i}=f{i}(1,2);" for i in range(3500))


def _js_with_base64() -> str:
    blob = base64.b64encode(random.Random(1).randbytes(45_000)).decode()
    code = "\n".join(f"export const x{i} = compute({i});" for i in range(80))
    return f"{code}\nconst icon = \"data:image/png;base64,{blob}\";\n{code}\n"


def _cjk() -> str:
    rng = random.Random(2)
    return "".join(chr(rng.randrange(0x4E00, 0x9FA5)) for _ in range(20_000))


def _paragraph() -> str:
    rng = random.Random(3)
    words = ["alpha", "beta", "gamma", "delta", "retrieval", "transformer", "gradient", "kernel"]
    return " ".join(rng.choice(words) + str(rng.randrange(50)) for _ in range(30_000))


def _python_source() -> str:
    return (REPO / "app" / "services" / "code_kb" / "service.py").read_text(encoding="utf-8")


INPUTS = {
    "minified_js": _minified_js,
    "js_with_base64": _js_with_base64,
    "cjk": _cjk,
    "paragraph": _paragraph,
    "python": _python_source,
}


def _assert_partition(text: str, spans: list[tuple[int, int]]) -> None:
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a < b for a, b in spans)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
    assert "".join(text[a:b] for a, b in spans) == text


@pytest.mark.parametrize("budget", [512, 2048])
@pytest.mark.parametrize("name", sorted(INPUTS))
def test_every_span_fits_the_budget_and_the_spans_partition_the_text(name, budget):
    from core.indexing.token_budget import split_to_budget

    text = INPUTS[name]()
    for prefer in ("line", "space"):
        spans = split_to_budget(text, max_tokens=budget, prefer=prefer)
        _assert_partition(text, spans)
        worst = max(_count(text[a:b]) for a, b in spans)
        assert worst <= budget, f"{name}/{prefer}: a span has {worst} tokens over {budget}"
    assert len(spans) > 1


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_line_mode_cuts_inside_a_line_only_when_the_line_alone_is_over_budget(name):
    from core.indexing.token_budget import split_to_budget

    text, budget = INPUTS[name](), 512
    for boundary in (b for _, b in split_to_budget(text, max_tokens=budget, prefer="line")[:-1]):
        if text[boundary - 1] == "\n":
            continue
        line_start = text.rfind("\n", 0, boundary) + 1
        line_end = text.find("\n", boundary)
        line = text[line_start: len(text) if line_end < 0 else line_end + 1]
        assert _count(line) > budget, "cut inside a line that fits the budget on its own"


def test_normal_python_is_packed_close_to_the_budget_on_line_boundaries():
    from core.indexing.token_budget import split_to_budget

    text = _python_source()
    spans = split_to_budget(text, max_tokens=512, prefer="line")
    sizes = [_count(text[a:b]) for a, b in spans]
    assert all(text[b - 1] == "\n" for _, b in spans[:-1])
    assert all(size > 300 for size in sizes[:-1]), sizes      # greedy: spans are nearly full


def test_a_text_within_budget_is_one_span_and_trivial_inputs_are_handled():
    from core.indexing.token_budget import split_to_budget

    text = "def f(x):\n    return x + 1\n"
    assert split_to_budget(text, max_tokens=512) == [(0, len(text))]
    assert split_to_budget("é" * 20 + " ok", max_tokens=512) == [(0, 23)]      # non-ASCII: counted, still fits
    assert split_to_budget("", max_tokens=512) == [(0, 0)]
    assert split_to_budget("   \n\n  ", max_tokens=512) == [(0, 7)]
    with pytest.raises(ValueError):
        split_to_budget("x", max_tokens=2)
    with pytest.raises(ValueError):
        split_to_budget("x", max_tokens=512, prefer="word")


def test_exactly_at_the_budget_is_one_span_and_one_over_is_two():
    from core.indexing.token_budget import split_to_budget
    from core.indexing.embedding_profile import get_profile, load_tokenizer
    from tests.embedding_model_support import text_of_tokens

    tokenizer = load_tokenizer(get_profile())
    assert len(split_to_budget(text_of_tokens(tokenizer, 512), max_tokens=512)) == 1
    spans = split_to_budget(text_of_tokens(tokenizer, 513), max_tokens=512, prefer="space")
    assert len(spans) == 2


def test_token_offsets_start_at_the_word_not_at_the_space_before_it():
    from core.indexing.token_budget import token_offsets

    text = "Hello  world\nfoo"
    ranges = token_offsets(text)
    assert [text[a:b].strip() for a, b in ranges][0] == "Hello"
    assert all(not text[a].isspace() for a, b in ranges if b > a)
