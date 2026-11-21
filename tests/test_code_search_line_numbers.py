"""Search numbering is presentation-only across both MCP implementations."""
import asyncio
import inspect

import pytest

from mcp_server import backend, server
from tests.test_mcp_plugin import _load


CODE = "def forward(x):\n\n    return x < 2\n"
HIT = {"path": "src/model.py", "start_line": 455, "end_line": 457,
       "language": "python", "label": "forward", "code": CODE, "score": 0.5,
       "origin": "public", "source_id": "demo", "revision": "abc123"}


@pytest.fixture(scope="module")
def plugin():
    return _load()


@pytest.mark.parametrize("kind", ["private", "public", "all"])
def test_all_formatters_default_to_source_line_numbers_and_allow_plain_code(plugin, kind):
    if kind == "private":
        implementations = [plugin._fmt_code_hits, server._format_code_hits]
        args = ([HIT], "Code matches:")
    elif kind == "public":
        implementations = [plugin._format_public_code_search, backend.format_public_code_search]
        args = ({"query": "forward", "hits": [HIT]},)
    else:
        implementations = [plugin._format_federated_code_search, server._format_federated_code_search]
        args = ({"query": "forward", "hits": [HIT]},)
    rendered = []
    for format_hits in implementations:
        text = format_hits(*args)
        assert "455 | def forward(x):\n456 | \n457 |     return x < 2" in text
        assert "src/model.py:455-457" in text
        plain = format_hits(*args, show_line_numbers=False)
        assert f"```python\n{CODE.rstrip(chr(10))}\n```" in plain
        assert "455 |" not in plain
        rendered.append(text)
    assert rendered[0] == rendered[1]
    assert HIT["code"] == CODE


def test_truncation_notice_is_not_given_a_source_line_number(plugin):
    text = plugin._clip_code_hit_body("a" * (plugin._CODE_HIT_CODE_CHARS + 100),
                                     path="large.py", start_line=700, end_line=700)
    assert text.startswith("700 | ")
    assert "truncated" in text
    assert "701 |" not in text


def test_missing_source_location_does_not_invent_line_one(plugin):
    text = plugin._fmt_code_hits([{**HIT, "start_line": None}], "Code matches:")
    assert CODE.rstrip("\n") in text
    assert "1 |" not in text


def test_numbering_overhead_still_obeys_the_single_chunk_output_cap(plugin):
    code = "x\n" * 1999 + "x"
    text = plugin._clip_code_hit_body(code, path="short_lines.py", start_line=900000, end_line=901999)
    assert len(text) < plugin._CODE_HIT_CODE_CHARS + 200
    assert "truncated" in text
    assert not any(line.strip().isdigit() for line in text.splitlines())


def test_windows_newlines_and_a_number_width_boundary(plugin):
    hit = {**HIT, "start_line": 9, "end_line": 10, "code": "one\r\ntwo\r\n"}
    text = plugin._fmt_code_hits([hit], "Code matches:")
    assert " 9 | one\n10 | two" in text


@pytest.mark.parametrize("module", [server])
def test_search_tool_advertises_numbering_enabled_by_default(module):
    param = inspect.signature(module.search_code).parameters["show_line_numbers"]
    assert param.default is True


@pytest.mark.parametrize("implementation", ["plugin", "repository"])
@pytest.mark.parametrize("sources", ["private", "public", "all"])
@pytest.mark.parametrize("query", ["forward", ["forward", "forward"]])
def test_search_numbering_flag_stays_out_of_backend_request(
    plugin, monkeypatch, implementation, sources, query,
):
    bodies = []

    async def post(path, body):
        bodies.append(body)
        return {"query": "forward", "hits": [HIT]}

    module = plugin if implementation == "plugin" else server
    if implementation == "plugin":
        monkeypatch.setattr(module, "_post", post)
    else:
        class Client:
            async def code_search(self, **body):
                return await post("private", body)

            async def search_public_code(self, **body):
                return await post("public", body)

            async def federated_code_search(self, **body):
                return await post("all", body)

        monkeypatch.setattr(module, "_client", Client())
    monkeypatch.setattr(module, "_local_repo_root", lambda _: None)
    default = asyncio.run(module.search_code(query, repo_root="/remote/demo", sources=sources))
    plain = asyncio.run(module.search_code(query, repo_root="/remote/demo", sources=sources,
                                          show_line_numbers=False))
    assert "455 | def forward" in default
    assert "455 |" not in plain
    middle = len(bodies) // 2
    assert bodies[:middle] == bodies[middle:]
    assert all("show_line_numbers" not in body for body in bodies)


def test_numbering_helpers_match_between_vendored_plugin_and_repo(plugin):
    assert inspect.getsource(plugin._number_code_lines) == inspect.getsource(backend._number_code_lines)
    assert inspect.getsource(plugin._clip_code_hit_body) == inspect.getsource(server._clip_code_hit_body)
