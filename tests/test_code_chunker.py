"""Tests for the code chunker (chonkie CodeChunker over tree-sitter grammars)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from tests.conftest import git_init_repo

# Code KB chunking needs chonkie[code] (bundles the tree-sitter grammar pack); skip
# cleanly where it's absent (it's a required dep in requirements.txt, optional on CI).
pytest.importorskip("chonkie")

from core.code_kb.chunker import (
    _builder, chunk_code, chunk_file, chunk_source,
    create_chunk_executor,
    file_content_hash, gitignore_allowed,
    iter_repo_chunks, iter_repo_files, LANG_BY_EXT,
)

SAMPLE = '''\
import os
from typing import List


def top_level(x, y):
    """add two numbers"""
    return x + y


class Foo:
    """a foo"""

    def __init__(self, a):
        self.a = a

    def bar(self, n):
        # multiply
        return self.a * n
'''


def test_produces_chunks_with_path_header_and_lines():
    chunks = chunk_code("pkg/mod.py", SAMPLE)
    assert chunks
    for c in chunks:
        assert c.content.startswith("# path: pkg/mod.py")
        assert c.start_line >= 1 and c.end_line >= c.start_line


def test_covers_all_definitions():
    joined = "\n".join(c.code for c in chunk_code("m.py", SAMPLE))
    # chonkie CodeChunker is lossless — every def survives, none split away
    assert "def top_level" in joined
    assert "def __init__" in joined
    assert "def bar" in joined
    assert "return self.a * n" in joined


def test_label_extracted():
    labels = [c.label for c in chunk_code("m.py", SAMPLE)]
    assert any(labels), "expected at least one chunk to carry a symbol label"


def test_no_brace_only_fragment_chunks():
    """chonkie leaves `{`/`}` slivers behind when a function exceeds chunk_size; those
    embed as nothing but the path header and pollute search (W157). They must be merged
    into a neighbor, never emitted as standalone chunks."""
    fn = (
        "func doThing%d(ctx context.Context, req *Request) (*Response, error) {\n"
        + "".join(f'    log.Debugf("padding line {j}")\n' for j in range(60))
        + "    return &Response{}, nil\n}\n"
    )
    src = "package util\n\n" + "\n".join(fn % i for i in range(6))
    chunks = chunk_code("drivers/139/util.go", src, language="go")
    assert chunks
    for c in chunks:
        nws = len("".join(c.code.split()))
        assert nws >= 24, f"fragment chunk survived: {c.start_line}-{c.end_line} {c.code!r}"
    # merging must not lose source: every padded line still present exactly once
    joined = "".join(c.code for c in chunks)
    assert joined.count("doThing3") == 1


def test_unsupported_extension_skipped(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# just markdown\nnot code\n", encoding="utf-8")
    assert chunk_file(str(p)) == []


def test_language_map_has_python():
    assert LANG_BY_EXT[".py"] == "python"


def test_cuda_extensions_use_cpp_grammar():
    assert LANG_BY_EXT[".cu"] == "cpp"
    assert LANG_BY_EXT[".cuh"] == "cpp"


def test_builder_is_cached_per_language():
    # A fresh CodeChunker rebuilds the tree-sitter grammar; on a repo of N files that
    # is N re-inits. Cache per (language, size) so chunking a whole repo builds each
    # grammar once. Same key → same instance; different language → different.
    b1 = _builder("python", 1500)
    b2 = _builder("python", 1500)
    assert b1 is b2, "same (language, size) must return the cached builder"
    assert _builder("java", 1500) is not b1, "different language must be a distinct builder"


def test_builder_cache_is_isolated_per_worker_thread():
    barrier = threading.Barrier(2)

    def get_builder():
        builder = _builder("c", 1500)
        barrier.wait(timeout=5)
        return builder

    with ThreadPoolExecutor(max_workers=2) as executor:
        builders = list(executor.map(lambda _: get_builder(), range(2)))

    assert builders[0] is not builders[1]


def test_chunk_executor_restores_process_default_stack_size():
    original_size = threading.stack_size()

    with create_chunk_executor(2) as executor:
        assert threading.stack_size() == original_size
        assert list(executor.map(lambda value: value + 1, [1, 2])) == [2, 3]


def assert_chunks_cite_their_source(chunks, source):
    """Every chunk's code lies on the lines it cites, and together the chunks hold the
    source's text exactly once."""
    lines = source.splitlines(keepends=True)
    for chunk in chunks:
        cited = "".join(lines[chunk.start_line - 1:chunk.end_line])
        assert chunk.code.strip() in cited, (chunk.start_line, chunk.end_line, chunk.code[:80])
    assert "\n".join(chunk.code for chunk in chunks).split() == source.split()


# Non-ASCII before a chunk: chonkie's CodeChunker reports UTF-8 byte offsets, and
# slicing the str with them cut every later chunk short at the front ("evious real
# chunk", "allback for grammarless") and cited the wrong lines (2026-10-13).
NON_ASCII_PY = "".join(
    f'def step_{i}(x):\n    """Copy the D3D12 output → D3D11 backbuffer — pass {i}; 中文注释。"""\n'
    f"    return x + {i}\n\n\n" for i in range(200))

# A large hand-written source: 9,255-line dlss5-feed.cpp (567K chars) came back as one
# chunk citing lines 1-9255, because sources over 384K chars got a window as large as
# the whole file (2026-10-13).
LARGE_CPP = "".join(
    f"// pass {i}: copy the D3D12 output → D3D11 backbuffer — blit\n"
    f"static int blit_{i}(int width, int height) {{\n"
    f"    if (width > {i}) {{\n        return width * height + {i};\n    }}\n"
    f"    return height - {i};\n}}\n\n" for i in range(4000))

# FFmpeg dcadata.c shape: one initializer far larger than any segment.
HUGE_TABLE_C = "const unsigned table[] = {\n" + "1,\n" * 150_000 + "};\n"


def test_chunks_after_non_ascii_text_cite_their_own_lines():
    chunks = chunk_code("steps.py", NON_ASCII_PY, language="python")

    assert len(chunks) > 5
    assert_chunks_cite_their_source(chunks, NON_ASCII_PY)
    assert all(chunk.code.lstrip().startswith("def step_") for chunk in chunks)


def test_large_source_is_chunked_into_bounded_ast_chunks():
    assert len(LARGE_CPP) > 384 * 1024

    chunks = chunk_code("dlss5-feed.cpp", LARGE_CPP, language="cpp")

    assert len(chunks) > 100
    assert max(len(chunk.code) for chunk in chunks) < 8 * 1024
    assert_chunks_cite_their_source(chunks, LARGE_CPP)
    # AST chunks begin at a declaration; line windows would begin anywhere.
    assert all(chunk.code.lstrip().startswith(("// pass", "static int")) for chunk in chunks)


def test_huge_initializer_is_line_windowed_not_one_chunk():
    started = time.monotonic()
    chunks = chunk_code("dcadata.c", HUGE_TABLE_C, language="c")

    assert time.monotonic() - started < 20
    assert len(chunks) > 50
    assert max(len(chunk.code) for chunk in chunks) < 8 * 1024
    assert chunks[0].code.startswith("const unsigned table[]")
    assert chunks[-1].code.rstrip().endswith("};")
    assert_chunks_cite_their_source(chunks, HUGE_TABLE_C)


def test_chunk_spans_accept_byte_or_character_offsets_and_reject_neither():
    from types import SimpleNamespace

    from core.code_kb.chunker import _char_spans

    source = "α = 1\nβ = 2\n"
    second = "β = 2\n"
    by_bytes = [SimpleNamespace(text="α = 1\n", start_index=0),
                SimpleNamespace(text=second, start_index=len("α = 1\n".encode("utf-8")))]
    by_chars = [SimpleNamespace(text="α = 1\n", start_index=0),
                SimpleNamespace(text=second, start_index=len("α = 1\n"))]

    assert _char_spans(source, by_bytes) == [[0, 6], [6, 12]]
    assert _char_spans(source, by_chars) == [[0, 6], [6, 12]]
    with pytest.raises(ValueError, match="locate neither"):
        _char_spans(source, [SimpleNamespace(text="γ = 3\n", start_index=0)])


def test_deeply_nested_source_falls_back_to_line_windows():
    # Regression (W155b): tree-sitter's Python tree traversal recurses per nesting level;
    # at ~1200 bracket depth it C-stack-overflows on Windows — a process-killing crash
    # (rc=127) that Python except cannot catch, so the MCP server dies and index_code
    # hangs forever. Deeply nested sources (minified JS, generated JSON) must be routed
    # to the line-window fallback BEFORE any AST parse.
    blob = "var a=" + "{'x':[" * 3000 + "1" + "]}" * 3000 + ";"
    chunks = chunk_code("bundle.min.js", blob, language="javascript")
    assert chunks, "deep-nested source must produce fallback chunks, not crash the process"
    for c in chunks:
        assert c.content.startswith("# path: bundle.min.js")


def test_shared_source_chunk_entrypoint_falls_back_when_ast_chunking_fails(monkeypatch):
    monkeypatch.setattr(
        "core.code_kb.chunker.chunk_code",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("parser failed")),
    )

    chunks = chunk_source("src/module.py", "print('still indexed')\n", language="python")

    assert len(chunks) == 1
    assert chunks[0].path == "src/module.py"
    assert chunks[0].code == "print('still indexed')"


def test_minified_long_line_is_hard_split():
    # A minified bundle is one enormous line; the line-window fallback must hard-split it
    # instead of emitting a single multi-megabyte chunk (which chokes embedding and the
    # backend's request limits).
    blob = "var b=[" + ",".join(str(i) for i in range(40000)) + "];"  # one ~230KB line
    chunks = chunk_code("flat.min.js", blob, language="javascript")
    assert len(chunks) > 1, "one huge line must be split into multiple chunks"
    for c in chunks:
        assert len(c.code) <= 4 * 1500, f"chunk too large: {len(c.code)} chars"


def test_bash_script_does_not_crash():
    # Regression (W155): astchunk 0.1.0 held long-lived tree-sitter Node refs for window
    # reassembly; on a bash grammar its node.text deref hit freed memory and SEGFAULTED
    # the whole process (uncatchable by Python except → index_code hung forever). chonkie
    # with include_nodes=False never retains nodes, so a bash script chunks cleanly.
    bash = "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(
        f"function step_{i}() {{\n  echo 'do {i}'\n  cat <<EOF\n multi\n line {i}\nEOF\n}}"
        for i in range(60))
    chunks = chunk_code("build.sh", bash, language="bash")
    assert chunks, "bash script must chunk without crashing (astchunk segfaulted here)"
    for c in chunks:
        assert c.content.startswith("# path: build.sh")
        assert c.start_line >= 1 and c.end_line >= c.start_line


def test_incomplete_grammar_cache_falls_back_without_starting_download(monkeypatch):
    """Indexing is an offline hot path; a partial grammar cache must fail fast.

    Chonkie's constructor calls ``download_all()``. Without this guard a compute host
    with no public egress waits for the language-pack network timeout for every worker,
    racing the isolated-chunker timeout and failing the entire index job.
    """

    constructor_called = False

    from core.code_kb import chunker as chunker_module

    monkeypatch.setattr(chunker_module._BUILDER_LOCAL, "builders", {}, raising=False)

    def fail_if_chonkie_is_constructed(*_args, **_kwargs):
        nonlocal constructor_called
        constructor_called = True
        raise AssertionError("partial grammar cache must not start Chonkie")

    monkeypatch.setattr(
        "tree_sitter_language_pack.downloaded_languages", lambda: ["python"]
    )
    monkeypatch.setattr("core.code_kb.chunker.CodeChunker", fail_if_chonkie_is_constructed)

    chunks = chunk_source("ack_timer.go", "package sctp\n", language="go")

    assert constructor_called is False
    assert len(chunks) == 1
    assert chunks[0].code == "package sctp"


# --------------------------- multi-language coverage --------------------------- #
# chonkie's CodeChunker sources parsers from the bundled tree-sitter grammar pack, so
# cAST-style structural chunking works across 40+ languages. One representative sample
# per language family the user cares about (systems, JVM, web, styling, config).

_MULTI_LANG_SAMPLES = {
    "a.rs":  "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\nstruct P { x: f64 }\n",
    "a.go":  "package main\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n",
    "a.c":   "int add(int a, int b) {\n    return a + b;\n}\n",
    "a.cpp": "class P {\npublic:\n    double norm() const { return 0.0; }\n};\n",
    "a.lua": "local function add(a, b)\n  return a + b\nend\n",
    "a.kt":  "fun add(a: Int, b: Int): Int {\n    return a + b\n}\n",
    "a.tsx": "export const App = () => {\n  return <div className=\"x\">hi</div>;\n};\n",
    "a.scss": ".card {\n  color: red;\n  .title { font-size: 14px; }\n}\n",
}


def test_many_languages_chunked_with_path_header(tmp_path):
    for name, src in _MULTI_LANG_SAMPLES.items():
        p = tmp_path / name
        p.write_text(src, encoding="utf-8")
        chunks = chunk_file(str(p), repo_root=str(tmp_path))
        assert chunks, f"{name}: expected AST chunks, got none (language unsupported?)"
        for c in chunks:
            assert c.content.startswith(f"# path: {name}"), f"{name}: missing path header"
            assert c.start_line >= 1 and c.end_line >= c.start_line, f"{name}: bad line range"


def test_csharp_is_not_silently_dropped(tmp_path):
    # Regression: the ext map said "c_sharp" but the grammar name is "csharp", so every
    # .cs file raised and was swallowed to zero chunks. Guard it stays fixed.
    p = tmp_path / "Program.cs"
    p.write_text("class P {\n    public int Add(int a, int b) { return a + b; }\n}\n",
                 encoding="utf-8")
    chunks = chunk_file(str(p))
    assert chunks, ".cs must produce chunks (csharp grammar, not 'c_sharp')"
    assert LANG_BY_EXT[".cs"] == "csharp"


def test_grammarless_code_file_falls_back_to_line_windows(tmp_path):
    # An extension we index but that has no tree-sitter grammar (config/plain text)
    # must still be chunked via a line-window fallback, not silently dropped.
    p = tmp_path / ".env"
    body = "\n".join(f"KEY_{i}=value_{i}" for i in range(200))
    p.write_text(body, encoding="utf-8")
    chunks = chunk_file(str(p))
    assert chunks, "grammarless code/config file must fall back to line-window chunks"
    for c in chunks:
        assert c.content.startswith("# path: ")
        assert c.start_line >= 1 and c.end_line >= c.start_line


def test_non_code_extension_still_skipped(tmp_path):
    # Docs (.md) belong to the journal, not the code KB; images are binary. Both skip.
    for name, data in [("notes.md", "# doc\ntext\n"), ("logo.png", "\x89PNG\r\n")]:
        p = tmp_path / name
        p.write_text(data, encoding="utf-8")
        assert chunk_file(str(p)) == [], f"{name} must not enter the code KB"


def test_iter_repo_files_lists_candidates_without_parsing(tmp_path):
    # Lists indexable files (resolvable language or fallback ext) as (abs, rel), skips
    # junk dirs + non-code, and mirrors the chunker's file gate — the manifest source.
    (tmp_path / "keep.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.go").write_text("package x\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# doc\n", encoding="utf-8")   # not code
    junk = tmp_path / "node_modules"
    junk.mkdir()
    (junk / "dep.js").write_text("var z=1\n", encoding="utf-8")       # pruned dir
    rels = {rel for _, rel in iter_repo_files(str(tmp_path))}
    assert rels == {"keep.py", "sub/b.go"}
    for abs_path, rel in iter_repo_files(str(tmp_path)):
        assert abs_path.endswith(rel.replace("/", __import__("os").sep))


def test_iter_repo_files_includes_cuda_sources(tmp_path):
    cuda_source = tmp_path / "kernel.cu"
    cuda_source.write_text(
        "__global__ void scale(float *values) { values[threadIdx.x] *= 2; }\n",
        encoding="utf-8",
    )
    (tmp_path / "kernel.cuh").write_text(
        "void launch_scale(float *values);\n",
        encoding="utf-8",
    )

    assert {rel for _, rel in iter_repo_files(str(tmp_path))} == {
        "kernel.cu",
        "kernel.cuh",
    }
    chunks = chunk_file(str(cuda_source), str(tmp_path))
    assert chunks
    assert "__global__ void scale" in "".join(chunk.code for chunk in chunks)


def test_iter_repo_files_keeps_web_source_under_a_frontend_directory(tmp_path):
    """A project directory name is not evidence that its source is generated."""
    source_dir = tmp_path / "frontend" / "mobile-web" / "src"
    source_dir.mkdir(parents=True)
    for filename, source in {
        "publisher.ts": "export const publish = () => true;\n",
        "Page.tsx": "export const Page = () => <main />;\n",
        "legacy.js": "export const legacy = true;\n",
        "Widget.jsx": "export const Widget = () => <div />;\n",
    }.items():
        (source_dir / filename).write_text(source, encoding="utf-8")

    assert {rel for _, rel in iter_repo_files(str(tmp_path))} == {
        "frontend/mobile-web/src/publisher.ts",
        "frontend/mobile-web/src/Page.tsx",
        "frontend/mobile-web/src/legacy.js",
        "frontend/mobile-web/src/Widget.jsx",
    }


def test_iter_repo_files_keeps_source_under_a_vendor_directory(tmp_path):
    """Vendored source stays indexable unless the project ignore rules exclude it."""
    source_file = tmp_path / "vendor" / "example.dev" / "codec" / "decoder.go"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("package codec\n", encoding="utf-8")

    assert {rel for _, rel in iter_repo_files(str(tmp_path))} == {
        "vendor/example.dev/codec/decoder.go",
    }


def test_chunk_and_file_walks_share_non_git_exclusions(tmp_path):
    (tmp_path / ".retrievalignore").write_text("generated/**\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "private.py").write_text("secret = 1\n", encoding="utf-8")
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "drop.py").write_text("y = 2\n", encoding="utf-8")

    files = {rel for _, rel in iter_repo_files(
        str(tmp_path), exclude_globs=["private.py"])}
    chunks = list(iter_repo_chunks(
        str(tmp_path), exclude_globs=["private.py"]))

    assert files == {"keep.py"}
    assert {chunk.path for chunk in chunks} == files


def test_file_content_hash_stable_and_change_sensitive(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("def a():\n    return 1\n", encoding="utf-8")
    h1 = file_content_hash(str(p))
    assert h1 and len(h1) == 64 and file_content_hash(str(p)) == h1  # stable
    p.write_text("def a():\n    return 2\n", encoding="utf-8")
    assert file_content_hash(str(p)) != h1                            # content-sensitive
    assert file_content_hash(str(tmp_path / "missing.py")) is None    # unreadable → None


def test_plugin_language_maps_match_core():
    # The standalone PyPI plugin (mcp_plugin/retrieval_mcp.py) ships as a single file and
    # cannot import from the repo, so it hand-duplicates the language maps + fallback set.
    # Guard against silent drift: the two copies must stay byte-equal or the same repo
    # indexed from the plugin vs the in-repo server would chunk different files.
    import ast
    import pathlib

    plugin_src = (pathlib.Path(__file__).resolve().parent.parent
                  / "mcp_plugin" / "retrieval_mcp.py").read_text(encoding="utf-8")
    tree = ast.parse(plugin_src)
    plugin: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in {"LANG_BY_EXT", "LANG_BY_NAME", "FALLBACK_EXTS"}:
                plugin[name] = ast.literal_eval(node.value)

    from core.code_kb import chunker
    assert plugin["LANG_BY_EXT"] == chunker.LANG_BY_EXT, "LANG_BY_EXT drifted between core and plugin"
    assert plugin["LANG_BY_NAME"] == chunker.LANG_BY_NAME, "LANG_BY_NAME drifted between core and plugin"
    assert plugin["FALLBACK_EXTS"] == chunker.FALLBACK_EXTS, "FALLBACK_EXTS drifted between core and plugin"


# ------------------------------- gitignore filtering ------------------------------- #
# Indexing a repo should, by default, skip the files git itself ignores (build output,
# secrets, vendored deps) instead of embedding them as search noise — delegating the
# ignore resolution to git so nested .gitignore / negation / global excludes all hold.


def test_gitignore_allowed_returns_none_outside_git_repo(tmp_path):
    # A directory that isn't a git working tree can't resolve .gitignore → None, which
    # the callers read as "apply no filtering" (graceful degrade, current behavior).
    assert gitignore_allowed(str(tmp_path)) is None


def test_gitignore_allowed_lists_only_non_ignored_paths(tmp_path):
    git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.py\ngen/\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "gen").mkdir()
    (tmp_path / "gen" / "out.py").write_text("z = 3\n", encoding="utf-8")
    allowed = gitignore_allowed(str(tmp_path))
    assert allowed is not None
    assert "keep.py" in allowed
    assert "ignored.py" not in allowed          # matched .gitignore
    assert "gen/out.py" not in allowed          # matched dir rule, forward-slash rel


def test_iter_repo_files_respects_gitignore_by_default(tmp_path):
    git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.py\ngen/\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    (tmp_path / "gen").mkdir()
    (tmp_path / "gen" / "out.py").write_text("gen = 1\n", encoding="utf-8")
    rels = {rel for _, rel in iter_repo_files(str(tmp_path))}
    assert rels == {"keep.py"}, rels          # ignored.py + gen/ excluded via .gitignore


def test_iter_repo_files_gitignore_can_be_disabled(tmp_path):
    git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    rels = {rel for _, rel in iter_repo_files(str(tmp_path), respect_gitignore=False)}
    assert rels == {"keep.py", "ignored.py"}, rels   # opt-out includes the ignored file


def test_iter_repo_files_no_git_repo_indexes_everything(tmp_path):
    # respect_gitignore=True is the default, but with no .git the resolution is impossible
    # → degrade to no filtering so a plain (non-repo) directory still indexes fully.
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    rels = {rel for _, rel in iter_repo_files(str(tmp_path))}
    assert rels == {"keep.py", "ignored.py"}, rels   # no git → nothing filtered
