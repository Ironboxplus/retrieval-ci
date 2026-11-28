"""Native-free source discovery and language policy for Code KB."""

from __future__ import annotations

import os
from typing import Optional


LANG_BY_EXT = {
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cu": "cpp", ".cuh": "cpp",
    ".rs": "rust", ".go": "go", ".zig": "zig", ".nim": "nim",
    ".swift": "swift", ".m": "objc", ".mm": "objc",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala", ".sc": "scala", ".groovy": "groovy", ".gradle": "groovy",
    ".cs": "csharp",
    ".py": "python", ".pyi": "python", ".rb": "ruby", ".php": "php",
    ".lua": "lua", ".pl": "perl", ".pm": "perl", ".r": "r", ".jl": "julia",
    ".dart": "dart", ".ex": "elixir", ".exs": "elixir",
    ".erl": "erlang", ".hrl": "erlang", ".hs": "haskell",
    ".ml": "ocaml", ".mli": "ocaml",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx", ".jsx": "tsx",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".vue": "vue", ".svelte": "svelte",
    ".css": "css", ".scss": "scss", ".html": "html", ".htm": "html",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".ps1": "powershell",
    ".tf": "terraform", ".hcl": "hcl", ".dockerfile": "dockerfile",
    ".cmake": "cmake", ".sol": "solidity",
    ".sql": "sql", ".graphql": "graphql", ".gql": "graphql",
    ".proto": "proto", ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".toml": "toml", ".xml": "xml", ".ini": "ini",
}

LANG_BY_NAME = {
    "dockerfile": "dockerfile",
    "makefile": "make",
    "cmakelists.txt": "cmake",
}

FALLBACK_EXTS = {".env", ".properties", ".cfg", ".conf", ".editorconfig", ".sass"}
# Documentation files the Journal document pipeline owns; index_code walks them with the code.
DOC_EXTS = frozenset({".md", ".mdx", ".markdown", ".rst", ".adoc", ".txt"})
DOC_GLOBS = ["*.md", "*.mdx", "*.markdown", "*.rst", "*.adoc", "*.txt"]
DEFAULT_MAX_CHUNK_SIZE = 1500
MAX_FILE_BYTES = 2 * 1024 * 1024


def source_extension(path: str) -> str:
    """Return a lowercased extension while preserving extensionless dotfiles."""

    name = os.path.basename(path).lower()
    extension = os.path.splitext(name)[1]
    if not extension and name.startswith("."):
        return name
    return extension


def resolve_language(path: str) -> Optional[str]:
    """Return the configured parser grammar without importing a native parser."""

    name = os.path.basename(path).lower()
    if name in LANG_BY_NAME:
        return LANG_BY_NAME[name]
    return LANG_BY_EXT.get(source_extension(path))


__all__ = [
    "DEFAULT_MAX_CHUNK_SIZE",
    "DOC_EXTS",
    "DOC_GLOBS",
    "FALLBACK_EXTS",
    "LANG_BY_EXT",
    "LANG_BY_NAME",
    "MAX_FILE_BYTES",
    "resolve_language",
    "source_extension",
]
