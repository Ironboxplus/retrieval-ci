import pytest

from core.code_kb.source_text import BinarySourceTextError, decode_source_bytes


def test_decode_source_bytes_prefers_utf8():
    assert decode_source_bytes("变量 = 1\n".encode("utf-8")) == "变量 = 1\n"


def test_decode_source_bytes_accepts_legacy_western_source_comments():
    source = "// Copyright Hervé Brönnimann\nint value = 1;\n"
    assert decode_source_bytes(source.encode("cp1252")) == source


def test_decode_source_bytes_rejects_nul_delimited_binary_content():
    with pytest.raises(BinarySourceTextError, match="NUL"):
        decode_source_bytes(b"int value = 1;\x00binary")
