from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_clipboard_falls_back_when_modern_api_rejects_and_always_cleans_up():
    source = (ROOT / "frontend" / "src" / "utils" / "clipboard.ts").read_text(
        encoding="utf-8"
    )

    assert "await navigator.clipboard.writeText(text)" in source
    assert "catch" in source
    assert "finally" in source
    assert "document.body.removeChild(textarea)" in source

    for consumer in (
        ROOT / "frontend" / "src" / "components" / "AccountModal.tsx",
        ROOT / "frontend" / "src" / "pages" / "PaperDetail.tsx",
        ROOT / "frontend" / "src" / "pages" / "Search.tsx",
    ):
        consumer_source = consumer.read_text(encoding="utf-8")
        assert "utils/clipboard" in consumer_source
        assert "navigator.clipboard" not in consumer_source


def test_metered_paper_queries_send_stable_idempotency_headers():
    source = (
        ROOT / "frontend" / "src" / "api" / "hooks" / "usePapers.ts"
    ).read_text(encoding="utf-8")

    assert "useId" in source
    assert "paper-detail-${paperId}-${requestId}" in source
    assert "paper-fulltext-${paperId}-${requestId}" in source
    assert source.count("'X-Idempotency-Key'") >= 2
