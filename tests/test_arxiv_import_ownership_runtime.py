"""import_bundle writes only what the sending host owns.

A bundle row is a whole ``papers`` row from the sender's harvest DB, but the sender owns two
groups of its columns: the OAI metadata it enumerated and the HTML download state it
produced. ``convert_*`` (HTML -> Markdown, the ingest host's converter) and ``pdf_*`` (a PDF
fallback whose files never travel in a bundle) belong to the host that ingests, so an
import must leave them as they are. Real SQLite files and real bundles written by
``write_rows_bundle``; nothing is faked.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from pipeline.paper_crawler.arxiv_batches import import_bundle, write_rows_bundle
from pipeline.paper_crawler.arxiv_html import ArxivRawRecord, HarvestStore, html_path

ARXIV_ID = "2601.00001"


def raw(arxiv_id: str, *, version="v1", title=None, abstract="abstract") -> ArxivRawRecord:
    return ArxivRawRecord(
        arxiv_id=arxiv_id, datestamp="2026-04-03", categories="cs.LG",
        title=title or f"title {arxiv_id}", authors="A B", abstract=abstract, comments=None,
        license=None, latest_version=version, latest_version_date="2026-04-02T00:00:00+00:00",
        first_version_date="2026-04-01T00:00:00+00:00", latest_size="10kb", content_version=version,
        content_version_date="2026-04-02T00:00:00+00:00", withdrawn=False)


def _row(store: HarvestStore, arxiv_id: str) -> dict:
    cur = store.db.execute("SELECT * FROM papers WHERE arxiv_id=?", (arxiv_id,))
    names = [d[0] for d in cur.description]
    return dict(zip(names, cur.fetchone()))


def _send(workdir: Path, records: list[ArxivRawRecord], *, ok=()) -> Path:
    """What a downloading host does: harvest the records, mark the pages it fetched, bundle the rows."""
    root = workdir / "sender"
    sender = HarvestStore(root / "arxiv_cs.sqlite3")
    sender.upsert(records)
    for record in records:
        if record.arxiv_id in ok:
            sender.mark(record.arxiv_id, "ok", html_version=record.content_version, raw_bytes=10, gz_bytes=5)
            page = html_path(root, record.arxiv_id)
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_bytes(gzip.compress(f"<html>{record.arxiv_id}</html>".encode()))
    cur = sender.db.execute("SELECT * FROM papers ORDER BY arxiv_id")
    names = [d[0] for d in cur.description]
    rows = [dict(zip(names, r)) for r in cur]
    sender.close()
    out = workdir / "bundle.tar"
    write_rows_bundle(rows, root, out)
    return out


def _ingest_host_with_local_outcomes(tmp_path: Path, *, version="v1", convert="rejected") -> HarvestStore:
    """A row the ingest host already converted (``convert``: ok, rejected or error) and tried the
    PDF fallback on."""
    ingest = HarvestStore(tmp_path / "ingest" / "arxiv_cs.sqlite3")
    ingest.upsert([raw(ARXIV_ID, version=version, title="old title")])
    ingest.mark(ARXIV_ID, "no_html", error="HTTP 404")
    ingest.mark_pdf(ARXIV_ID, "ok", pdf_bytes=1234)
    ingest.mark_pdf(ARXIV_ID, "ok", pdf_bytes=1234)  # a second attempt: pdf_attempts = 2
    ingest.mark_convert([(ARXIV_ID, convert, version, None if convert == "ok" else "no article body")])
    return ingest


@pytest.mark.parametrize("convert", ["ok", "rejected", "error"])
def test_import_keeps_the_ingest_hosts_convert_and_pdf_columns(tmp_path, convert):
    ingest = _ingest_host_with_local_outcomes(tmp_path, convert=convert)
    before = _row(ingest, ARXIV_ID)
    assert before["convert_status"] == convert and before["pdf_status"] == "ok" and before["pdf_attempts"] == 2

    bundle = _send(tmp_path, [raw(ARXIV_ID, title="new title", abstract="new abstract")], ok={ARXIV_ID})
    assert import_bundle(ingest, tmp_path / "ingest", bundle) == {"rows": 1, "html": 1}

    after = _row(ingest, ARXIV_ID)
    # what the sender owns is the sender's
    assert (after["title"], after["abstract"]) == ("new title", "new abstract")
    assert (after["status"], after["html_version"], after["raw_bytes"], after["gz_bytes"]) == ("ok", "v1", 10, 5)
    sent = _row(HarvestStore(tmp_path / "sender" / "arxiv_cs.sqlite3"), ARXIV_ID)
    for column in ("attempts", "error", "fetched_at", "latest_size", "withdrawn", "content_version"):
        assert after[column] == sent[column], column
    assert after["attempts"] == 1 and after["fetched_at"]
    # what the ingest host owns is untouched
    for column in ("convert_status", "convert_version", "convert_error", "pdf_status", "pdf_attempts",
                   "pdf_bytes", "pdf_fetched_at", "pdf_error"):
        assert after[column] == before[column], column
    assert html_path(tmp_path / "ingest", ARXIV_ID).read_bytes()


def test_import_inserts_a_new_row_with_the_senders_columns_and_no_local_outcomes(tmp_path):
    ingest = HarvestStore(tmp_path / "ingest" / "arxiv_cs.sqlite3")
    bundle = _send(tmp_path, [raw("2601.00002")], ok={"2601.00002"})
    import_bundle(ingest, tmp_path / "ingest", bundle)

    row = _row(ingest, "2601.00002")
    assert (row["title"], row["status"], row["html_version"]) == ("title 2601.00002", "ok", "v1")
    assert row["convert_status"] is None and row["pdf_status"] is None and row["pdf_attempts"] == 0


def test_a_new_content_version_requeues_the_pdf_fallback_but_not_the_conversion_record(tmp_path):
    """HarvestStore.upsert resets pdf_status/pdf_attempts when the content version changes (the
    old PDF is of an older paper); an import does the same. convert_* needs no reset: an outcome
    only counts while convert_version equals the row's content version."""
    ingest = _ingest_host_with_local_outcomes(tmp_path, version="v1")
    bundle = _send(tmp_path, [raw(ARXIV_ID, version="v2")])  # not downloaded yet: the sender's row is pending
    import_bundle(ingest, tmp_path / "ingest", bundle)

    row = _row(ingest, ARXIV_ID)
    assert (row["content_version"], row["status"], row["html_version"]) == ("v2", "pending", None)
    assert row["pdf_status"] is None and row["pdf_attempts"] == 0
    assert (row["convert_status"], row["convert_version"]) == ("rejected", "v1")


def test_a_row_the_lean_seed_started_does_not_blank_the_page_sizes_the_ingest_host_knows(tmp_path):
    """A lean seed row carries the status and version columns but not raw_bytes/gz_bytes. When
    enumerate lists the paper again its metadata is rewritten and the row travels; its unknown
    sizes must not erase the sizes the ingest host recorded when it fetched the page."""
    ingest = HarvestStore(tmp_path / "ingest" / "arxiv_cs.sqlite3")
    ingest.upsert([raw(ARXIV_ID, title="old title")])
    ingest.mark(ARXIV_ID, "ok", html_version="v1", raw_bytes=1000, gz_bytes=400)

    root = tmp_path / "sender"
    sender = HarvestStore(root / "arxiv_cs.sqlite3")
    sender.db.execute(
        "INSERT INTO papers(arxiv_id, datestamp, categories, primary_category, title, authors, abstract,"
        " latest_version, latest_version_date, first_version_date, status, attempts, html_version,"
        " fetched_at, content_version, content_version_date)"
        " VALUES(?, '2026-04-03', '', 'cs.LG', '', '', '', 'v1', '2026-04-02T00:00:00+00:00',"
        " '2026-04-01T00:00:00+00:00', 'ok', 1, 'v1', '2026-04-02T01:00:00+00:00', 'v1',"
        " '2026-04-02T00:00:00+00:00')", (ARXIV_ID,))
    sender.upsert([raw(ARXIV_ID, title="new title")])  # listed again: real metadata, sizes still unknown
    cur = sender.db.execute("SELECT * FROM papers")
    names = [d[0] for d in cur.description]
    rows = [dict(zip(names, r)) for r in cur]
    sender.close()
    assert rows[0]["raw_bytes"] is None and rows[0]["status"] == "ok"
    bundle = tmp_path / "bundle.tar"
    write_rows_bundle(rows, root, bundle, html_ids=set())
    import_bundle(ingest, tmp_path / "ingest", bundle)

    row = _row(ingest, ARXIV_ID)
    assert row["title"] == "new title" and row["status"] == "ok"
    assert (row["raw_bytes"], row["gz_bytes"]) == (1000, 400)


def test_importing_the_same_bundle_twice_changes_nothing(tmp_path):
    ingest = _ingest_host_with_local_outcomes(tmp_path)
    bundle = _send(tmp_path, [raw(ARXIV_ID, title="new title")], ok={ARXIV_ID})
    import_bundle(ingest, tmp_path / "ingest", bundle)
    once = _row(ingest, ARXIV_ID)
    import_bundle(ingest, tmp_path / "ingest", bundle)
    assert _row(ingest, ARXIV_ID) == once
