"""arXiv HTML -> Markdown -> SQL -> Qdrant payload, on real data, in isolation.

Nothing is faked: a real arXiv LaTeXML page (Mixtral, 2401.04088), the real
conversion CLI, the real preindex script writing a real SQLite database, the
real chunk dump payload builder and the one papers payload projector. Every
root (harvest, corpus, DATA_DIR/db) lives in a temporary directory, so no
production database or collection is touched.
"""
from __future__ import annotations

import gzip
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from app.services.pipeline.arxiv_stages import ArxivStageContext, arxiv_extract_probe, validate_slice_key
from app.services.pipeline.stages import build_preindex_cmd
from core.db.models import Paper as DBPaper
from core.storage.papers_payload import project_papers_payload
from pipeline.paper_crawler.arxiv_corpus import (
    LEGACY_SLICE,
    paper_id_for,
    slice_of,
    split_authors,
)
from pipeline.paper_crawler.arxiv_html import ArxivRawRecord, HarvestStore, html_path
from pipeline.paper_crawler.arxiv_html_convert import parse_arxiv_html, to_markdown

REPO = Path(__file__).resolve().parents[1]
MIXTRAL = "2401.04088"
LOCAL_HTML = REPO / "data" / "arxiv" / "cs_html" / "html" / "2401" / f"{MIXTRAL}.html.gz"


def _first_hop_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 10808), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def mixtral_html() -> bytes:
    """The real page: the harvested copy when present, else one request
    through the local first hop to arXiv's programmatic host."""
    if LOCAL_HTML.exists():
        return gzip.decompress(LOCAL_HTML.read_bytes())
    if not _first_hop_up():
        pytest.skip("no harvested copy and local first hop 10808 is down")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": "http://127.0.0.1:10808"}))
    return opener.open(f"https://export.arxiv.org/html/{MIXTRAL}", timeout=120).read()


def test_slice_keys():
    assert slice_of("2409.00001") == 202409
    assert slice_of("2312.12345") == 202312
    assert slice_of("2311.99999") == LEGACY_SLICE
    assert slice_of("cs/0101001") == LEGACY_SLICE
    validate_slice_key(202409)
    validate_slice_key(LEGACY_SLICE)
    for bad in (2024, 202413, 202310, 202300):
        with pytest.raises(ValueError):
            validate_slice_key(bad)
    assert paper_id_for("cs/0101001") == "arxiv-cs-0101001"
    assert split_authors("A. B, C D, and E F") == ["A. B", "C D", "E F"]
    assert split_authors("Wei Zhang and Taejoon Kim") == ["Wei Zhang", "Taejoon Kim"]


def test_real_page_structure(mixtral_html):
    doc = parse_arxiv_html(mixtral_html, MIXTRAL)
    assert doc.title == "Mixtral of Experts"
    assert doc.version == "v1"
    titles = [s.title for s in doc.sections]
    assert titles[:3] == ["Introduction", "Architectural details", "Sparse Mixture of Experts"]
    assert titles[-1] == "Figures and tables"  # floats LaTeXML parked after the body
    assert len(doc.references) == 35
    assert any(b.kind == "equation" and b.text == r"$$\sum_{i=0}^{n-1}G(x)_{i}\cdot E_{i}(x).$$"
               for b in doc.blocks)
    table2 = next(b for b in doc.blocks if b.kind == "table" and b.label == "Table 2")
    assert table2.caption.startswith("Comparison of Mixtral with Llama")
    assert "| Mixtral 8x7B | 13B | 70.6% |" in table2.text
    assert all(b.section is not None for b in doc.blocks)

    md = to_markdown(doc)
    assert md.startswith("# Mixtral of Experts\n\n## Abstract\n\nWe introduce Mixtral 8x7B")
    assert "## 2.1 Sparse Mixture of Experts" in md
    assert md.index("## 6 Conclusion") < md.index("## Figures and tables")
    # The bibliography stays in the JSON and out of the indexed Markdown.
    assert any("Program synthesis with large language models" in r.text for r in doc.references)
    assert "Program synthesis with large language models" not in md


def _real_page(arxiv_id: str) -> bytes:
    local = REPO / "data" / "arxiv" / "cs_html" / "html" / arxiv_id[:4] / f"{arxiv_id}.html.gz"
    if local.exists():
        return gzip.decompress(local.read_bytes())
    if not _first_hop_up():
        pytest.skip(f"no harvested copy of {arxiv_id} and local first hop 10808 is down")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": "http://127.0.0.1:10808"}))
    return opener.open(f"https://export.arxiv.org/html/{arxiv_id}", timeout=120).read()


@pytest.mark.parametrize("arxiv_id, must_contain", [
    # An unclosed float: LaTeXML nested section 4 onwards inside a figure panel.
    ("2609.01515", ["## 4.1.2 Configurations", "By default, we uniformly sample 16 frames"]),
    # Framed prompt boxes carry ltx_break, which once meant "drop as a line break".
    ("2609.01575", ["Analyze this document page image and classify it",
                    "STEP 2: Modify the current prompt to align with the modified expected answer"]),
    # tcolorbox prompt drawn as an svg: the text lives in foreignObject.
    ("2609.00455", ["You are in the middle of a room. Looking quickly around you"]),
    # Acronym glossary: a headingless section holding a dl.
    ("2609.01478", ["- 3GPP: Third Generation Partnership Project"]),
])
def test_real_page_content_is_not_dropped(arxiv_id, must_contain):
    md = to_markdown(parse_arxiv_html(_real_page(arxiv_id), arxiv_id))
    for text in must_contain:
        assert text in md, f"{arxiv_id}: missing {text!r}"


def _run(cmd: list[str], env: dict) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=900)
    assert proc.returncode == 0, f"{cmd}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}"
    return proc


def test_html_to_sql_to_papers_payload(mixtral_html):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        hroot, croot, data_dir = tmp / "harvest", tmp / "corpus", tmp / "data"
        (data_dir / "db").mkdir(parents=True)

        store = HarvestStore(hroot / "arxiv_cs.sqlite3")
        store.upsert([ArxivRawRecord(
            arxiv_id=MIXTRAL, datestamp="2024-01-09", categories="cs.LG cs.CL",
            title="Mixtral of Experts",
            authors="Albert Q. Jiang, Alexandre Sablayrolles, and Antoine Roux",
            abstract="We introduce Mixtral 8x7B, a Sparse Mixture of Experts language model.",
            comments=None, license=None, latest_version="v1",
            latest_version_date="2024-01-08T18:47:34+00:00",
            first_version_date="2024-01-08T18:47:34+00:00",
            latest_size="100kb", content_version="v1",
            content_version_date="2024-01-08T18:47:34+00:00", withdrawn=False,
        )])
        store.mark(MIXTRAL, "ok", html_version="v1", raw_bytes=len(mixtral_html), gz_bytes=0)
        store.close()
        dest = html_path(hroot, MIXTRAL)
        dest.parent.mkdir(parents=True)
        dest.write_bytes(gzip.compress(mixtral_html))

        env = dict(os.environ, DATA_DIR=str(data_dir), ARXIV_HARVEST_ROOT=str(hroot),
                   ARXIV_CORPUS_ROOT=str(croot), PYTHONIOENCODING="utf-8")
        ctx = ArxivStageContext(venue="arXiv", year=20240101, store=None, task_id=0,
                                corpus_root=croot, harvest_root=hroot)
        assert (arxiv_extract_probe(ctx).done, arxiv_extract_probe(ctx).total) == (0, 1)
        _run([sys.executable, "scripts/collect/arxiv_cs_corpus.py", "convert", "--slice", "20240101",
              "--workers", "1"], env)
        assert (arxiv_extract_probe(ctx).done, arxiv_extract_probe(ctx).total) == (1, 1)
        assert ctx.manifest_slice_extracted() == 1
        assert ctx.harvest_counts() == {"ok": 1}

        # The real schema, created in the temporary DATA_DIR's database.
        _run([sys.executable, "-c",
              "from core.db.database import engine, Base; import core.db.models; "
              "Base.metadata.create_all(engine)"], env)
        preindex = build_preindex_cmd(ctx)
        assert preindex[preindex.index("--manifest-path") + 1] == str(ctx.manifest_path)
        assert "--year" not in preindex
        _run(preindex, env)

        db_file = data_dir / "db" / "retrieval.db"
        db = sqlite3.connect(db_file)
        row = db.execute(
            "SELECT paper_id, venue, year, venue_tier, track_type, arxiv_id, authors_json "
            "FROM papers").fetchall()
        assert len(row) == 1
        paper_id, venue, year, tier, track, arxiv_id, authors = row[0]
        assert (paper_id, venue, year, tier, track, arxiv_id) == (
            "arxiv-2401.04088", "arXiv", 2024, "preprint", "preprint", MIXTRAL)
        assert "Alexandre Sablayrolles" in authors
        n_chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert n_chunks > 5

        # The slice's own count clause (the preindex probe) sees exactly this paper.
        engine = create_engine(f"sqlite:///{db_file}")
        with Session(engine) as session:
            assert session.query(func.count(DBPaper.paper_id)).filter(
                ctx.paper_count_clause(DBPaper)).scalar() == 1
            other = ArxivStageContext(venue="arXiv", year=20240100, store=None, task_id=0,
                                      corpus_root=croot, harvest_root=hroot)
            assert session.query(func.count(DBPaper.paper_id)).filter(
                other.paper_count_clause(DBPaper)).scalar() == 0
        engine.dispose()  # release the file so the temporary directory can go

        # Chunk export -> the papers collection's payload, as ingest would write it.
        sys.path.insert(0, str(REPO / "scripts" / "remote_embed"))
        try:
            from dump_chunks import build_payload, iter_chunks
        finally:
            sys.path.pop(0)
        payloads = [build_payload(r) for r in iter_chunks(db)]
        db.close()
        assert len(payloads) == n_chunks
        sections = {p.get("section_name") for p in payloads}
        assert "2.1 Sparse Mixture of Experts" in sections
        assert not any("Program synthesis with large language models" in p.get("content", "")
                       for p in payloads)
        for raw in payloads:
            projected = project_papers_payload({**raw, "content": raw.get("content", "x"),
                                                "doc_id": raw.get("doc_id", "1"), "level": 2})
            assert projected["venue"] == "arXiv"
            assert projected["venue_tier"] == "preprint"
            assert projected["paper_id"] == "arxiv-2401.04088"
            assert projected["year"] == 2024
