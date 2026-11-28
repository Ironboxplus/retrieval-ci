import argparse
import json
import sqlite3

from scripts.reconcile_paper_chunk_schema import reconcile_slice


def test_reconcile_can_select_an_exact_paper_id_file(tmp_path):
    db = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
      create table documents (id integer primary key, title text, file_path text);
      create table papers (paper_id text primary key, title text, doc_id integer);
      create table chunks (id integer primary key, doc_id integer, content text, level integer,
        chunk_index integer, start_idx integer, end_idx integer, token_count integer, metadata_json text);
    """)
    md = tmp_path / "paper.md"
    md.write_text("# Title\n\n## Abstract\n\n" + "abstract text " * 20 +
                   "\n\n## Body\n\n" + "body text " * 40, encoding="utf-8")
    conn.execute("insert into documents values (1, 'Title', ?)", (str(md),))
    conn.execute("insert into papers values ('arxiv-2601.1', 'Title', 1)")
    conn.execute("insert into documents values (2, 'Other', ?)", (str(md),))
    conn.execute("insert into papers values ('arxiv-2601.2', 'Other', 2)")
    conn.commit(); conn.close()
    ids = tmp_path / "ids.txt"; ids.write_text("arxiv-2601.1\n", encoding="utf-8")
    report = reconcile_slice(argparse.Namespace(
        db=db, paper_ids_file=ids, conf=None, year=None, limit=0, execute=False,
        commit_batch=50, checkpoint_interval=100, report_path=tmp_path / "report.json",
    ))
    assert report["papers"] == 1 and report["failed"] == 0
