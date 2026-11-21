"""TDD: the 5 stage builders (download / extract / preindex / embed / ingest).

Each stage is decomposed into PURE, offline-testable pieces:
  - build_*_cmd(ctx)      → exact command (list/str) — asserted arg-by-arg
  - *_probe(ctx)          → Progress from temp dirs / fake store state
  - *_gate / *_preflight  → GateResult verdicts
  - make_*_stage(ctx)     → StageSpec assembling them

NO real download / ssh / GPU / Qdrant runs here. The thin run() wiring is checked
by monkeypatching the runner and asserting it is called with the BUILT command.
Everything is generic for ANY (venue, year).
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from core.db import models  # noqa: F401
from core.db.models import Chunk as DBChunk, Document as DBDocument, Paper as DBPaper, SourceType
from app.services.pipeline.spec import StageSpec, Progress, GateResult
from app.services.pipeline.store import TaskStore
from app.services.pipeline.context import StageContext
from app.services.pipeline import stages as st


# --------------------------------------------------------------------------- db
@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        s = Local()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return factory


@pytest.fixture
def store(session_factory):
    return TaskStore(session_factory)


# ---------------------------------------------------------------------- corpus
def _rec(conf, year, *, paper_id, markdown_path=None, pdf_path=None):
    return {
        "paper_id": paper_id, "title": "t", "authors": [],
        "conference": conf, "year": year, "pdf_url": "u", "paper_page_url": "p",
        "source_name": "s", "markdown_path": markdown_path, "pdf_path": pdf_path,
    }


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    (root / "manifests").mkdir(parents=True)
    records = [
        _rec("CVPR", 2026, paper_id="cvpr-2026-a", markdown_path="/m/a.md"),
        _rec("CVPR", 2026, paper_id="cvpr-2026-b", markdown_path="/m/b.md"),
        _rec("CVPR", 2026, paper_id="cvpr-2026-c", markdown_path=None),
        _rec("CVPR", 2026, paper_id="cvpr-2026-d", markdown_path="/m/d.md"),
    ]
    (root / "manifests" / "paper_manifest.json").write_text(json.dumps(records), encoding="utf-8")
    return root


@pytest.fixture
def ctx(corpus, store, tmp_path):
    """A CVPR-2026 context with SMB md dir pointed at a real temp dir."""
    task_id = store.create_task(task_type="ingest_pipeline")
    smb_root = tmp_path / "smb"
    return StageContext(
        venue="CVPR", year=2026, store=store, task_id=task_id,
        corpus_root=corpus, project_root=tmp_path / "proj",
        arc_smb_unc=str(smb_root), concurrency=8, world_size=8,
    )


def _make_md(ctx, name, body):
    d = Path(ctx.smb_md_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")


def _make_pdfs(ctx, n):
    ctx.pdf_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (ctx.pdf_dir / f"p{i}.pdf").write_bytes(b"%PDF-1.4")


def _make_slice_pdfs(ctx, n):
    """The PDFs of the first n manifest papers of the slice, under the names the downloader gives them."""
    ctx.pdf_dir.mkdir(parents=True, exist_ok=True)
    for record in ctx.manifest_slice()[:n]:
        (ctx.pdf_dir / f"{record['paper_id']}.pdf").write_bytes(b"%PDF-1.4")


# ======================================================================= STAGE 1
def test_download_cmd_is_generic(ctx):
    cmd = st.build_download_cmd(ctx)
    assert cmd == [
        sys.executable, "scripts/download_subset.py",
        "--conf", "CVPR", "--year", "2026", "--concurrency", "8",
    ]


def test_icml_2026_download_cmd_is_rate_limited(corpus, store, tmp_path):
    task_id = store.create_task(task_type="ingest_pipeline")
    c = StageContext(
        venue="ICML", year=2026, store=store, task_id=task_id,
        corpus_root=corpus, project_root=tmp_path / "proj",
        concurrency=8, world_size=8,
    )
    cmd = st.build_download_cmd(c)
    assert cmd == [
        sys.executable, "scripts/download_subset.py",
        "--conf", "ICML", "--year", "2026",
        "--concurrency", "1",
        "--min-interval-seconds", "5",
    ]


def test_download_probe_counts_pdf_vs_manifest(ctx):
    _make_pdfs(ctx, 3)
    pr = st.download_probe(ctx)
    assert pr.done == 3 and pr.total == 4  # 4 CVPR records in manifest


def test_download_gate_passes_at_90pct(ctx):
    _make_slice_pdfs(ctx, 4)  # 4/4
    assert st.download_gate(ctx).ok


def test_download_gate_fails_below_90pct(ctx):
    _make_slice_pdfs(ctx, 2)  # 2/4 = 50%
    g = st.download_gate(ctx)
    assert not g.ok and "2/4" in g.reason


def test_download_gate_passes_when_total_zero(corpus, store, tmp_path):
    # A venue/year with no manifest records → nothing to download → ok.
    task_id = store.create_task(task_type="ingest_pipeline")
    c = StageContext(venue="NOPE", year=1999, store=store, task_id=task_id, corpus_root=corpus)
    assert st.download_gate(c).ok


# ---- a slice whose missing papers were all tried on the downloading host may go on
@pytest.fixture
def ten_ctx(store, tmp_path):
    """Ten CVPR 2026 papers in the manifest, nothing downloaded yet."""
    root = tmp_path / "corpus10"
    (root / "manifests").mkdir(parents=True)
    records = [_rec("CVPR", 2026, paper_id=f"cvpr-2026-{n:02d}") for n in range(10)]
    (root / "manifests" / "paper_manifest.json").write_text(json.dumps(records), encoding="utf-8")
    return StageContext(
        venue="CVPR", year=2026, store=store, task_id=store.create_task(task_type="ingest_pipeline"),
        corpus_root=root, project_root=tmp_path / "proj",
    )


def _put_pdfs(ctx, numbers):
    ctx.pdf_dir.mkdir(parents=True, exist_ok=True)
    for n in numbers:
        (ctx.pdf_dir / f"cvpr-2026-{n:02d}.pdf").write_bytes(b"%PDF-1.4")


def _put_record(ctx, kind, n, body):
    """A record file of the downloading host: kind is the directory name under corpus_root."""
    directory = ctx.corpus_root / kind / "cvpr" / "2026"
    directory.mkdir(parents=True, exist_ok=True)
    text = body if isinstance(body, str) else json.dumps(body)
    (directory / f"cvpr-2026-{n:02d}.json").write_text(text, encoding="utf-8")


def _tried(ctx, n):
    _put_record(ctx, "download_attempts", n, {
        "paper_id": f"cvpr-2026-{n:02d}", "conference": "CVPR", "year": 2026,
        "attempts": [{"at": "2026-11-16T01:00:00+00:00", "error": "HTTP 403"}]})


def test_download_gate_counts_only_the_slices_own_pdfs(ten_ctx):
    _put_pdfs(ten_ctx, range(8))
    for i in range(5):  # PDFs of other papers sharing the directory
        (ten_ctx.pdf_dir / f"another-slice-{i}.pdf").write_bytes(b"%PDF-1.4")
    g = st.download_gate(ten_ctx)
    assert not g.ok and g.reason.startswith("only 8/10 PDFs downloaded"), g.reason


def test_download_gate_passes_when_every_missing_paper_was_tried_and_says_so(ten_ctx):
    _put_pdfs(ten_ctx, range(8))
    _tried(ten_ctx, 8)
    _tried(ten_ctx, 9)
    g = st.download_gate(ten_ctx)
    assert g.ok
    assert g.reason == "8/10 PDFs; 2 still missing after download attempts (kept on the refill list)"


def test_download_gate_fails_while_one_missing_paper_was_never_tried(ten_ctx):
    _put_pdfs(ten_ctx, range(8))
    _tried(ten_ctx, 8)
    g = st.download_gate(ten_ctx)
    assert not g.ok
    assert g.reason == "only 8/10 PDFs downloaded (1 tried, 1 never tried)"


def test_download_gate_does_not_count_a_record_that_proves_no_try(ten_ctx):
    _put_pdfs(ten_ctx, range(6))
    _put_record(ten_ctx, "download_attempts", 6, {"paper_id": "cvpr-2026-09", "attempts": [{"error": "x"}]})  # another paper
    _put_record(ten_ctx, "download_attempts", 7, {"paper_id": "cvpr-2026-07", "attempts": []})  # no attempt
    _put_record(ten_ctx, "download_attempts", 8, "{not json")  # does not parse
    _put_record(ten_ctx, "download_attempts", 9, {"paper_id": "cvpr-2026-09", "attempts": "once"})  # not a list
    g = st.download_gate(ten_ctx)
    assert not g.ok and g.reason == "only 6/10 PDFs downloaded (0 tried, 4 never tried)"


def test_download_gate_counts_an_identity_rejection_as_a_try(ten_ctx):
    _put_pdfs(ten_ctx, range(8))
    _put_record(ten_ctx, "rejected_identity", 8, {
        "paper_id": "cvpr-2026-08", "rejections": [{"url": "u", "full_text_source": "arxiv", "reason": "title"}]})
    _tried(ten_ctx, 9)
    assert st.download_gate(ten_ctx).ok
    # A rejection record that names another paper or holds no rejection is not a try.
    _put_record(ten_ctx, "rejected_identity", 9, {"paper_id": "cvpr-2026-08", "rejections": [{"url": "u"}]})
    (ten_ctx.corpus_root / "download_attempts" / "cvpr" / "2026" / "cvpr-2026-09.json").unlink()
    g = st.download_gate(ten_ctx)
    assert not g.ok and g.reason == "only 8/10 PDFs downloaded (1 tried, 1 never tried)"


def test_download_gate_at_90pct_without_any_record_is_unchanged(ten_ctx):
    _put_pdfs(ten_ctx, range(9))
    g = st.download_gate(ten_ctx)
    assert g.ok and g.reason == ""


def test_make_download_stage(ctx):
    spec = st.make_download_stage(ctx)
    assert isinstance(spec, StageSpec)
    assert spec.key == "download"
    assert spec.preflight is None


def test_download_run_invokes_runner_with_built_cmd(ctx, monkeypatch):
    seen = {}

    def fake_run_local(cmd, cwd, sink, **kw):
        seen["cmd"] = cmd
        seen["cwd"] = cwd
        return 0

    monkeypatch.setattr(st, "run_local", fake_run_local)
    # make_download_stage never builds this stage on this host (it refuses 'local'); the builder is called directly.
    st.make_download_stage_local(ctx).run(ctx, lambda *a, **k: None)
    assert seen["cmd"] == st.build_download_cmd(ctx)
    assert seen["cwd"] == ctx.project_root


# ======================================================================= STAGE 2
def test_extract_remote_cmd_is_generic(ctx):
    cmd = st.build_extract_remote_cmd(ctx)
    assert cmd == (
        "bash /home/kss/arc/retrival/scripts/remote_marker/run_marker_kss.sh "
        "/mnt/lsb/retrival/papers/pdf/cvpr/2026 "
        "/mnt/lsb/retrival/papers/markdown/cvpr/2026 8"
    )


def test_run_marker_kss_quotes_input_output_inside_tmux_cmd():
    """W80b: even with shlex-quoted shell cmd reaching 4090, the launcher script
    itself re-tokenizes ${INPUT}/${OUTPUT} when building the tmux child cmd.
    Without quoting inside the script, --input gets split on the embedded
    space and argparse silently fails. Regression: USENIX Security 2025 (task
    91) — 6 tmux ranks launched, 0 MDs produced, 20-min stall timeout."""
    from pathlib import Path as _P
    script = _P(__file__).resolve().parent.parent / "scripts" / "remote_marker" / "run_marker_kss.sh"
    body = script.read_text(encoding="utf-8")
    # The python invocation inside the tmux cmd must pass INPUT/OUTPUT quoted
    assert '--input \\"${INPUT}\\"' in body, body
    assert '--output \\"${OUTPUT}\\"' in body, body


def test_extract_remote_cmd_quotes_paths_with_spaces(corpus, store, tmp_path):
    """W80: venue with space (USENIX Security) must shell-quote the SMB paths,
    otherwise `bash run_marker.sh /mnt/.../usenix security 6` is parsed as 4 args
    and the launcher fails with `((: ...: syntax error`."""
    task_id = store.create_task(task_type="ingest_pipeline")
    c = StageContext(
        venue="USENIX Security", year=2024, store=store, task_id=task_id,
        corpus_root=corpus, project_root=tmp_path / "proj",
        arc_smb_unc=str(tmp_path / "smb"), concurrency=8, world_size=6,
    )
    cmd = st.build_extract_remote_cmd(c)
    # SMB paths contain a space → must appear single-quoted or backslash-escaped
    assert "'/mnt/lsb/retrival/papers/pdf/usenix security/2024'" in cmd, cmd
    assert "'/mnt/lsb/retrival/papers/markdown/usenix security/2024'" in cmd, cmd


def test_extract_world_size_stacks_ranks_per_gpu(ctx):
    """ranks_per_gpu × #preflight-passed GPUs decides effective world size.

    4090 case: 3 idle GPUs preflight pass, ranks_per_gpu=2 → WORLD=6
    (sh stripes ranks 0..5 across GPUs 1,2,3 → 2 marker procs per card).
    """
    ctx.ranks_per_gpu = 2
    ctx.state_set(gpu_ids="1,2,3")
    cmd = st.build_extract_remote_cmd(ctx)
    assert cmd.endswith(" 6"), f"expected world=6, got: {cmd}"


def test_extract_world_size_single_rank_per_gpu(ctx):
    """ranks_per_gpu=1 (kss default) → world == #idle_gpus, no stacking."""
    ctx.ranks_per_gpu = 1
    ctx.state_set(gpu_ids="0,1,2,3,4,5,6,7")
    cmd = st.build_extract_remote_cmd(ctx)
    assert cmd.endswith(" 8")


def test_extract_world_size_fallback_when_no_gpu_state(ctx):
    """Before preflight runs (no state.gpu_ids), fall back to ctx.world_size."""
    ctx.ranks_per_gpu = 2  # would normally stack, but no GPU state yet
    cmd = st.build_extract_remote_cmd(ctx)
    assert cmd.endswith(" 8")  # ctx.world_size=8 fallback


def test_extract_env_prefix_from_state_gpu_ids(ctx):
    ctx.state_set(gpu_ids="0,1,3")
    pre = st.extract_env_prefix(ctx)
    assert pre == "GPU_IDS=0,1,3 STAGGER=3 "


def test_extract_preflight_sets_gpu_ids(ctx, monkeypatch):
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: [0, 1, 2])
    g = st.extract_preflight(ctx)
    assert g.ok
    assert ctx.state_get("gpu_ids") == "0,1,2"


def test_extract_preflight_fails_when_no_gpu(ctx, monkeypatch):
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: [])
    g = st.extract_preflight(ctx)
    assert not g.ok and "GPU" in g.reason


def test_extract_probe_counts_md_vs_pdf(ctx):
    _make_pdfs(ctx, 4)
    _make_md(ctx, "a.md", "# A\nbody")
    _make_md(ctx, "b.md", "# B\nbody")
    pr = st.extract_probe(ctx)
    assert pr.done == 2 and pr.total == 4


def test_extract_gate_passes_on_good_md(ctx):
    _make_pdfs(ctx, 4)
    for i in range(4):
        _make_md(ctx, f"g{i}.md", "# Header\n" + ("x" * 300))
    assert st.extract_gate(ctx).ok


def test_extract_gate_fails_on_empty_md(ctx):
    _make_pdfs(ctx, 4)
    for i in range(3):
        _make_md(ctx, f"g{i}.md", "# Header\n" + ("x" * 300))
    _make_md(ctx, "bad.md", "tiny")  # < 200 bytes, no header
    g = st.extract_gate(ctx)
    assert not g.ok and "garbage" in g.reason.lower() or "empty" in g.reason.lower()


def test_extract_gate_fails_on_no_header(ctx):
    _make_pdfs(ctx, 1)
    _make_md(ctx, "nohdr.md", "x" * 500)  # big enough but no '#' header
    g = st.extract_gate(ctx)
    assert not g.ok


def test_extract_gate_fails_below_95pct(ctx):
    _make_pdfs(ctx, 10)
    # only 5 good md (50%) → ratio gate fails even though samples are fine
    for i in range(5):
        _make_md(ctx, f"g{i}.md", "# Header\n" + ("x" * 300))
    assert not st.extract_gate(ctx).ok


def test_extract_run_polls_until_complete(ctx, monkeypatch):
    """run() does: SMB warmup (ssh find) → ssh-launch marker → poll → sync MD
    paths into manifest. Each sub-step is exactly one call; a regression that
    drops any of them is caught here."""
    calls = {"remote_cmds": [], "local_cmds": [], "sleep": 0}

    def fake_run_remote(target, cmd, sink, **kw):
        calls["remote_cmds"].append(cmd)
        return 0

    def fake_run_local(cmd, cwd, sink, **kw):
        calls["local_cmds"].append(cmd)
        return 0

    seq = iter([Progress(0, 4), Progress(2, 4), Progress(4, 4)])
    monkeypatch.setattr(st, "run_remote", fake_run_remote)
    monkeypatch.setattr(st, "run_local", fake_run_local)
    monkeypatch.setattr(st, "gpu_health", lambda *a, **k: [0])

    def fake_sleep(_s):
        calls["sleep"] += 1

    spec = st.make_extract_stage(ctx, probe_fn=lambda c: next(seq), sleep_fn=fake_sleep, poll_s=0.0)
    spec.run(ctx, lambda *a, **k: None)
    # 2 ssh calls: warmup `find` + marker launch
    assert len(calls["remote_cmds"]) == 2
    assert "find " in calls["remote_cmds"][0]
    assert calls["sleep"] >= 1
    # 1 local call: sync_new_conf_md_smb.py to populate manifest.markdown_path
    assert len(calls["local_cmds"]) == 1
    assert "sync_new_conf_md_smb.py" in " ".join(calls["local_cmds"][0])


# ======================================================================= STAGE 3
def test_preindex_cmd_is_generic(ctx):
    cmd = st.build_preindex_cmd(ctx)
    assert cmd == [
        sys.executable, "scripts/preindex_md_to_sql_only.py",
        "--conf", "CVPR", "--year", "2026",
        "--report-path", str(st.preindex_report_path(ctx)), "--chunker-type", "section_l1",
    ]
    # Per task, so concurrent tasks never overwrite each other's replaced/new lists.
    assert st.preindex_report_path(ctx).name == f"preindex_{ctx.task_id}.json"


def _seed_papers(session_factory, venue, year, n):
    with session_factory() as s:
        for i in range(n):
            doc = DBDocument(title=f"t{i}", source="x", source_type=SourceType.paper)
            s.add(doc)
            s.flush()
            s.add(DBPaper(paper_id=f"{venue.lower()}-{year}-{i}", doc_id=doc.id, title=f"t{i}", year=year, venue=venue))


def test_preindex_probe_counts_papers(ctx, session_factory):
    _seed_papers(session_factory, "CVPR", 2026, 3)
    pr = st.preindex_probe(ctx, session_factory=session_factory)
    assert pr.done == 3
    # total = manifest_slice_extracted = 3 (a,b,d have markdown_path)
    assert pr.total == 3


def test_preindex_gate_passes_at_95pct(ctx, session_factory):
    _seed_papers(session_factory, "CVPR", 2026, 3)  # 3/3
    assert st.preindex_gate(ctx, session_factory=session_factory).ok


def test_preindex_gate_fails_below_95pct(ctx, session_factory):
    _seed_papers(session_factory, "CVPR", 2026, 1)  # 1/3
    assert not st.preindex_gate(ctx, session_factory=session_factory).ok


def _slice_manifest(ctx, *, papers, with_markdown):
    records = [
        _rec("CVPR", 2026, paper_id=f"cvpr-2026-{i}", markdown_path=f"/m/{i}.md" if i < with_markdown else None)
        for i in range(papers)
    ]
    ctx.manifest_path.write_text(json.dumps(records), encoding="utf-8")


def test_preindex_is_complete_when_every_paper_that_has_markdown_is_in_sql(ctx, session_factory):
    # task 1302: 297 PDFs, 259 verified Markdown, 38 refused. The refused ones have no Markdown to index.
    _slice_manifest(ctx, papers=10, with_markdown=8)
    _make_pdfs(ctx, 10)
    _seed_papers(session_factory, "CVPR", 2026, 8)
    pr = st.preindex_probe(ctx, session_factory=session_factory)
    assert (pr.done, pr.total) == (8, 8) and pr.is_complete
    assert st.preindex_gate(ctx, session_factory=session_factory).ok


def test_preindex_gate_still_fails_when_papers_with_markdown_are_missing_from_sql(ctx, session_factory):
    _slice_manifest(ctx, papers=10, with_markdown=8)
    _make_pdfs(ctx, 10)
    _seed_papers(session_factory, "CVPR", 2026, 6)  # 6/8: a real gap, whatever the refusals
    gate = st.preindex_gate(ctx, session_factory=session_factory)
    assert not gate.ok and "6/8" in gate.reason


def test_preindex_with_no_recorded_markdown_is_measured_against_the_pdfs(ctx, session_factory):
    # nothing recorded yet must not read as "nothing owed"
    _slice_manifest(ctx, papers=10, with_markdown=0)
    _make_pdfs(ctx, 10)
    assert st.preindex_probe(ctx, session_factory=session_factory).total == 10
    assert not st.preindex_gate(ctx, session_factory=session_factory).ok


def test_preindex_run_runs_the_script_and_forgets_the_earlier_dump(ctx, session_factory, monkeypatch):
    # An earlier run left a chunk-id range and a shard range behind: this run may change the
    # task's text, so none of it is kept (the next embed dumps from the preindex report).
    ctx.state_set(chunk_id_before=5, chunk_id_after=9, dump_start_shard=0, shard_start=0, shard_end=31)
    seen = {}
    monkeypatch.setattr(st, "run_local", lambda cmd, cwd, sink, **kw: seen.update(cmd=cmd) or 0)
    spec = st.make_preindex_stage(ctx, session_factory=session_factory)
    assert spec.skip_when_satisfied is False  # an upsert: papers already in SQL may need a new text
    spec.run(ctx, lambda *a, **k: None)
    assert seen["cmd"] == st.build_preindex_cmd(ctx)
    assert [ctx.state_get(k) for k in ("chunk_id_before", "chunk_id_after", "dump_start_shard",
                                      "shard_start", "shard_end")] == [None] * 5


# ======================================================================= STAGE 4
def test_embed_dump_cmd_without_a_preindex_report_uses_the_state_chunk_range_and_shard(ctx):
    # No preindex report: an external start at embed, whose range is the task state's.
    ctx.state_set(chunk_id_before=100)
    cmd = st.build_embed_dump_cmd(ctx, start_shard=5)
    assert cmd == [
        sys.executable, "scripts/remote_embed/dump_chunks.py",
        "--out", str(st._embed_chunks_dir(ctx)),
        "--min-chunk-id", "101", "--start-shard", "5",
    ]
    # Task-scoped: <project>/data/remote_embed/tasks/<task_id>/chunks, never the shared directory.
    assert st._embed_chunks_dir(ctx) == (
        ctx.project_root / "data" / "remote_embed" / "tasks" / str(ctx.task_id) / "chunks")


def test_embed_remote_cmd_and_env(ctx):
    ctx.state_set(gpu_ids="0,2")
    assert st.build_embed_remote_cmd(ctx) == "bash /home/kss/arc/retrival/scripts/remote_embed/run_embed_kss.sh"
    assert st.embed_env_prefix(ctx) == 'GPUS="0 2" '


def test_embed_preflight_sets_gpu(ctx, monkeypatch):
    monkeypatch.setattr(st, "gpu_health", lambda *a, **k: [1, 2])
    assert st.embed_preflight(ctx).ok
    assert ctx.state_get("gpu_ids") == "1,2"


def test_embed_gate_passes_when_new_parquet(ctx):
    ctx.state_set(shard_start=2, shard_end=7)
    assert st.embed_gate(ctx).ok


def test_embed_gate_fails_when_no_new_parquet(ctx):
    ctx.state_set(shard_start=5, shard_end=5)
    assert not st.embed_gate(ctx).ok


def test_count_shard_files_helper(ctx, tmp_path):
    d = tmp_path / "vec"
    d.mkdir()
    (d / "vectors_0.parquet").write_bytes(b"x")
    (d / "vectors_1.parquet").write_bytes(b"x")
    (d / "other.txt").write_text("ignore")
    assert st._count_files(d, "vectors_*.parquet") == 2
    assert st._count_files(tmp_path / "missing", "vectors_*.parquet") == 0


# ======================================================================= STAGE 5
def test_ingest_cmd_uses_shard_start(ctx):
    ctx.state_set(shard_start=3)
    cmd = st.build_ingest_cmd(ctx)
    assert cmd == [
        sys.executable, "scripts/remote_embed/local_ingest.py",
        "--vectors", str(st._embed_vectors_dir(ctx)), "--shard-start", "3",
    ]


def test_ingest_preflight_qdrant_reachable(ctx, monkeypatch):
    monkeypatch.setattr(st, "_qdrant_reachable", lambda: True)
    assert st.ingest_preflight(ctx).ok


def test_ingest_preflight_qdrant_down(ctx, monkeypatch):
    monkeypatch.setattr(st, "_qdrant_reachable", lambda: False)
    g = st.ingest_preflight(ctx)
    assert not g.ok and "qdrant" in g.reason.lower()


def test_ingest_probe_and_gate(ctx, tmp_path, monkeypatch):
    vec = st._embed_vectors_dir(ctx)
    vec.mkdir(parents=True)
    for i in range(5):
        (vec / f"vectors_{i}.parquet").write_bytes(b"x")
    # ingest_done.txt holds parquet FILENAMES (local_ingest.py writes path.name),
    # not bare indices — shards 3 and 4 done (>= shard_start=3)
    (vec / "ingest_done.txt").write_text("vectors_3.parquet\nvectors_4.parquet\n", encoding="utf-8")
    ctx.state_set(shard_start=3)
    pr = st.ingest_probe(ctx)
    assert pr.done == 2  # shards 3,4 done
    assert pr.total == 2  # 5 parquet - shard_start 3 = 2
    assert st.ingest_gate(ctx).ok


def test_ingest_gate_fails_when_incomplete(ctx, tmp_path):
    vec = st._embed_vectors_dir(ctx)
    vec.mkdir(parents=True)
    for i in range(5):
        (vec / f"vectors_{i}.parquet").write_bytes(b"x")
    (vec / "ingest_done.txt").write_text("vectors_3.parquet\n", encoding="utf-8")  # only 1 of 2 done
    ctx.state_set(shard_start=3)
    assert not st.ingest_gate(ctx).ok


# ============================================================ W81 shard accounting
def test_count_shards_in_range_helper(tmp_path):
    d = tmp_path / "vec"
    d.mkdir()
    for i in (120, 121, 125, 130):
        (d / f"vectors_{i:04d}.parquet").write_bytes(b"x")
    (d / "other.txt").write_text("ignore")
    # [121, 126) → 121, 125 (NOT 130, NOT 120)
    assert st._count_shards_in_range(d, "vectors_*.parquet", st._SHARD_RE, 121, 126) == 2
    # open upper bound → 121, 125, 130
    assert st._count_shards_in_range(d, "vectors_*.parquet", st._SHARD_RE, 121, None) == 3
    # missing dir → 0
    assert st._count_shards_in_range(tmp_path / "missing", "vectors_*.parquet", st._SHARD_RE, 0, None) == 0


def test_embed_run_anchors_shard_end_to_dumped_chunks(ctx, monkeypatch):
    """W81: shard_end = dump anchor + #chunk shards THIS task dumped, and
    shard_start is UNIFIED to the dump anchor — NOT a recount of the shared
    vectors/ dir after ship-down (which pulls every venue's parquet)."""
    chunks = st._embed_chunks_dir(ctx)
    chunks.mkdir(parents=True)
    for i in range(120):  # 0..119 already present → next free anchor = 120
        (chunks / f"chunks_{i:04d}.jsonl.gz").write_bytes(b"x")
    # Pre-seed the SHARED vectors/ dir with another venue's parquet far above our
    # range, to prove shard_end does NOT come from counting vectors/.
    vectors = st._embed_vectors_dir(ctx)
    vectors.mkdir(parents=True)
    (vectors / "vectors_0500.parquet").write_bytes(b"x")

    def fake_run_local(cmd, cwd, sink, **kw):
        if "dump_chunks.py" in " ".join(map(str, cmd)):
            for i in (120, 121, 122):  # dump produces 3 new chunk shards
                (chunks / f"chunks_{i:04d}.jsonl.gz").write_bytes(b"x")
        return 0

    monkeypatch.setattr(st, "run_local", fake_run_local)
    monkeypatch.setattr(st, "run_remote", lambda *a, **k: 0)
    monkeypatch.setattr(st, "_poll_until_complete", lambda *a, **k: None)
    st.make_embed_stage(ctx).run(ctx, lambda *a, **k: None)

    assert ctx.state_get("dump_start_shard") == 120
    assert ctx.state_get("shard_start") == 120          # unified to dump anchor
    assert ctx.state_get("shard_end") == 123            # 120 + 3 dumped (NOT 501)


def test_ingest_probe_number_based_with_shard_end(ctx, tmp_path):
    """total = shard_end - shard_start; done counts ONLY our shards in range.
    Another venue's vectors_0140 in the shared dir must not inflate either."""
    vec = st._embed_vectors_dir(ctx)
    vec.mkdir(parents=True)
    for i in (125, 126, 128, 140):  # gap at 127; 140 is another venue's
        (vec / f"vectors_{i:04d}.parquet").write_bytes(b"x")
    (vec / "ingest_done.txt").write_text(
        "vectors_0125.parquet\nvectors_0126.parquet\nvectors_0140.parquet\n", encoding="utf-8"
    )
    ctx.state_set(shard_start=125, shard_end=128)  # our range [125,128) = 3
    pr = st.ingest_probe(ctx)
    assert pr.total == 3        # shard_end - shard_start, NOT count(4) - 125
    assert pr.done == 2         # 125,126 in range & done; 140 excluded (>= end)
    assert not st.ingest_gate(ctx).ok  # 2/3 → incomplete


def test_ingest_done_shards_excludes_above_shard_end(ctx, tmp_path):
    vec = st._embed_vectors_dir(ctx)
    vec.mkdir(parents=True)
    (vec / "ingest_done.txt").write_text(
        "vectors_0125.parquet\nvectors_0140.parquet\n", encoding="utf-8"
    )
    ctx.state_set(shard_start=125, shard_end=128)
    assert st._ingest_done_shards(ctx, 125, 128) == 1   # only 125; 140 excluded
    assert st._ingest_done_shards(ctx, 125, None) == 2  # open bound → both


# ============================== W82 embed poll watches the mounted SMB share
def test_smb_vectors_dir_is_arc_unc_vectors(ctx):
    assert ctx.smb_vectors_dir == f"{ctx.arc_smb_unc}/vectors"


def test_embed_smb_probe_counts_smb_vectors_in_range(ctx, tmp_path, monkeypatch):
    """W82: the poll counts the locally-mounted SMB share (where the KSS worker
    writes), NUMBER-bounded to our shard range — so step 4 finishes the instant
    our shards land instead of dead-waiting (the C-drive copy stays empty until
    ship-down)."""
    smb_vec = tmp_path / "smb" / "vectors"
    smb_vec.mkdir(parents=True)
    for i in (120, 121, 140):  # ours: 120,121; 140 belongs to another venue
        (smb_vec / f"vectors_{i:04d}.parquet").write_bytes(b"x")
    monkeypatch.setattr(st, "_embed_smb_vectors_dir", lambda c: smb_vec)
    ctx.state_set(shard_start=120, shard_end=122)  # range [120,122) = 2
    pr = st.embed_smb_probe(ctx)
    assert pr.total == 2
    assert pr.done == 2          # 120,121 present on SMB; 140 excluded (>= end)


def test_embed_smb_probe_partial(ctx, tmp_path, monkeypatch):
    smb_vec = tmp_path / "smb" / "vectors"
    smb_vec.mkdir(parents=True)
    (smb_vec / "vectors_0120.parquet").write_bytes(b"x")  # only 1 of 3 produced
    monkeypatch.setattr(st, "_embed_smb_vectors_dir", lambda c: smb_vec)
    ctx.state_set(shard_start=120, shard_end=123)
    pr = st.embed_smb_probe(ctx)
    assert pr.done == 1 and pr.total == 3 and not pr.is_complete


def test_embed_stage_probe_is_smb_share(ctx):
    """The embed StageSpec probe (used by the orchestrator poller AND the step-4
    poll) watches the mounted SMB share, not the empty C-drive copy (W82)."""
    spec = st.make_embed_stage(ctx)
    assert spec.probe is st.embed_smb_probe


# ===================================================== W106 LOCAL extract (self-hosted)
def _make_md_local(ctx, name, body):
    d = ctx.local_md_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")


def test_extract_local_cmd_is_the_registered_worker_per_rank(ctx):
    cmd = st.build_extract_local_cmd(ctx, gpu=1, rank=0, world_size=2)
    assert cmd == [
        Path(sys.executable).as_posix(), "scripts/remote_pvl/pvl_worker.py",
        "--input", str(ctx.pdf_dir),
        "--output", str(ctx.local_md_dir),
        "--gpu", "1", "--rank", "0", "--world-size", "2", "--recursive",
    ]


def test_extract_local_preflight_uses_local_gpu_health(ctx, monkeypatch):
    seen = {}

    def fake_gpu_health(target, **kw):
        seen["target"] = target
        seen["kw"] = kw
        return [0, 1]

    monkeypatch.setattr(st, "gpu_health", fake_gpu_health)
    g = st.extract_local_preflight(ctx)
    assert g.ok
    assert seen["target"] is None  # local: NO ssh target
    assert seen["kw"]["min_free_mb"] == 16384 and seen["kw"].get("min_free_ratio", 0.0) == 0.0   # no idle-ratio gate
    assert ctx.state_get("gpu_ids") == "0,1"


def test_extract_local_preflight_fails_when_no_gpu(ctx, monkeypatch):
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: [])
    g = st.extract_local_preflight(ctx)
    assert not g.ok and "GPU" in g.reason


def test_extract_local_probe_counts_local_md_vs_pdf(ctx):
    _make_pdfs(ctx, 4)
    _make_md_local(ctx, "a.md", "# A\nbody")
    _make_md_local(ctx, "b.md", "# B\nbody")
    pr = st.extract_local_probe(ctx)
    assert pr.done == 2 and pr.total == 4


def test_extract_local_gate_passes_on_good_md(ctx):
    _make_pdfs(ctx, 4)
    for i in range(4):
        _make_md_local(ctx, f"g{i}.md", "# Header\n" + ("x" * 300))
    assert st.extract_local_gate(ctx).ok


def test_extract_local_gate_fails_on_empty_md(ctx):
    _make_pdfs(ctx, 4)
    for i in range(3):
        _make_md_local(ctx, f"g{i}.md", "# Header\n" + ("x" * 300))
    _make_md_local(ctx, "bad.md", "tiny")
    assert not st.extract_local_gate(ctx).ok


def test_extract_local_gate_fails_below_95pct(ctx):
    _make_pdfs(ctx, 10)
    for i in range(5):
        _make_md_local(ctx, f"g{i}.md", "# Header\n" + ("x" * 300))
    assert not st.extract_local_gate(ctx).ok


def test_plan_local_ranks_stacks_per_gpu(ctx):
    ctx.ranks_per_gpu = 2
    ctx.state_set(gpu_ids="1,2,3")
    plan = st._plan_local_ranks(ctx)
    # world = 3 gpus * 2 = 6; rank r → gpus[r % 3]
    assert [(r, g, w) for (r, g, w) in plan] == [
        (0, 1, 6), (1, 2, 6), (2, 3, 6), (3, 1, 6), (4, 2, 6), (5, 3, 6),
    ]


def test_extract_local_run_launches_one_proc_per_rank(ctx, monkeypatch):
    """run() launches N local extractor procs (one per rank, distributed across the
    preflight GPUs) then syncs MD paths via the LOCAL slice script — never ssh,
    never run_marker_kss.sh, never the SMB sync."""
    ctx.ranks_per_gpu = 1
    ctx.state_set(gpu_ids="0,1")
    calls = {"local_cmds": [], "remote_cmds": []}

    monkeypatch.setattr(st, "run_local", lambda cmd, cwd, sink, **kw: calls["local_cmds"].append(cmd) or 0)
    monkeypatch.setattr(st, "run_remote", lambda *a, **k: calls["remote_cmds"].append(a) or 0)

    st.make_extract_stage_local(ctx).run(ctx, lambda *a, **k: None)

    assert calls["remote_cmds"] == []  # no ssh in local mode
    marker_cmds = [c for c in calls["local_cmds"] if "pvl_worker.py" in " ".join(map(str, c))]
    assert len(marker_cmds) == 2  # world_size = 2 gpus * 1 = 2 ranks
    # ranks 0,1 on gpus 0,1
    assert ["--rank", "0"] == marker_cmds[0][7:9] or "0" in marker_cmds[0]
    # exactly one sync, and it's the LOCAL slice script (not SMB)
    sync_cmds = [c for c in calls["local_cmds"] if "sync_local_conf_md.py" in " ".join(map(str, c))]
    assert len(sync_cmds) == 1
    assert all("sync_new_conf_md_smb.py" not in " ".join(map(str, c)) for c in calls["local_cmds"])


def test_make_extract_stage_local_spec_shape(ctx):
    spec = st.make_extract_stage_local(ctx)
    assert spec.key == "extract"
    assert spec.resource_key == "gpu-local"
    assert spec.preflight is st.extract_local_preflight
    assert spec.probe is st.extract_local_probe
    assert spec.gate is st.extract_local_gate


# ============================== LOCAL extract with a registered extractor (PIPELINE_LOCAL_EXTRACTOR)
REPO_ROOT = Path(__file__).resolve().parents[1]
STUB_WORKER = REPO_ROOT / "tests" / "fixtures" / "pvl" / "stub_worker.py"


def _extractor_file(tmp_path, **overrides) -> Path:
    block = {
        "python": Path(sys.executable).as_posix(),
        "worker": "scripts/remote_pvl/pvl_worker.py",
        "env": {"STUB_ENV": "from-config"},
        "args": ["--trace", (tmp_path / "trace.jsonl").as_posix()],
        "extract_min_free_mb": 9216,
        "ranks_per_gpu": 1,
        **overrides,
    }
    path = tmp_path / "local_extractor.json"
    path.write_text(json.dumps(block), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _registered_default_extractor(tmp_path, monkeypatch):
    """Marker is retired: the local extract stage runs only a registered worker. Tests that are not about
    which worker runs get one (the registry's own defaults for everything but the program); a test that
    needs none takes ``no_extractor``."""
    path = tmp_path / "default_extractor.json"
    path.write_text(json.dumps({"python": Path(sys.executable).as_posix(),
                                "worker": "scripts/remote_pvl/pvl_worker.py"}), encoding="utf-8")
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(path))


@pytest.fixture
def no_extractor(monkeypatch):
    monkeypatch.delenv("PIPELINE_LOCAL_EXTRACTOR", raising=False)


@pytest.fixture
def stub_project(tmp_path, monkeypatch, no_extractor):
    """A project root whose registered worker and slice-sync script are real scripts: the stub
    honours the worker contract, the sync is the repo's own."""
    proj = tmp_path / "stubproj"
    for rel, source in (
        ("scripts/remote_pvl/pvl_worker.py", STUB_WORKER),
        ("scripts/sync/sync_local_conf_md.py", REPO_ROOT / "scripts" / "sync" / "sync_local_conf_md.py"),
    ):
        (proj / rel).parent.mkdir(parents=True, exist_ok=True)
        (proj / rel).write_bytes(source.read_bytes())
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))  # the sync script imports the repo's `pipeline`
    # the one registered extractor (Marker is retired and there is no default): the stub, with the registry's defaults
    registered = tmp_path / "stub_extractor.json"
    registered.write_text(json.dumps({"python": Path(sys.executable).as_posix(),
                                      "worker": "scripts/remote_pvl/pvl_worker.py"}), encoding="utf-8")
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(registered))
    return proj


@pytest.fixture
def local_ctx(corpus, store, tmp_path, stub_project):
    """CVPR 2026 with three PDFs on disk and a task whose preflight chose card 0."""
    task_id = store.create_task(task_type="ingest_pipeline")
    c = StageContext(
        venue="CVPR", year=2026, store=store, task_id=task_id,
        corpus_root=corpus, project_root=stub_project, concurrency=8, world_size=8,
    )
    c.pdf_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("cvpr-2026-a", "cvpr-2026-b", "cvpr-2026-c"):
        (c.pdf_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4")
    c.state_set(gpu_ids="0")
    return c


def _manifest_md_paths(ctx) -> dict:
    records = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    return {r["paper_id"]: r["markdown_path"] for r in records}


def test_load_local_extractor_is_none_unless_configured(no_extractor):
    assert st.load_local_extractor() is None


def test_load_local_extractor_parses_the_registry_marker_block(tmp_path, monkeypatch, no_extractor):
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(_extractor_file(tmp_path)))
    marker = st.load_local_extractor()
    assert marker == st.MarkerHost(
        python=Path(sys.executable).as_posix(), worker="scripts/remote_pvl/pvl_worker.py",
        env={"STUB_ENV": "from-config"}, args=("--trace", (tmp_path / "trace.jsonl").as_posix()),
        extract_min_free_mb=9216, ranks_per_gpu=1,
    )


@pytest.mark.parametrize("bad", [
    {"python": 'py"; rm -rf /'}, {"worker": "C:\\x\\w.py"}, {"env": {"A": "`id`"}}, {"ranks_per_gpu": 0},
])
def test_load_local_extractor_rejects_what_the_registry_rejects(tmp_path, monkeypatch, no_extractor, bad):
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(_extractor_file(tmp_path, **bad)))
    with pytest.raises(ValueError):
        st.load_local_extractor()


def test_load_local_extractor_fails_loud_when_the_file_is_unreadable(tmp_path, monkeypatch, no_extractor):
    # A typo must never silently turn into "use Marker instead".
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="local extractor"):
        st.load_local_extractor()


def test_a_registry_block_with_only_a_program_keeps_the_defaults_of_the_stage(ctx):
    """(Without any registered extractor the stage is refused: tests/test_retired_marker.py.)"""
    cmd = st.build_extract_local_cmd(ctx, gpu=1, rank=0, world_size=2)
    assert cmd == [
        Path(sys.executable).as_posix(), "scripts/remote_pvl/pvl_worker.py",
        "--input", str(ctx.pdf_dir), "--output", str(ctx.local_md_dir),
        "--gpu", "1", "--rank", "0", "--world-size", "2", "--recursive",
    ]
    assert st.local_extractor_env(None) is None
    assert st.extract_local_min_free_mb() == 16384
    ctx.ranks_per_gpu = 2
    ctx.state_set(gpu_ids="1,2")
    assert len(st._plan_local_ranks(ctx)) == 4
    assert st.make_extract_stage_local(ctx).title.endswith("(local pvl_worker)")


def test_local_extract_uses_the_registered_extractor(ctx, tmp_path, monkeypatch, no_extractor):
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(_extractor_file(tmp_path)))
    marker = st.load_local_extractor()
    cmd = st.build_extract_local_cmd(ctx, gpu=0, rank=0, world_size=1)
    assert cmd == [
        marker.python, marker.worker,
        "--input", str(ctx.pdf_dir), "--output", str(ctx.local_md_dir),
        "--gpu", "0", "--rank", "0", "--world-size", "1", "--recursive", *marker.args,
    ]
    env = st.local_extractor_env(marker)
    assert env["STUB_ENV"] == "from-config" and env["PATH"] == os.environ["PATH"]
    assert st.extract_local_min_free_mb() == 9216
    assert "pvl_worker" in st.make_extract_stage_local(ctx).title


def test_registered_ranks_per_gpu_outranks_the_context(ctx, tmp_path, monkeypatch, no_extractor):
    """One server per card: a worker that owns the card cannot be stacked."""
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(_extractor_file(tmp_path, ranks_per_gpu=1)))
    ctx.ranks_per_gpu = 3
    ctx.state_set(gpu_ids="0")
    assert st._plan_local_ranks(ctx) == [(0, 0, 1)]


def test_local_extract_keeps_what_it_converted_when_a_paper_fails(local_ctx):
    """The worker exits 1 for the paper it refuses. The Markdown it did write must still
    reach the manifest (the gate decides about the rest), not be dropped with the batch."""
    messages = []
    st.make_extract_stage_local(local_ctx).run(local_ctx, lambda m, *a, **k: messages.append(m))
    paths = _manifest_md_paths(local_ctx)
    assert paths["cvpr-2026-a"].endswith("cvpr-2026-a.md") and Path(paths["cvpr-2026-a"]).is_file()
    assert Path(paths["cvpr-2026-b"]).is_file()
    assert not Path(paths["cvpr-2026-c"] or "/nonexistent").is_file()
    assert any("exit 1" in m for m in messages)


def test_local_extract_runs_the_registered_worker_with_its_env_and_args(local_ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(_extractor_file(tmp_path)))
    st.make_extract_stage_local(local_ctx).run(local_ctx, lambda *a, **k: None)
    runs = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(runs) == 1  # ranks_per_gpu 1 on one card
    assert runs[0]["stub_env"] == "from-config" and runs[0]["gpu"] == 0 and runs[0]["world_size"] == 1
    assert runs[0]["argv"][-2] == "--trace"
    assert Path(_manifest_md_paths(local_ctx)["cvpr-2026-a"]).is_file()


def test_local_extract_abort_is_still_cancelled(local_ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(_extractor_file(
        tmp_path, args=["--sleep", "30"])))
    local_ctx.abort.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        st.make_extract_stage_local(local_ctx).run(local_ctx, lambda *a, **k: None)
    assert not any(_manifest_md_paths(local_ctx).get(k) and Path(_manifest_md_paths(local_ctx)[k]).is_file()
                   for k in ("cvpr-2026-a", "cvpr-2026-b"))  # no sync after a cancel


def test_extract_gate_names_the_papers_that_failed_verification(local_ctx):
    st.make_extract_stage_local(local_ctx).run(local_ctx, lambda *a, **k: None)
    gate = st.extract_local_gate(local_ctx)
    assert not gate.ok
    assert "only 2/3 MD extracted" in gate.reason
    assert "1 paper(s) failed" in gate.reason
    assert "cvpr-2026-c" in gate.reason and "fabrication" in gate.reason


def test_extract_gate_lists_at_most_five_failures(local_ctx):
    quality = local_ctx.local_md_dir / "_quality"
    quality.mkdir(parents=True)
    for i in range(8):
        (quality / f"p{i}.json").write_text(json.dumps(
            {"paper": f"p{i}", "status": "failed", "failure_reasons": [f"page {i}: omission"]}), encoding="utf-8")
    gate = st.extract_local_gate(local_ctx)
    assert not gate.ok and "8 paper(s) failed" in gate.reason
    assert "p4" in gate.reason and "p5" not in gate.reason


def test_quality_reports_are_not_counted_as_markdown(local_ctx):
    _make_md_local(local_ctx, "cvpr-2026-a.md", "# A\nbody")
    quality = local_ctx.local_md_dir / "_quality"
    quality.mkdir()
    (quality / "cvpr-2026-a.json").write_text("{}", encoding="utf-8")
    (quality / "cvpr-2026-c.json").write_text("{}", encoding="utf-8")
    assert local_ctx.md_count_local() == 1
    assert st.extract_local_probe(local_ctx).done == 1


# ------------------------------------------------ what a verifying extractor leaves behind
def _verified_slice(ctx, *, written, refused, unaccounted=0):
    """`written` papers with Markdown and a passing report, `refused` with only a failed report (what
    pvl_worker leaves for a paper it will not vouch for), `unaccounted` with neither (a crash, a paper
    the worker never reached)."""
    ctx.pdf_dir.mkdir(parents=True, exist_ok=True)
    quality = ctx.local_md_dir / "_quality"
    quality.mkdir(parents=True, exist_ok=True)
    for i in range(written + refused + unaccounted):
        stem = f"v{i:03d}"
        (ctx.pdf_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4")
        if i < written:
            (ctx.local_md_dir / f"{stem}.md").write_text("# Header\n" + "x" * 300, encoding="utf-8")
            report = {"paper": stem, "status": "passed", "failure_reasons": []}
        elif i < written + refused:
            report = {"paper": stem, "status": "failed", "failure_reasons": [f"page {i}: number: not on the page: ['0']"]}
        else:
            continue
        (quality / f"{stem}.json").write_text(json.dumps(report), encoding="utf-8")


def test_extract_gate_accepts_a_slice_whose_only_gap_is_refused_papers(ctx):
    # task 1302: 259 written, 38 refused, and the gate stopped the pipeline at "only 259/297 MD extracted"
    _verified_slice(ctx, written=259, refused=38)
    gate = st.extract_local_gate(ctx)
    assert gate.ok, gate.reason
    assert "38/297 refused" in gate.reason
    assert "v259" in gate.reason and "v263" in gate.reason and "v264" not in gate.reason  # the first five reasons


def test_extract_gate_fails_for_a_paper_with_neither_markdown_nor_report(ctx):
    _verified_slice(ctx, written=9, refused=1, unaccounted=1)
    gate = st.extract_local_gate(ctx)
    assert not gate.ok
    unaccounted = gate.reason.split(";")[0]
    assert "neither" in unaccounted and "v010" in unaccounted and "v009" not in unaccounted


def test_extract_gate_fails_when_a_third_of_the_slice_is_refused(ctx):
    _verified_slice(ctx, written=7, refused=3)  # 30%
    gate = st.extract_local_gate(ctx)
    assert not gate.ok and "only 7/10 MD extracted" in gate.reason and "3 paper(s) failed" in gate.reason


@pytest.mark.parametrize("written, refused, ok", [(3, 1, True), (6, 2, True), (5, 2, False)])
def test_extract_gate_cap_is_a_quarter_of_the_slice(ctx, written, refused, ok):
    _verified_slice(ctx, written=written, refused=refused)
    assert st.extract_local_gate(ctx).ok is ok


def test_a_refused_paper_converted_by_a_later_run_is_not_counted_as_refused(ctx):
    _verified_slice(ctx, written=9, refused=1)
    (ctx.local_md_dir / "v009.md").write_text("# Header\n" + "x" * 300, encoding="utf-8")  # the retry wrote it
    gate = st.extract_local_gate(ctx)
    assert gate.ok and "refused" not in gate.reason


def test_extract_gate_without_quality_reports_keeps_the_95pct_rule(ctx):
    _make_pdfs(ctx, 100)
    for i in range(96):
        _make_md_local(ctx, f"g{i}.md", "# Header\n" + ("x" * 300))
    assert st.extract_local_gate(ctx).ok  # Marker: counts only, no names to match
    (ctx.local_md_dir / "g0.md").unlink()
    (ctx.local_md_dir / "g1.md").unlink()
    gate = st.extract_local_gate(ctx)
    assert not gate.ok and "only 94/100 MD extracted" in gate.reason


def test_extract_gate_counts_refusals_when_the_pdfs_were_not_shipped(ctx):
    # Markdown produced elsewhere and pushed alone: the manifest slice (4 papers) is the total
    for name in ("cvpr-2026-a", "cvpr-2026-b", "cvpr-2026-d"):
        _make_md_local(ctx, f"{name}.md", "# Header\n" + ("x" * 300))
    assert not st.extract_local_gate(ctx).ok  # 3/4, nothing explains the fourth
    quality = ctx.local_md_dir / "_quality"
    quality.mkdir()
    (quality / "cvpr-2026-c.json").write_text(json.dumps(
        {"paper": "cvpr-2026-c", "status": "failed", "failure_reasons": ["page 2: omission"]}), encoding="utf-8")
    gate = st.extract_local_gate(ctx)
    assert gate.ok and "1/4 refused" in gate.reason


def test_extract_gate_names_pipeline_crashes_apart_from_verification_refusals(ctx):
    # TPAMI 2022 (2026-11-21): "59 paper(s) failed verification" when 45 were GPU out-of-memory crashes
    _verified_slice(ctx, written=6, refused=1)
    quality = ctx.local_md_dir / "_quality"
    for stem in ("v007", "v008", "v009"):
        (ctx.pdf_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4")
        (quality / f"{stem}.json").write_text(json.dumps({"paper": stem, "status": "failed", "failure_reasons": [
            "page 2: omission: page text 'x' is 0% in the Markdown (< 50%)",
            "pipeline stream failed: GPU out of memory: 366 MB needed, 340 MB free (RuntimeError)"]}), encoding="utf-8")
    gate = st.extract_local_gate(ctx)
    assert not gate.ok  # 4 of 10 not written: over the cap either way
    assert "; 1 paper(s) failed verification: v006: page 6: number" in gate.reason
    assert ("; 3 paper(s) were not converted, the extraction pipeline failed: v007: pipeline stream failed: "
            "GPU out of memory: 366 MB needed, 340 MB free (RuntimeError); v008: ") in gate.reason
    assert "v009" not in gate.reason  # two named per kind of failure


def test_extract_probe_reports_the_refused_papers(ctx):
    _verified_slice(ctx, written=8, refused=2)
    pr = st.extract_local_probe(ctx)
    assert (pr.done, pr.total) == (8, 10) and "2 refused" in pr.message


# ===================================================== W106 LOCAL embed (self-hosted)
def test_the_local_embed_stage_has_no_second_model_loader_left():
    """The local stage encodes with the backend's resident model: the helpers that built a
    second embed_worker process (command, model dir, GPU pinning) are gone."""
    for name in ("build_embed_worker_local_cmd", "_bge_m3_model_dir", "_BGE_M3_FALLBACK", "_local_gpu_env"):
        assert not hasattr(st, name), name


def test_embed_local_probe_counts_local_vectors_in_range(ctx, tmp_path):
    vec = st._embed_vectors_dir(ctx)
    vec.mkdir(parents=True)
    for i in (120, 121, 140):  # 140 = another venue's
        (vec / f"vectors_{i:04d}.parquet").write_bytes(b"x")
    ctx.state_set(shard_start=120, shard_end=122)
    pr = st.embed_local_probe(ctx)
    assert pr.total == 2 and pr.done == 2  # 120,121; 140 excluded


def test_embed_local_run_anchors_shard_end_and_no_ship(ctx, monkeypatch):
    """LOCAL embed: dump → in-process embed, SAME shard-anchor state as remote, but
    NO ship.sh up/down, NO ssh and NO embed_worker subprocess."""
    import gzip
    import json

    chunks = st._embed_chunks_dir(ctx)
    chunks.mkdir(parents=True)
    for i in range(120):    # earlier shards of the directory: valid, empty
        (chunks / f"chunks_{i:04d}.jsonl.gz").write_bytes(gzip.compress(b""))
    calls = {"local_cmds": [], "remote_cmds": []}

    def fake_run_local(cmd, cwd, sink, **kw):
        calls["local_cmds"].append(cmd)
        if "dump_chunks.py" in " ".join(map(str, cmd)):
            for i in (120, 121, 122):
                record = {"point_id": i, "chunk_id_str": f"c{i}", "doc_id": 1, "content": f"text {i}",
                          "level": 2, "start_idx": 0, "end_idx": 6}
                (chunks / f"chunks_{i:04d}.jsonl.gz").write_bytes(gzip.compress(json.dumps(record).encode()))
        return 0

    seen: list[str] = []

    def encode(texts):
        import numpy as np
        seen.extend(texts)
        return np.zeros((len(texts), 1024), dtype=np.float32), [{1: 0.5} for _ in texts]

    monkeypatch.setattr(st, "run_local", fake_run_local)
    monkeypatch.setattr(st, "run_remote", lambda *a, **k: calls["remote_cmds"].append(a) or 0)

    st.make_embed_stage_local(ctx, encode=encode).run(ctx, lambda *a, **k: None)

    assert ctx.state_get("dump_start_shard") == 120
    assert ctx.state_get("shard_start") == 120
    assert ctx.state_get("shard_end") == 123  # 120 + 3 dumped
    assert sorted(seen) == ["text 120", "text 121", "text 122"]
    assert (st._embed_vectors_dir(ctx) / "vectors_0122.parquet").exists()
    assert calls["remote_cmds"] == []  # no ssh
    joined = [" ".join(map(str, c)) for c in calls["local_cmds"]]
    assert any("dump_chunks.py" in j for j in joined)
    assert all("embed_worker.py" not in j for j in joined)  # the model is the backend's, not a second process
    assert all("ship.sh" not in j for j in joined)  # NO ship up/down


def test_make_embed_stage_local_spec_shape(ctx):
    spec = st.make_embed_stage_local(ctx)
    assert spec.key == "embed"
    assert spec.resource_key == "embed-service"      # a client of the vLLM service: it does not hold the extract's card
    assert spec.skip_when_satisfied is False
    assert spec.probe is st.embed_local_probe
    # preflight and gate are the local ones / the remote gate (shard_end > shard_start), behind the
    # nothing-to-embed short-circuit: with no preindex watermark they answer as the originals do.
    assert not st.nothing_to_embed(ctx)
    assert spec.gate(ctx) == st.embed_gate(ctx)
    assert spec.preflight is not None


# ============================================ W106 LOCAL shared-box GPU safety (W70)
def test_local_gpu_allowlist_parses_env(monkeypatch):
    monkeypatch.delenv("PIPELINE_LOCAL_GPU_IDS", raising=False)
    assert st._local_gpu_allowlist() is None  # unset → no restriction (dedicated box)
    monkeypatch.setenv("PIPELINE_LOCAL_GPU_IDS", "3")
    assert st._local_gpu_allowlist() == {3}
    monkeypatch.setenv("PIPELINE_LOCAL_GPU_IDS", "2,3")
    assert st._local_gpu_allowlist() == {2, 3}
    monkeypatch.setenv("PIPELINE_LOCAL_GPU_IDS", "2 3")  # space-separated also ok
    assert st._local_gpu_allowlist() == {2, 3}
    monkeypatch.setenv("PIPELINE_LOCAL_GPU_IDS", "   ")  # blank → None
    assert st._local_gpu_allowlist() is None


def test_local_idle_gpus_intersects_allowlist(monkeypatch):
    """A card must be BOTH idle AND in the allowlist before a local stage may use
    it — the W106 shared-box gate. GPU0 (co-tenant, W70) is idle but excluded."""
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: [0, 1, 2, 3])
    monkeypatch.setenv("PIPELINE_LOCAL_GPU_IDS", "2,3")
    assert st._local_idle_gpus(min_free_mb=16384) == [2, 3]
    # idle ∩ allow: only GPU3 idle, but allow={2,3} → keep [3]
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: [0, 3])
    assert st._local_idle_gpus(min_free_mb=16384) == [3]
    # no allowlist → all idle cards (dedicated box)
    monkeypatch.delenv("PIPELINE_LOCAL_GPU_IDS", raising=False)
    assert st._local_idle_gpus(min_free_mb=16384) == [0, 3]


def test_local_idle_gpus_asks_for_the_floor_and_never_for_an_idle_ratio(monkeypatch):
    """The card always carries the resident vLLM services, so a local stage is gated by absolute free
    memory only; no environment variable brings a ratio back."""
    seen = {}
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: seen.update(target=target, **kw) or [3])
    monkeypatch.delenv("PIPELINE_LOCAL_GPU_IDS", raising=False)
    monkeypatch.setenv("PIPELINE_LOCAL_MIN_FREE_RATIO", "0.99")
    monkeypatch.setenv("PIPELINE_LOCAL_MIN_FREE_MB", "1")
    assert st._local_idle_gpus(min_free_mb=16384) == [3]
    assert seen["target"] is None and seen["min_free_mb"] == 16384
    assert seen.get("min_free_ratio", 0.0) == 0.0


def test_extract_local_preflight_excludes_disallowed_gpu(ctx, monkeypatch):
    """gpu_health offers GPU0 (hty's) + GPU3 (idle); allowlist=3 → state records
    ONLY GPU3, never GPU0."""
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: [0, 3])
    monkeypatch.setenv("PIPELINE_LOCAL_GPU_IDS", "3")
    assert st.extract_local_preflight(ctx).ok
    assert ctx.state_get("gpu_ids") == "3"


def test_extract_local_preflight_fails_when_allowlist_empties_idle(ctx, monkeypatch):
    """Only GPU0 idle but it's disallowed → preflight fails loud (no silent GPU0)."""
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: [0])
    monkeypatch.setenv("PIPELINE_LOCAL_GPU_IDS", "3")
    g = st.extract_local_preflight(ctx)
    assert not g.ok and "PIPELINE_LOCAL_GPU_IDS" in g.reason
    assert ctx.state_get("gpu_ids") in (None, "")


def test_plan_local_ranks_raises_without_gpu_ids(ctx):
    """No gpu_ids in state (preflight never ran / was bypassed) → fail loud rather
    than silently defaulting to GPU0 on a shared box (W70)."""
    with pytest.raises(RuntimeError, match="no gpu_ids"):
        st._plan_local_ranks(ctx)
