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
    _make_pdfs(ctx, 4)  # 4/4
    assert st.download_gate(ctx).ok


def test_download_gate_fails_below_90pct(ctx):
    _make_pdfs(ctx, 2)  # 2/4 = 50%
    g = st.download_gate(ctx)
    assert not g.ok and "2/4" in g.reason


def test_download_gate_passes_when_total_zero(corpus, store, tmp_path):
    # A venue/year with no manifest records → nothing to download → ok.
    task_id = store.create_task(task_type="ingest_pipeline")
    c = StageContext(venue="NOPE", year=1999, store=store, task_id=task_id, corpus_root=corpus)
    assert st.download_gate(c).ok


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
    st.make_download_stage(ctx).run(ctx, lambda *a, **k: None)
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
        "--conf", "CVPR", "--year", "2026", "--chunker-type", "section_l1",
    ]


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


def test_preindex_run_captures_watermark_and_runs(ctx, session_factory, monkeypatch):
    # seed some chunks so MAX(id) is meaningful
    with session_factory() as s:
        doc = DBDocument(title="t", source="x", source_type=SourceType.paper)
        s.add(doc)
        s.flush()
        s.add(DBChunk(doc_id=doc.id, chunk_index=0, content="c", level=1))
    seen = {}
    monkeypatch.setattr(st, "run_local", lambda cmd, cwd, sink, **kw: seen.update(cmd=cmd) or 0)
    spec = st.make_preindex_stage(ctx, session_factory=session_factory)
    spec.run(ctx, lambda *a, **k: None)
    assert seen["cmd"] == st.build_preindex_cmd(ctx)
    assert ctx.state_get("chunk_id_before") == 1  # one chunk seeded → MAX(id)=1


# ======================================================================= STAGE 4
def test_embed_dump_cmd_uses_watermark_and_shard(ctx):
    cmd = st.build_embed_dump_cmd(ctx, chunk_id_before=100, start_shard=5)
    assert cmd == [
        sys.executable, "scripts/remote_embed/dump_chunks.py",
        "--min-chunk-id", "101", "--start-shard", "5",
    ]


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
        sys.executable, "scripts/remote_embed/local_ingest.py", "--shard-start", "3",
    ]


def test_ingest_preflight_qdrant_reachable(ctx, monkeypatch):
    monkeypatch.setattr(st, "_qdrant_reachable", lambda: True)
    assert st.ingest_preflight(ctx).ok


def test_ingest_preflight_qdrant_down(ctx, monkeypatch):
    monkeypatch.setattr(st, "_qdrant_reachable", lambda: False)
    g = st.ingest_preflight(ctx)
    assert not g.ok and "qdrant" in g.reason.lower()


def test_ingest_probe_and_gate(ctx, tmp_path, monkeypatch):
    vec = tmp_path / "proj" / "data" / "remote_embed" / "vectors"
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
    vec = tmp_path / "proj" / "data" / "remote_embed" / "vectors"
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
    vec = tmp_path / "proj" / "data" / "remote_embed" / "vectors"
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
    vec = tmp_path / "proj" / "data" / "remote_embed" / "vectors"
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


def test_extract_local_cmd_is_marker_worker_per_rank(ctx):
    cmd = st.build_extract_local_cmd(ctx, gpu=1, rank=0, world_size=2)
    assert cmd == [
        sys.executable, "scripts/remote_marker/marker_worker.py",
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
    assert seen["kw"]["min_free_mb"] == 16384 and seen["kw"]["min_free_ratio"] == 0.85
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
    """run() launches N local marker procs (one per rank, distributed across the
    preflight GPUs) then syncs MD paths via the LOCAL slice script — never ssh,
    never run_marker_kss.sh, never the SMB sync."""
    ctx.ranks_per_gpu = 1
    ctx.state_set(gpu_ids="0,1")
    calls = {"local_cmds": [], "remote_cmds": []}

    monkeypatch.setattr(st, "run_local", lambda cmd, cwd, sink, **kw: calls["local_cmds"].append(cmd) or 0)
    monkeypatch.setattr(st, "run_remote", lambda *a, **k: calls["remote_cmds"].append(a) or 0)

    st.make_extract_stage_local(ctx).run(ctx, lambda *a, **k: None)

    assert calls["remote_cmds"] == []  # no ssh in local mode
    marker_cmds = [c for c in calls["local_cmds"] if "marker_worker.py" in " ".join(map(str, c))]
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


# ===================================================== W106 LOCAL embed (self-hosted)
def test_embed_worker_local_cmd_is_generic(ctx):
    cmd = st.build_embed_worker_local_cmd(ctx, model_dir="/m/bge-m3")
    assert cmd == [
        sys.executable, "scripts/remote_embed/embed_worker.py",
        "--input", "data/remote_embed/chunks",
        "--output", "data/remote_embed/vectors",
        "--model-dir", "/m/bge-m3",
        "--rank", "0", "--world-size", "1",
    ]


def test_embed_local_model_dir_honors_env(monkeypatch):
    monkeypatch.setenv("BGE_M3_PATH", "/custom/bge")
    assert st._bge_m3_model_dir() == "/custom/bge"
    monkeypatch.delenv("BGE_M3_PATH", raising=False)
    assert st._bge_m3_model_dir() == st._BGE_M3_FALLBACK


def test_embed_local_preflight_uses_local_gpu_health(ctx, monkeypatch):
    seen = {}
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: seen.update(target=target, kw=kw) or [2])
    assert st.embed_local_preflight(ctx).ok
    assert seen["target"] is None
    assert seen["kw"]["min_free_mb"] == 12288 and seen["kw"]["min_free_ratio"] == 0.85
    assert ctx.state_get("gpu_ids") == "2"


def test_embed_local_probe_counts_local_vectors_in_range(ctx, tmp_path):
    vec = st._embed_vectors_dir(ctx)
    vec.mkdir(parents=True)
    for i in (120, 121, 140):  # 140 = another venue's
        (vec / f"vectors_{i:04d}.parquet").write_bytes(b"x")
    ctx.state_set(shard_start=120, shard_end=122)
    pr = st.embed_local_probe(ctx)
    assert pr.total == 2 and pr.done == 2  # 120,121; 140 excluded


def test_embed_local_run_anchors_shard_end_and_no_ship(ctx, monkeypatch):
    """LOCAL embed: dump → embed_worker, SAME shard-anchor state as remote, but
    NO ship.sh up/down and NO ssh."""
    chunks = st._embed_chunks_dir(ctx)
    chunks.mkdir(parents=True)
    for i in range(120):
        (chunks / f"chunks_{i:04d}.jsonl.gz").write_bytes(b"x")
    calls = {"local_cmds": [], "remote_cmds": []}

    def fake_run_local(cmd, cwd, sink, **kw):
        calls["local_cmds"].append(cmd)
        if "dump_chunks.py" in " ".join(map(str, cmd)):
            for i in (120, 121, 122):
                (chunks / f"chunks_{i:04d}.jsonl.gz").write_bytes(b"x")
        return 0

    monkeypatch.setattr(st, "run_local", fake_run_local)
    monkeypatch.setattr(st, "run_remote", lambda *a, **k: calls["remote_cmds"].append(a) or 0)

    st.make_embed_stage_local(ctx).run(ctx, lambda *a, **k: None)

    assert ctx.state_get("dump_start_shard") == 120
    assert ctx.state_get("shard_start") == 120
    assert ctx.state_get("shard_end") == 123  # 120 + 3 dumped
    assert calls["remote_cmds"] == []  # no ssh
    joined = [" ".join(map(str, c)) for c in calls["local_cmds"]]
    assert any("dump_chunks.py" in j for j in joined)
    assert any("embed_worker.py" in j for j in joined)
    assert all("ship.sh" not in j for j in joined)  # NO ship up/down


def test_make_embed_stage_local_spec_shape(ctx):
    spec = st.make_embed_stage_local(ctx)
    assert spec.key == "embed"
    assert spec.resource_key == "gpu-local"
    assert spec.skip_when_satisfied is False
    assert spec.preflight is st.embed_local_preflight
    assert spec.gate is st.embed_gate  # reuses remote gate (shard_end > shard_start)
    assert spec.probe is st.embed_local_probe


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
    assert st._local_idle_gpus(min_free_mb=16384, min_free_ratio=0.85) == [2, 3]
    # idle ∩ allow: only GPU3 idle, but allow={2,3} → keep [3]
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: [0, 3])
    assert st._local_idle_gpus(min_free_mb=16384, min_free_ratio=0.85) == [3]
    # no allowlist → all idle cards (dedicated box)
    monkeypatch.delenv("PIPELINE_LOCAL_GPU_IDS", raising=False)
    assert st._local_idle_gpus(min_free_mb=16384, min_free_ratio=0.85) == [0, 3]


def test_local_idle_gpus_env_overrides_thresholds(monkeypatch):
    """A shared-box operator trades the politeness RATIO for absolute usable
    memory: PIPELINE_LOCAL_MIN_FREE_RATIO=0 + a MB floor lets a card with enough
    FREE memory qualify even when it's <85% idle — so we use a card that has room
    WITHOUT crowding out the co-tenant's job (the MB floor is the headroom)."""
    seen = {}
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: seen.update(kw) or [1, 2])
    monkeypatch.delenv("PIPELINE_LOCAL_GPU_IDS", raising=False)
    monkeypatch.setenv("PIPELINE_LOCAL_MIN_FREE_RATIO", "0")
    monkeypatch.setenv("PIPELINE_LOCAL_MIN_FREE_MB", "12288")
    out = st._local_idle_gpus(min_free_mb=16384, min_free_ratio=0.85)
    assert seen["min_free_ratio"] == 0.0   # env beat the 0.85 default arg
    assert seen["min_free_mb"] == 12288    # env beat the 16384 default arg
    assert out == [1, 2]


def test_local_idle_gpus_uses_defaults_without_env(monkeypatch):
    seen = {}
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: seen.update(kw) or [3])
    for v in ("PIPELINE_LOCAL_MIN_FREE_RATIO", "PIPELINE_LOCAL_MIN_FREE_MB", "PIPELINE_LOCAL_GPU_IDS"):
        monkeypatch.delenv(v, raising=False)
    st._local_idle_gpus(min_free_mb=16384, min_free_ratio=0.85)
    assert seen["min_free_ratio"] == 0.85   # unchanged when no env override
    assert seen["min_free_mb"] == 16384


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


def test_embed_local_preflight_intersects_allowlist(ctx, monkeypatch):
    monkeypatch.setattr(st, "gpu_health", lambda target, **kw: [1, 3])
    monkeypatch.setenv("PIPELINE_LOCAL_GPU_IDS", "3")
    assert st.embed_local_preflight(ctx).ok
    assert ctx.state_get("gpu_ids") == "3"


def test_plan_local_ranks_raises_without_gpu_ids(ctx):
    """No gpu_ids in state (preflight never ran / was bypassed) → fail loud rather
    than silently defaulting to GPU0 on a shared box (W70)."""
    with pytest.raises(RuntimeError, match="no gpu_ids"):
        st._plan_local_ranks(ctx)


def test_local_gpu_env_pins_first_idle_gpu(ctx):
    ctx.state_set(gpu_ids="3,2")
    env = st._local_gpu_env(ctx)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"  # first preflight-selected card
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"  # physical index (W70)


def test_local_gpu_env_none_without_gpu_ids(ctx):
    assert st._local_gpu_env(ctx) is None  # inherit env when state empty


def test_embed_local_run_pins_cuda_visible_devices(ctx, monkeypatch):
    """The embed_worker subprocess (hardcoded cuda:0) MUST be launched with
    CUDA_VISIBLE_DEVICES pinned to gpu_ids[0] — else it embeds on the backend's
    live-search GPU (W106/W70)."""
    ctx.state_set(gpu_ids="3")
    chunks = st._embed_chunks_dir(ctx)
    chunks.mkdir(parents=True)
    for i in range(120):
        (chunks / f"chunks_{i:04d}.jsonl.gz").write_bytes(b"x")
    seen = {"embed_env": "MISSING"}

    def fake_run_local(cmd, cwd, sink, **kw):
        joined = " ".join(map(str, cmd))
        if "dump_chunks.py" in joined:
            for i in (120, 121):
                (chunks / f"chunks_{i:04d}.jsonl.gz").write_bytes(b"x")
        if "embed_worker.py" in joined:
            seen["embed_env"] = kw.get("env")
        return 0

    monkeypatch.setattr(st, "run_local", fake_run_local)
    st.make_embed_stage_local(ctx).run(ctx, lambda *a, **k: None)

    env = seen["embed_env"]
    assert env is not None and env != "MISSING"  # env was passed (not inherited)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
