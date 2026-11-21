"""The control flow of scripts/remote_pvl/pvl_worker.py, as a real subprocess.

tests/fixtures/pvl/worker_harness.py runs the worker's own ``main()``; only the model server and the
extractor (which need a GPU) are scripted. Everything below happened in task 1302 (VLDB 2022, 2026-11-13):
the worker printed ``worker.done ok=0 fail=0`` and exited 0 with 165 papers untouched, nothing held for the
second model was written until the end, and the progress line counted only written pages.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "fixtures" / "pvl" / "worker_harness.py"


def run_worker(tmp_path, papers, script, *extra, env=None):
    input_dir, output = tmp_path / "pdf", tmp_path / "md"
    input_dir.mkdir(exist_ok=True)
    for i in range(papers):
        (input_dir / f"p{i:02d}.pdf").write_bytes(b"%PDF-1.4")
    trace = tmp_path / "trace.jsonl"
    environment = {**os.environ, "PVL_HARNESS": json.dumps({**script, "trace": str(trace)}), **(env or {})}
    done = subprocess.run(
        [sys.executable, str(HARNESS), "--input", str(input_dir), "--output", str(output), "--gpu", "0", "--recursive",
         "--server-config", str(tmp_path / "unused.yaml"), "--server-command", "unused {port}", "--batch-pdfs", "2",
         "--progress-every", "2", *extra],
        capture_output=True, text=True, env=environment, timeout=120)
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()] if trace.exists() else []
    summary_path = output / "_quality" / "_run.rank0.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    written = sorted(p.stem for p in output.glob("*.md"))
    return done, events, summary, written


def kinds(events, name):
    return [e for e in events if e["event"] == name]


def test_a_failed_restart_does_not_end_the_worker_and_the_counts_are_true(tmp_path):
    # batch 2 breaks, the restart finds no room on the card; the other three batches still run
    done, events, summary, written = run_worker(
        tmp_path, 8, {"broken_calls": [2], "healthy": False, "start_fails": [2]})
    assert [e["call"] for e in kinds(events, "convert")] == [1, 2, 3, 4], done.stderr[-2000:]
    assert len(written) == 6
    assert (summary["ok"], summary["fail"], summary["unwritten"]) == (6, 2, 0)
    assert "worker.done ok=6 fail=2 unwritten=0" in done.stderr
    assert done.returncode == 1  # two papers failed: the stage must not read this as success


def test_a_server_that_never_comes_back_is_given_up_with_the_true_counts(tmp_path):
    done, events, summary, written = run_worker(
        tmp_path, 12, {"broken_calls": list(range(1, 40)), "healthy": False, "start_fails": list(range(2, 40))})
    assert len(kinds(events, "convert")) == 3  # three failed revivals in a row, then stop
    assert written == []
    assert (summary["ok"], summary["fail"], summary["unwritten"]) == (0, 6, 6)
    assert "giving up" in done.stderr and "worker.done ok=0 fail=6 unwritten=6" in done.stderr
    assert done.returncode == 1


def test_a_worker_that_dies_still_reports_what_it_did_and_exits_nonzero(tmp_path):
    done, events, summary, written = run_worker(tmp_path, 8, {"die_calls": [3]})
    assert done.returncode != 0 and "scripted checker failure" in done.stderr
    assert len(written) == 4
    assert (summary["ok"], summary["fail"], summary["unwritten"]) == (4, 0, 4)


def test_a_refused_paper_is_not_checkpointed_and_the_next_run_offers_it_again(tmp_path):
    # the extract gate now lets a slice through with refused papers; that is only safe while they stay retryable
    done, events, summary, written = run_worker(tmp_path, 6, {"refused": ["p01", "p04"]})
    assert written == ["p00", "p02", "p03", "p05"]
    checkpoint = (tmp_path / "md" / "done.rank0.txt").read_text(encoding="utf-8").splitlines()
    assert sorted(Path(line).stem for line in checkpoint) == written
    for stem in ("p01", "p04"):
        report = json.loads((tmp_path / "md" / "_quality" / f"{stem}.json").read_text(encoding="utf-8"))
        assert report["status"] == "failed" and "omission" in report["failure_reasons"][0]
    assert (summary["ok"], summary["fail"], summary["unwritten"]) == (4, 2, 0)
    assert done.returncode == 1

    again, events_again, _, _ = run_worker(tmp_path, 6, {})
    offered = [e["papers"] for e in events_again[len(events):] if e["event"] == "convert"]
    assert sorted(stem for batch in offered for stem in batch) == ["p01", "p04"], again.stderr[-2000:]
    assert again.returncode == 0


def test_a_pipeline_failure_with_a_healthy_server_rebuilds_the_pipeline_and_keeps_the_server(tmp_path):
    # task 1302: Paddle ran out of memory, the vLLM server was fine; restarting it cost 96 s and a failed sizing
    done, events, summary, written = run_worker(tmp_path, 8, {"broken_calls": [2], "healthy": True})
    assert done.returncode == 0, done.stderr[-2000:]
    assert len(written) == 8
    assert len(kinds(events, "server.start")) == 1
    assert [e["event"] for e in events].index("server.stop") > max(i for i, e in enumerate(events) if e["event"] == "convert")
    assert len(kinds(events, "release_pipeline")) >= 1
    assert [e["call"] for e in kinds(events, "convert")] == [1, 2, 3, 4, 5]


def test_held_papers_are_written_in_chunks_and_the_rate_counts_converted_pages(tmp_path):
    # every paper waits for the second model; with a chunk of 4 the first four are written before the next batch
    done, events, summary, written = run_worker(
        tmp_path, 10, {"unresolved": True, "pages": 5}, "--no-formula-witness", "--fallback-command", "unused",
        "--defer-flush-papers", "4")
    assert done.returncode == 0, done.stderr[-2000:]
    assert len(written) == 10 and summary["ok"] == 10 and summary["unwritten"] == 0
    assert [e["papers"] for e in kinds(events, "fallback")] == [4, 4, 2]
    assert [e["md_written"] for e in kinds(events, "fallback")] == [0, 4, 8]
    converts = kinds(events, "convert")
    assert converts[2]["md_written"] == 4 and converts[4]["md_written"] == 8  # papers 1-4 were on disk by batch 3
    assert len(kinds(events, "server.start")) == 3  # the server comes back after each of the two flushes
    assert "progress 2/10 ok=0 skip=0 fail=0 held=2 pages=10" in done.stderr  # converted pages, not written ones
    assert summary["pages"] == 50


def test_the_worker_pins_what_paddle_wants_before_it_loads(tmp_path):
    done, events, summary, written = run_worker(tmp_path, 2, {}, env={"OMP_NUM_THREADS": "2"})
    assert done.returncode == 0, done.stderr[-2000:]
    (extractor,) = kinds(events, "extractor")
    assert extractor["omp"] == "1" and extractor["allocator"] == "auto_growth"


def test_the_chatty_client_libraries_are_kept_at_warning(tmp_path):
    # task 1302 had 4,000+ events, mostly mineru_vl_utils DEBUG "Layout raw output" and openai "Retrying request"
    done, events, summary, written = run_worker(tmp_path, 2, {})
    (extractor,) = kinds(events, "extractor")
    assert set(extractor["loggers"].values()) == {30}, extractor["loggers"]  # logging.WARNING


def test_model_servers_keep_idle_connections_open_longer_than_the_pipeline_idles(monkeypatch):
    # task 1302: 160+ "Retrying request" events, always 16 at once, after the pipeline had been idle for more than the
    # 5 s vLLM keeps an idle connection; the server log had no error and no non-200 answer
    spec = importlib.util.spec_from_file_location("pvl_worker_env", ROOT / "scripts" / "remote_pvl" / "pvl_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.delenv("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", raising=False)
    environment = worker.server_environment()
    assert int(environment["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"]) >= 600 and environment["PATH"] == os.environ["PATH"]


@pytest.fixture
def worker_module():
    spec = importlib.util.spec_from_file_location("pvl_worker_sizing", ROOT / "scripts" / "remote_pvl" / "pvl_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def test_a_restarted_server_is_not_charged_twice_for_what_the_worker_already_holds(worker_module):
    # task 1306 (2026-11-15): first start free=7441 MiB -> 0.264; the restart after the formula phase saw free=7194 MiB
    # because the worker's own CUDA context (247 MiB) was taken out of it, and 3,700 MiB were reserved on top again
    first = worker_module.server_memory_fraction(7441, 12282, 3700)
    restart = worker_module.server_memory_fraction(7194, 12282, 3700, held_mib=247)
    assert round(restart, 3) == round(first, 3) == 0.264


def test_what_the_worker_holds_beyond_the_reserve_leaves_only_the_margin(worker_module):
    assert worker_module.server_memory_fraction(5000, 12282, 3700, held_mib=4000) == round((5000 - 500) / 12282, 3)


def test_the_compute_apps_listing_gives_this_process_its_memory(worker_module):
    # nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
    sample = "4242, 2800\n31337, 247\n"
    assert worker_module._parse_compute_apps(sample, 31337) == 247
    assert worker_module._parse_compute_apps(sample, 4242) == 2800
    assert worker_module._parse_compute_apps(sample, 999) == 0  # no CUDA context yet
    assert worker_module._parse_compute_apps("", 31337) == 0


def test_the_reference_ocr_runs_on_the_cpu_where_it_takes_no_card_memory():
    from pipeline.paper_crawler.converters.verify.ocr import OcrReference

    options = OcrReference()._options
    assert options["device"] == "cpu" and options["cpu_threads"] >= 1


@pytest.fixture
def katex():
    from pipeline.paper_crawler.converters.verify.katex import shared_renderer

    return shared_renderer()


def test_a_page_retry_that_dies_makes_the_paper_retryable_not_failed(katex):
    # task 1302: the 'cv' worker ran out of memory during the retry; four papers were failed for good
    from pipeline.paper_crawler.converters.extractors.paddleocr_vl import PaddleOCRVLExtractor

    fixtures = ROOT / "tests" / "fixtures" / "pvl"
    pdf = fixtures / "unrescued.pdf"
    data = json.loads((fixtures / "unrescued.page0.pvl.json").read_text(encoding="utf-8"))
    data.update(input_path=str(pdf), page_count=1, page_index=0)

    class Page:
        json = {"res": data}

        def _to_markdown(self, pretty=False):
            return {"markdown_texts": ""}

    class Pipeline:
        calls = 0

        def predict_iter(self, paths, **kwargs):
            Pipeline.calls += 1
            if Pipeline.calls > 1:
                raise RuntimeError("Exception from the 'cv' worker: Out of memory error on GPU 0")
            yield Page()

    extractor = PaddleOCRVLExtractor("http://127.0.0.1:1/v1", katex=katex, pipeline=Pipeline())
    batch = extractor.convert([pdf])
    assert Pipeline.calls == 2
    result = batch.results[pdf]
    assert not result.succeeded and result.extra["retryable"] is True
    assert "page retry failed" in result.error
