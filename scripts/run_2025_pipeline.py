"""
Run the full 2025 paper collection pipeline:
  1. collect_manifest (add 2025 papers to existing manifest)
  2. download PDFs
  3. extract to markdown
  4. preindex into Qdrant + SQLite

Usage:
    python scripts/run_2025_pipeline.py                  # full pipeline
    python scripts/run_2025_pipeline.py --step collect   # only collect manifest
    python scripts/run_2025_pipeline.py --step download   # only download PDFs
    python scripts/run_2025_pipeline.py --step extract    # only extract markdown
    python scripts/run_2025_pipeline.py --step preindex   # only preindex
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    MainConferenceCorpusBuilder,
    CorpusPaper,
    load_json_with_recovery,
)


def _load_manifest(builder: MainConferenceCorpusBuilder) -> list[CorpusPaper]:
    records = load_json_with_recovery(builder.manifest_path, default_value=[])
    return [CorpusPaper.from_manifest_record(r) for r in records]


def step_collect(builder: MainConferenceCorpusBuilder):
    print("=" * 60)
    print("STEP 1: Collect manifest (adding 2025 papers)")
    print("=" * 60)
    t0 = time.perf_counter()
    papers = builder.collect_manifest()
    elapsed = time.perf_counter() - t0
    print(f"\nManifest: {len(papers)} total papers ({elapsed:.0f}s)")

    from collections import Counter
    year_counts = Counter(p.year for p in papers)
    for year in sorted(year_counts):
        print(f"  {year}: {year_counts[year]} papers")


def step_download(builder: MainConferenceCorpusBuilder):
    print("=" * 60)
    print("STEP 2: Download PDFs")
    print("=" * 60)
    papers = _load_manifest(builder)
    pending = sum(1 for p in papers if p.pdf_url and not (builder.pdf_root / p.conference / str(p.year) / f"{p.paper_id}.pdf").exists())
    print(f"Total: {len(papers)}, pending download: {pending}")
    t0 = time.perf_counter()
    asyncio.run(builder.download_papers(papers))
    elapsed = time.perf_counter() - t0
    print(f"\nDownload complete ({elapsed:.0f}s)")


def step_extract(builder: MainConferenceCorpusBuilder):
    print("=" * 60)
    print("STEP 3: Extract PDF → Markdown")
    print("=" * 60)
    papers = _load_manifest(builder)
    pending = [p for p in papers if p.pdf_path and Path(p.pdf_path).exists() and not (p.markdown_path and Path(p.markdown_path).exists())]
    print(f"Total: {len(papers)}, pending extract: {len(pending)}")
    t0 = time.perf_counter()
    builder.extract_markdown(pending)
    elapsed = time.perf_counter() - t0
    print(f"\nExtraction complete ({elapsed:.0f}s)")


def step_preindex(builder: MainConferenceCorpusBuilder):
    print("=" * 60)
    print("STEP 4: Preindex into Qdrant + SQLite")
    print("=" * 60)
    from scripts.preindex_existing_extracted_corpus_papers import run_preindex, _default_heartbeat

    report_path = builder.report_dir / "preindex_existing_extracted_corpus_papers.json"
    asyncio.run(run_preindex(
        manifest_path=builder.manifest_path,
        report_path=report_path,
        checkpoint_interval=200,
        batch_size=50,
        limit=0,
        embedding_device="cuda",
        require_gpu=True,
        heartbeat_callback=_default_heartbeat,
    ))


STEPS = {
    "collect": step_collect,
    "download": step_download,
    "extract": step_extract,
    "preindex": step_preindex,
}


def main():
    parser = argparse.ArgumentParser(description="Run 2025 paper collection pipeline")
    parser.add_argument("--step", choices=list(STEPS.keys()), default=None,
                        help="Run a single step (default: all steps in order)")
    args = parser.parse_args()

    builder = MainConferenceCorpusBuilder()
    print(f"Corpus root: {builder.corpus_root}")
    print(f"Years: {builder.years}")
    print(f"Conferences: {builder.conferences}")
    print()

    if args.step:
        STEPS[args.step](builder)
    else:
        for name, fn in STEPS.items():
            fn(builder)
            print()


if __name__ == "__main__":
    main()
