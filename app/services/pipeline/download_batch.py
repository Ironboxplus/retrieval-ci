"""Multi-venue PDF download batch: one ingest_pipeline task, one download stage.

``venue="papers-download"`` with ``year=<batch number>`` selects this pipeline.
``options["targets"]`` lists the manifest slices ("ECCV:2024", "IEEE S&P:2023",
...) that download together as ONE serial queue: ``download_subset.py
--host-turns`` never sends two consecutive requests to the same host (or venue)
while another has work, and cools down a host that refuses instead of
retrying it hot. Holding the shared ``net`` lock for the whole batch keeps
every other paper download off the proxy meanwhile.

The batch only downloads. Each slice is ingested later by its own
``(venue, year)`` pipeline, whose download stage then finds the PDFs on disk
and skips.

``options["collect"]`` (default off) puts a ``collect`` stage in front: the targets
that have no row in this host's manifest yet are collected from their publishers
(``scripts/collect/collect_2026_to_manifest.py``, which merges by paper_id into the
manifest after taking a backup), so a slice that was never collected here can be
downloaded in the same task. Targets that already have rows are left alone,
because a re-collect replaces the slice's records and drops the PDF/Markdown
paths and full-text provenance recorded on them.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from app.services.pipeline.context import StageContext
from app.services.pipeline.runners import EventSink, run_local
from app.services.pipeline.spec import GateResult, Progress, StageSpec
from pipeline.paper_crawler.runtime_files import load_json_with_recovery

DOWNLOAD_BATCH_VENUE = "papers-download"
DEFAULT_MIN_INTERVAL_SECONDS = 3.0
DEFAULT_HOST_COOLDOWN_SECONDS = 300.0
GATE_MIN_FRACTION = 0.9


def is_download_batch_venue(venue: str) -> bool:
    return venue.lower() == DOWNLOAD_BATCH_VENUE


def parse_download_targets(raw_targets) -> list[tuple[str, int]]:
    """["ECCV:2024", ...] -> [("ECCV", 2024), ...]; raises ValueError when malformed or empty."""
    targets: list[tuple[str, int]] = []
    for raw in raw_targets or []:
        name, sep, raw_year = str(raw).rpartition(":")
        if not sep or not name.strip() or not raw_year.isdigit():
            raise ValueError(f"download targets look like CONF:YEAR, got {raw!r}")
        target = (name.strip(), int(raw_year))
        if target not in targets:
            targets.append(target)
    if not targets:
        raise ValueError("a download batch needs at least one CONF:YEAR target")
    return targets


@dataclass
class DownloadBatchContext(StageContext):
    """StageContext for one download batch; ``year`` carries the batch number."""

    def __post_init__(self) -> None:
        if self.year < 1:
            raise ValueError(f"download batch number must be >= 1, got {self.year}")
        parse_download_targets(self.options.get("targets"))

    @property
    def targets(self) -> list[tuple[str, int]]:
        return parse_download_targets(self.options.get("targets"))

    @property
    def min_interval_seconds(self) -> float:
        return max(0.0, float(self.options.get("min_interval_seconds", DEFAULT_MIN_INTERVAL_SECONDS)))

    @property
    def host_cooldown_seconds(self) -> float:
        return max(0.0, float(self.options.get("host_cooldown_seconds", DEFAULT_HOST_COOLDOWN_SECONDS)))

    @property
    def collect_enabled(self) -> bool:
        return bool(self.options.get("collect", False))

    def target_row_counts(self) -> list[tuple[str, int, int]]:
        """(conference, year, manifest rows) per target: what collect has already put here."""
        records = load_json_with_recovery(self.manifest_path, [])
        if not isinstance(records, list):
            records = []
        counts: dict[tuple, int] = {}
        for r in records:
            key = (r.get("conference"), r.get("year"))
            counts[key] = counts.get(key, 0) + 1
        return [(conference, year, counts.get((conference, year), 0)) for conference, year in self.targets]

    def target_pdf_dir(self, conference: str, year: int) -> Path:
        # Same layout as MainConferenceCorpusBuilder.download_papers writes.
        return self.corpus_root / "pdf" / conference.lower() / str(year)

    def target_progress(self) -> list[tuple[str, int, int, int, int]]:
        """(conference, year, downloaded, rejected, total) per target, from the
        manifest and the PDF directories: a slice paper counts once its own PDF
        file exists, so unrelated files in a reused directory never inflate the
        count. ``rejected`` papers have no PDF because every copy tried failed the
        full-text identity check -- settled, not missing."""
        from pipeline.paper_crawler.ccf_a_main_corpus import REJECTED_IDENTITY_DIR, sanitize_file_component

        records = load_json_with_recovery(self.manifest_path, [])
        if not isinstance(records, list):
            records = []
        rows = []
        for conference, year in self.targets:
            names = {
                f"{sanitize_file_component(r['paper_id'])}.pdf"
                for r in records
                if r.get("conference") == conference and r.get("year") == year and r.get("paper_id")
            }
            present = self._pdf_names(self.target_pdf_dir(conference, year))
            rejected = self._pdf_names(self.corpus_root / REJECTED_IDENTITY_DIR / conference.lower() / str(year))
            rows.append((conference, year, len(names & present), len((names & rejected) - present), len(names)))
        return rows

    @staticmethod
    def _pdf_names(directory: Path) -> set[str]:
        try:
            return {p.name for p in directory.glob("*.pdf")} if directory.exists() else set()
        except OSError as exc:
            logger.warning(f"download_batch.pdf_dir.oserror dir={directory}: {exc}")
            return set()


def maybe_download_batch_context(**kw) -> DownloadBatchContext | None:
    if not is_download_batch_venue(kw.get("venue", "")):
        return None
    return DownloadBatchContext(**kw)


def build_download_batch_cmd(ctx: DownloadBatchContext) -> list[str]:
    cmd = [sys.executable, "scripts/download_subset.py"]
    for conference, year in ctx.targets:
        cmd.extend(["--target", f"{conference}:{year}"])
    cmd.extend([
        "--host-turns",
        "--min-interval-seconds", f"{ctx.min_interval_seconds:g}",
        "--host-cooldown-seconds", f"{ctx.host_cooldown_seconds:g}",
        "--corpus-root", str(ctx.corpus_root),
    ])
    return cmd


def build_collect_cmd(ctx: DownloadBatchContext) -> list[str]:
    """Collect the targets that have no manifest rows yet, into this context's manifest."""
    cmd = [sys.executable, "scripts/collect/collect_2026_to_manifest.py"]
    for conference, year, rows in ctx.target_row_counts():
        if not rows:
            cmd.extend(["--target", f"{conference}:{year}"])
    cmd.extend(["--corpus-root", str(ctx.corpus_root)])
    return cmd


def collect_probe(ctx: DownloadBatchContext) -> Progress:
    """Targets with at least one manifest row, out of all targets."""
    rows = ctx.target_row_counts()
    return Progress(
        done=sum(1 for _, _, n in rows if n),
        total=len(rows),
        message=", ".join(f"{conf} {year} {n} rows" for conf, year, n in rows),
    )


def collect_gate(ctx: DownloadBatchContext) -> GateResult:
    empty = [f"{conf} {year}" for conf, year, n in ctx.target_row_counts() if not n]
    if empty:
        return GateResult(False, "no manifest rows after collect for: " + ", ".join(empty))
    return GateResult(True)


def download_batch_probe(ctx: DownloadBatchContext) -> Progress:
    rows = ctx.target_progress()
    message = ", ".join(
        f"{conf} {year} {done}/{total}" + (f" ({rejected} rejected)" if rejected else "")
        for conf, year, done, rejected, total in rows
    )
    return Progress(
        done=sum(done for _, _, done, _, _ in rows),
        total=sum(total for _, _, _, _, total in rows),
        message=message,
    )


def download_batch_gate(ctx: DownloadBatchContext) -> GateResult:
    # A paper whose every copy failed the identity check is settled: no full text.
    short = [
        f"{conf} {year} {done}+{rejected} rejected/{total}"
        for conf, year, done, rejected, total in ctx.target_progress()
        if total and (done + rejected) / total < GATE_MIN_FRACTION
    ]
    if short:
        return GateResult(False, "below 90% downloaded or rejected: " + ", ".join(short))
    return GateResult(True)


def build_download_batch_pipeline(ctx: DownloadBatchContext) -> list[StageSpec]:
    def run(c: DownloadBatchContext, sink: EventSink) -> None:
        sink(
            f"download batch {c.year}: {len(c.targets)} slices, host turns, "
            f"min_interval={c.min_interval_seconds:g}s, host cooldown={c.host_cooldown_seconds:g}s"
        )
        run_local(build_download_batch_cmd(c), c.project_root, sink, abort=c.abort)

    def collect_run(c: DownloadBatchContext, sink: EventSink) -> None:
        missing = [f"{conf} {year}" for conf, year, n in c.target_row_counts() if not n]
        sink(f"collect: {len(missing)} of {len(c.targets)} slices have no manifest rows yet: {', '.join(missing)}")
        run_local(build_collect_cmd(c), c.project_root, sink, abort=c.abort)

    specs = []
    if ctx.collect_enabled:
        specs.append(StageSpec(
            key="collect",
            title=f"Collect batch {ctx.year} manifest rows ({len(ctx.targets)} slices)",
            run=collect_run,
            probe=collect_probe,
            gate=collect_gate,
            # Satisfied when every target already has rows: nothing is re-collected.
            # The probe reads the whole manifest, so no background sampling.
            poll_interval_s=0,
            resource_key="net",  # publisher listings go through the shared proxy
        ))
    specs.append(StageSpec(
        key="download",
        title=f"Download batch {ctx.year} PDFs ({len(ctx.targets)} slices)",
        run=run,
        probe=download_batch_probe,
        gate=download_batch_gate,
        resource_key="net",
        # The task's own progress mirrors the PDF count only when download is its
        # sole stage; with a collect stage it is the usual stage count.
        parent_progress_from_probe=not ctx.collect_enabled,
    ))
    logger.info(
        f"registry.build_download_batch_pipeline batch={ctx.year} targets={ctx.targets} "
        f"stages={[s.key for s in specs]}")
    return specs
