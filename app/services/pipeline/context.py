"""StageContext — the opaque carrier the orchestrator forwards to every callback.

The orchestrator never introspects `ctx`; it just hands it to run/probe/gate/
preflight. So this is a plain data object: venue/year + resolved paths + a few
manifest/filesystem helpers + a thin state proxy onto TaskStore. EVERYTHING here
is generic for ANY (venue, year) — no conference is special-cased.

Path model (matches the project's local-symlink → SMB layout):
  - pdf_dir is the LOCAL corpus path (a symlink into SMB on Arc) used by the
    local download stage + local pdf_count probe.
  - kss_pdf_dir / kss_md_dir are the KSS-side SMB mount paths handed to remote
    marker / embed launchers over ssh.
  - smb_md_dir is the Arc-side UNC view of marker output, used to PROBE extracted
    markdown without paying for an ssh round-trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from app.services.pipeline.store import TaskStore
from pipeline.paper_crawler.runtime_files import load_json_with_recovery

# Repo root = 4 parents up from this file (app/services/pipeline/context.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CORPUS_ROOT = _PROJECT_ROOT / "data" / "papers" / "ccf_a_main_conference_corpus"


@dataclass
class StageContext:
    """Opaque per-run context: (venue, year) + resolved paths + helpers.

    Forwarded to stage callbacks unchanged. Construct one per pipeline run.
    """

    venue: str
    year: int
    store: TaskStore
    task_id: int

    project_root: Path = _PROJECT_ROOT
    corpus_root: Path = _DEFAULT_CORPUS_ROOT

    concurrency: int = 8
    world_size: int = 8
    # Execution topology for the GPU stages (extract + embed):
    #   "remote" — orchestrator on Arc, GPU work on a KSS-style box over
    #              ssh + SMB-UNC polling + ship.sh (cwrsync). Original default.
    #   "local"  — orchestrator runs ON the GPU box (self-hosted, post-4090
    #              migration); extract/embed run via run_local against local
    #              filesystem dirs. No ssh / SMB / ship.sh (W106).
    # download / preindex / ingest already run local in BOTH modes — only
    # extract + embed branch on this. Default "remote" keeps existing callers.
    execution_mode: str = "remote"
    # Stacking factor: how many marker/embed ranks to launch per GPU that
    # passes preflight. Effective ranks = len(state.gpu_ids) * ranks_per_gpu.
    # 1 = original behavior (1 rank/GPU). 2 fits comfortably on a 24GB 4090
    # for marker (~10-12GB/rank). Preflight still controls which GPUs make
    # the cut — this only changes how many ranks land on each.
    ranks_per_gpu: int = 1
    ssh_target: str = "kss"

    # Where the retrival repo lives on the remote box (KSS default; override
    # for any other ssh target e.g. "/data/arc/retrival" on 4090). Marker /
    # embed launcher script paths are derived from this — no other hardcodes.
    remote_repo: str = "/home/kss/arc/retrival"

    # KSS-side SMB mount (what the remote launchers see) and the Arc git-bash UNC
    # view of the same share (what Arc-side probes glob).
    kss_smb_root: str = "/mnt/lsb/retrival"
    arc_smb_unc: str = "//10.100.100.9/lsb/retrival"

    # Optional manifest path override (tests inject a temp manifest). When None,
    # falls back to corpus_root/manifests/paper_manifest.json.
    manifest_path_override: Optional[Path] = None

    # ----------------------------------------------------------- computed paths
    @property
    def conf_lower(self) -> str:
        return self.venue.lower()

    @property
    def manifest_path(self) -> Path:
        if self.manifest_path_override is not None:
            return self.manifest_path_override
        return self.corpus_root / "manifests" / "paper_manifest.json"

    @property
    def pdf_dir(self) -> Path:
        """LOCAL corpus PDF dir (symlink → SMB) for this (venue, year)."""
        return self.corpus_root / "pdf" / self.conf_lower / str(self.year)

    @property
    def kss_pdf_dir(self) -> str:
        return f"{self.kss_smb_root}/papers/pdf/{self.conf_lower}/{self.year}"

    @property
    def kss_md_dir(self) -> str:
        return f"{self.kss_smb_root}/papers/markdown/{self.conf_lower}/{self.year}"

    @property
    def smb_md_dir(self) -> str:
        """Arc-side UNC view of marker output (probed without ssh)."""
        return f"{self.arc_smb_unc}/papers/markdown/{self.conf_lower}/{self.year}"

    @property
    def smb_vectors_dir(self) -> str:
        """Locally-mounted SMB share holding the embed output (vectors_*.parquet).

        FLAT + shared across venues (no conf/year subdir). Same NAS the KSS
        worker writes to, so Arc reads it WITHOUT ssh (a plain glob over the
        mount) and sees shards the instant they land. The embed poll watches
        this so step 4 finishes on real progress, instead of dead-waiting on the
        C-drive copy that ship-down only fills at step 5 (W82). Mirrors how
        smb_md_dir backs the extract poll."""
        return f"{self.arc_smb_unc}/vectors"

    # ------------------------------------------------------ local (self-hosted)
    @property
    def local_md_dir(self) -> Path:
        """LOCAL markdown dir where the self-hosted marker worker writes, for
        this (venue, year). Mirrors the corpus pdf layout: corpus_root/markdown/
        <conf_lower>/<year>. Used by the local extract probe/gate (W106)."""
        return self.corpus_root / "markdown" / self.conf_lower / str(self.year)

    @property
    def local_vectors_dir(self) -> Path:
        """LOCAL embed output dir (vectors_*.parquet). Same flat, cross-venue
        dir the ingest stage already reads (project_root/data/remote_embed/
        vectors); exposed here so the local embed probe can glob it without
        re-deriving the path (W106)."""
        return self.project_root / "data" / "remote_embed" / "vectors"

    def md_count_local(self) -> int:
        """Count *.md in the LOCAL markdown dir. Missing dir → 0. The local
        analogue of md_count() (which globs the SMB UNC dir); used by the
        local extract probe so self-hosted runs track real on-disk output."""
        d = self.local_md_dir
        try:
            if not d.exists():
                return 0
            return sum(1 for _ in d.glob("*.md"))
        except OSError as exc:
            logger.warning(f"context.md_count_local.oserror dir={d}: {exc}")
            return 0

    # --------------------------------------------------------------- manifest
    def manifest_slice(self) -> list[dict]:
        """Manifest records for this exact (venue, year). Generic filter."""
        records = load_json_with_recovery(self.manifest_path, [])
        if not isinstance(records, list):
            logger.warning(f"context.manifest.non_list path={self.manifest_path}")
            return []
        slice_ = [
            r for r in records
            if r.get("conference") == self.venue and r.get("year") == self.year
        ]
        logger.debug(
            f"context.manifest.slice venue={self.venue} year={self.year} "
            f"total={len(records)} slice={len(slice_)}"
        )
        return slice_

    def manifest_slice_total(self) -> int:
        return len(self.manifest_slice())

    def manifest_slice_extracted(self) -> int:
        """Count slice records whose markdown_path is set (extraction recorded)."""
        return sum(1 for r in self.manifest_slice() if r.get("markdown_path"))

    # ------------------------------------------------------------- fs counts
    def pdf_count(self) -> int:
        """Count *.pdf in the LOCAL pdf_dir. Missing dir → 0."""
        d = self.pdf_dir
        if not d.exists():
            return 0
        return sum(1 for _ in d.glob("*.pdf"))

    def md_count(self) -> int:
        """Count *.md in the Arc-side SMB markdown dir. Missing dir → 0."""
        d = Path(self.smb_md_dir)
        try:
            if not d.exists():
                return 0
            return sum(1 for _ in d.glob("*.md"))
        except OSError as exc:  # SMB share unreachable → treat as 0, best-effort probe
            logger.warning(f"context.md_count.oserror dir={d}: {exc}")
            return 0

    # ----------------------------------------------------------- state proxy
    def state_get(self, key: str, default: Any = None) -> Any:
        return self.store.get_state(self.task_id).get(key, default)

    def state_set(self, **kv: Any) -> None:
        self.store.update_state(self.task_id, **kv)
        logger.debug(f"context.state.set task={self.task_id} keys={list(kv)}")
