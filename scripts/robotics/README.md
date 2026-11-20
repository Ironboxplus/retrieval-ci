# Robotics corpus preparation (ICRA / CoRL / RSS)

Prepare **full bulk** catalogs and PDF downloads for the three flagship robotics
venues over **2022–2026 inclusive**. Network I/O always uses
`http://127.0.0.1:10808`.

## Scope (default = full, not sample)

| Venue | Source | PDF | Notes (probed 2026-07-26) |
|-------|--------|-----|---------------------------|
| **RSS** | `roboticsproceedings.org` (`rss18`…`rss21`) | OA direct PDF | 2022–2025 online; **2026 (`rss22`) not published yet** → collect skips cell |
| **CoRL** | PMLR volumes | OA PDF (GitHub raw / mlr) | 2022=`v205` … 2025=`v305`; **2026 volume TBD** (DBLP auto-discover when available) |
| **ICRA** | DBLP HTML TOC (lxml) → IEEE stamp / arXiv | IEEE often **418/CF** from proxy | Full year catalog (~8s/year with lxml); CF bypass on download |

Default CLI years: `2022 2023 2024 2025 2026`.  
Default venues: `ICRA CoRL RSS`.  
Missing editions are recorded under `skipped_cells` and **do not abort** other cells.

## CLI

```powershell
$env:HTTP_PROXY = 'http://127.0.0.1:10808'
$env:HTTPS_PROXY = 'http://127.0.0.1:10808'

# Readiness + default full scope
.\.venv\Scripts\python.exe scripts/robotics/prepare_robotics_corpus.py status

# FULL catalog: all venues × all years (no PDF bytes)
.\.venv\Scripts\python.exe scripts/robotics/prepare_robotics_corpus.py collect

# Smoke only (do not use for production bulk)
.\.venv\Scripts\python.exe scripts/robotics/prepare_robotics_corpus.py collect --sample-per-cell 5

# Plan every PDF path from the full manifest
.\.venv\Scripts\python.exe scripts/robotics/prepare_robotics_corpus.py download

# ONE-SHOT full download of the entire manifest (after collect is green)
.\.venv\Scripts\python.exe scripts/robotics/prepare_robotics_corpus.py download --execute-download
```

`--download-limit 0` (default) means **entire manifest**. Only set a positive
limit for smoke. Resumable: download re-skips rows that already have a valid PDF.

## Data layout

```
data/robotics/
  manifest/robotics_manifest.jsonl
  manifest/robotics_prepare_summary.json
  pdf/{icra,corl,rss}/{year}/{paper_id}.pdf
```

## CF / IEEE

ICRA PDFs on IEEE Xplore often return 418 + Cloudflare from the proxy exit IP.
Downloader reuses medical `TurnstileBypass` minimal headless (`--no-cf-bypass`
to disable). RSS and CoRL/PMLR are normally open without CF.
