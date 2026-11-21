# Medical corpus preparation

Prepare a **legal open-access medical PDF** pipeline without mixing into the CS
`papers` corpus. Network I/O **always** uses the Arc local proxy
`http://127.0.0.1:10808` (no silent direct fallback).

## Components

| Path | Role |
|------|------|
| `pipeline/medical_corpus/` | EuropePMC client, shield classifier, PDF planner/downloader, CF adapter |
| `vendor/TurnstileBypass/` | Ironboxplus dual-path CF SDK (**minimal** headless Chrome) |
| `scripts/medical/prepare_medical_corpus.py` | CLI |
| `MEDICAL_DATA_ROOT` | Deployment-owned manifest + PDF root; `.88` uses `/mnt/lsb/retrival/medical` |
| `app/services/medical_corpus/service.py` | Durable `TaskStore` / `PipelineOrchestrator` download stage and resume logic |
| `/api/pipeline/medical-download` | Internal-admin start API; status, details, events, cancellation and retry all use the existing `/api/tasks` surface |

## .88 execution contract

The internal UI and API are **control-plane only**. They never accept a target
directory, proxy URL, or bandwidth value from a browser. The compute service
must set the following service environment before the endpoint is allowed to
start a transfer:

```ini
MEDICAL_DOWNLOAD_HOST=10.100.100.88
MEDICAL_DATA_ROOT=/mnt/lsb/retrival/medical
MEDICAL_MIN_FREE_BYTES=21474836480
```

Every actual transfer is forced through `http://127.0.0.1:10808`. The medical
stage has both a TaskStore resource lock and an OS file lock at
`$MEDICAL_DATA_ROOT/manifest/medical-download.lock`, so two backend processes
cannot each open a 2 MB/s lane. A second task-id lease at
`$MEDICAL_DATA_ROOT/manifest/medical-download-task-<id>.lock` makes duplicate
restart recovery stand down without writing a false failure event. The one process-wide limiter uses a strict
decimal `2,000,000 B/s` ceiling (`2.00 MB/s`, not MiB/s). State shown in the
UI is persisted in the `Task` record; manifest JSONL is atomically checkpointed
after every completed source. A restart resumes from the manifest and skips
already verified `%PDF` + size + SHA-256 artifacts.

Before every new file, at `Content-Length` preflight, and before each stream
chunk is written, the worker checks free space on `MEDICAL_DATA_ROOT`. If the
reserve would be crossed, it deletes the `.write_in_progress` file, keeps the
manifest record pending, safely stops new transfers, and records the reason in
the task summary. It never falls back to `/home/arc`, a Windows path, or a
direct network connection.

## Cloudflare / Turnstile

On `403` / HTML interstitial / captcha-like bodies, the downloader can call:

```python
from pipeline.medical_corpus.cf_bypass import solve_cloudflare_for_url_sync
res = solve_cloudflare_for_url_sync(url, proxy="http://127.0.0.1:10808", headless=True)
# res.cookie_header / res.cf_clearance / res.token
```

Uses **`solve_cf_dual_path`** with backend preference **minimal** (system Chrome
headless via `selenium-driverless`). Not captcha SaaS, not headed click theater.

Install FO deps once on Arc:

```powershell
pip install selenium-driverless
# optional TLS fingerprint for soft/HTTP:
pip install curl_cffi
```

Vendor clone (already under `vendor/TurnstileBypass`, gitignored):

```powershell
git clone --depth 1 https://github.com/Ironboxplus/TurnstileBypass.git vendor/TurnstileBypass
```

## CLI

```powershell
$env:HTTP_PROXY = 'http://127.0.0.1:10808'
$env:HTTPS_PROXY = 'http://127.0.0.1:10808'

# .88 only: all runtime artifacts on the artifact mount, never Windows.
$env:MEDICAL_DATA_ROOT = '/mnt/lsb/retrival/medical'
$env:MEDICAL_DOWNLOAD_HOST = '10.100.100.88'
$env:MEDICAL_MIN_FREE_BYTES = '21474836480'

# Readiness
.\.venv\Scripts\python.exe scripts/medical/prepare_medical_corpus.py status

# Catalog smoke (network, no PDF bytes) — default = OA+PDF MED **top journals only**
.\.venv\Scripts\python.exe scripts/medical/prepare_medical_corpus.py collect --max-records 25

# Plan downloads from manifest
.\.venv\Scripts\python.exe scripts/medical/prepare_medical_corpus.py download

# Real download of a few PDFs (only after smoke)
.\.venv\Scripts\python.exe scripts/medical/prepare_medical_corpus.py download --execute-download --download-limit 3

# CF dual-path probe
.\.venv\Scripts\python.exe scripts/medical/prepare_medical_corpus.py cf-probe --url https://www.linshiyouxiang.net/
```

## Phases (do not skip)

1. **collect** — EuropePMC cursor → `data/medical/manifest/medical_manifest.jsonl`
2. **download dry-run** — annotate planned paths / missing URLs
3. **download --execute-download --download-limit N** — small real PDF smoke
4. Scale `max-records` / query only after (3) is green

## Top journals only (default)

Non-flagship OA dumps are **dropped**. Whitelist lives in
`pipeline/medical_corpus/top_journals.py` (NEJM, Lancet family, JAMA family,
BMJ, Nature Medicine / Nature / Science / Cell, Annals, plus high-IF specialty
flagships). EuropePMC query includes a `JOURNAL:"..."` OR-clause; client-side
`filter_top_journal_records` is the authority.

```powershell
# Explicitly disable whitelist (not recommended)
.\.venv\Scripts\python.exe scripts/medical/prepare_medical_corpus.py collect --no-top-journals --max-records 25
```

## Out of scope for this prep

- Full subscription PDF dumps for paywalled top journals (only legal OA PDFs)
- SCI-Hub or other unauthorized mirrors
- Merging into production CS `papers` Qdrant without a separate collection design
