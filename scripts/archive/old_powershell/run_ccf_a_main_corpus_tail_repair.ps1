param(
    [string]$ProxyUrl = "http://127.0.0.1:10808",
    [string]$CorpusRoot = "data/papers/ccf_a_main_conference_corpus",
    [int]$ManifestCheckpointInterval = 200,
    [int]$DownloadRetryAttempts = 10,
    [int]$DownloadConcurrency = 1,
    [int]$MaxRepairCycles = 5,
    [int]$MaxNoProgressCycles = 2,
    [int]$PollIntervalSeconds = 60,
    [string]$PipelineStatePath = ".omx/state/ccf_a_main_corpus_pipeline_state.json",
    [string]$TailRepairLogPath = ".omx/logs/ccf_a_main_corpus_tail_repair.log",
    [string]$TailRepairErrorLogPath = ".omx/logs/ccf_a_main_corpus_tail_repair.err.log"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $projectRoot = Split-Path -Parent $PSScriptRoot
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

function Ensure-ParentDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath
    )

    $parentDirectory = Split-Path -Parent $FilePath
    if ($parentDirectory -and -not (Test-Path $parentDirectory)) {
        New-Item -ItemType Directory -Path $parentDirectory -Force | Out-Null
    }
}

function Read-PipelineState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        return $null
    }

    return Get-Content -Raw -Encoding UTF8 $FilePath | ConvertFrom-Json
}

function Write-TailRepairLogLine {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $timestampedMessage = "{0} {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $resolvedTailRepairLogPath -Value $timestampedMessage -Encoding UTF8
}

function Get-CorpusSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestFilePath
    )

    $snapshotJson = & $pythonExecutable -c @"
import json
import pathlib
import sys

papers = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))

downloaded = sum(1 for paper in papers if paper.get("pdf_path"))
download_pending = sum(1 for paper in papers if not paper.get("pdf_path"))
extracted_ok = sum(1 for paper in papers if paper.get("markdown_path") and not paper.get("extract_error"))
extract_fail = sum(1 for paper in papers if paper.get("extract_error"))
indexed_ok = sum(1 for paper in papers if paper.get("indexed_doc_id") and not paper.get("index_error"))
index_fail = sum(1 for paper in papers if paper.get("index_error"))

snapshot = {
    "total": len(papers),
    "downloaded": downloaded,
    "download_pending": download_pending,
    "extracted_ok": extracted_ok,
    "extract_fail": extract_fail,
    "indexed_ok": indexed_ok,
    "index_fail": index_fail,
    "indexed_backlog": max(extracted_ok - indexed_ok, 0),
}

print(json.dumps(snapshot))
"@ $ManifestFilePath

    return $snapshotJson | ConvertFrom-Json
}

function Test-CorpusConverged {
    param(
        [Parameter(Mandatory = $true)]
        $Snapshot
    )

    return (
        $Snapshot.download_pending -eq 0 -and
        $Snapshot.extract_fail -eq 0 -and
        $Snapshot.index_fail -eq 0 -and
        $Snapshot.indexed_backlog -eq 0
    )
}

function Test-CorpusProgressed {
    param(
        [Parameter(Mandatory = $true)]
        $BeforeSnapshot,
        [Parameter(Mandatory = $true)]
        $AfterSnapshot
    )

    return (
        $AfterSnapshot.downloaded -ne $BeforeSnapshot.downloaded -or
        $AfterSnapshot.download_pending -ne $BeforeSnapshot.download_pending -or
        $AfterSnapshot.extracted_ok -ne $BeforeSnapshot.extracted_ok -or
        $AfterSnapshot.extract_fail -ne $BeforeSnapshot.extract_fail -or
        $AfterSnapshot.indexed_ok -ne $BeforeSnapshot.indexed_ok -or
        $AfterSnapshot.index_fail -ne $BeforeSnapshot.index_fail
    )
}

function Invoke-RebuildStage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StageName
    )

    Write-TailRepairLogLine -Message ("Starting stage: {0}" -f $StageName)

    $process = Start-Process `
        -FilePath $pythonExecutable `
        -ArgumentList (@($rebuildScriptPath, "--stage", $StageName) + $commonArguments[1..($commonArguments.Count - 1)]) `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $resolvedTailRepairLogPath `
        -RedirectStandardError $resolvedTailRepairErrorLogPath `
        -PassThru

    $process.WaitForExit()

    Write-TailRepairLogLine -Message ("Finished stage: {0} (exit={1})" -f $StageName, $process.ExitCode)

    if ($process.ExitCode -ne 0) {
        throw "Stage $StageName failed with exit code $($process.ExitCode)."
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
$rebuildScriptPath = Join-Path $projectRoot "scripts\rebuild_ccf_a_main_corpus.py"
$resolvedPipelineStatePath = Resolve-ProjectPath -Path $PipelineStatePath
$resolvedTailRepairLogPath = Resolve-ProjectPath -Path $TailRepairLogPath
$resolvedTailRepairErrorLogPath = Resolve-ProjectPath -Path $TailRepairErrorLogPath
$resolvedManifestPath = Resolve-ProjectPath -Path (Join-Path $CorpusRoot "manifests/paper_manifest.json")

@(
    $resolvedTailRepairLogPath,
    $resolvedTailRepairErrorLogPath
) | ForEach-Object { Ensure-ParentDirectory -FilePath $_ }

$commonArguments = @(
    $rebuildScriptPath,
    "--proxy", $ProxyUrl,
    "--corpus-root", $CorpusRoot,
    "--manifest-checkpoint-interval", $ManifestCheckpointInterval,
    "--download-retry-attempts", $DownloadRetryAttempts,
    "--download-concurrency", $DownloadConcurrency
)

$terminalStages = @("index_completed", "index_failed", "extract_failed")

while ($true) {
    $state = Read-PipelineState -FilePath $resolvedPipelineStatePath
    if ($state -ne $null -and $terminalStages -contains [string]$state.stage) {
        break
    }
    Start-Sleep -Seconds $PollIntervalSeconds
}

$stages = @("repair_urls", "download", "extract", "index")
$noProgressCycles = 0

for ($cycle = 1; $cycle -le $MaxRepairCycles; $cycle++) {
    $beforeSnapshot = Get-CorpusSnapshot -ManifestFilePath $resolvedManifestPath
    Write-TailRepairLogLine -Message (
        "Repair cycle {0}/{1} snapshot before run: downloaded={2}, pending={3}, extracted_ok={4}, extract_fail={5}, indexed_ok={6}, index_fail={7}, indexed_backlog={8}" -f
        $cycle, $MaxRepairCycles, $beforeSnapshot.downloaded, $beforeSnapshot.download_pending, $beforeSnapshot.extracted_ok,
        $beforeSnapshot.extract_fail, $beforeSnapshot.indexed_ok, $beforeSnapshot.index_fail, $beforeSnapshot.indexed_backlog
    )

    if (Test-CorpusConverged -Snapshot $beforeSnapshot) {
        Write-TailRepairLogLine -Message "Corpus is already converged. Tail repair loop exits without additional work."
        exit 0
    }

    foreach ($stage in $stages) {
        Invoke-RebuildStage -StageName $stage
    }

    $afterSnapshot = Get-CorpusSnapshot -ManifestFilePath $resolvedManifestPath
    Write-TailRepairLogLine -Message (
        "Repair cycle {0}/{1} snapshot after run: downloaded={2}, pending={3}, extracted_ok={4}, extract_fail={5}, indexed_ok={6}, index_fail={7}, indexed_backlog={8}" -f
        $cycle, $MaxRepairCycles, $afterSnapshot.downloaded, $afterSnapshot.download_pending, $afterSnapshot.extracted_ok,
        $afterSnapshot.extract_fail, $afterSnapshot.indexed_ok, $afterSnapshot.index_fail, $afterSnapshot.indexed_backlog
    )

    if (Test-CorpusConverged -Snapshot $afterSnapshot) {
        Write-TailRepairLogLine -Message "Corpus converged after tail repair."
        exit 0
    }

    if (Test-CorpusProgressed -BeforeSnapshot $beforeSnapshot -AfterSnapshot $afterSnapshot) {
        $noProgressCycles = 0
        continue
    }

    $noProgressCycles += 1
    Write-TailRepairLogLine -Message (
        "Repair cycle {0}/{1} produced no measurable progress. no_progress_cycles={2}/{3}" -f
        $cycle, $MaxRepairCycles, $noProgressCycles, $MaxNoProgressCycles
    )

    if ($noProgressCycles -ge $MaxNoProgressCycles) {
        throw "Tail repair stopped after repeated no-progress cycles."
    }
}

throw "Tail repair exhausted all configured repair cycles without converging."
