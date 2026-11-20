param(
    [string]$ProxyUrl = "http://127.0.0.1:10808",
    [string]$CorpusRoot = "data/papers/ccf_a_main_conference_corpus",
    [int]$ManifestCheckpointInterval = 200,
    [int]$DownloadRetryAttempts = 5,
    [int]$DownloadConcurrency = 4,
    [string]$CollectLogPath = ".omx/logs/ccf_a_main_corpus_collect.log",
    [string]$CollectErrorLogPath = ".omx/logs/ccf_a_main_corpus_collect.err.log",
    [string]$DownloadLogPath = ".omx/logs/ccf_a_main_corpus_download.log",
    [string]$DownloadErrorLogPath = ".omx/logs/ccf_a_main_corpus_download.err.log",
    [string]$DownloadMonitorLogPath = ".omx/logs/ccf_a_main_corpus_download_followup.jsonl",
    [string]$DownloadMonitorStatePath = ".omx/state/ccf_a_main_corpus_download_followup_state.json",
    [string]$PipelineStatePath = ".omx/state/ccf_a_main_corpus_pipeline_state.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $scriptRoot = Split-Path -Parent $PSScriptRoot
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $scriptRoot $Path))
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

function Write-PipelineState {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$State
    )

    $State.timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $State | ConvertTo-Json -Depth 5 | Set-Content -Path $resolvedPipelineStatePath -Encoding UTF8
}

function Get-CompletedCollectSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [string]$CorpusRootPath
    )

    $manifestPath = Resolve-ProjectPath -Path (Join-Path $CorpusRootPath "manifests/paper_manifest.json")
    $summaryPath = Resolve-ProjectPath -Path (Join-Path $CorpusRootPath "reports/build_summary.json")

    if (-not (Test-Path $manifestPath) -or -not (Test-Path $summaryPath)) {
        return $null
    }

    $manifestRecords = Get-Content $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $summary = Get-Content $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $manifestCount = @($manifestRecords).Count
    $summaryCount = $summary.paper_count

    if ($manifestCount -gt 0 -and $summaryCount -eq $manifestCount) {
        return [PSCustomObject]@{
            manifest_path = $manifestPath
            summary_path = $summaryPath
            manifest_count = $manifestCount
            summary_count = $summaryCount
        }
    }

    return $null
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
$rebuildScriptPath = Join-Path $projectRoot "scripts\rebuild_ccf_a_main_corpus.py"
$monitorScriptPath = Join-Path $projectRoot "scripts\monitor_ccf_a_main_corpus_download.ps1"

$resolvedCollectLogPath = Resolve-ProjectPath -Path $CollectLogPath
$resolvedCollectErrorLogPath = Resolve-ProjectPath -Path $CollectErrorLogPath
$resolvedDownloadLogPath = Resolve-ProjectPath -Path $DownloadLogPath
$resolvedDownloadErrorLogPath = Resolve-ProjectPath -Path $DownloadErrorLogPath
$resolvedDownloadMonitorLogPath = Resolve-ProjectPath -Path $DownloadMonitorLogPath
$resolvedDownloadMonitorStatePath = Resolve-ProjectPath -Path $DownloadMonitorStatePath
$resolvedPipelineStatePath = Resolve-ProjectPath -Path $PipelineStatePath

@(
    $resolvedCollectLogPath,
    $resolvedCollectErrorLogPath,
    $resolvedDownloadLogPath,
    $resolvedDownloadErrorLogPath,
    $resolvedDownloadMonitorLogPath,
    $resolvedDownloadMonitorStatePath,
    $resolvedPipelineStatePath
) | ForEach-Object { Ensure-ParentDirectory -FilePath $_ }

Write-PipelineState @{
    stage = "collect_running"
    proxy_url = $ProxyUrl
    corpus_root = $CorpusRoot
}

$commonArguments = @(
    $rebuildScriptPath,
    "--proxy", $ProxyUrl,
    "--corpus-root", $CorpusRoot,
    "--manifest-checkpoint-interval", $ManifestCheckpointInterval,
    "--download-retry-attempts", $DownloadRetryAttempts,
    "--download-concurrency", $DownloadConcurrency
)

$completedCollectSnapshot = Get-CompletedCollectSnapshot -ProjectRoot $projectRoot -CorpusRootPath $CorpusRoot
$collectProcess = $null
$collectExitCode = 0

if ($completedCollectSnapshot -eq $null) {
    $collectProcess = Start-Process `
        -FilePath $pythonExecutable `
        -ArgumentList (@($rebuildScriptPath, "--stage", "collect") + $commonArguments[1..($commonArguments.Count - 1)]) `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $resolvedCollectLogPath `
        -RedirectStandardError $resolvedCollectErrorLogPath `
        -PassThru

    $collectProcess.WaitForExit()
    $collectExitCode = $collectProcess.ExitCode
}

if ($collectExitCode -ne 0) {
    Write-PipelineState @{
        stage = "collect_failed"
        proxy_url = $ProxyUrl
        corpus_root = $CorpusRoot
        collect_exit_code = $collectExitCode
        collect_process_id = if ($collectProcess) { $collectProcess.Id } else { $null }
    }
    exit $collectExitCode
}

Write-PipelineState @{
    stage = "download_starting"
    proxy_url = $ProxyUrl
    corpus_root = $CorpusRoot
    collect_exit_code = $collectExitCode
    collect_process_id = if ($collectProcess) { $collectProcess.Id } else { $null }
    collect_skipped = [bool]($completedCollectSnapshot -ne $null)
    manifest_count = if ($completedCollectSnapshot) { $completedCollectSnapshot.manifest_count } else { $null }
}

$downloadProcess = Start-Process `
    -FilePath $pythonExecutable `
    -ArgumentList (@($rebuildScriptPath, "--stage", "download") + $commonArguments[1..($commonArguments.Count - 1)]) `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $resolvedDownloadLogPath `
    -RedirectStandardError $resolvedDownloadErrorLogPath `
    -PassThru

$monitorProcess = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $monitorScriptPath,
        "-ManifestPath", (Join-Path $CorpusRoot "manifests/paper_manifest.json"),
        "-PdfRoot", (Join-Path $CorpusRoot "pdf"),
        "-CorpusRootToken", ($CorpusRoot -replace "/", "\"),
        "-LogPath", $DownloadMonitorLogPath,
        "-StatePath", $DownloadMonitorStatePath,
        "-MonitoringWindowHours", 10,
        "-SampleIntervalSeconds", 300
    ) `
    -WorkingDirectory $projectRoot `
    -PassThru

Write-PipelineState @{
    stage = "download_running"
    proxy_url = $ProxyUrl
    corpus_root = $CorpusRoot
    collect_exit_code = $collectExitCode
    collect_process_id = if ($collectProcess) { $collectProcess.Id } else { $null }
    collect_skipped = [bool]($completedCollectSnapshot -ne $null)
    download_process_id = $downloadProcess.Id
    monitor_process_id = $monitorProcess.Id
}

$downloadProcess.WaitForExit()
$downloadExitCode = $downloadProcess.ExitCode

if (-not $monitorProcess.HasExited) {
    $monitorProcess.WaitForExit()
}

if ($downloadExitCode -ne 0) {
    Write-PipelineState @{
        stage = "download_failed"
        proxy_url = $ProxyUrl
        corpus_root = $CorpusRoot
        collect_exit_code = $collectExitCode
        collect_process_id = if ($collectProcess) { $collectProcess.Id } else { $null }
        collect_skipped = [bool]($completedCollectSnapshot -ne $null)
        download_exit_code = $downloadExitCode
        download_process_id = $downloadProcess.Id
        monitor_process_id = $monitorProcess.Id
    }
    exit $downloadExitCode
}

Write-PipelineState @{
    stage = "download_completed"
    proxy_url = $ProxyUrl
    corpus_root = $CorpusRoot
    collect_exit_code = $collectExitCode
    collect_process_id = if ($collectProcess) { $collectProcess.Id } else { $null }
    collect_skipped = [bool]($completedCollectSnapshot -ne $null)
    download_exit_code = $downloadExitCode
    download_process_id = $downloadProcess.Id
    monitor_process_id = $monitorProcess.Id
}
