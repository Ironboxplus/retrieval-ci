param(
    [string]$ManifestPath = "data/papers/ccf_a_main_conference_corpus/manifests/paper_manifest.json",
    [string]$ReportPath = "data/papers/ccf_a_main_conference_corpus/reports/preindex_existing_extracted_corpus_papers.json",
    [int]$CheckpointInterval = 100,
    [int]$Limit = 0,
    [string]$EmbeddingDevice = "cuda",
    [switch]$AllowCpu,
    [string]$StdoutLogPath = ".omx/logs/preindex_existing_extracted_corpus_papers.out.log",
    [string]$StderrLogPath = ".omx/logs/preindex_existing_extracted_corpus_papers.err.log"
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

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $projectRoot "scripts\preindex_existing_extracted_corpus_papers.py"
$resolvedManifestPath = Resolve-ProjectPath -Path $ManifestPath
$resolvedReportPath = Resolve-ProjectPath -Path $ReportPath
$resolvedStdoutLogPath = Resolve-ProjectPath -Path $StdoutLogPath
$resolvedStderrLogPath = Resolve-ProjectPath -Path $StderrLogPath

@(
    $resolvedReportPath,
    $resolvedStdoutLogPath,
    $resolvedStderrLogPath
) | ForEach-Object { Ensure-ParentDirectory -FilePath $_ }

$arguments = @(
    $scriptPath,
    "--manifest-path", $resolvedManifestPath,
    "--report-path", $resolvedReportPath,
    "--checkpoint-interval", $CheckpointInterval,
    "--limit", $Limit,
    "--embedding-device", $EmbeddingDevice
)

if ($AllowCpu) {
    $arguments += "--allow-cpu"
}

$process = Start-Process `
    -FilePath $pythonExecutable `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $resolvedStdoutLogPath `
    -RedirectStandardError $resolvedStderrLogPath `
    -PassThru

[pscustomobject]@{
    process_id = $process.Id
    manifest_path = $resolvedManifestPath
    report_path = $resolvedReportPath
    stdout_log_path = $resolvedStdoutLogPath
    stderr_log_path = $resolvedStderrLogPath
    checkpoint_interval = $CheckpointInterval
    limit = $Limit
    embedding_device = $EmbeddingDevice
    allow_cpu = [bool]$AllowCpu
} | ConvertTo-Json -Depth 4
