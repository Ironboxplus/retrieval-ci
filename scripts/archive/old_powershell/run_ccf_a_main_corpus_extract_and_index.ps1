param(
    [string]$ProxyUrl = "http://127.0.0.1:10808",
    [string]$CorpusRoot = "data/papers/ccf_a_main_conference_corpus",
    [int]$ManifestCheckpointInterval = 200,
    [int]$DownloadRetryAttempts = 5,
    [int]$DownloadConcurrency = 4,
    [string]$ExtractLogPath = ".omx/logs/ccf_a_main_corpus_extract.log",
    [string]$ExtractErrorLogPath = ".omx/logs/ccf_a_main_corpus_extract.err.log",
    [string]$IndexLogPath = ".omx/logs/ccf_a_main_corpus_index.log",
    [string]$IndexErrorLogPath = ".omx/logs/ccf_a_main_corpus_index.err.log",
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

function Write-PipelineState {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$State
    )

    $State.timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $State | ConvertTo-Json -Depth 6 | Set-Content -Path $resolvedPipelineStatePath -Encoding UTF8
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
$rebuildScriptPath = Join-Path $projectRoot "scripts\rebuild_ccf_a_main_corpus.py"
$manifestPath = Resolve-ProjectPath -Path (Join-Path $CorpusRoot "manifests/paper_manifest.json")
$summaryPath = Resolve-ProjectPath -Path (Join-Path $CorpusRoot "reports/build_summary.json")

$resolvedExtractLogPath = Resolve-ProjectPath -Path $ExtractLogPath
$resolvedExtractErrorLogPath = Resolve-ProjectPath -Path $ExtractErrorLogPath
$resolvedIndexLogPath = Resolve-ProjectPath -Path $IndexLogPath
$resolvedIndexErrorLogPath = Resolve-ProjectPath -Path $IndexErrorLogPath
$resolvedPipelineStatePath = Resolve-ProjectPath -Path $PipelineStatePath

@(
    $resolvedExtractLogPath,
    $resolvedExtractErrorLogPath,
    $resolvedIndexLogPath,
    $resolvedIndexErrorLogPath,
    $resolvedPipelineStatePath
) | ForEach-Object { Ensure-ParentDirectory -FilePath $_ }

if (-not (Test-Path $manifestPath) -or -not (Test-Path $summaryPath)) {
    throw "Formal manifest or summary is missing. Cannot start extract/index."
}

$commonArguments = @(
    $rebuildScriptPath,
    "--proxy", $ProxyUrl,
    "--corpus-root", $CorpusRoot,
    "--manifest-checkpoint-interval", $ManifestCheckpointInterval,
    "--download-retry-attempts", $DownloadRetryAttempts,
    "--download-concurrency", $DownloadConcurrency
)

Write-PipelineState @{
    stage = "extract_running"
    proxy_url = $ProxyUrl
    corpus_root = $CorpusRoot
}

$extractProcess = Start-Process `
    -FilePath $pythonExecutable `
    -ArgumentList (@($rebuildScriptPath, "--stage", "extract") + $commonArguments[1..($commonArguments.Count - 1)]) `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $resolvedExtractLogPath `
    -RedirectStandardError $resolvedExtractErrorLogPath `
    -PassThru

$extractProcess.WaitForExit()
$extractExitCode = $extractProcess.ExitCode

if ($extractExitCode -ne 0) {
    Write-PipelineState @{
        stage = "extract_failed"
        proxy_url = $ProxyUrl
        corpus_root = $CorpusRoot
        extract_exit_code = $extractExitCode
        extract_process_id = $extractProcess.Id
    }
    exit $extractExitCode
}

Write-PipelineState @{
    stage = "index_running"
    proxy_url = $ProxyUrl
    corpus_root = $CorpusRoot
    extract_exit_code = $extractExitCode
    extract_process_id = $extractProcess.Id
}

$indexProcess = Start-Process `
    -FilePath $pythonExecutable `
    -ArgumentList (@($rebuildScriptPath, "--stage", "index") + $commonArguments[1..($commonArguments.Count - 1)]) `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $resolvedIndexLogPath `
    -RedirectStandardError $resolvedIndexErrorLogPath `
    -PassThru

$indexProcess.WaitForExit()
$indexExitCode = $indexProcess.ExitCode

if ($indexExitCode -ne 0) {
    Write-PipelineState @{
        stage = "index_failed"
        proxy_url = $ProxyUrl
        corpus_root = $CorpusRoot
        extract_exit_code = $extractExitCode
        extract_process_id = $extractProcess.Id
        index_exit_code = $indexExitCode
        index_process_id = $indexProcess.Id
    }
    exit $indexExitCode
}

Write-PipelineState @{
    stage = "index_completed"
    proxy_url = $ProxyUrl
    corpus_root = $CorpusRoot
    extract_exit_code = $extractExitCode
    extract_process_id = $extractProcess.Id
    index_exit_code = $indexExitCode
    index_process_id = $indexProcess.Id
}
