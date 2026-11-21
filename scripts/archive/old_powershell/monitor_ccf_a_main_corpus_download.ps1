# Monitors the formal CCF-A main-conference PDF download job and writes
# machine-readable progress snapshots for long blocking follow-up windows.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/monitor_ccf_a_main_corpus_download.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/monitor_ccf_a_main_corpus_download.ps1 `
#       -SampleIntervalSeconds 300 `
#       -MonitoringWindowHours 10

param(
    [string]$ManifestPath = "data/papers/ccf_a_main_conference_corpus/manifests/paper_manifest.json",
    [string]$PdfRoot = "data/papers/ccf_a_main_conference_corpus/pdf",
    [string]$CorpusRootToken = "data\papers\ccf_a_main_conference_corpus",
    [string]$LogPath = ".omx/logs/ccf_a_main_corpus_download_followup.jsonl",
    [string]$StatePath = ".omx/state/ccf_a_main_corpus_download_followup_state.json",
    [int]$SampleIntervalSeconds = 300,
    [int]$MonitoringWindowHours = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
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

function Get-CorpusRootPattern {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Token
    )

    $escapedToken = [regex]::Escape($Token)
    return $escapedToken.Replace("\\", "[\\/]")
}

function Get-DownloadProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CorpusPattern
    )

    return @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match "python" -and
                $_.CommandLine -match "scripts\\rebuild_ccf_a_main_corpus\.py" -and
                $_.CommandLine -match "--stage download" -and
                $_.CommandLine -match $CorpusPattern
            }
    )
}

function New-Snapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestFile,
        [Parameter(Mandatory = $true)]
        [string]$PdfDirectory,
        [Parameter(Mandatory = $true)]
        [string]$CorpusPattern,
        [Parameter(Mandatory = $true)]
        [datetime]$Deadline
    )

    if (-not (Test-Path $ManifestFile)) {
        throw "Manifest file not found: $ManifestFile"
    }

    $records = Get-Content -Raw -Encoding UTF8 $ManifestFile | ConvertFrom-Json
    $downloadedCount = @(
        $records | Where-Object {
            $_.pdf_path -and $_.pdf_path.Trim() -ne ""
        }
    ).Count
    $failedCount = @(
        $records | Where-Object {
            $_.download_error -and $_.download_error.Trim() -ne ""
        }
    ).Count

    $pdfFileCount = 0
    $temporaryFileCount = 0
    if (Test-Path $PdfDirectory) {
        $pdfFileCount = @(Get-ChildItem $PdfDirectory -Recurse -File -Filter *.pdf).Count
        $temporaryFileCount = @(
            Get-ChildItem $PdfDirectory -Recurse -File |
                Where-Object { $_.Name -like "*.write_in_progress.pdf" }
        ).Count
    }

    $downloadProcesses = @(Get-DownloadProcesses -CorpusPattern $CorpusPattern)
    $now = Get-Date

    $status = if ($downloadedCount -ge $records.Count) {
        "download_complete"
    }
    elseif (@($downloadProcesses).Count -gt 0 -and $now -lt $Deadline) {
        "download_running"
    }
    elseif (@($downloadProcesses).Count -gt 0 -and $now -ge $Deadline) {
        "monitor_window_complete"
    }
    else {
        "download_process_missing"
    }

    return [PSCustomObject]@{
        timestamp = $now.ToString("yyyy-MM-dd HH:mm:ss")
        status = $status
        manifest_total = $records.Count
        downloaded = $downloadedCount
        failed = $failedCount
        pending = $records.Count - $downloadedCount
        pdf_files = $pdfFileCount
        write_in_progress = $temporaryFileCount
        process_ids = @($downloadProcesses | ForEach-Object { $_.ProcessId })
    }
}

function Write-Snapshot {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Snapshot,
        [Parameter(Mandatory = $true)]
        [string]$ProgressLogFile,
        [Parameter(Mandatory = $true)]
        [string]$CurrentStateFile
    )

    $jsonLine = $Snapshot | ConvertTo-Json -Compress -Depth 4
    Add-Content -Path $ProgressLogFile -Value $jsonLine -Encoding UTF8
    Set-Content -Path $CurrentStateFile -Value ($Snapshot | ConvertTo-Json -Depth 4) -Encoding UTF8
}

$resolvedManifestPath = Resolve-ProjectPath -Path $ManifestPath
$resolvedPdfRoot = Resolve-ProjectPath -Path $PdfRoot
$resolvedLogPath = Resolve-ProjectPath -Path $LogPath
$resolvedStatePath = Resolve-ProjectPath -Path $StatePath
$corpusPattern = Get-CorpusRootPattern -Token $CorpusRootToken

Ensure-ParentDirectory -FilePath $resolvedLogPath
Ensure-ParentDirectory -FilePath $resolvedStatePath

$deadline = (Get-Date).AddHours($MonitoringWindowHours)

do {
    $snapshot = New-Snapshot `
        -ManifestFile $resolvedManifestPath `
        -PdfDirectory $resolvedPdfRoot `
        -CorpusPattern $corpusPattern `
        -Deadline $deadline

    Write-Snapshot `
        -Snapshot $snapshot `
        -ProgressLogFile $resolvedLogPath `
        -CurrentStateFile $resolvedStatePath

    if ($snapshot.status -ne "download_running") {
        break
    }

    Start-Sleep -Seconds $SampleIntervalSeconds
}
while ($true)
