[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$DestinationRoot,

    [string]$CatalogPath,

    [string[]]$SourceId,

    [ValidateRange(1, 2)]
    [int]$MaxConcurrentMirrors = 2,

    [ValidateNotNullOrEmpty()]
    [string]$ProxyUrl = "http://127.0.0.1:10808"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-PublicRepositoryCatalog {
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Public source catalog does not exist: $Path"
    }

    $catalog = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($catalog.catalog_id -ne "public-code-first-wave") {
        throw "Unexpected catalog_id '$($catalog.catalog_id)' in $Path"
    }
    return @($catalog.repositories)
}

function Resolve-SafeMirrorTargetPath {
    param(
        [Parameter(Mandatory)][string]$DestinationRoot,
        [Parameter(Mandatory)][string]$RelativePath
    )

    $normalizedRelativePath = $RelativePath.Replace("\", "/")
    $pathParts = @($normalizedRelativePath.Split("/"))
    if ([string]::IsNullOrWhiteSpace($normalizedRelativePath) `
            -or [System.IO.Path]::IsPathRooted($normalizedRelativePath) `
            -or $pathParts | Where-Object { $_ -in @("", ".", "..") }) {
        throw "Catalog relative_path must be a normalized path below DestinationRoot: '$RelativePath'"
    }

    $resolvedRoot = [System.IO.Path]::GetFullPath($DestinationRoot)
    $targetPath = [System.IO.Path]::GetFullPath(
        (Join-Path $resolvedRoot "$normalizedRelativePath.git")
    )
    $rootPrefix = $resolvedRoot.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $targetPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Catalog relative_path escapes DestinationRoot: '$RelativePath'"
    }
    return $targetPath
}

function Invoke-GitInspection {
    param([Parameter(Mandatory)][string[]]$GitArguments)

    # A failed Git probe is data for the caller to classify, not a clone failure.
    # PowerShell otherwise turns native stderr into a terminating error under the
    # script-wide ErrorActionPreference=Stop before we can inspect $LASTEXITCODE.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& git @GitArguments 2>$null)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    return [PSCustomObject]@{
        Output = $output
        ExitCode = $exitCode
    }
}

function Assert-ExistingMirrorMatchesCatalog {
    param(
        [Parameter(Mandatory)][string]$RepositoryPath,
        [Parameter(Mandatory)][string]$ExpectedUpstreamUrl
    )

    $isBareRepositoryInspection = Invoke-GitInspection -GitArguments @(
        "-C", $RepositoryPath, "rev-parse", "--is-bare-repository"
    )
    $isBareRepositoryOutput = @($isBareRepositoryInspection.Output)
    $isBareRepositoryExitCode = $isBareRepositoryInspection.ExitCode
    $isBareRepository = if ($isBareRepositoryOutput.Count -gt 0) {
        ([string]$isBareRepositoryOutput[0]).Trim()
    } else {
        ""
    }
    if ($isBareRepositoryExitCode -ne 0 -or $isBareRepository -ne "true") {
        throw "Destination exists but is not a bare Git mirror: $RepositoryPath"
    }

    $actualUpstreamUrlInspection = Invoke-GitInspection -GitArguments @(
        "-C", $RepositoryPath, "remote", "get-url", "origin"
    )
    $actualUpstreamUrlOutput = @($actualUpstreamUrlInspection.Output)
    $actualUpstreamUrlExitCode = $actualUpstreamUrlInspection.ExitCode
    $actualUpstreamUrl = if ($actualUpstreamUrlOutput.Count -gt 0) {
        ([string]$actualUpstreamUrlOutput[0]).Trim()
    } else {
        ""
    }
    if ($actualUpstreamUrlExitCode -ne 0) {
        throw "Cannot read origin URL from existing repository: $RepositoryPath"
    }
    if ($actualUpstreamUrl -ne $ExpectedUpstreamUrl) {
        throw "Existing repository origin differs at $RepositoryPath. Expected '$ExpectedUpstreamUrl', found '$actualUpstreamUrl'."
    }
}

function Get-VerifiedMirrorHead {
    param([Parameter(Mandatory)][string]$RepositoryPath)

    $headOutput = @(& git -C $RepositoryPath rev-parse --verify --quiet --short "HEAD^{commit}" 2>$null)
    $exitCode = $LASTEXITCODE
    $head = if ($headOutput.Count -gt 0) { [string]$headOutput[0] } else { "" }
    if ($exitCode -ne 0 -or [string]::IsNullOrWhiteSpace($head)) {
        throw "Destination exists but bare Git mirror has no verified HEAD commit: $RepositoryPath"
    }
    return $head.Trim()
}

function Start-PublicSourceMirrorJob {
    param(
        [Parameter(Mandatory)][string]$SourceId,
        [Parameter(Mandatory)][string]$UpstreamUrl,
        [Parameter(Mandatory)][string]$TargetPath,
        [Parameter(Mandatory)][string]$ProxyUrl
    )

    return Start-Job -Name "public-source-mirror-$SourceId" -ArgumentList `
        $SourceId, $UpstreamUrl, $TargetPath, $ProxyUrl -ScriptBlock {
            param($jobSourceId, $jobUpstreamUrl, $jobTargetPath, $jobProxyUrl)

            $gitOutput = @(
                & git -c "core.autocrlf=false" -c "http.proxy=$jobProxyUrl" `
                    -c "https.proxy=$jobProxyUrl" clone --mirror --depth 1 --no-tags `
                    $jobUpstreamUrl $jobTargetPath 2>&1 |
                    ForEach-Object { $_.ToString() }
            )
            $exitCode = $LASTEXITCODE
            [PSCustomObject]@{
                SourceId = $jobSourceId
                TargetPath = $jobTargetPath
                ExitCode = $exitCode
                GitOutput = $gitOutput
            }
        }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required to clone public source repositories."
}

if ([string]::IsNullOrWhiteSpace($CatalogPath)) {
    $CatalogPath = Join-Path $PSScriptRoot "..\..\config\public_sources\first_wave_repositories.json"
}

$resolvedCatalogPath = (Resolve-Path -LiteralPath $CatalogPath).Path
$repositories = @(Get-PublicRepositoryCatalog -Path $resolvedCatalogPath)
$requestedSourceIds = @(
    $SourceId | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)

if ($requestedSourceIds.Count -gt 0) {
    $requestedIds = [System.Collections.Generic.HashSet[string]]::new()
    foreach ($requestedSourceId in $requestedSourceIds) {
        [void]$requestedIds.Add($requestedSourceId)
    }
    $repositories = @($repositories | Where-Object { $requestedIds.Contains($_.source_id) })
    $missingIds = @($requestedIds | Where-Object { $_ -notin $repositories.source_id })
    if ($missingIds.Count -gt 0) {
        throw "Unknown source_id values: $($missingIds -join ', ')"
    }
}

$resolvedDestinationRoot = [System.IO.Path]::GetFullPath($DestinationRoot)
$pendingMirrors = [System.Collections.Generic.List[object]]::new()

for ($index = 0; $index -lt $repositories.Count; $index++) {
    $repository = $repositories[$index]
    $targetPath = Resolve-SafeMirrorTargetPath `
        -DestinationRoot $resolvedDestinationRoot `
        -RelativePath $repository.relative_path
    $targetParent = Split-Path -Parent $targetPath
    $position = "$($index + 1)/$($repositories.Count)"

    if (Test-Path -LiteralPath $targetPath) {
        Assert-ExistingMirrorMatchesCatalog -RepositoryPath $targetPath -ExpectedUpstreamUrl $repository.upstream_url
        $head = Get-VerifiedMirrorHead -RepositoryPath $targetPath
        Write-Host "[$position] $($repository.source_id) bare mirror already present at $targetPath ($head)"
        continue
    }

    if (-not $PSCmdlet.ShouldProcess($targetPath, "Mirror $($repository.source_id) from $($repository.upstream_url)")) {
        continue
    }

    New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
    $pendingMirrors.Add([PSCustomObject]@{
        Repository = $repository
        TargetPath = $targetPath
        Position = $position
    })
}

$nextMirrorIndex = 0
$runningJobs = @{}
$mirrorFailure = $false

while ($runningJobs.Count -gt 0 -or (-not $mirrorFailure -and $nextMirrorIndex -lt $pendingMirrors.Count)) {
    while (-not $mirrorFailure -and $runningJobs.Count -lt $MaxConcurrentMirrors `
            -and $nextMirrorIndex -lt $pendingMirrors.Count) {
        $pending = $pendingMirrors[$nextMirrorIndex]
        $nextMirrorIndex += 1
        $repository = $pending.Repository
        Write-Host "[$($pending.Position)] mirroring $($repository.source_id) into $($pending.TargetPath)"
        $job = Start-PublicSourceMirrorJob -SourceId $repository.source_id `
            -UpstreamUrl $repository.upstream_url -TargetPath $pending.TargetPath -ProxyUrl $ProxyUrl
        $runningJobs[$job.Id] = [PSCustomObject]@{
            Job = $job
            Pending = $pending
        }
    }

    $completed = @(
        $runningJobs.Values | Where-Object {
            $_.Job.State -in @("Completed", "Failed", "Stopped")
        }
    )
    if ($completed.Count -eq 0) {
        Start-Sleep -Seconds 1
        continue
    }

    foreach ($completedJob in $completed) {
        $job = $completedJob.Job
        $pending = $completedJob.Pending
        $repository = $pending.Repository
        [void]$runningJobs.Remove($job.Id)
        $result = @(Receive-Job -Job $job -ErrorAction SilentlyContinue)
        Remove-Job -Job $job -Force

        $jobResult = $result | Where-Object { $_.PSObject.Properties.Name -contains "ExitCode" } |
            Select-Object -Last 1
        if ($job.State -ne "Completed" -or $null -eq $jobResult -or $jobResult.ExitCode -ne 0) {
            $details = @($jobResult.GitOutput) -join [Environment]::NewLine
            if ([string]::IsNullOrWhiteSpace($details)) {
                $details = "Git child job ended with state '$($job.State)'."
            }
            Write-Error "Mirror failed for $($repository.source_id). Partial destination was retained for inspection: $($pending.TargetPath)`n$details"
            $mirrorFailure = $true
            continue
        }

        Assert-ExistingMirrorMatchesCatalog -RepositoryPath $pending.TargetPath `
            -ExpectedUpstreamUrl $repository.upstream_url
        $head = Get-VerifiedMirrorHead -RepositoryPath $pending.TargetPath
        Write-Host "[$($pending.Position)] mirrored $($repository.source_id) at $head"
    }
}

if ($mirrorFailure) {
    throw "Public source mirror run stopped after a failed source. No additional mirrors were started."
}

Write-Host "Public source mirror run completed for $($repositories.Count) repository entries with concurrency $MaxConcurrentMirrors."
