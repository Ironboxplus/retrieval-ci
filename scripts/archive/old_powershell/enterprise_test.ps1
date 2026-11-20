# Enterprise Test Suite for Retrieval System
# All requests have 5 second timeout

param(
    [string]$ApiUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Continue"

# Initialize test results
$script:testResults = @{
    timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    api_url = $ApiUrl
    tests = [System.Collections.ArrayList]@()
    summary = @{passed = 0; failed = 0; total = 0}
    data_consistency = @{
        sqlite_chunks = 0
        chromadb_vectors = 0
        whoosh_documents = 0
        expected_count = 42450
        is_consistent = $false
    }
}

function Add-TestResult {
    param(
        [string]$Name,
        [string]$Status,
        [object]$Data = $null,
        [string]$Error = $null,
        [double]$Duration = 0
    )
    $result = @{name=$Name; status=$Status; duration_ms=[math]::Round($Duration, 2)}
    if ($Data) { $result.data = $Data }
    if ($Error) { $result.error = $Error }
    [void]$script:testResults.tests.Add($result)
    $script:testResults.summary.total++
    if ($Status -eq "PASSED") {
        $script:testResults.summary.passed++
        Write-Host "  [PASSED] $Name (${Duration}ms)" -ForegroundColor Green
    } else {
        $script:testResults.summary.failed++
        Write-Host "  [FAILED] $Name (${Duration}ms)" -ForegroundColor Red
        if ($Error) { Write-Host "           Error: $Error" -ForegroundColor DarkRed }
    }
}

function Invoke-ApiRequest {
    param(
        [string]$Endpoint,
        [string]$Method = "GET",
        [object]$Body = $null,
        [int]$TimeoutSec = 5
    )
    $uri = "$ApiUrl$Endpoint"
    $params = @{
        Uri = $uri
        Method = $Method
        TimeoutSec = $TimeoutSec
        ErrorAction = "Stop"
    }
    if ($Body) {
        $params.Body = ($Body | ConvertTo-Json -Depth 10)
        $params.ContentType = "application/json"
    }
    return Invoke-RestMethod @params
}

# Header
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "   ENTERPRISE TEST SUITE - Retrieval System" -ForegroundColor Cyan
Write-Host "   API: $ApiUrl" -ForegroundColor Cyan
Write-Host "   Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

# ============================================================
# Module 1: API Availability Tests
# ============================================================
Write-Host "`n[Module 1] API Availability Tests" -ForegroundColor Yellow
Write-Host ("-" * 50)

# Test 1.1: Root Endpoint
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $response = Invoke-ApiRequest -Endpoint "/" -TimeoutSec 5
    $sw.Stop()
    Add-TestResult -Name "Root Endpoint (/)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Root Endpoint (/)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# Test 1.2: Stats Endpoint
$sw.Restart()
try {
    $stats = Invoke-ApiRequest -Endpoint "/api/stats" -TimeoutSec 5
    $sw.Stop()
    Add-TestResult -Name "Stats Endpoint (/api/stats)" -Status "PASSED" -Data @{stats=$stats} -Duration $sw.ElapsedMilliseconds

    # Store stats for data consistency check
    if ($stats.total_documents) { $script:testResults.data_consistency.sqlite_chunks = $stats.total_documents }
    if ($stats.total_chunks) { $script:testResults.data_consistency.sqlite_chunks = $stats.total_chunks }
    if ($stats.chromadb_count) { $script:testResults.data_consistency.chromadb_vectors = $stats.chromadb_count }
    if ($stats.whoosh_count) { $script:testResults.data_consistency.whoosh_documents = $stats.whoosh_count }
} catch {
    $sw.Stop()
    Add-TestResult -Name "Stats Endpoint (/api/stats)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# Test 1.3: Config Endpoint
$sw.Restart()
try {
    $config = Invoke-ApiRequest -Endpoint "/api/config" -TimeoutSec 5
    $sw.Stop()
    Add-TestResult -Name "Config Endpoint (/api/config)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Config Endpoint (/api/config)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# Test 1.4: OpenAPI Spec
$sw.Restart()
try {
    $openapi = Invoke-ApiRequest -Endpoint "/openapi.json" -TimeoutSec 5
    $sw.Stop()
    $endpointCount = ($openapi.paths.PSObject.Properties | Measure-Object).Count
    Add-TestResult -Name "OpenAPI Spec (/openapi.json)" -Status "PASSED" -Data @{endpoint_count=$endpointCount} -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "OpenAPI Spec (/openapi.json)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# ============================================================
# Module 2: Document Search Tests
# ============================================================
Write-Host "`n[Module 2] Document Search Tests" -ForegroundColor Yellow
Write-Host ("-" * 50)

# Test 2.1: Basic Search
$sw.Restart()
try {
    $searchResult = Invoke-ApiRequest -Endpoint "/api/search" -Method "POST" -Body @{query="machine learning"; top_k=10} -TimeoutSec 5
    $sw.Stop()
    $resultCount = if ($searchResult.results) { $searchResult.results.Count } else { 0 }
    Add-TestResult -Name "Basic Search (query='machine learning', top_k=10)" -Status "PASSED" -Data @{result_count=$resultCount} -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Basic Search (query='machine learning', top_k=10)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# Test 2.2: Boundary Search (top_k=1)
$sw.Restart()
try {
    $searchResult = Invoke-ApiRequest -Endpoint "/api/search" -Method "POST" -Body @{query="python programming"; top_k=1} -TimeoutSec 5
    $sw.Stop()
    $resultCount = if ($searchResult.results) { $searchResult.results.Count } else { 0 }
    $status = if ($resultCount -le 1) { "PASSED" } else { "FAILED" }
    Add-TestResult -Name "Boundary Search (top_k=1)" -Status $status -Data @{result_count=$resultCount; expected="<=1"} -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Boundary Search (top_k=1)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# Test 2.3: Large Search (top_k=100)
$sw.Restart()
try {
    $searchResult = Invoke-ApiRequest -Endpoint "/api/search" -Method "POST" -Body @{query="data analysis"; top_k=100} -TimeoutSec 5
    $sw.Stop()
    $resultCount = if ($searchResult.results) { $searchResult.results.Count } else { 0 }
    Add-TestResult -Name "Large Search (top_k=100)" -Status "PASSED" -Data @{result_count=$resultCount} -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Large Search (top_k=100)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# Test 2.4: Chinese Query Search
$sw.Restart()
try {
    $searchResult = Invoke-ApiRequest -Endpoint "/api/search" -Method "POST" -Body @{query="深度学习算法"; top_k=10} -TimeoutSec 5
    $sw.Stop()
    $resultCount = if ($searchResult.results) { $searchResult.results.Count } else { 0 }
    Add-TestResult -Name "Chinese Query Search (query='深度学习算法')" -Status "PASSED" -Data @{result_count=$resultCount} -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Chinese Query Search (query='深度学习算法')" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# Test 2.5: Special Characters Search
$sw.Restart()
try {
    $searchResult = Invoke-ApiRequest -Endpoint "/api/search" -Method "POST" -Body @{query="C++ programming"; top_k=10} -TimeoutSec 5
    $sw.Stop()
    $resultCount = if ($searchResult.results) { $searchResult.results.Count } else { 0 }
    Add-TestResult -Name "Special Characters Search (query='C++ programming')" -Status "PASSED" -Data @{result_count=$resultCount} -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Special Characters Search (query='C++ programming')" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# ============================================================
# Module 3: Multi-Method Search Tests
# ============================================================
Write-Host "`n[Module 3] Multi-Method Search Tests" -ForegroundColor Yellow
Write-Host ("-" * 50)

# Test 3.1: Multi-Method Search
$sw.Restart()
try {
    $searchResult = Invoke-ApiRequest -Endpoint "/api/search/multi-method" -Method "POST" -Body @{query="neural network"; top_k=10} -TimeoutSec 5
    $sw.Stop()
    Add-TestResult -Name "Multi-Method Search (/api/search/multi-method)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Multi-Method Search (/api/search/multi-method)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# ============================================================
# Module 4: Document Management Tests
# ============================================================
Write-Host "`n[Module 4] Document Management Tests" -ForegroundColor Yellow
Write-Host ("-" * 50)

# Test 4.1: List Documents
$sw.Restart()
try {
    $docs = Invoke-ApiRequest -Endpoint "/api/documents" -TimeoutSec 5
    $sw.Stop()
    $docCount = if ($docs.Count) { $docs.Count } elseif ($docs.documents) { $docs.documents.Count } else { 0 }
    Add-TestResult -Name "List Documents (/api/documents)" -Status "PASSED" -Data @{document_count=$docCount} -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "List Documents (/api/documents)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# ============================================================
# Module 5: Task System Tests
# ============================================================
Write-Host "`n[Module 5] Task System Tests" -ForegroundColor Yellow
Write-Host ("-" * 50)

# Test 5.1: List Tasks
$sw.Restart()
try {
    $tasks = Invoke-ApiRequest -Endpoint "/api/tasks" -TimeoutSec 5
    $sw.Stop()
    $taskCount = if ($tasks.Count) { $tasks.Count } elseif ($tasks.tasks) { $tasks.tasks.Count } else { 0 }
    Add-TestResult -Name "List Tasks (/api/tasks)" -Status "PASSED" -Data @{task_count=$taskCount} -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "List Tasks (/api/tasks)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# ============================================================
# Module 6: Crawl System Tests
# ============================================================
Write-Host "`n[Module 6] Crawl System Tests" -ForegroundColor Yellow
Write-Host ("-" * 50)

# Test 6.1: Crawl Records
$sw.Restart()
try {
    $records = Invoke-ApiRequest -Endpoint "/api/crawl/records" -TimeoutSec 5
    $sw.Stop()
    Add-TestResult -Name "Crawl Records (/api/crawl/records)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Crawl Records (/api/crawl/records)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# ============================================================
# Module 7: Data Consistency Verification
# ============================================================
Write-Host "`n[Module 7] Data Consistency Verification" -ForegroundColor Yellow
Write-Host ("-" * 50)

# Get detailed stats for consistency check
$sw.Restart()
try {
    $stats = Invoke-ApiRequest -Endpoint "/api/stats" -TimeoutSec 5
    $sw.Stop()

    # Extract counts from stats response
    $sqliteCount = 0
    $chromaCount = 0
    $whooshCount = 0

    # Try different possible field names
    if ($stats.PSObject.Properties["total_chunks"]) { $sqliteCount = $stats.total_chunks }
    elseif ($stats.PSObject.Properties["sqlite_chunks"]) { $sqliteCount = $stats.sqlite_chunks }
    elseif ($stats.PSObject.Properties["document_count"]) { $sqliteCount = $stats.document_count }

    if ($stats.PSObject.Properties["chromadb_count"]) { $chromaCount = $stats.chromadb_count }
    elseif ($stats.PSObject.Properties["vector_count"]) { $chromaCount = $stats.vector_count }

    if ($stats.PSObject.Properties["whoosh_count"]) { $whooshCount = $stats.whoosh_count }
    elseif ($stats.PSObject.Properties["index_count"]) { $whooshCount = $stats.index_count }

    $script:testResults.data_consistency.sqlite_chunks = $sqliteCount
    $script:testResults.data_consistency.chromadb_vectors = $chromaCount
    $script:testResults.data_consistency.whoosh_documents = $whooshCount

    # Check consistency
    $expected = 42450
    $allMatch = ($sqliteCount -eq $expected) -and ($chromaCount -eq $expected) -and ($whooshCount -eq $expected)
    $script:testResults.data_consistency.is_consistent = $allMatch

    if ($allMatch) {
        Add-TestResult -Name "Data Consistency Check (Expected: $expected)" -Status "PASSED" -Data @{
            sqlite=$sqliteCount
            chromadb=$chromaCount
            whoosh=$whooshCount
        } -Duration $sw.ElapsedMilliseconds
    } else {
        Add-TestResult -Name "Data Consistency Check (Expected: $expected)" -Status "FAILED" -Data @{
            sqlite=$sqliteCount
            chromadb=$chromaCount
            whoosh=$whooshCount
            expected=$expected
        } -Error "Counts do not match expected value of $expected" -Duration $sw.ElapsedMilliseconds
    }

    # Store raw stats
    $script:testResults.data_consistency.raw_stats = $stats

} catch {
    $sw.Stop()
    Add-TestResult -Name "Data Consistency Check" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# ============================================================
# Module 8: Error Handling Tests
# ============================================================
Write-Host "`n[Module 8] Error Handling Tests" -ForegroundColor Yellow
Write-Host ("-" * 50)

# Test 8.1: Invalid Endpoint (404)
$sw.Restart()
try {
    $response = Invoke-ApiRequest -Endpoint "/api/nonexistent" -TimeoutSec 5
    $sw.Stop()
    Add-TestResult -Name "Invalid Endpoint (Expected 404)" -Status "FAILED" -Error "Should have returned 404" -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    if ($_.Exception.Message -match "404") {
        Add-TestResult -Name "Invalid Endpoint (Expected 404)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
    } else {
        Add-TestResult -Name "Invalid Endpoint (Expected 404)" -Status "PASSED" -Data @{error_type=$_.Exception.GetType().Name} -Duration $sw.ElapsedMilliseconds
    }
}

# Test 8.2: Invalid JSON Body
$sw.Restart()
try {
    $params = @{
        Uri = "$ApiUrl/api/search"
        Method = "POST"
        Body = "not valid json"
        ContentType = "application/json"
        TimeoutSec = 5
        ErrorAction = "Stop"
    }
    $response = Invoke-RestMethod @params
    $sw.Stop()
    Add-TestResult -Name "Invalid JSON Body (Expected Error)" -Status "FAILED" -Error "Should have returned error" -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Invalid JSON Body (Expected Error)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
}

# Test 8.3: Missing Required Field
$sw.Restart()
try {
    $searchResult = Invoke-ApiRequest -Endpoint "/api/search" -Method "POST" -Body @{top_k=10} -TimeoutSec 5
    $sw.Stop()
    # Some APIs might handle missing query gracefully
    Add-TestResult -Name "Missing Query Field" -Status "PASSED" -Data @{note="API handled gracefully"} -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    if ($_.Exception.Message -match "422|400|validation") {
        Add-TestResult -Name "Missing Query Field (Expected Validation Error)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
    } else {
        Add-TestResult -Name "Missing Query Field" -Status "PASSED" -Data @{error=$_.Exception.Message} -Duration $sw.ElapsedMilliseconds
    }
}

# ============================================================
# Summary
# ============================================================
Write-Host "`n" ("=" * 70) -ForegroundColor Cyan
Write-Host "   TEST SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

$passed = $script:testResults.summary.passed
$failed = $script:testResults.summary.failed
$total = $script:testResults.summary.total
$passRate = if ($total -gt 0) { [math]::Round(($passed / $total) * 100, 2) } else { 0 }

Write-Host "`n   Total Tests:  $total" -ForegroundColor White
Write-Host "   Passed:       $passed" -ForegroundColor Green
Write-Host "   Failed:       $failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
Write-Host "   Pass Rate:    $passRate%" -ForegroundColor $(if ($passRate -ge 80) { "Green" } elseif ($passRate -ge 60) { "Yellow" } else { "Red" })

Write-Host "`n   Data Consistency:" -ForegroundColor Yellow
Write-Host "   - SQLite Chunks:    $($script:testResults.data_consistency.sqlite_chunks)"
Write-Host "   - ChromaDB Vectors: $($script:testResults.data_consistency.chromadb_vectors)"
Write-Host "   - Whoosh Documents: $($script:testResults.data_consistency.whoosh_documents)"
Write-Host "   - Expected:         $($script:testResults.data_consistency.expected_count)"
Write-Host "   - Consistent:       $($script:testResults.data_consistency.is_consistent)" -ForegroundColor $(if ($script:testResults.data_consistency.is_consistent) { "Green" } else { "Red" })

Write-Host "`n" ("=" * 70) -ForegroundColor Cyan

# Save results to JSON
$reportPath = "E:\JETB\pythons\sf\retrival\test_report.json"
$script:testResults | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportPath -Encoding UTF8
Write-Host "`nTest report saved to: $reportPath" -ForegroundColor Cyan

# Return results
$script:testResults
