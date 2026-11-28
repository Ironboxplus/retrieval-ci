# Enterprise Test Suite for Retrieval System V2
# All requests have 5 second timeout
# Corrected endpoints and field names

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
        vector_index_size = 0
        bm25_index_size = 0
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
Write-Host "   ENTERPRISE TEST SUITE V2 - Retrieval System" -ForegroundColor Cyan
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
    Add-TestResult -Name "Stats Endpoint (/api/stats)" -Status "PASSED" -Data @{
        total_documents = $stats.total_documents
        total_chunks = $stats.total_chunks
        vector_index_size = $stats.vector_index_size
        bm25_index_size = $stats.bm25_index_size
    } -Duration $sw.ElapsedMilliseconds

    # Store for consistency check
    $script:testResults.data_consistency.sqlite_chunks = $stats.total_chunks
    $script:testResults.data_consistency.vector_index_size = $stats.vector_index_size
    $script:testResults.data_consistency.bm25_index_size = $stats.bm25_index_size
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
# Module 2: Document Search Tests (POST /api/search)
# ============================================================
Write-Host "`n[Module 2] Document Search Tests (POST)" -ForegroundColor Yellow
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
    Add-TestResult -Name "Chinese Query Search" -Status "PASSED" -Data @{result_count=$resultCount} -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Chinese Query Search" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# Test 2.5: Special Characters Search
$sw.Restart()
try {
    $searchResult = Invoke-ApiRequest -Endpoint "/api/search" -Method "POST" -Body @{query="C++ programming"; top_k=10} -TimeoutSec 5
    $sw.Stop()
    $resultCount = if ($searchResult.results) { $searchResult.results.Count } else { 0 }
    Add-TestResult -Name "Special Characters Search (C++)" -Status "PASSED" -Data @{result_count=$resultCount} -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Special Characters Search (C++)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# ============================================================
# Module 3: Multi-Method Search Tests (GET with query params)
# ============================================================
Write-Host "`n[Module 3] Multi-Method Search Tests (GET)" -ForegroundColor Yellow
Write-Host ("-" * 50)

# Test 3.1: Basic Multi-Method Search
$sw.Restart()
try {
    $encodedQuery = [System.Web.HttpUtility]::UrlEncode("neural network")
    $searchResult = Invoke-RestMethod -Uri "$ApiUrl/api/search/multi-method?query=$encodedQuery&top_k=10" -Method GET -TimeoutSec 5
    $sw.Stop()
    Add-TestResult -Name "Multi-Method Search (basic)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Multi-Method Search (basic)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# Test 3.2: Multi-Method with source_type filter (webpage)
$sw.Restart()
try {
    $encodedQuery = [System.Web.HttpUtility]::UrlEncode("python tutorial")
    $searchResult = Invoke-RestMethod -Uri "$ApiUrl/api/search/multi-method?query=$encodedQuery&top_k=10&source_type=webpage" -Method GET -TimeoutSec 5
    $sw.Stop()
    Add-TestResult -Name "Multi-Method Search (source_type=webpage)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Multi-Method Search (source_type=webpage)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# Test 3.3: Multi-Method with source_type filter (paper)
$sw.Restart()
try {
    $encodedQuery = [System.Web.HttpUtility]::UrlEncode("research methodology")
    $searchResult = Invoke-RestMethod -Uri "$ApiUrl/api/search/multi-method?query=$encodedQuery&top_k=10&source_type=paper" -Method GET -TimeoutSec 5
    $sw.Stop()
    Add-TestResult -Name "Multi-Method Search (source_type=paper)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Multi-Method Search (source_type=paper)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
}

# Test 3.4: Multi-Method with data_source filter
$sw.Restart()
try {
    $encodedQuery = [System.Web.HttpUtility]::UrlEncode("algorithm design")
    $searchResult = Invoke-RestMethod -Uri "$ApiUrl/api/search/multi-method?query=$encodedQuery&top_k=10&data_source=system" -Method GET -TimeoutSec 5
    $sw.Stop()
    Add-TestResult -Name "Multi-Method Search (data_source=system)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Multi-Method Search (data_source=system)" -Status "FAILED" -Error $_.Exception.Message -Duration $sw.ElapsedMilliseconds
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
    $docCount = if ($docs -is [array]) { $docs.Count } elseif ($docs.documents) { $docs.documents.Count } elseif ($docs.total) { $docs.total } else { 0 }
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
    $taskCount = if ($tasks -is [array]) { $tasks.Count } elseif ($tasks.tasks) { $tasks.tasks.Count } else { 0 }
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

$sw.Restart()
try {
    $stats = Invoke-ApiRequest -Endpoint "/api/stats" -TimeoutSec 5
    $sw.Stop()

    $sqliteChunks = $stats.total_chunks
    $vectorSize = $stats.vector_index_size
    $bm25Size = $stats.bm25_index_size
    $expected = 42450

    $script:testResults.data_consistency.sqlite_chunks = $sqliteChunks
    $script:testResults.data_consistency.vector_index_size = $vectorSize
    $script:testResults.data_consistency.bm25_index_size = $bm25Size

    # Check if all three match expected
    $allMatch = ($sqliteChunks -eq $expected) -and ($vectorSize -eq $expected) -and ($bm25Size -eq $expected)
    $script:testResults.data_consistency.is_consistent = $allMatch

    if ($allMatch) {
        Add-TestResult -Name "Data Consistency (All = $expected)" -Status "PASSED" -Data @{
            sqlite_chunks=$sqliteChunks
            vector_index=$vectorSize
            bm25_index=$bm25Size
        } -Duration $sw.ElapsedMilliseconds
    } else {
        Add-TestResult -Name "Data Consistency (Expected: $expected)" -Status "FAILED" -Data @{
            sqlite_chunks=$sqliteChunks
            vector_index=$vectorSize
            bm25_index=$bm25Size
            expected=$expected
        } -Error "Some counts do not match $expected" -Duration $sw.ElapsedMilliseconds
    }
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
    Add-TestResult -Name "Invalid Endpoint (Expected 404)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
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
    Add-TestResult -Name "Missing Query Field" -Status "FAILED" -Error "Should have returned validation error" -Duration $sw.ElapsedMilliseconds
} catch {
    $sw.Stop()
    Add-TestResult -Name "Missing Query Field (Expected 422)" -Status "PASSED" -Duration $sw.ElapsedMilliseconds
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
Write-Host "   Pass Rate:    $passRate%" -ForegroundColor $(if ($passRate -ge 90) { "Green" } elseif ($passRate -ge 70) { "Yellow" } else { "Red" })

Write-Host "`n   Data Consistency:" -ForegroundColor Yellow
Write-Host "   - SQLite Chunks:     $($script:testResults.data_consistency.sqlite_chunks)"
Write-Host "   - Vector Index Size: $($script:testResults.data_consistency.vector_index_size)"
Write-Host "   - BM25 Index Size:   $($script:testResults.data_consistency.bm25_index_size)"
Write-Host "   - Expected:          $($script:testResults.data_consistency.expected_count)"
$consistentColor = if ($script:testResults.data_consistency.is_consistent) { "Green" } else { "Red" }
Write-Host "   - All Consistent:    $($script:testResults.data_consistency.is_consistent)" -ForegroundColor $consistentColor

Write-Host "`n" ("=" * 70) -ForegroundColor Cyan

# Save results to JSON
$reportPath = "E:\JETB\pythons\sf\retrival\test_report.json"
$script:testResults | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportPath -Encoding UTF8
Write-Host "`nTest report saved to: $reportPath" -ForegroundColor Cyan

# Return results object
$script:testResults
