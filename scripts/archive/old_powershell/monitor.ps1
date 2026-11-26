# Huey Worker 日志监控脚本
# 使用方法: .\scripts\monitor.ps1
# 或者: powershell -ExecutionPolicy Bypass -File scripts\monitor.ps1

param(
    [string]$LogFile = "E:\JETB\pythons\sf\retrival\huey_worker.log",
    [int]$Tail = 50
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Huey Worker 日志监控" -ForegroundColor Cyan
Write-Host "  日志文件: $LogFile" -ForegroundColor Gray
Write-Host "  按 Ctrl+C 退出" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 检查文件是否存在
if (-not (Test-Path $LogFile)) {
    Write-Host "[WARN] 日志文件不存在，等待创建..." -ForegroundColor Yellow
    while (-not (Test-Path $LogFile)) {
        Start-Sleep -Seconds 1
    }
    Write-Host "[INFO] 日志文件已创建，开始监控" -ForegroundColor Green
}

# 监控日志文件
Get-Content -Path $LogFile -Wait -Tail $Tail | ForEach-Object {
    $line = $_

    # 根据内容着色
    if ($line -match "ERROR|Exception|FAILED") {
        Write-Host $line -ForegroundColor Red
    }
    elseif ($line -match "WARNING|WARN") {
        Write-Host $line -ForegroundColor Yellow
    }
    elseif ($line -match "SUCCESS|executed|completed") {
        Write-Host $line -ForegroundColor Green
    }
    else {
        Write-Host $line
    }
}
