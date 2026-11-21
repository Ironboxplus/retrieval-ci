$procs = Get-CimInstance Win32_Process -Filter 'Name="python.exe"' | Where-Object { $_.CommandLine -match 'uvicorn|multiprocessing-fork' }
foreach ($p in $procs) {
    Write-Host "Stopping PID $($p.ProcessId)"
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3
$listen = (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "port_8000_listening=$listen"
