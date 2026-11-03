# Boot the API + dashboard and open it in the browser.
#   scripts\launch_dashboard.ps1 [-Port 8000] [-NoBrowser] [-Foreground]
param([int]$Port = 8000, [switch]$NoBrowser, [switch]$Foreground)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "Run scripts\setup.ps1 first (venv missing)."; exit 1 }
$env:PYTHONPATH = "src"

# one-time assets: decoder DLL, kernel latency numbers, interleaved schedules, small INT8 GNNs
& $py -c "from decoder.fast import build; build()"
if (-not (Test-Path "results\kernel_latency.json")) { & $py scripts\bench_kernel.py | Out-Null }
if (-not (Test-Path "results\gnn\BB72_hz.npz") -or -not (Test-Path "results\schedules\BB288.json")) {
    Write-Host "Precomputing schedules and GNN models (first run only, a few minutes)..."
    & $py scripts\precompute_assets.py --gnn BB72
}

$url = "http://localhost:$Port"
$args_ = @("-m", "uvicorn", "api.server:app", "--app-dir", "src", "--host", "127.0.0.1", "--port", $Port, "--log-level", "warning")
if ($Foreground) {
    if (-not $NoBrowser) { Start-Job { Start-Sleep 3; Start-Process $using:url } | Out-Null }
    & $py @args_
} else {
    $proc = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $root -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "server.log" -RedirectStandardError "server.err"
    for ($i = 0; $i -lt 60; $i++) {
        try { if ((Invoke-RestMethod "$url/api/v1/health" -TimeoutSec 2).status -eq "ok") { break } } catch { Start-Sleep 1 }
    }
    Write-Host "Dashboard: $url   (API docs: $url/docs)   PID $($proc.Id)  -> stop with: Stop-Process -Id $($proc.Id)"
    if (-not $NoBrowser) { Start-Process $url }
}
