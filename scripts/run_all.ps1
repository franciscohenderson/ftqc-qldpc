# End-to-end: code table -> Monte Carlo sweep -> figures. Edit the sweep arguments for longer runs.
param([string]$Out = "results/full", [int]$Workers = 0)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$env:PYTHONPATH = "src"
$py = ".\.venv\Scripts\python.exe"
$w = @(); if ($Workers -gt 0) { $w = @("--workers", $Workers) }
& $py -m algebra.report --out exports --time-limit 300
& $py -m benchmarks.run sweep --codes BB72 BB90 BB108 BB144 --ps 3e-4 6e-4 1e-3 2e-3 3e-3 5e-3 --out $Out @w
& $py -m benchmarks.run figures --out $Out
