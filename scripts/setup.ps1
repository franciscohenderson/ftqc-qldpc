# Create the venv, install dependencies, build the C++20 decoder and run the test-suite.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
py -3.14 -m venv .venv 2>$null; if ($LASTEXITCODE -ne 0) { py -3.12 -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet stim numpy scipy networkx pandas pyarrow h5py matplotlib seaborn cffi "openqasm3[parser]" pytest ziglang fastapi uvicorn httpx websockets
& $py -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
$env:PYTHONPATH = "src"
& $py -c "from decoder.fast import build; print('built', build(force=True))"
& $py -m pytest -q
