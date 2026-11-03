# Build dist\outreach_ready\LAUNCHPAD.html and open it in Chrome.  It only PREPARES drafts; it never sends anything.
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
& (Join-Path $root ".venv\Scripts\python.exe") scripts\stage_outreach.py @args
