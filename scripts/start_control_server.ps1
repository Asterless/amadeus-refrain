# Start the host control service in the background (hidden window).
# Run:  powershell -ExecutionPolicy Bypass -File scripts/start_control_server.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'control_server.py'

if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "venv python not found: $python"
    exit 1
}

Start-Process -FilePath $python -ArgumentList $script -WorkingDirectory $root -WindowStyle Hidden
Write-Host 'Control server starting in the background (port from [control].port in config.toml).'
