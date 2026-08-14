# Stop the host control service.
# Run:  powershell -ExecutionPolicy Bypass -File scripts/stop_control_server.ps1
$conn = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($conn) {
    Stop-Process -Id $conn.OwningProcess -Force
    Write-Host "Control server stopped (PID $($conn.OwningProcess))."
} else {
    Write-Host 'Control server is not running.'
}
