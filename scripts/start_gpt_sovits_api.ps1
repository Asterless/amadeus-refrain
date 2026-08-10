# Start the GPT-SoVITS API server for the trained "shigeju" model.
# Run:  powershell -ExecutionPolicy Bypass -File scripts/start_gpt_sovits_api.ps1
$root = 'D:\GPT-SoVITS\GPT-SoVITS-v4-20250529\GPT-SoVITS-v4-20250529'
$python = Join-Path $root 'runtime\python.exe'
$outLog = Join-Path $root 'tts_api.stdout.log'
$errLog = Join-Path $root 'tts_api.stderr.log'

Start-Process -FilePath $python `
    -ArgumentList @('api_v2.py', '-a', '0.0.0.0', '-p', '9880', '-c', 'GPT_SoVITS/configs/tts_infer_shigeju.yaml') `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog

Write-Host 'GPT-SoVITS API starting on http://0.0.0.0:9880 (logs: tts_api.stdout.log / tts_api.stderr.log)'
