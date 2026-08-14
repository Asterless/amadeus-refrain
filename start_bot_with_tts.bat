@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem One-click launcher for GPT-SoVITS API + Docker QQ bot.
rem Keep this file next to docker-compose.yml.

set "BOT_DIR=%~dp0"
set "GSV_DIR=D:\GPT-SoVITS\GPT-SoVITS-v4-20250529\GPT-SoVITS-v4-20250529"
set "GSV_CONFIG=%GSV_DIR%\GPT_SoVITS\configs\tts_infer_shigeju.yaml"
set "GSV_BIND=0.0.0.0"
set "GSV_PORT=9880"
set "GSV_URL=http://127.0.0.1:%GSV_PORT%"
set "BOT_TTS_URL=http://host.docker.internal:%GSV_PORT%"
set "REF_AUDIO=%GSV_DIR%\output\slicer_opt\simple.mp4_0000144960_0000273600.wav"
set "REF_TEXT_B64=6aaW5YWI77yM5L2g6KaB5YeG5aSH5LiA5Lu95bmy5YeA5pegQkdN55qE57qv5Lq65aOw54mH5q6144CC"

title Amadeus Bot Launcher
cd /d "%BOT_DIR%" || goto :fail

echo [1/6] Checking bot files...
if not exist "docker-compose.yml" (
  echo [ERROR] docker-compose.yml not found in "%BOT_DIR%".
  goto :fail
)

if not exist ".env" (
  if exist ".env.example" (
    echo [WARN] .env not found. Copying .env.example to .env.
    copy ".env.example" ".env" >nul
  ) else (
    echo [ERROR] .env not found and .env.example is missing.
    goto :fail
  )
)

if not exist "config.toml" (
  if exist "config.example.toml" (
    echo [WARN] config.toml not found. Copying config.example.toml to config.toml.
    copy "config.example.toml" "config.toml" >nul
  ) else (
    echo [ERROR] config.toml not found and config.example.toml is missing.
    goto :fail
  )
)

echo [INFO] Updating config.toml TTS settings...
powershell -NoProfile -ExecutionPolicy Bypass -File "%BOT_DIR%scripts\update_tts_config.ps1" -ConfigPath "config.toml" -BaseUrl "%BOT_TTS_URL%" -RefAudioPath "%REF_AUDIO%" -PromptTextBase64 "%REF_TEXT_B64%"
if errorlevel 1 (
  echo [ERROR] Failed to update config.toml TTS settings.
  goto :fail
)

echo [2/6] Checking GPT-SoVITS files...
if not exist "%GSV_DIR%\runtime\python.exe" (
  echo [ERROR] GPT-SoVITS runtime python not found:
  echo         "%GSV_DIR%\runtime\python.exe"
  goto :fail
)

if not exist "%GSV_DIR%\api_v2.py" (
  echo [ERROR] GPT-SoVITS api_v2.py not found:
  echo         "%GSV_DIR%\api_v2.py"
  goto :fail
)

if not exist "%GSV_CONFIG%" (
  echo [WARN] "%GSV_CONFIG%" not found. Falling back to default tts_infer.yaml.
  set "GSV_CONFIG=%GSV_DIR%\GPT_SoVITS\configs\tts_infer.yaml"
)

if not exist "%GSV_CONFIG%" (
  echo [ERROR] GPT-SoVITS tts config not found.
  goto :fail
)

findstr /C:"GPT_weights_v2/shigeju-e10.ckpt" "%GSV_CONFIG%" >nul
if errorlevel 1 (
  echo [WARN] Config does not point to GPT_weights_v2/shigeju-e10.ckpt.
)

findstr /C:"gsv-v2final-pretrained/s2G2333k.pth" "%GSV_CONFIG%" >nul
if not errorlevel 1 (
  echo [WARN] Config still uses the official pretrained SoVITS weight.
  echo        If you trained SoVITS, update vits_weights_path in:
  echo        "%GSV_CONFIG%"
)

echo [3/6] Checking Docker Desktop...
docker version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Docker is not reachable. Start Docker Desktop first, then run this bat again.
  goto :fail
)

echo [4/6] Starting GPT-SoVITS API at %GSV_URL% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url='%GSV_URL%/docs'; try { $r=Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>nul
if errorlevel 1 (
  start "GPT-SoVITS API :%GSV_PORT%" /D "%GSV_DIR%" cmd /k "".\runtime\python.exe" -I ".\api_v2.py" -a %GSV_BIND% -p %GSV_PORT% -c "%GSV_CONFIG%""
) else (
  echo [INFO] GPT-SoVITS API is already running.
)

echo [5/6] Waiting for GPT-SoVITS API...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ok=$false; for($i=0;$i -lt 60;$i++){ try { Invoke-WebRequest -UseBasicParsing -Uri '%GSV_URL%/docs' -TimeoutSec 2 | Out-Null; $ok=$true; break } catch { Start-Sleep -Seconds 2 } }; if($ok){ exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo [ERROR] GPT-SoVITS API did not become ready within 120 seconds.
  echo         Check the GPT-SoVITS API window for the real error.
  goto :fail
)

echo [6/6] Starting Docker bot...
docker compose up -d --build
if errorlevel 1 (
  echo [ERROR] docker compose up failed.
  goto :fail
)

echo.
echo [OK] Started.
echo      GPT-SoVITS API: %GSV_URL%
echo      NapCat WebUI:   http://127.0.0.1:6099
echo      Bot container:  qq-bot
echo.
echo Useful commands:
echo   docker compose logs -f bot
echo   docker compose logs -f napcat
echo.
pause
exit /b 0

:fail
echo.
echo Startup failed. Fix the message above and run this bat again.
pause
exit /b 1
