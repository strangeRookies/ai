@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

set "LOCAL_CONFIG=%~dp0AI_LOCAL_CONFIG.bat"
if not exist "%LOCAL_CONFIG%" (
  echo [ERROR] Missing AI_LOCAL_CONFIG.bat at %LOCAL_CONFIG%
  exit /b 1
)
call "%LOCAL_CONFIG%"

set "DRY_RUN=false"
if "%~1"=="--dry-run" (
  set "DRY_RUN=true"
)

if "%DRY_RUN%"=="true" (
  echo ======================================================================
  echo             [DRY-RUN] AI PROCESS CLEANUP PREVIEW
  echo ======================================================================
  echo GPU Server: %GPU_USER%@%GPU_HOST%
  echo.
  echo [1] Remote processes that WOULD be terminated:
  ssh %GPU_USER%@%GPU_HOST% "ps -eo pid,cmd | grep -E 'serve_ai_overlay.py|run_registered_cameras.py|start_simulated_rtsp_from_folder.py|ffmpeg' | grep -v grep || echo 'No active remote AI processes found.'"
  echo.
  echo [2] Remote MediaMTX docker container that WOULD be stopped/removed:
  ssh %GPU_USER%@%GPU_HOST% "docker ps --filter 'name=^mediamtx$' --format '{{.ID}} {{.Names}} {{.Status}}' || echo 'No active MediaMTX container.'"
  echo.
  echo [3] Local Windows SSH tunnels that WOULD be killed:
  tasklist /FI "IMAGENAME eq ssh.exe" 2>nul | findstr /i "ssh.exe"
  if errorlevel 1 (
    echo No active local ssh.exe tunnels found.
  )
  echo.
  echo [4] Camera worker registry file that WOULD be deleted:
  ssh %GPU_USER%@%GPU_HOST% "if [ -f %STABLE_ROOT%/runs/camera_worker_registry.json ]; then echo '%STABLE_ROOT%/runs/camera_worker_registry.json exists'; else echo 'No registry file found.'; fi"
  echo.
  echo ======================================================================
  echo Dry-run finished. No processes were modified.
  echo ======================================================================
  goto END
)

echo ======================================================================
echo             PERFORMING GRADUAL AI PROCESS CLEANUP (LOCAL & REMOTE)
echo ======================================================================
echo GPU Server: %GPU_USER%@%GPU_HOST%
echo ======================================================================
echo.

echo [1/5] Stopping remote AI workers gently (SIGTERM)...
ssh %GPU_USER%@%GPU_HOST% "pkill -15 -f 'serve_ai_overlay.py' 2>/dev/null; pkill -15 -f 'run_registered_cameras.py' 2>/dev/null; true"
sleep 2

echo [2/5] Stopping remote RTSP simulated publishers and FFmpeg gently (SIGTERM)...
ssh %GPU_USER%@%GPU_HOST% "pkill -15 -f 'start_simulated_rtsp_from_folder.py' 2>/dev/null; pkill -15 -f 'ffmpeg' 2>/dev/null; true"
sleep 2

echo [3/5] Force killing any remaining remote processes (SIGKILL)...
ssh %GPU_USER%@%GPU_HOST% "pkill -9 -f 'serve_ai_overlay.py' 2>/dev/null; pkill -9 -f 'run_registered_cameras.py' 2>/dev/null; pkill -9 -f 'start_simulated_rtsp_from_folder.py' 2>/dev/null; pkill -9 -f 'ffmpeg' 2>/dev/null; true"
ssh %GPU_USER%@%GPU_HOST% "docker rm -f mediamtx 2>/dev/null || true; rm -f %STABLE_ROOT%/runs/camera_worker_registry.json 2>/dev/null || true"

echo [4/5] Terminating local SSH tunnel processes gently...
taskkill /IM ssh.exe 2>nul
timeout /t 2 /nobreak >nul

echo [5/5] Force terminating remaining local SSH tunnel processes...
taskkill /F /IM ssh.exe 2>nul
echo.

echo ======================================================================
echo Cleanup complete. Checking process status...
echo ======================================================================
echo Local ssh tunnels remaining:
tasklist /FI "IMAGENAME eq ssh.exe" 2>nul | findstr /i "ssh.exe" || echo None.
echo.
echo Remote GPU processes remaining:
ssh %GPU_USER%@%GPU_HOST% "ps -eo pid,cmd | grep -E 'serve_ai_overlay.py|run_registered_cameras.py|start_simulated_rtsp_from_folder.py|ffmpeg' | grep -v grep || echo 'None.'"
echo.
echo ======================================================================

:END
endlocal
