@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

set "LOCAL_CONFIG=%~dp0AI_LOCAL_CONFIG.bat"
if not exist "%LOCAL_CONFIG%" (
  echo [ERROR] Missing AI_LOCAL_CONFIG.bat at %LOCAL_CONFIG%
  exit /b 1
)
call "%LOCAL_CONFIG%"

echo ======================================================================
echo             AI PROCESS ^& PORT STATUS DIAGNOSIS (LOCAL ^& REMOTE)
echo ======================================================================
echo GPU Server: %GPU_USER%@%GPU_HOST%
echo Remote Root: %STABLE_ROOT%
echo ======================================================================
echo.

echo ----------------------------------------------------------------------
echo [1] LOCAL WINDOWS PROCESSES & PORTS CHECK
echo ----------------------------------------------------------------------
echo checking active ports: 8080(Backend), 5173(Frontend), 8888(HLS), 8889(WebRTC)...
netstat -ano | findstr /R "LISTENING.*:8080 LISTENING.*:5173 LISTENING.*:8888 LISTENING.*:8889 LISTENING.*:18080"
echo.
echo checking running local processes (ssh, java, node)...
tasklist | findstr /i "ssh.exe java.exe node.exe git.exe"
echo.

echo ----------------------------------------------------------------------
echo [2] REMOTE GPU SERVER PROCESSES CHECK (via SSH)
echo ----------------------------------------------------------------------
echo Checking Python, FFmpeg, and MediaMTX processes...
ssh %GPU_USER%@%GPU_HOST% "ps -eo pid,ppid,cmd | grep -E 'python|ffmpeg|run_registered|serve_ai|demo_streamer|start_simulated|mediamtx' | grep -v grep"
if errorlevel 1 (
  echo [WARNING] Failed to query remote processes. SSH session may have timed out or failed.
)
echo.

echo ----------------------------------------------------------------------
echo [3] REMOTE PORT LISTENERS CHECK (via SSH)
echo ----------------------------------------------------------------------
echo Checking remote port states...
ssh %GPU_USER%@%GPU_HOST% "ss -tlnp | grep -E '8554|8888|8889|8189|18080' || true"
echo.

echo ----------------------------------------------------------------------
echo [4] ACTIVE CAMERAS WORKER REGISTRY (`runs/camera_worker_registry.json`)
echo ----------------------------------------------------------------------
ssh %GPU_USER%@%GPU_HOST% "if [ -f %STABLE_ROOT%/runs/camera_worker_registry.json ]; then cat %STABLE_ROOT%/runs/camera_worker_registry.json; else echo 'No camera registry found (no active workers).'; fi"
echo.

echo ----------------------------------------------------------------------
echo [5] GPU MEMORY ^& COMPUTE STATUS (nvidia-smi)
echo ----------------------------------------------------------------------
ssh %GPU_USER%@%GPU_HOST% "nvidia-smi || echo '[WARNING] nvidia-smi command not available on GPU host'"
echo.

echo ======================================================================
echo Status check complete. Use `cleanup_ai_processes.bat` to stop runtime.
echo ======================================================================
endlocal
