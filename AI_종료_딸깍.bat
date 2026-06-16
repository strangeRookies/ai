@echo off
echo ========================================================
echo Terminating AI Environment (GPU PC Processes and Local Tunnels)
echo ========================================================

echo.
echo [1/2] Terminating background processes on GPU PC...
ssh welabs@58.127.241.84 "pkill -f 'scripts/run_registered_cameras.py' 2>/dev/null || true; pkill -f 'scripts/start_simulated_rtsp_from_folder.py' 2>/dev/null || true; pkill -f 'scripts/serve_ai_overlay.py' 2>/dev/null || true; pkill -f 'rtsp://127.0.0.1:8554' 2>/dev/null || true; fuser -k 8010/tcp 2>/dev/null || true; fuser -k 8011/tcp 2>/dev/null || true; fuser -k 8012/tcp 2>/dev/null || true; fuser -k 8013/tcp 2>/dev/null || true; docker rm -f mediamtx 2>/dev/null || true; echo 'Remote processes terminated.'"

echo.
echo [2/2] Terminating local SSH tunnel processes...
taskkill /F /IM ssh.exe 2>nul
if %errorlevel% equ 0 (
    echo Local SSH tunnels terminated successfully.
) else (
    echo No active local SSH tunnel processes found.
)

echo.
echo ========================================================
echo AI System cleanly terminated.
echo ========================================================
pause
