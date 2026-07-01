@echo off
setlocal
chcp 65001 >nul

set "LOCAL_CONFIG=%~dp0AI_LOCAL_CONFIG.bat"
if not exist "%LOCAL_CONFIG%" goto CONFIG_ERROR
call "%LOCAL_CONFIG%"
goto CONFIG_OK

:CONFIG_ERROR
echo [ERROR] Missing AI_LOCAL_CONFIG.bat
echo Copy AI_LOCAL_CONFIG.example.bat to AI_LOCAL_CONFIG.bat and set local values.
pause
exit /b 1

:CONFIG_OK
if not defined GPU_HOST exit /b 1
if not defined GPU_USER exit /b 1
if not defined STABLE_ROOT exit /b 1

set "REMOTE_ROOT=%STABLE_ROOT%"
set "MQTT_HOST=15.165.248.37"
set "MQTT_PORT=1883"

echo ========================================================
echo Starting STABLE AI environment from GPU develop branch
echo ========================================================
echo GPU: %GPU_USER%@%GPU_HOST%
echo Remote repo: %REMOTE_ROOT%
echo MQTT: %MQTT_HOST%:%MQTT_PORT% topic=safety/events
echo ========================================================
echo.
echo ========================================================
echo Select running mode:
echo   [1] Full restart and run AI processes (First run)
echo   [2] SSH Tunnels only (For sharing running streams)
echo   [3] Stop remote AI processes only
echo ========================================================
set /p "MODE=Choose (1, 2, or 3): "

if "%MODE%"=="3" (
  goto MODE_STOP
) else if "%MODE%"=="2" (
  goto MODE_TUNNEL
) else (
  goto MODE_FULL
)

:MODE_TUNNEL
echo.
echo === [Tunnel Only Mode] ===
echo [1/5] Skipping repo sync.
echo [2/5] Keeping existing AI processes active.
set "RUN_MODE=tunnel_only"
goto DO_TUNNEL

:MODE_FULL
echo.
echo === [Full Restart Mode] ===
echo [1/5] Syncing GPU stable repo to origin/develop...
ssh %GPU_USER%@%GPU_HOST% "cd %REMOTE_ROOT% && git stash push -u -m auto-stash-before-ai-stable-run-$(date +%%Y%%m%%d-%%H%%M%%S) || true && git fetch origin && git checkout develop && git pull --ff-only origin develop"
if errorlevel 1 (
  echo [ERROR] Failed to sync GPU stable repo.
  pause
  exit /b 1
)

echo.
echo [2/5] Stopping previous AI runtime processes...
ssh %GPU_USER%@%GPU_HOST% "pkill -f '%REMOTE_ROOT%/scripts/run_registered_cameras.py' 2>/dev/null || true; pkill -f '%REMOTE_ROOT%/scripts/start_simulated_rtsp_from_folder.py' 2>/dev/null || true; pkill -f '%REMOTE_ROOT%/scripts/serve_ai_overlay.py' 2>/dev/null || true; pkill -f 'ffmpeg' 2>/dev/null || true; rm -f %REMOTE_ROOT%/runs/camera_worker_registry.json 2>/dev/null || true; docker rm -f mediamtx 2>/dev/null || true"
if errorlevel 1 (
  echo [ERROR] Failed while stopping old GPU runtime.
  pause
  exit /b 1
)
set "RUN_MODE=full_run"

:DO_TUNNEL
echo.
echo [3/5] Starting SSH tunnel in a new window...
if "%RUN_MODE%"=="tunnel_only" (
  start "AI STABLE SSH Tunnel - keep open" cmd /k ssh -o ExitOnForwardFailure=yes -N -L 8888:127.0.0.1:8888 -L 8889:127.0.0.1:8889 -L 8189:127.0.0.1:8189 %GPU_USER%@%GPU_HOST%
) else (
  start "AI STABLE SSH Tunnel - keep open" cmd /k ssh -o ExitOnForwardFailure=yes -N -L 8888:127.0.0.1:8888 -L 8889:127.0.0.1:8889 -L 8189:127.0.0.1:8189 -R 18080:127.0.0.1:8080 %GPU_USER%@%GPU_HOST%
)

echo Enter the SSH password in the tunnel window and keep it open.
pause

echo.
echo [4/5] Checking GPU access to Windows backend through reverse tunnel...
if "%RUN_MODE%"=="tunnel_only" (
  echo Skipping backend check (Tunnel Only mode).
) else (
  ssh %GPU_USER%@%GPU_HOST% "curl -fsS http://127.0.0.1:18080/api/cameras/active >/dev/null"
  if errorlevel 1 (
    echo [ERROR] GPU PC cannot reach the Windows backend through port 18080.
    echo Make sure the tunnel window is open and the backend is running on localhost:8080.
    pause
    exit /b 1
  )
)

if "%RUN_MODE%"=="tunnel_only" (
  echo.
  echo [5/5] Skipping AI runtime startup.
  goto FINISH
)

echo.
echo [5/5] Starting MediaMTX, RTSP publisher, then AI runner on GPU stable repo...
ssh %GPU_USER%@%GPU_HOST% "cd %REMOTE_ROOT% && ( docker ps --filter 'name=^mediamtx$' --format '{{.Names}}' | grep -q '^mediamtx$' || nohup bash scripts/run_rtsp_server.sh > rtsp_server.log 2>&1 </dev/null & ) && sleep 3 && source .venv/bin/activate && ( nohup python scripts/start_simulated_rtsp_from_folder.py --video-dir /home/%GPU_USER%/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos --backend-url http://127.0.0.1:18080 --rtsp-host 127.0.0.1 --rtsp-port 8554 --poll-interval 30 --ffmpeg-mode nvenc > publisher.log 2>&1 </dev/null & ) && sleep 8 && ( nohup python scripts/run_registered_cameras.py --backend-base-url http://127.0.0.1:18080 --rtsp-base-url rtsp://127.0.0.1:8554 --video-pool /home/%GPU_USER%/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos --overlay-report-enabled --detector-mode real --yolo-model yolo26n-pose.pt --publisher mqtt --mqtt-host %MQTT_HOST% --mqtt-port %MQTT_PORT% --mqtt-topic safety/events --skip-simulated-ffmpeg > ai_runner.log 2>&1 </dev/null & )"
if errorlevel 1 (
  echo [ERROR] Failed to start GPU stable runtime.
  pause
  exit /b 1
)

:MODE_STOP
echo.
echo Stopping remote AI runtime processes...
ssh %GPU_USER%@%GPU_HOST% "pkill -f '%REMOTE_ROOT%/scripts/run_registered_cameras.py' 2>/dev/null || true; pkill -f '%REMOTE_ROOT%/scripts/start_simulated_rtsp_from_folder.py' 2>/dev/null || true; pkill -f '%REMOTE_ROOT%/scripts/serve_ai_overlay.py' 2>/dev/null || true; pkill -f 'ffmpeg' 2>/dev/null || true; rm -f %REMOTE_ROOT%/runs/camera_worker_registry.json 2>/dev/null || true; docker rm -f mediamtx 2>/dev/null || true"
echo Remote AI processes stopped.
goto FINISH

:FINISH
echo.
if "%RUN_MODE%"=="tunnel_only" (
  echo SSH Tunnel established. Keep the tunnel window open.
) else (
  echo STABLE runtime started. Keep the tunnel window open.
)
echo.
echo Quick checks after startup:
echo   GPU: ss -lntup ^| grep -E "8554^|8888^|8889^|8189"
echo   GPU: docker ps ^| grep mediamtx
echo   Local: http://localhost:8888/cam_04/index.m3u8
echo.
echo Press any key to close this launcher window.
pause >nul
endlocal
