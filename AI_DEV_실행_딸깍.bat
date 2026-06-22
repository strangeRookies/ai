@echo off
setlocal

set "LOCAL_CONFIG=%~dp0AI_DEV_LOCAL_CONFIG.bat"
if not exist "%LOCAL_CONFIG%" (
  echo [ERROR] Missing AI_DEV_LOCAL_CONFIG.bat
  echo Copy AI_DEV_LOCAL_CONFIG.example.bat to AI_DEV_LOCAL_CONFIG.bat and set local values.
  pause
  exit /b 1
)
call "%LOCAL_CONFIG%"

if not defined GPU_HOST exit /b 1
if not defined GPU_USER exit /b 1
if not defined MQTT_HOST exit /b 1
if not defined MQTT_PORT exit /b 1
if not defined STABLE_ROOT exit /b 1
if not defined DEV_BASE exit /b 1

set "DEV_ROOT=%DEV_BASE%/current"

echo ========================================================
echo Starting isolated AI DEV environment
echo ========================================================

echo.
echo [1/4] Uploading local source code to GPU dev releases...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy_to_gpu_dev.ps1" -RemoteHost "%GPU_HOST%" -RemoteUser "%GPU_USER%" -StablePath "%STABLE_ROOT%" -DevBasePath "%DEV_BASE%"
if errorlevel 1 exit /b 1

echo.
echo [2/4] Stopping previous AI runtime processes...
ssh %GPU_USER%@%GPU_HOST% "pkill -f '[s]cripts/run_registered_cameras.py' 2>/dev/null || true; pkill -f '[s]cripts/start_simulated_rtsp_from_folder.py' 2>/dev/null || true; pkill -f '[s]cripts/serve_ai_overlay.py' 2>/dev/null || true; pkill -f '[r]tsp://127.0.0.1:8554' 2>/dev/null || true; fuser -k 8010/tcp 2>/dev/null || true; fuser -k 8011/tcp 2>/dev/null || true; fuser -k 8012/tcp 2>/dev/null || true; fuser -k 8013/tcp 2>/dev/null || true; docker rm -f mediamtx 2>/dev/null || true"

echo.
echo [3/4] Starting SSH tunnel in a new window...
start "AI DEV SSH Tunnel - keep open" cmd /k ssh -o ExitOnForwardFailure=yes -N -L 8888:127.0.0.1:8888 -L 8889:127.0.0.1:8889 -L 8189:127.0.0.1:8189 -L 8010:127.0.0.1:8010 -L 8011:127.0.0.1:8011 -L 8012:127.0.0.1:8012 -L 8013:127.0.0.1:8013 -R 18080:127.0.0.1:8080 %GPU_USER%@%GPU_HOST%

echo Enter the SSH password in the tunnel window and keep it open.
pause

ssh %GPU_USER%@%GPU_HOST% "curl -fsS http://127.0.0.1:18080/api/cameras/active >/dev/null"
if errorlevel 1 (
  echo [ERROR] GPU PC cannot reach the Windows backend through port 18080.
  exit /b 1
)

echo.
echo [4/4] Starting MediaMTX, RTSP publisher, then AI runner...
ssh %GPU_USER%@%GPU_HOST% "cd %DEV_ROOT% && ( nohup bash scripts/run_rtsp_server.sh > rtsp_server.log 2>&1 </dev/null & ) && sleep 3 && source %STABLE_ROOT%/.venv/bin/activate && ( nohup python scripts/start_simulated_rtsp_from_folder.py --video-dir /home/%GPU_USER%/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos --backend-url http://127.0.0.1:18080 --rtsp-host 127.0.0.1 --rtsp-port 8554 --poll-interval 30 > publisher.log 2>&1 </dev/null & ) && sleep 8 && ( nohup python scripts/run_registered_cameras.py --backend-base-url http://127.0.0.1:18080 --rtsp-base-url rtsp://127.0.0.1:8554 --video-pool /home/%GPU_USER%/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos --overlay-base-port 8010 --detector-mode real --yolo-model yolo26n-pose.pt --publisher mqtt --mqtt-host %MQTT_HOST% --mqtt-port %MQTT_PORT% --mqtt-topic safety/events --skip-simulated-ffmpeg > ai_runner.log 2>&1 </dev/null & )"

echo.
echo DEV runtime started. Keep the tunnel window open.
echo Press any key to close this launcher window.
pause >nul
endlocal
