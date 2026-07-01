@echo off
setlocal enabledelayedexpansion
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
set "BRANCH=develop"
set "MQTT_HOST=15.165.248.37"
set "MQTT_PORT=1883"

echo.
echo ========================================================
echo [1/5] 사전 상태 점검
echo ========================================================
echo Checking for busy local ports (8888, 8889, 8189, 18080)...
netstat -ano | findstr /R "LISTENING.*:8888 LISTENING.*:8889 LISTENING.*:8189 LISTENING.*:18080" >nul
if errorlevel 1 goto PORTS_FREE

echo.
echo [WARNING] Stale tunnel ports are already listening on your local machine.
echo This indicates another SSH tunnel or AI session might be active.
set /p "CLEAN_CHOICE=Do you want to clean up existing AI processes first? [y/n]: "
if /i "%CLEAN_CHOICE%"=="y" call "%~dp0cleanup_ai_processes.bat"
if not /i "%CLEAN_CHOICE%"=="y" echo Continuing anyway.

:PORTS_FREE
echo.
echo ========================================================
echo Starting STABLE AI environment from GPU %BRANCH% branch
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
set /p "MODE=Choose [1, 2, or 3]: "

if "%MODE%"=="3" goto MODE_STOP
if "%MODE%"=="2" goto MODE_TUNNEL
goto MODE_FULL

:MODE_TUNNEL
echo.
echo === [Tunnel Only Mode] ===
set "RUN_MODE=tunnel_only"
goto STEP3_TUNNEL

:MODE_FULL
set "RUN_MODE=full_run"
echo.
echo ========================================================
echo [2/5] GPU 서버 코드 동기화
echo ========================================================
echo Syncing GPU stable repo to origin/%BRANCH%...
ssh %GPU_USER%@%GPU_HOST% "cd %REMOTE_ROOT% && git stash push -u -m auto-stash-before-ai-stable-run || true && git fetch origin && git checkout %BRANCH% && git pull --ff-only origin %BRANCH%"
if errorlevel 1 (
  set "CURRENT_STEP=[2/5] GPU 서버 코드 동기화"
  goto FAIL
)

echo.
echo Stopping previous AI runtime processes...
ssh %GPU_USER%@%GPU_HOST% "pkill -f '%REMOTE_ROOT%/scripts/run_registered_cameras.py' 2>/dev/null || true; pkill -f '%REMOTE_ROOT%/scripts/start_simulated_rtsp_from_folder.py' 2>/dev/null || true; pkill -f '%REMOTE_ROOT%/scripts/serve_ai_overlay.py' 2>/dev/null || true; pkill -f 'ffmpeg' 2>/dev/null || true; rm -f %REMOTE_ROOT%/runs/camera_worker_registry.json 2>/dev/null || true; docker rm -f mediamtx 2>/dev/null || true"
if errorlevel 1 (
  set "CURRENT_STEP=[2/5] 기존 AI 프로세스 정리"
  goto FAIL
)

:STEP3_TUNNEL
echo.
echo ========================================================
echo [3/5] SSH 터널 창 실행
echo ========================================================
echo Cleaning up stale remote ports and starting SSH tunnel in a new window...
ssh %GPU_USER%@%GPU_HOST% "pkill -f '^sshd: %GPU_USER%$' 2>/dev/null || true; lsof -t -i:18080 | xargs kill -9 2>/dev/null || true"

if "%RUN_MODE%"=="tunnel_only" start "AI STABLE SSH Tunnel - keep open" cmd /k ssh -o ExitOnForwardFailure=yes -N -L 8888:127.0.0.1:8888 -L 8889:127.0.0.1:8889 -L 8189:127.0.0.1:8189 %GPU_USER%@%GPU_HOST%
if not "%RUN_MODE%"=="tunnel_only" start "AI STABLE SSH Tunnel - keep open" cmd /k ssh -o ExitOnForwardFailure=yes -N -L 8888:127.0.0.1:8888 -L 8889:127.0.0.1:8889 -L 8189:127.0.0.1:8189 -R 18080:127.0.0.1:8080 %GPU_USER%@%GPU_HOST%

if errorlevel 1 (
  echo [ERROR] Failed to launch SSH tunnel window.
  set "CURRENT_STEP=[3/5] SSH 터널 창 실행"
  goto FAIL
)

echo.
echo Enter the SSH password in the tunnel window and keep it open.
pause

if "%RUN_MODE%"=="tunnel_only" (
  echo Skipping backend check (Tunnel Only mode).
  goto STEP5_STATUS
)

:BACKEND_CHECK
echo.
echo Checking GPU access to Windows backend through reverse tunnel...
ssh %GPU_USER%@%GPU_HOST% "curl -fsS http://127.0.0.1:18080/api/cameras/active >/dev/null"
if not errorlevel 1 goto STEP4_START_REMOTE

echo.
echo [WARNING] GPU PC cannot reach the Windows backend through reverse tunnel port 18080.
echo Please make sure:
echo   1. The SSH Tunnel window is open and you entered the password.
echo   2. The backend service is running locally on localhost:8080.
echo.
echo Options:
echo   [1] Retry the backend check
echo   [2] Skip the check and continue AI startup anyway
echo   [3] Exit launcher
set /p "CHECK_FAIL_CHOICE=Choose [1, 2, or 3]: "
if "%CHECK_FAIL_CHOICE%"=="1" goto BACKEND_CHECK
if "%CHECK_FAIL_CHOICE%"=="2" echo [WARNING] User selected to skip backend check. Continuing anyway.
if "%CHECK_FAIL_CHOICE%"=="2" goto STEP4_START_REMOTE
if "%CHECK_FAIL_CHOICE%"=="3" goto SAFE_EXIT
echo Invalid choice.
goto BACKEND_CHECK

:STEP4_START_REMOTE
echo.
echo ========================================================
echo [4/5] 원격 AI 서비스 실행
echo ========================================================
echo Starting MediaMTX, RTSP publisher, then AI runner on GPU stable repo...
ssh %GPU_USER%@%GPU_HOST% "bash %REMOTE_ROOT%/scripts/start_ai_stable.sh %MQTT_HOST% %MQTT_PORT%"
if errorlevel 1 (
  set "CURRENT_STEP=[4/5] 원격 AI 서비스 실행"
  goto FAIL
)

:STEP5_STATUS
echo.
echo ========================================================
echo [5/5] 상태 점검
echo ========================================================
if "%RUN_MODE%"=="tunnel_only" echo SSH Tunnel established. Keep the tunnel window open.
if not "%RUN_MODE%"=="tunnel_only" echo STABLE runtime started. Keep the tunnel window open.
echo.
echo Quick checks after startup:
echo   GPU: ss -lntup ^| grep -E "8554^|8888^|8889^|8189"
echo   GPU: docker ps ^| grep mediamtx
echo   Local: http://localhost:8888/cam_04/index.m3u8
echo.
goto SUCCESS

:MODE_STOP
echo.
echo Stopping remote AI runtime processes...
ssh %GPU_USER%@%GPU_HOST% "pkill -f '%REMOTE_ROOT%/scripts/run_registered_cameras.py' 2>/dev/null || true; pkill -f '%REMOTE_ROOT%/scripts/start_simulated_rtsp_from_folder.py' 2>/dev/null || true; pkill -f '%REMOTE_ROOT%/scripts/serve_ai_overlay.py' 2>/dev/null || true; pkill -f 'ffmpeg' 2>/dev/null || true; rm -f %REMOTE_ROOT%/runs/camera_worker_registry.json 2>/dev/null || true; docker rm -f mediamtx 2>/dev/null || true"
echo Remote AI processes stopped.
goto SUCCESS

:SUCCESS
echo.
echo [DONE] AI launcher completed successfully.
pause
exit /b 0

:FAIL
echo.
echo [FAILED] AI launcher stopped at step: %CURRENT_STEP%
pause
exit /b 1

:SAFE_EXIT
echo.
echo User selected exit.
pause
exit /b 0
