@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

if not exist "%~dp0logs" mkdir "%~dp0logs"
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set "dt=%%I"
if not defined dt set "dt=20260701120000"
set "YYYY=%dt:~0,4%"
set "MM=%dt:~4,2%"
set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%"
set "Min=%dt:~10,2%"
set "Sec=%dt:~12,2%"
set "LAUNCHER_LOG=%~dp0logs\ai_launcher_trace_%YYYY%%MM%%DD%_%HH%%Min%%Sec%.log"

echo TRACE: launcher trace started > "%LAUNCHER_LOG%"

call :LOG "TRACE: Running script path: %~f0"
call :LOG "TRACE: Log file path: %LAUNCHER_LOG%"

set "LOCAL_CONFIG=%~dp0AI_LOCAL_CONFIG.bat"
if not exist "%LOCAL_CONFIG%" goto CONFIG_ERROR
call "%LOCAL_CONFIG%"
goto CONFIG_OK

:CONFIG_ERROR
call :LOG "[ERROR] Missing AI_LOCAL_CONFIG.bat"
pause
exit /b 1

:CONFIG_OK
call :LOG "TRACE: ENTER CONFIG_OK"
if not defined GPU_HOST (
  call :LOG "ERROR: GPU_HOST not defined"
  exit /b 1
)
if not defined GPU_USER (
  call :LOG "ERROR: GPU_USER not defined"
  exit /b 1
)
if not defined STABLE_ROOT (
  call :LOG "ERROR: STABLE_ROOT not defined"
  exit /b 1
)

set "REMOTE_ROOT=%STABLE_ROOT%"
if not defined AI_GIT_BRANCH (set "BRANCH=develop") else set "BRANCH=%AI_GIT_BRANCH%"
set "MQTT_HOST=15.165.248.37"
set "MQTT_PORT=1883"
if not defined MJPEG_TUNNEL_START_PORT set "MJPEG_TUNNEL_START_PORT=8010"
if not defined MJPEG_TUNNEL_END_PORT set "MJPEG_TUNNEL_END_PORT=8020"

call :LOG "TRACE: ENTER STEP1"
echo.
echo ========================================================
echo [1/5] 사전 상태 점검
echo ========================================================
call :LOG "Checking for busy local ports (8091, 8888, 8889, 8189, 18080, MJPEG %MJPEG_TUNNEL_START_PORT%-%MJPEG_TUNNEL_END_PORT%)..."
set "BUSY_PORTS="
for %%P in (8091 8888 8889 8189 18080) do call :CHECK_LOCAL_PORT %%P
for /L %%P in (%MJPEG_TUNNEL_START_PORT%,1,%MJPEG_TUNNEL_END_PORT%) do call :CHECK_LOCAL_PORT %%P
if not defined BUSY_PORTS goto PORTS_FREE

call :LOG "[WARNING] Stale tunnel ports are already listening on your local machine:%BUSY_PORTS%"
echo This indicates another SSH tunnel or AI session might be active.
set /p "CLEAN_CHOICE=Do you want to clean up existing AI processes first? [y/n]: "
call :LOG "TRACE: User chose clean_choice = %CLEAN_CHOICE%"
if /i "%CLEAN_CHOICE%"=="y" call "%~dp0cleanup_ai_processes.bat"
if not /i "%CLEAN_CHOICE%"=="y" call :LOG "Continuing anyway."

:PORTS_FREE
call :LOG "TRACE: PORTS_FREE reached"
echo.
echo ========================================================
echo Starting STABLE AI environment from GPU %BRANCH% branch
echo ========================================================
echo [DEPRECATED] This batch file is obsolete.
echo Please run the official launcher at the project root instead:
echo   ..\AI_실행_딸깍.bat
echo ========================================================
echo.
echo ========================================================
echo Select running mode:
echo   [1] Full restart and run AI processes (First run)
echo   [2] SSH Tunnels only (For sharing running streams)
echo   [3] Stop remote AI processes only
echo ========================================================
set /p "MODE=Choose [1, 2, or 3]: "
call :LOG "TRACE: MODE selected = %MODE%"

if "%MODE%"=="3" goto MODE_STOP
if "%MODE%"=="2" goto MODE_TUNNEL
goto MODE_FULL

:MODE_TUNNEL
call :LOG "TRACE: ENTER MODE_TUNNEL"
echo.
echo === [Tunnel Only Mode] ===
set "RUN_MODE=tunnel_only"
goto STEP3_TUNNEL

:MODE_FULL
call :LOG "TRACE: ENTER MODE_FULL"
set "RUN_MODE=full_run"
call :LOG "TRACE: ENTER STEP2"
echo.
echo ========================================================
echo [2/5] GPU 서버 코드 동기화
echo ========================================================
call :LOG "Syncing GPU stable repo to origin/%BRANCH% and stopping previous AI runtime processes (single SSH connection to avoid MaxStartups)..."
ssh %GPU_USER%@%GPU_HOST% "cd %REMOTE_ROOT% && git stash push -u -m auto-stash-before-ai-stable-run || true && git fetch origin && git checkout %BRANCH% && git pull --ff-only origin %BRANCH% && (pkill -f 'scripts/run_registered_cameras.py' 2>/dev/null || true; pkill -f 'scripts/start_simulated_rtsp_from_folder.py' 2>/dev/null || true; pkill -f 'scripts/serve_ai_overlay.py' 2>/dev/null || true; pkill -f 'ai.internal_vlm_api' 2>/dev/null || true; pkill -f 'ffmpeg' 2>/dev/null || true; rm -f %REMOTE_ROOT%/runs/camera_worker_registry.json 2>/dev/null || true; docker rm -f mediamtx 2>/dev/null || true)"
if errorlevel 1 (
  set "CURRENT_STEP=[2/5] GPU 서버 코드 동기화 및 기존 프로세스 정리"
  goto FAIL
)

:STEP3_TUNNEL
call :LOG "TRACE: ENTER STEP3"
echo.
echo ========================================================
echo [3/5] SSH 터널 창 실행
echo ========================================================
call :LOG "Cleaning up stale remote ports and starting SSH tunnel in a new window..."
ssh %GPU_USER%@%GPU_HOST% "fuser -k 18080/tcp 2>/dev/null || true; lsof -ti tcp:18080 2>/dev/null | xargs -r kill -9 2>/dev/null || true; ss -lntp 'sport = :18080' 2>/dev/null || true"

call :LOG "TRACE: BEFORE START SSH WINDOW"
set "TUNNEL_FORWARDS=-L 8091:127.0.0.1:8091 -L 8888:127.0.0.1:8888 -L 8889:127.0.0.1:8889 -L 8189:127.0.0.1:8189"
for /L %%P in (%MJPEG_TUNNEL_START_PORT%,1,%MJPEG_TUNNEL_END_PORT%) do set "TUNNEL_FORWARDS=!TUNNEL_FORWARDS! -L %%P:127.0.0.1:%%P"
call :LOG "MJPEG tunnel ports: %MJPEG_TUNNEL_START_PORT%-%MJPEG_TUNNEL_END_PORT% (cam_01 uses 8010, cam_02 uses 8011, ...)"
if "%RUN_MODE%"=="tunnel_only" start "AI STABLE SSH Tunnel - keep open" cmd /k ssh -o ExitOnForwardFailure=yes -t %TUNNEL_FORWARDS% %GPU_USER%@%GPU_HOST% "echo ==============================================; echo [SSH TUNNEL ACTIVE] Tunnel established successfully.; echo Keep this window open to maintain streams.; echo ==============================================; tail -f /dev/null"
if not "%RUN_MODE%"=="tunnel_only" start "AI STABLE SSH Tunnel - keep open" cmd /k ssh -o ExitOnForwardFailure=yes -t %TUNNEL_FORWARDS% -R 18080:127.0.0.1:8080 %GPU_USER%@%GPU_HOST% "echo ==============================================; echo [SSH TUNNEL ACTIVE] Tunnel established successfully.; echo Keep this window open to maintain/share streams.; echo ==============================================; tail -f /dev/null"

if errorlevel 1 (
  call :LOG "[ERROR] Failed to launch SSH tunnel window."
  set "CURRENT_STEP=[3/5] SSH 터널 창 실행"
  goto FAIL
)
call :LOG "TRACE: AFTER START SSH WINDOW"

echo.
call :LOG "Enter the SSH password in the tunnel window and keep it open."
call :LOG "TRACE: BEFORE PAUSE - waiting for user keypress after starting tunnel window"
pause
call :LOG "TRACE: AFTER PAUSE"

if "%RUN_MODE%"=="tunnel_only" (
  call :LOG "Skipping backend check (Tunnel Only mode)."
  goto STEP5_STATUS
)

:BACKEND_CHECK
call :LOG "TRACE: ENTER BACKEND_CHECK"
echo.
call :LOG "Checking GPU access to Windows backend through reverse tunnel..."
ssh %GPU_USER%@%GPU_HOST% "curl -fsS http://127.0.0.1:18080/api/cameras/active >/dev/null"
set "BACKEND_ERR=!errorlevel!"
call :LOG "TRACE: BACKEND_CHECK errorlevel = !BACKEND_ERR!"
if "!BACKEND_ERR!"=="0" goto STEP4_START_REMOTE

call :LOG "TRACE: ENTER VERIFY_TUNNEL_FAILED"
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
call :LOG "TRACE: VERIFY_FAIL CHOICE selected = %CHECK_FAIL_CHOICE%"
if "%CHECK_FAIL_CHOICE%"=="1" goto BACKEND_CHECK
if "%CHECK_FAIL_CHOICE%"=="2" call :LOG "[WARNING] User selected to skip backend check. Continuing anyway."
if "%CHECK_FAIL_CHOICE%"=="2" goto STEP4_START_REMOTE
if "%CHECK_FAIL_CHOICE%"=="3" goto SAFE_EXIT
echo Invalid choice.
goto BACKEND_CHECK

:STEP4_START_REMOTE
call :LOG "TRACE: ENTER STEP4"
echo.
echo ========================================================
echo [4/5] 원격 AI 서비스 실행
echo ========================================================
call :LOG "Starting MediaMTX, RTSP publisher, then AI runner on GPU stable repo..."
call :LOG "TRACE: STEP4 REMOTE COMMAND START"
ssh %GPU_USER%@%GPU_HOST% "bash %REMOTE_ROOT%/scripts/start_ai_stable.sh %MQTT_HOST% %MQTT_PORT%"
set "CMD_ERR=!errorlevel!"
call :LOG "TRACE: STEP4 REMOTE COMMAND END errorlevel = !CMD_ERR!"
if "!CMD_ERR!"=="0" goto STEP5_STATUS

set "CURRENT_STEP=[4/5] 원격 AI 서비스 실행"
goto FAIL

:STEP5_STATUS
call :LOG "TRACE: ENTER STEP5"
echo.
echo ========================================================
echo [5/5] 상태 점검
echo ========================================================
if "%RUN_MODE%"=="tunnel_only" call :LOG "SSH Tunnel established. Keep the tunnel window open."
if not "%RUN_MODE%"=="tunnel_only" call :LOG "STABLE runtime started. Keep the tunnel window open."
echo.
echo Quick checks after startup:
echo   GPU: ss -lntup ^| grep -E "8091^|8554^|8888^|8889^|8189^|8010^|8011^|8012^|8013^|8014"
echo   GPU: docker ps ^| grep mediamtx
echo   Local MJPEG cam_01: http://localhost:8010/mjpeg/cam_01
echo   Local MJPEG cam_05: http://localhost:8014/mjpeg/cam_05
echo   Forwarded MJPEG range: %MJPEG_TUNNEL_START_PORT%-%MJPEG_TUNNEL_END_PORT%
echo   Local VLM health: http://localhost:8091/internal/vlm/health
echo.
goto SUCCESS

:MODE_STOP
call :LOG "TRACE: ENTER MODE_STOP"
if /i not "%MODE%"=="3" (
  call :LOG "ERROR: MODE_STOP reached while MODE is not stop. Redirecting to FAIL."
  goto FAIL
)
echo.
call :LOG "Stopping remote AI runtime processes..."
ssh %GPU_USER%@%GPU_HOST% "pkill -f 'scripts/run_registered_cameras.py' 2>/dev/null || true; pkill -f 'scripts/start_simulated_rtsp_from_folder.py' 2>/dev/null || true; pkill -f 'scripts/serve_ai_overlay.py' 2>/dev/null || true; pkill -f 'ai.internal_vlm_api' 2>/dev/null || true; pkill -f 'ffmpeg' 2>/dev/null || true; rm -f %REMOTE_ROOT%/runs/camera_worker_registry.json 2>/dev/null || true; docker rm -f mediamtx 2>/dev/null || true"
call :LOG "Remote AI processes stopped."
goto SUCCESS

:SUCCESS
call :LOG "TRACE: ENTER SUCCESS"
echo.
echo [DONE] AI launcher completed successfully.
echo Log file: %LAUNCHER_LOG%
echo Please inspect the logs above before closing this window.
pause
exit /b 0

:FAIL
call :LOG "TRACE: ENTER FAIL at %CURRENT_STEP%"
echo.
echo [FAILED] AI launcher failed at: %CURRENT_STEP%
echo Log file: %LAUNCHER_LOG%
pause
exit /b 1

:SAFE_EXIT
call :LOG "TRACE: ENTER SAFE_EXIT"
echo.
echo User selected exit.
pause
exit /b 0

:LOG
echo %~1
echo [!DATE! !TIME!] %~1 >> "%LAUNCHER_LOG%"
exit /b 0

:CHECK_LOCAL_PORT
netstat -ano | findstr /R "LISTENING.*:%~1" >nul
if not errorlevel 1 set "BUSY_PORTS=!BUSY_PORTS! %~1"
exit /b 0
