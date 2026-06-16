@echo off
chcp 65001 >nul
echo ========================================================
echo AI 환경 원클릭 종료 (GPU PC 백그라운드 프로세스 및 로컬 터널 종료)
echo ========================================================

echo.
echo [1/2] GPU PC의 백그라운드 프로세스들을 종료합니다...
ssh welabs@58.127.241.84 "pkill -f 'scripts/run_registered_cameras.py' 2>/dev/null || true; pkill -f 'scripts/start_simulated_rtsp_from_folder.py' 2>/dev/null || true; pkill -f 'scripts/serve_ai_overlay.py' 2>/dev/null || true; fuser -k 8010/tcp 2>/dev/null || true; fuser -k 8011/tcp 2>/dev/null || true; fuser -k 8012/tcp 2>/dev/null || true; fuser -k 8013/tcp 2>/dev/null || true; docker stop mediamtx 2>/dev/null || true; echo 'GPU PC 프로세스 종료 완료.'"

echo.
echo [2/2] 로컬 Windows PC의 SSH 터널링 프로세스를 종료합니다...
taskkill /F /IM ssh.exe 2>nul
if %errorlevel% equ 0 (
    echo 로컬 SSH 터널링 종료 완료.
) else (
    echo 실행 중인 로컬 SSH 터널링 프로세스가 없습니다.
)

echo.
echo ========================================================
echo 모든 AI 시스템이 정상적으로 종료되었습니다.
echo ========================================================
pause
