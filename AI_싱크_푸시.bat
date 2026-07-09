@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

set "BRANCH=codex/ai-worker-flow-improvements"
set /p "MSG=커밋 메시지를 입력하세요: "
if "!MSG!"=="" set "MSG=auto-sync AI updates"

echo.
echo ========================================================
echo [1/2] AI 저장소 (strange_ai) 커밋 및 푸시
echo ========================================================
cd "%~dp0strange_ai"
git add .
git commit -m "%MSG%"
git push origin %BRANCH%

echo.
echo ========================================================
echo [2/2] 루트 저장소 포인터 커밋 및 푸시
echo ========================================================
cd "%~dp0."
git add strange_ai
git commit -m "chore(ai): sync strange_ai pointer - %MSG%"
git push origin %BRANCH%

echo.
echo ========================================================
echo [완료] 모든 저장소의 싱크 및 푸시가 완료되었습니다.
echo ========================================================
pause
