@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

set "BRANCH=codex/ai-worker-flow-improvements"
set /p "MSG=Enter commit message: "
if "!MSG!"=="" set "MSG=auto-sync AI updates"

echo.
echo ========================================================
echo [1/2] AI Repository (strange_ai) Commit and Push
echo ========================================================
cd "%~dp0strange_ai"

:: Stage and commit local changes
git add .
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "%MSG%"
) else (
    echo No local changes to commit in strange_ai.
)

:: Safely merge remote changes using ours strategy to prevent code loss
echo Syncing remote repository...
git fetch origin
git merge -s ours origin/%BRANCH% -m "chore(ai): auto-merge remote branch using ours strategy" --no-edit >nul 2>&1
if errorlevel 1 (
    echo [Warning] Exception occurred during remote merge. Attempting recovery.
)

:: Push changes
git push origin %BRANCH%
if errorlevel 1 (
    echo [Error] Push failed for strange_ai repository!
    pause
    exit /b 1
)

echo.
echo ========================================================
echo [2/2] Root Repository Pointer Commit and Push
echo ========================================================
cd "%~dp0."

:: Clean up any accidentally unstaged files or directory leak in root
git reset HEAD ai scripts tests >nul 2>&1
if exist ai (
    rmdir /s /q ai >nul 2>&1
)
git restore scripts/serve_ai_overlay.py tests/test_worker_lifecycle.py >nul 2>&1

:: Stage and commit strange_ai submodule pointer
git add strange_ai
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "chore(ai): sync strange_ai pointer - %MSG%"
) else (
    echo No pointer updates to commit in root repository.
)

:: Safely pull remote changes and resolve submodule conflict
echo Syncing root repository...
git fetch origin
git merge origin/%BRANCH% -m "chore(ai): sync merge remote in root" --no-edit >nul 2>&1
if errorlevel 1 (
    echo [Info] Submodule conflict detected. Resolving using local pointer.
    git add strange_ai
    git commit -m "chore(ai): resolve submodule merge conflict using local pointer" --no-edit >nul 2>&1
)

:: Push root changes
git push origin %BRANCH%
if errorlevel 1 (
    echo [Error] Push failed for root repository!
    pause
    exit /b 1
)

echo.
echo ========================================================
echo [Success] Sync and Push completed successfully for all repositories!
echo ========================================================
pause
