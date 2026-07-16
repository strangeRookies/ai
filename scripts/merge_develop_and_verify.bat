@echo off
setlocal
cd /d "%~dp0.."

echo === strange_ai: codex/vlm-eight-keyframes ===
git status -sb
if errorlevel 1 exit /b 1

git fetch origin develop
if errorlevel 1 exit /b 1

git merge origin/develop --no-edit
if errorlevel 1 (
  echo.
  echo MERGE CONFLICT. Resolve files, then: git add -A ^&^& git commit
  exit /b 1
)

python -m pytest tests/test_vlm_process.py -q
if errorlevel 1 exit /b 1

echo.
echo Tests OK. Push with:
echo   git push origin codex/vlm-eight-keyframes
endlocal