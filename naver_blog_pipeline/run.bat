@echo off
setlocal enabledelayedexpansion

if not exist "input" (
    mkdir "input"
    echo [ERROR] 'input' folder not found. Created 'input' folder.
    echo Please place your .md files in the 'input' folder and run again.
    pause
    exit /b 1
)

set "found_files=0"
for %%F in (input\*.md) do (
    set /a found_files+=1
)

if !found_files! equ 0 (
    echo [WARNING] No .md files found in 'input' folder.
    echo Please place your markdown files in the 'input' folder and run again.
    pause
    exit /b 1
)

if not exist "output" (
    mkdir "output"
)

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo [INFO] node_modules folder not found. Installing npm packages...
    call npm install
    if !errorlevel! neq 0 (
        echo [ERROR] npm package installation failed.
        pause
        exit /b 1
    )
)

echo.
echo ===================================================
echo   Naver Blog Conversion Pipeline
echo ===================================================
echo.

for %%F in (input\*.md) do (
    set "filename=%%~nF"
    echo [*] Converting %%F...
    
    python scripts/convert_to_blog.py --input "input\%%~nxF" --output "output\!filename!.md" --humanize
    if !errorlevel! equ 0 (
        python scripts/publish_to_naver.py --input "output\!filename!.md" --draft
        if !errorlevel! equ 0 (
            echo [OK] Successfully converted %%F!
            echo Preview file: output\!filename!.preview.html
            echo.
            
            start "" "output\!filename!.preview.html"
        ) else (
            echo [ERROR] Failed to generate preview HTML for %%F.
            echo.
        )
    ) else (
        echo [ERROR] Failed to convert %%F.
        echo.
    )
)

echo ===================================================
echo   All conversions completed.
echo ===================================================
pause
