@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_PYTHON="
if exist ".venv\Scripts\python.exe" (
    for /f "delims=" %%P in ('.venv\Scripts\python.exe -c "import sys; print(sys._base_executable)" 2^>nul') do set "PROJECT_PYTHON=%%P"
)
if defined PROJECT_PYTHON (
    "%PROJECT_PYTHON%" start.py %*
    goto finished
)
where py >nul 2>&1
if errorlevel 1 (
    python start.py %*
) else (
    py -3.12 start.py %*
)
:finished
if errorlevel 1 (
    echo.
    echo Install Python 3.12 and try again. See README.md for instructions.
    pause
    exit /b 1
)
