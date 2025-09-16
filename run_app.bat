@echo off
setlocal enabledelayedexpansion

REM Resolve script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Choose venv directory name
set "VENV_DIR=%SCRIPT_DIR%\.venv"

if exist "%VENV_DIR%" (
    echo Found existing virtual environment at: %VENV_DIR%
) else (
    echo Creating virtual environment at: %VENV_DIR%
    python -m venv "%VENV_DIR%"
)

echo Activating virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

echo Upgrading pip
python -m pip install --upgrade pip

echo Installing requirements
pip install -r "%SCRIPT_DIR%\requirements.txt"

echo Starting app.py
python "%SCRIPT_DIR%\app.py"

pause
