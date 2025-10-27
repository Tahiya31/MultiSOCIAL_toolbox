@echo off
setlocal enabledelayedexpansion

REM Resolve script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Choose venv directory name
set "VENV_DIR=%SCRIPT_DIR%\.venv"

REM Check if Python 3.11 is available
echo Checking for Python 3.11...
python3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 is not installed or not in PATH
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Display Python version for confirmation
echo Found Python 3.11:
python3.11 --version

if exist "%VENV_DIR%" (
    echo Found existing virtual environment at: %VENV_DIR%
    echo Checking Python version in virtual environment...
    call "%VENV_DIR%\Scripts\activate.bat"
    python --version
    call "%VENV_DIR%\Scripts\deactivate.bat"
) else (
    echo Creating virtual environment with Python 3.11 at: %VENV_DIR%
    python3.11 -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo Activating virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

echo Upgrading pip
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

echo Installing requirements
pip install -r "%SCRIPT_DIR%\requirements.txt"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo Verifying MediaPipe installation...
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
if %errorlevel% neq 0 (
    echo WARNING: MediaPipe verification failed, but continuing...
)

echo Starting MultiSOCIAL Toolbox...
python "%SCRIPT_DIR%\app.py"

pause
