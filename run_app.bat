@echo off
setlocal enabledelayedexpansion

REM Resolve script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Choose venv directory name
set "VENV_DIR=%SCRIPT_DIR%\.venv"

REM Desired Python version (major.minor). Override by setting DESIRED_PYTHON env var before running.
if "%DESIRED_PYTHON%"=="" (
    set "DESIRED_PYTHON=3.11"
)

REM Check if Python 3.11 is available
echo Checking for Python %DESIRED_PYTHON%...
python%DESIRED_PYTHON% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python %DESIRED_PYTHON% is not installed or not in PATH
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Display Python version for confirmation
echo Found Python %DESIRED_PYTHON%:
python%DESIRED_PYTHON% --version

if exist "%VENV_DIR%" (
    echo Found existing virtual environment at: %VENV_DIR%
    for /f "usebackq tokens=*" %%v in (`"%VENV_DIR%\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"`) do set VENV_PY_VER=%%v
    echo venv Python: %VENV_PY_VER%
    if not "%VENV_PY_VER%"=="%DESIRED_PYTHON%" (
        echo Virtual environment uses %VENV_PY_VER%, expected %DESIRED_PYTHON%. Recreating venv...
        rmdir /S /Q "%VENV_DIR%"
    )
)

if not exist "%VENV_DIR%" (
    echo Creating virtual environment with Python %DESIRED_PYTHON% at: %VENV_DIR%
    python%DESIRED_PYTHON% -m venv "%VENV_DIR%"
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
