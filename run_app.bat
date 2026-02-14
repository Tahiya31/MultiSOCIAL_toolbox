@echo off
setlocal enabledelayedexpansion

REM Ensure Unicode support for output
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM Resolve script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Choose venv directory name
set "VENV_DIR=%SCRIPT_DIR%\.venv"

REM Desired Python version (major.minor). Override by setting DESIRED_PYTHON env var before running.
if "%DESIRED_PYTHON%"=="" (
    set "DESIRED_PYTHON=3.11"
)

REM Check for Python 3.11
set "PYTHON_EXEC=python%DESIRED_PYTHON%"
%PYTHON_EXEC% --version >nul 2>&1
if %errorlevel% neq 0 (
    REM Try just 'python' and check version
    python -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor == 11 else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXEC=python"
    ) else (
        REM Try 'py' launcher
        py -%DESIRED_PYTHON% --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_EXEC=py -%DESIRED_PYTHON%"
        ) else (
            echo ERROR: Python %DESIRED_PYTHON% is not installed or not in PATH
            echo Please install Python 3.11 from https://www.python.org/downloads/
            echo Make sure to check "Add Python to PATH" during installation
            pause
            exit /b 1
        )
    )
)

REM Display Python version for confirmation
echo Found Python %DESIRED_PYTHON% using command: %PYTHON_EXEC%
%PYTHON_EXEC% --version

if exist "%VENV_DIR%" (
    echo Found existing virtual environment at: %VENV_DIR%
    set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
    if not exist "!VENV_PYTHON!" (
        echo Venv Python executable not found at: !VENV_PYTHON!
        set "VENV_PY_VER=0.0"
    ) else (
        "!VENV_PYTHON!" -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))" > "%TEMP%\_multi_social_venv_ver.txt" 2>&1
        if !errorlevel! neq 0 (
            echo Verify command failed. Output:
            type "%TEMP%\_multi_social_venv_ver.txt"
            set "VENV_PY_VER=ERROR"
        ) else (
            set "VENV_PY_VER="
            for /f "usebackq tokens=*" %%v in ("%TEMP%\_multi_social_venv_ver.txt") do set "VENV_PY_VER=%%v"
        )
        if exist "%TEMP%\_multi_social_venv_ver.txt" del "%TEMP%\_multi_social_venv_ver.txt"
    )
    echo venv Python: !VENV_PY_VER!
    if not "!VENV_PY_VER!"=="%DESIRED_PYTHON%" (
        echo Virtual environment uses !VENV_PY_VER!, expected %DESIRED_PYTHON%. Recreating venv...
        rmdir /S /Q "%VENV_DIR%"
    )
)

if not exist "%VENV_DIR%" (
    echo Creating virtual environment with Python %DESIRED_PYTHON% at: %VENV_DIR%
    %PYTHON_EXEC% -m venv "%VENV_DIR%"
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
python "%SCRIPT_DIR%\src\app.py"
set "APP_EXIT_CODE=!errorlevel!"

if !APP_EXIT_CODE! neq 0 (
    echo.
    echo ============================================================
    echo The application exited with an error.
    echo If you see "ModuleNotFoundError" or import errors, try:
    echo   1. Delete the .venv folder in this directory
    echo   2. Run this script again to recreate the environment
    echo ============================================================
)

pause
