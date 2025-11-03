#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory (works for spaces in path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Choose venv directory name
VENV_DIR="$SCRIPT_DIR/.venv"
# Desired Python version (major.minor). Override by exporting DESIRED_PYTHON=3.10
DESIRED_PYTHON="${DESIRED_PYTHON:-3.11}"

# Check if Python 3.11 is available
echo "Checking for Python ${DESIRED_PYTHON}..."
if ! command -v python${DESIRED_PYTHON} &> /dev/null; then
    echo "ERROR: Python ${DESIRED_PYTHON} is not installed or not in PATH"
    echo "Please install Python 3.11:"
    echo "  macOS: brew install python@3.11"
    echo "  Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "  Or download from: https://www.python.org/downloads/"
    exit 1
fi

# Display Python version for confirmation
echo "Found Python ${DESIRED_PYTHON}:"
python${DESIRED_PYTHON} --version

if [ -d "$VENV_DIR" ]; then
  echo "Found existing virtual environment at: $VENV_DIR"
  echo "Checking Python version in virtual environment..."
  VENV_PY_VER=$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' || echo "")
  echo "venv Python: $VENV_PY_VER"
  if [ "$VENV_PY_VER" != "$DESIRED_PYTHON" ]; then
    echo "Virtual environment uses $VENV_PY_VER, expected $DESIRED_PYTHON. Recreating venv..."
    rm -rf "$VENV_DIR"
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment with Python ${DESIRED_PYTHON} at: $VENV_DIR"
  if ! python${DESIRED_PYTHON} -m venv "$VENV_DIR"; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
  fi
fi

echo "Activating virtual environment"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip"
if ! python -m pip install --upgrade pip; then
    echo "ERROR: Failed to upgrade pip"
    exit 1
fi

echo "Installing requirements"
if ! pip install -r "$SCRIPT_DIR/requirements.txt"; then
    echo "ERROR: Failed to install requirements"
    echo "Please check your internet connection and try again"
    exit 1
fi

echo "Verifying MediaPipe installation..."
if ! python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"; then
    echo "WARNING: MediaPipe verification failed, but continuing..."
fi

echo "Starting MultiSOCIAL Toolbox..."
python "$SCRIPT_DIR/app.py"


