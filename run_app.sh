#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory (works for spaces in path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Choose venv directory name
VENV_DIR="$SCRIPT_DIR/.venv"

# Check if Python 3.11 is available
echo "Checking for Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    echo "ERROR: Python 3.11 is not installed or not in PATH"
    echo "Please install Python 3.11:"
    echo "  macOS: brew install python@3.11"
    echo "  Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "  Or download from: https://www.python.org/downloads/"
    exit 1
fi

# Display Python version for confirmation
echo "Found Python 3.11:"
python3.11 --version

if [ -d "$VENV_DIR" ]; then
  echo "Found existing virtual environment at: $VENV_DIR"
  echo "Checking Python version in virtual environment..."
  source "$VENV_DIR/bin/activate"
  python --version
  deactivate
else
  echo "Creating virtual environment with Python 3.11 at: $VENV_DIR"
  if ! python3.11 -m venv "$VENV_DIR"; then
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


