#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory (works for spaces in path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Choose venv directory name
VENV_DIR="$SCRIPT_DIR/.venv"
INSTALL_STAMP_FILE="$VENV_DIR/.multisocial-install-stamp"
# Desired Python version (major.minor). Override by exporting DESIRED_PYTHON=3.10
DESIRED_PYTHON="${DESIRED_PYTHON:-3.10}"

# Install profile: standard or complete. complete includes optional diarization support.
INSTALL_PROFILE="${MULTISOCIAL_INSTALL_PROFILE:-}"
if [ -z "$INSTALL_PROFILE" ] && [ -t 0 ]; then
    echo "Choose install profile:"
    echo "  1) Standard toolbox"
    echo "  2) Complete toolbox (includes optional speaker diarization support)"
    read -r -p "Selection [1/2]: " profile_choice
    if [ "$profile_choice" = "2" ]; then
        INSTALL_PROFILE="complete"
    else
        INSTALL_PROFILE="standard"
    fi
fi
INSTALL_PROFILE="${INSTALL_PROFILE:-standard}"

# Check if Python 3.10 is available
echo "Checking for Python ${DESIRED_PYTHON}..."
if ! command -v python${DESIRED_PYTHON} &> /dev/null; then
    echo "ERROR: Python ${DESIRED_PYTHON} is not installed or not in PATH"
    echo "Please install Python 3.10:"
    echo "  macOS: brew install python@3.10"
    echo "  Ubuntu/Debian: sudo apt install python3.10 python3.10-venv"
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

CURRENT_STAMP="$(python -c "from pathlib import Path; import hashlib; print('${INSTALL_PROFILE}|${DESIRED_PYTHON}|' + hashlib.sha256(Path(r'''$SCRIPT_DIR/pyproject.toml''').read_bytes()).hexdigest())")"
NEEDS_INSTALL=1
if [ -f "$INSTALL_STAMP_FILE" ]; then
    EXISTING_STAMP="$(cat "$INSTALL_STAMP_FILE" 2>/dev/null || true)"
    if [ "$EXISTING_STAMP" = "$CURRENT_STAMP" ]; then
        NEEDS_INSTALL=0
    fi
fi

echo "Upgrading pip"
if ! python -m pip install --upgrade pip; then
    echo "ERROR: Failed to upgrade pip"
    exit 1
fi

echo "Pinning setuptools for yolov5 compatibility"
if ! python -m pip install "setuptools==80.10.2"; then
    echo "ERROR: Failed to install the required setuptools version"
    exit 1
fi

if [ "$NEEDS_INSTALL" -eq 1 ]; then
    echo "Installing MultiSOCIAL Toolbox ($INSTALL_PROFILE profile)"
    pushd "$SCRIPT_DIR" >/dev/null
    if [ "$INSTALL_PROFILE" = "complete" ]; then
        INSTALL_CMD=(python -m pip install -e ".[complete]")
    else
        INSTALL_CMD=(python -m pip install -e .)
    fi
    if ! "${INSTALL_CMD[@]}"; then
        echo "ERROR: Failed to install toolbox dependencies"
        echo "Please check your internet connection and try again"
        popd >/dev/null
        exit 1
    fi
    popd >/dev/null
    printf '%s\n' "$CURRENT_STAMP" > "$INSTALL_STAMP_FILE"
else
    echo "Environment already matches the $INSTALL_PROFILE profile. Skipping reinstall."
fi

echo "Verifying MediaPipe installation..."
if ! python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"; then
    echo "WARNING: MediaPipe verification failed, but continuing..."
fi

echo "Starting MultiSOCIAL Toolbox..."
export MULTISOCIAL_INSTALL_PROFILE="$INSTALL_PROFILE"
python "$SCRIPT_DIR/src/app.py"
