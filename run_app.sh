#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory (works for spaces in path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Choose venv directory name
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
  echo "Found existing virtual environment at: $VENV_DIR"
else
  echo "Creating virtual environment at: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip"
python -m pip install --upgrade pip

echo "Installing requirements"
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Starting app.py"
python "$SCRIPT_DIR/app.py"


