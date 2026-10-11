#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x "$HOME/miniforge3/bin/python3" ]; then
    PYTHON_CMD="$HOME/miniforge3/bin/python3"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python3" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python3"
elif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python3"
elif [ -x "$HOME/miniconda3/bin/python3" ]; then
    PYTHON_CMD="$HOME/miniconda3/bin/python3"
elif [ -x "$HOME/anaconda3/bin/python3" ]; then
    PYTHON_CMD="$HOME/anaconda3/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo "Error: python3 not found." >&2
    exit 1
fi

exec "$PYTHON_CMD" Drowsiness_Detection.py "$@"
