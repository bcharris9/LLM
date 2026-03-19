#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${1:-.venv312}"

pick_python() {
  if command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  return 1
}

if [[ -x "$VENV_DIR/bin/python" ]]; then
  echo "Using existing virtual environment: $ROOT_DIR/$VENV_DIR"
else
  PYTHON_BIN="$(pick_python)" || {
    echo "No Python interpreter found (tried python3.12, python3, python)." >&2
    exit 1
  }
  echo "Creating virtual environment with $PYTHON_BIN at $ROOT_DIR/$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

cat <<EOF

Virtual environment ready: $ROOT_DIR/$VENV_DIR
Activate it with:
  source "$VENV_DIR/bin/activate"

Or run commands directly, for example:
  "$VENV_DIR/bin/python" -m uvicorn server:app --host 127.0.0.1 --port 8000
EOF
