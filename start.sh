#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "╔═══════════════════════════════════════════╗"
echo "║    triMind-BitNet — Setup & Launch       ║"
echo "╚═══════════════════════════════════════════╝"

# --- Detect Python ---
PYTHON=""
for cmd in python3 python python3.11 python3.10; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found. Install Python 3.10+ first."
    exit 1
fi
echo "[1/3] Using $($PYTHON --version)"

# --- Virtual environment ---
VENV_DIR="$REPO_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/3] Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "[2/3] Virtual environment exists — skipping creation."
fi

source "$VENV_DIR/bin/activate"

# --- Install dependencies ---
echo "[3/3] Installing dependencies..."
pip install -q --upgrade pip setuptools wheel 2>/dev/null
pip install -q -r "$REPO_DIR/requirements.txt" 2>/dev/null

# --- Check PyTorch separately (platform-dependent) ---
if ! $PYTHON -c "import torch" 2>/dev/null; then
    echo "PyTorch not found. Installing PyTorch (CPU version)..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "All dependencies installed."
echo ""

# --- Parse args ---
ARGS=("${@:1}")
if [ ${#ARGS[@]} -eq 0 ]; then
    ARGS=("--max-tokens" "512")
fi

# --- Launch chat interface ---
echo "Starting chat interface..."
exec $PYTHON -m src.chat_interface "${ARGS[@]}"
