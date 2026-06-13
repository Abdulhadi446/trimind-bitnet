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

# --- Virtual environment (skip if venv unavailable, e.g. Colab) ---
VENV_DIR="$REPO_DIR/.venv"
USE_VENV=true
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/3] Attempting virtual environment..."
    $PYTHON -m venv "$VENV_DIR" 2>/dev/null || { echo "  (venv failed — installing directly)"; USE_VENV=false; }
else
    echo "[2/3] Virtual environment exists."
fi

if [ "$USE_VENV" = true ]; then
    source "$VENV_DIR/bin/activate"
    PIP="$VENV_DIR/bin/pip"
else
    PIP="pip3"
fi

# --- Install dependencies ---
echo "[3/3] Installing dependencies..."
$PIP install -q --upgrade pip setuptools wheel 2>/dev/null || true
$PIP install -q -r "$REPO_DIR/requirements.txt" 2>/dev/null || true

# --- Install bitsandbytes on GPU systems ---
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo "GPU detected — ensuring bitsandbytes..."
    $PIP install -q -U bitsandbytes>=0.46.1 2>/dev/null || true
fi

# --- Check PyTorch separately (platform-dependent) ---
if ! $PYTHON -c "import torch" 2>/dev/null; then
    echo "PyTorch not found. Installing..."
    $PIP install -q torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true
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
