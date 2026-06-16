import os, sys, subprocess

MODEL_NAME = "google/gemma-4-12B-it"
HF_BF16_REPO = MODEL_NAME
SAVE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "gemma-4-12B-it-bitnet"))
BITNET_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "BitNet"))

GGUF_FILE = os.path.join(SAVE_DIR, "ggml-model-i2_s.gguf")
if os.path.exists(GGUF_FILE):
    print(f"Quantized model exists at {GGUF_FILE}, skipping.")
    sys.exit(0)

os.makedirs(SAVE_DIR, exist_ok=True)

def run(cmd, **kw):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kw)

# ── 1. Clone BitNet framework ──────────────────────────────────────────────
if not os.path.exists(BITNET_DIR):
    run("git clone --recursive https://github.com/microsoft/BitNet.git", cwd=os.path.dirname(BITNET_DIR))

# ── 2. Install deps ────────────────────────────────────────────────────────
run(f"pip install -r {BITNET_DIR}/requirements.txt -q")

# ── 3. Download raw bf16 model ─────────────────────────────────────────────
BF16_DIR = os.path.join(SAVE_DIR, "bf16")
if not os.path.exists(os.path.join(BF16_DIR, "config.json")):
    os.makedirs(BF16_DIR, exist_ok=True)
    run(f"hf download {HF_BF16_REPO} --local-dir {BF16_DIR} --quiet")

# ── 4. Convert to GGUF via official BitNet script ──────────────────────────
run(f"python {BITNET_DIR}/utils/convert-helper-bitnet.py {BF16_DIR}")

# ── 5. Setup env ───────────────────────────────────────────────────────────
run(f"python {BITNET_DIR}/setup_env.py -md {BF16_DIR} -q i2_s")

# ── 6. Copy GGUF to SAVE_DIR ───────────────────────────────────────────────
import glob, shutil
for f in glob.glob(os.path.join(BF16_DIR, "ggml-*.gguf")):
    shutil.copy(f, SAVE_DIR)
    print(f"Copied {f} -> {SAVE_DIR}")

print(f"Done. GGUF model at {GGUF_FILE}")
