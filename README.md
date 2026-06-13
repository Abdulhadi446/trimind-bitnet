# triMind-BitNet

**Qwen3-VL-8B → 1.58-bit Ternary Quantization Pipeline**

Compress the [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) multimodal model to BitNet-style ternary weights (`{-1, 0, +1}`) using post-training quantization (PTQ). This reduces each parameter from 16 bits to ~1.58 bits, drastically cutting memory while preserving most of the model's capabilities.

## How It Works

1. **Hardware auto-detection** — Detects TPU (torch-xla), CUDA GPU, or CPU and selects optimal precision (BF16, 8-bit, or 4-bit).
2. **Model loading** — Loads Qwen3-VL-8B-Instruct with the chosen precision.
3. **Ternary quantization** — Replaces all `nn.Linear` layers with `TernaryLinear`: weights are clamped to `{-α, 0, +α}` via absmean scaling. This is a *generic* transformation that works on any Hugging Face model (Qwen, Gemma, Llama, etc.).
4. **Inference test** — Runs text prompts, multi-turn Q&A, and optional vision inference through the quantized model.
5. **Save** — Optionally persists the quantized weights.

## Quick Start

```bash
# One-command setup + chat interface
./start.sh

# Or with custom max tokens
./start.sh --max-tokens 1024
```

## Setup

### One-command (auto-installs everything)

```bash
git clone https://github.com/Abdulhadi446/trimind-bitnet.git && cd trimind-bitnet && ./start.sh
```

### Google Colab — GPU (T4, A100, etc.)

```python
!git clone https://github.com/Abdulhadi446/trimind-bitnet.git
%cd trimind-bitnet
!pip install -q -r requirements.txt
!pip install -q -U bitsandbytes>=0.46.1
!python src/run_pipeline.py
```

### Google Colab — TPU (v5e, v5p)

```python
!git clone https://github.com/Abdulhadi446/trimind-bitnet.git
%cd trimind-bitnet
!pip install -r requirements.txt
# Install torch-xla TPU wheels:
!pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
!python src/run_pipeline.py
```

On TPU, `hardware_utils.py` detects torch-xla and uses BF16 automatically.

## Usage

```bash
# Full pipeline with automatic hardware detection
python src/run_pipeline.py

# Force 4-bit base precision (for low-memory GPUs)
python src/run_pipeline.py --dtype 4bit

# Save quantized model to ./output/
python src/run_pipeline.py --save

# Test with a vision image
python src/run_pipeline.py --image /path/to/photo.jpg

# Override device
python src/run_pipeline.py --device cpu
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── start.sh               — One-click venv setup + chat launch
├── requirements.txt
├── .gitignore
└── src/
    ├── hardware_utils.py   — Device/dtype auto-detection + memory logging
    ├── load_model.py       — Load Qwen3-VL-8B-Instruct with auto-config
    ├── quantize.py         — Generic TernaryLinear for any nn.Linear layer
    ├── test_inference.py   — Text, multi-turn, and vision test prompts
    ├── run_pipeline.py     — Main entry point: load → quantize → test → save
    └── chat_interface.py   — Interactive CLI chat interface
```

## Known Limitations

- **PTQ vs. QAT** — This is post-training quantization (PTQ), not quantization-aware training (QAT). BitNet papers achieve best results with training from scratch or QAT fine-tuning. Expect some quality degradation, especially on complex reasoning.
- **Vision encoder** — The vision encoder's Linear layers are also quantized; for best results, consider excluding the vision tower with a custom `exclude_names` set.
- **No backward pass** — `TernaryLinear` supports forward inference only. For QAT, extend the class with a custom autograd `Function` that uses a straight-through estimator.

## License & Attribution

This project is licensed under **Apache 2.0**.

The base model [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) is also Apache 2.0. Full attribution:
- **Qwen Team** — *Qwen3-VL-8B-Instruct* (2026). https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

BitNet 1.58-bit quantization concept from:
- Wang et al., *BitNet: Scaling 1-bit Transformers for Large Language Models* (2023)
- Ma et al., *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (2024)
