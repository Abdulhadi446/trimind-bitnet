#!/usr/bin/env python3
"""Load and run Qwen3-8B-1Q (1.58-bit ternary quantized).

Usage:
    python load_ternary.py --prompt "Hello"
    python load_ternary.py --prompt "Write fibonacci" --max-tokens 512
"""
import argparse
import json
import logging
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def pack_ternary(w: torch.Tensor) -> tuple[bytes, float]:
    scale = w.abs().mean().item()
    if scale == 0:
        scale = 1.0
    idx = (w.view(-1) / scale).to(torch.int8).add_(1).clamp_(0, 2)
    n = idx.numel()
    pad = (4 - n % 4) % 4
    if pad:
        idx = torch.cat([idx, torch.zeros(pad, dtype=torch.int8)])
    idx = idx.view(-1, 4).to(torch.uint8)
    packed = (idx[:, 0] << 6) | (idx[:, 1] << 4) | (idx[:, 2] << 2) | idx[:, 3]
    return packed.numpy().tobytes(), scale


def unpack_ternary(packed: bytes, shape: tuple, scale: float, dtype=torch.bfloat16, device="cpu") -> torch.Tensor:
    packed_t = torch.frombuffer(packed, dtype=torch.uint8, device=device)
    idx = torch.stack([
        (packed_t >> 6) & 3,
        (packed_t >> 4) & 3,
        (packed_t >> 2) & 3,
        packed_t & 3,
    ], dim=1).view(-1)
    n = shape[0] * shape[1]
    idx = idx[:n]
    w = (idx.to(torch.int8).sub_(1)).to(dtype) * scale
    return w.view(shape)


def load_ternary_model(model_dir: str, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load a ternary-quantized model from directory with ternary_packed.bin, etc."""
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    with open(os.path.join(model_dir, "ternary_metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(model_dir, "ternary_offsets.json")) as f:
        offsets = json.load(f)
    with open(os.path.join(model_dir, "ternary_packed.bin"), "rb") as f:
        data = f.read()

    model = model.to(device)
    for name, module in model.named_modules():
        key = name + ".weight"
        if key not in metadata:
            continue
        m = metadata[key]
        pos = offsets[key]
        n = m["shape"][0] * m["shape"][1]
        byte_len = (n + 3) // 4
        packed = data[pos:pos + byte_len]
        dtype = getattr(torch, m["dtype"], torch.bfloat16)
        w = unpack_ternary(packed, tuple(m["shape"]), m["scale"], dtype=dtype, device=device)
        module.weight.data = w.to(device)

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return model, processor


@torch.no_grad()
def generate(model, processor, prompt: str, max_tokens=256, device="cuda"):
    inputs = processor(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=processor.eos_token_id,
    )
    return processor.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-8B-1Q ternary model")
    parser.add_argument("--model-dir", default=".", help="Path to model directory")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model from %s on %s ...", args.model_dir, device)
    model, processor = load_ternary_model(args.model_dir, device=device)
    logger.info("Generating...")
    text = generate(model, processor, args.prompt, max_tokens=args.max_tokens, device=device)
    print(text)


if __name__ == "__main__":
    main()
