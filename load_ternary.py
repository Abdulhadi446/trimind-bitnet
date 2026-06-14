#!/usr/bin/env python3
"""Load and run a ternary-quantized model (2-bit packed in safetensors).

Usage:
    python load_ternary.py --model-dir /path/to/model --prompt "Hello"
"""
import argparse
import json
import logging
import os
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_ternary_weights(model, model_dir: str, device="cpu"):
    with open(os.path.join(model_dir, "ternary_packed_info.json")) as f:
        info = json.load(f)
    packed_info = info["packed_layers"]

    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as sf:
        for name, module in model.named_modules():
            key = name + ".weight"
            if key not in packed_info:
                continue
            packed = sf.get_tensor(key + ".ternary_packed").to(device)
            scale_t = sf.get_tensor(key + ".ternary_scale").to(device)
            scale = scale_t.item()
            shape = packed_info[key]["shape"]

            idx = torch.stack([
                (packed >> 6) & 3, (packed >> 4) & 3, (packed >> 2) & 3, packed & 3,
            ], dim=1).view(-1)
            n = shape[0] * shape[1]
            idx = idx[:n]
            w = (idx.to(torch.int8).sub_(1)).to(torch.bfloat16) * scale
            module.weight.data = w.view(shape).to(device)

    logger.info("Loaded ternary weights for %d layers.", len(packed_info))


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading from %s on %s ...", args.model_dir, device)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.to(device)
    load_ternary_weights(model, args.model_dir, device=device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    print(generate(model, processor, args.prompt, max_tokens=args.max_tokens, device=device))


if __name__ == "__main__":
    main()
