#!/usr/bin/env python3
"""Load and run Qwen3-8B-1Q (ternary-quantized, safetensors format).

Usage:
    python load_ternary.py --prompt "Hello"
    python load_ternary.py --prompt "Write fibonacci" --max-tokens 512
"""
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading from %s on %s ...", args.model_dir, device)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    logger.info("Generating...")
    print(generate(model, processor, args.prompt, max_tokens=args.max_tokens, device=device))


if __name__ == "__main__":
    main()
