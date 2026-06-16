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


def load_quantized(model, model_dir: str, device="cpu"):
    """Load all weights from safetensors, unpacking ternary layers."""
    with open(os.path.join(model_dir, "ternary_packed_info.json")) as f:
        packed_info = json.load(f).get("packed_layers", {})

    state_dict = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as sf:
        for key in sf.keys():
            tensor = sf.get_tensor(key)

            # If it's a packed ternary key, unpack it
            if key.endswith(".ternary_packed"):
                base_key = key[: -len(".ternary_packed")]
                meta = packed_info.get(base_key)
                if meta is None:
                    continue
                scale_key = base_key + ".ternary_scale"
                scale_t = sf.get_tensor(scale_key)
                scale = scale_t.item()
                shape = meta["shape"]

                idx = torch.stack([
                    (tensor >> 6) & 3, (tensor >> 4) & 3,
                    (tensor >> 2) & 3, tensor & 3,
                ], dim=1).view(-1)
                n = shape[0] * shape[1]
                idx = idx[:n]
                w = (idx.to(torch.int8).sub_(1)).to(torch.bfloat16) * scale
                state_dict[base_key] = w.view(shape)
            elif not key.endswith(".ternary_scale"):
                # Non-quantized weight (embed, norm, bias, lm_head)
                state_dict[key] = tensor.to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys: %d", len(unexpected))
    model.to(device)
    logger.info("Loaded %d tensors (%d packed).", len(state_dict), len(packed_info))
    return model


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
    load_quantized(model, args.model_dir, device=device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    print(generate(model, processor, args.prompt, max_tokens=args.max_tokens, device=device))


if __name__ == "__main__":
    main()
