#!/usr/bin/env python3
"""
triMind-BitNet: Qwen3.6-35B-A3B → 1.58-bit Ternary Quantization Pipeline.

Usage:
    python src/run_pipeline.py                          # full pipeline
    python src/run_pipeline.py --dtype 4bit              # force 4-bit base
    python src/run_pipeline.py --image /path/to/img.jpg  # include vision test
    python src/run_pipeline.py --save                    # save quantized model
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.load_model import load_model
from src.quantize import replace_linear_with_ternary, revert_ternary_to_linear
from src.test_inference import (
    run_text_inference,
    run_multi_turn_inference,
    run_vision_inference,
)
from src.hardware_utils import log_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
MODEL_SAVE_NAME = "qwen3.6-35b-a3b-ternary"


def parse_args():
    parser = argparse.ArgumentParser(description="triMind-BitNet quantization pipeline")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "8bit", "4bit"],
        default=None,
        help="Override automatic precision selection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file for vision inference test.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the quantized model to disk.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 50)
    logger.info("triMind-BitNet Pipeline")
    logger.info("Model: Qwen/Qwen3.6-35B-A3B → 1.58-bit Ternary")
    logger.info("=" * 50)

    # Step 1: Load
    logger.info("Step 1/4: Loading base model...")
    model, processor, device = load_model(
        device_override=args.device, dtype_override=args.dtype
    )

    # Step 2: Quantize
    logger.info("Step 2/4: Applying ternary quantization to Linear layers...")
    n_replaced = replace_linear_with_ternary(
        model, exclude_names={"lm_head"}
    )
    logger.info("Replaced %d Linear layers with TernaryLinear.", n_replaced)
    log_memory("after_quantize", device)

    # Step 3: Test
    logger.info("Step 3/4: Running test inference...")
    run_text_inference(model, processor, device)
    run_multi_turn_inference(model, processor, device)

    if args.image:
        run_vision_inference(model, processor, device, image_path=args.image)
    else:
        logger.info("Skipping vision test (no --image provided).")

    # Step 4: Save (optional)
    if args.save:
        logger.info("Step 4/4: Saving quantized model...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, MODEL_SAVE_NAME)
        revert_ternary_to_linear(model)
        model.save_pretrained(save_path, safe_serialization=True)
        processor.save_pretrained(save_path)
        logger.info("Quantized model saved to: %s", save_path)
    else:
        logger.info("Step 4/4: Skipped (use --save to persist).")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
