#!/usr/bin/env python3
"""
triMind-BitNet: Gemma4-12B → 1.58-bit Ternary Quantization Pipeline.

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
from src.quantize import quantize_inplace
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
MODEL_SAVE_NAME = "gemma4-12b-ternary"


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
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save the quantized model (default: ./output).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 50)
    logger.info("triMind-BitNet Pipeline")
    logger.info("Model: google/gemma-4-12B-it → 1.58-bit Ternary")
    logger.info("=" * 50)

    # Step 1: Load
    logger.info("Step 1/4: Loading base model...")
    model, processor, device = load_model(
        device_override=args.device, dtype_override=args.dtype
    )

    # Step 2: Quantize
    logger.info("Step 2/4: Applying ternary quantization to Linear layers...")
    n_replaced = quantize_inplace(
        model, exclude_names={"lm_head"}
    )
    logger.info("Quantized %d Linear layers in-place.", n_replaced)
    log_memory("after_quantize", device)

    # Step 3: Save (before inference — save even if inference crashes)
    if args.save:
        logger.info("Step 3/4: Saving quantized model...")
        save_dir = os.path.join(args.output_dir, MODEL_SAVE_NAME)
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, safe_serialization=True)
        processor.save_pretrained(save_dir)
        logger.info("Quantized model saved to: %s", save_dir)
    else:
        logger.info("Step 3/4: Skipped (use --save to persist).")

    # Step 4: Test (may fail on quantized model — that's expected)
    logger.info("Step 4/4: Running test inference...")
    run_text_inference(model, processor, device)
    run_multi_turn_inference(model, processor, device)

    if args.image:
        run_vision_inference(model, processor, device, image_path=args.image)
    else:
        logger.info("Skipping vision test (no --image provided).")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
