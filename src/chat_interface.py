#!/usr/bin/env python3
"""Interactive CLI chat interface for triMind-BitNet quantized models."""

import argparse
import sys
import os
import readline
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.load_model import load_model, MODEL_ID

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("chat")
logging.getLogger("pipeline").setLevel(logging.WARNING)
logging.getLogger("src").setLevel(logging.WARNING)

WELCOME = """
╔══════════════════════════════════════════════╗
║         triMind-BitNet Chat Interface        ║
║         Model: Gemma4-12B-it                   ║
║         Type /help for commands              ║
╚══════════════════════════════════════════════╝
"""

HELP = """
Commands:
  /help       Show this help
  /clear      Clear conversation history
  /reset      Reload model from scratch
  /quit       Exit
  /info       Show model and device info

Just type anything to chat. Press Ctrl+C to interrupt generation.
"""


def main():
    parser = argparse.ArgumentParser(description="triMind-BitNet interactive chat")
    parser.add_argument("--dtype", choices=["bf16", "8bit", "4bit"], default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response")
    args = parser.parse_args()

    print(WELCOME)
    model = processor = device = None

    def load():
        nonlocal model, processor, device
        print(f"Loading {MODEL_ID}...")
        model, processor, device = load_model(
            device_override=args.device, dtype_override=args.dtype
        )
        print(f"Loaded on {device}. Ready.\n")

    try:
        load()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    history = []

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input == "/quit":
            break
        elif user_input == "/help":
            print(HELP)
            continue
        elif user_input == "/clear":
            history.clear()
            print("History cleared.\n")
            continue
        elif user_input == "/reset":
            history.clear()
            print()
            load()
            continue
        elif user_input == "/info":
            print(f"Model: {MODEL_ID}")
            print(f"Device: {device}")
            print(f"History turns: {len(history) // 2}")
            print()
            continue

        history.append({"role": "user", "content": user_input})
        text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(device)

        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            response = processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            print(response.strip())
            history.append({"role": "assistant", "content": response.strip()})
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as e:
            print(f"\n[Error: {e}]")

        print()

    print("Goodbye!")


if __name__ == "__main__":
    main()
