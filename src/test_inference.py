import logging
import torch
from PIL import Image

logger = logging.getLogger(__name__)

TEXT_PROMPTS = [
    "Hello, how are you?",
    "Write a Python function that reverses a linked list.",
    "Explain what's wrong with this code: def factorial(n): return n * factorial(n-1)",
]

MULTI_TURN_PROMPTS = [
    ("What is the capital of France?", None),
    ("And what is its population?", None),
    ("Tell me a fun fact about that city.", None),
]

VISION_PROMPT = (
    "Describe this image in detail."
)


def run_text_inference(model, processor, device, max_new_tokens: int = 256):
    """Run text-only prompts through the model."""
    logger.info("Running text inference...")

    for i, prompt in enumerate(TEXT_PROMPTS):
        logger.info("Prompt %d: %s", i + 1, prompt)
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n[Prompt {i+1}] {prompt}\n{'-'*40}\n{response}\n")


def run_multi_turn_inference(model, processor, device, max_new_tokens: int = 256):
    """Run multi-turn conversation."""
    logger.info("Running multi-turn inference...")
    history = []

    for turn_idx, (user_msg, _) in enumerate(MULTI_TURN_PROMPTS):
        logger.info("Turn %d: %s", turn_idx + 1, user_msg)
        history.append({"role": "user", "content": user_msg})
        text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        history.append({"role": "assistant", "content": response})
        print(f"\n[Turn {turn_idx+1}] User: {user_msg}\nAssistant: {response}\n")

    return history


def run_vision_inference(model, processor, device, image_path: str | None = None):
    """Run vision-language inference if an image is provided."""
    if image_path is None:
        logger.warning(
            "No image path provided for vision inference. "
            "To test vision, pass --image /path/to/image.jpg to run_pipeline.py."
        )
        print("\n[Vision] Skipped — no image provided. Pass --image <path> to test.")
        return

    logger.info("Running vision inference with image: %s", image_path)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error("Failed to load image: %s", e)
        return

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[Vision] Prompt: {VISION_PROMPT}\n{'-'*40}\n{response}\n")
