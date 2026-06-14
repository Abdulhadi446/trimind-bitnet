"""Chat with fine-tuned Qwen3-8B-BitNet + LoRA adapters."""
import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "codys12/Qwen3-8B-BitNet"

# Auto-detect latest adapter dir
adapter_dirs = sorted([d for d in os.listdir(".") if d.startswith("adapters-")])
if not adapter_dirs:
    print("No adapter directories found (adapters-*).")
    sys.exit(1)

print("Available adapters:")
for i, d in enumerate(adapter_dirs, 1):
    print(f"  {i}. {d}")

while True:
    try:
        choice = int(input(f"\nPick (1-{len(adapter_dirs)}): "))
        if 1 <= choice <= len(adapter_dirs):
            break
    except ValueError:
        pass
    print("Invalid choice.")

adapter_path = adapter_dirs[choice - 1]

print(f"\nLoading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading base model (this downloads ~3 GB on first run) ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto", trust_remote_code=True,
)

print(f"Loading LoRA adapters from {adapter_path} ...")
from peft import PeftModel
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

print(f"\n=== Chat with {adapter_path} (type 'quit' to exit) ===\n")
while True:
    prompt = input("You: ").strip()
    if prompt.lower() in ("quit", "exit", "q"):
        break

    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Bot: {reply}\n")
