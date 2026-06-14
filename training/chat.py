"""Chat with fine-tuned Qwen3-8B-BitNet + LoRA adapters."""
import os, sys, torch, warnings
warnings.filterwarnings("ignore", message=".*does not support bfloat16.*")

from transformers import AutoModelForCausalLM, AutoTokenizer


def patch_peft_for_bitnet():
    import torch.nn as nn
    from peft.tuners.lora.model import LoraModel
    from peft.tuners.lora import Linear as LoraLinear

    _orig_create = LoraModel._create_new_module

    def _patched_create(self, lora_config, adapter_name, target, **kwargs):
        if not isinstance(target, nn.Linear) and hasattr(target, 'in_features'):
            return LoraLinear(target, adapter_name, config=lora_config, **kwargs)
        return _orig_create(self, lora_config, adapter_name, target, **kwargs)

    LoraModel._create_new_module = _patched_create
    warnings.filterwarnings("ignore", message="Unsupported layer type")


MODEL_ID = "codys12/Qwen3-8B-BitNet"
MODEL_DIR = os.path.abspath("../model")
adapter_dirs = sorted([
    os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR)
    if os.path.isdir(os.path.join(MODEL_DIR, d))
])

if not adapter_dirs:
    print(f"No adapter directories found in {MODEL_DIR}/")
    sys.exit(1)

print("Available adapters:")
for i, d in enumerate(adapter_dirs, 1):
    print(f"  {i}. {os.path.basename(d)}")

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
    MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)

print(f"Loading LoRA adapters from {adapter_path} ...")
patch_peft_for_bitnet()
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

    print("Generating...", end=" ", flush=True)
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Bot: {reply}\n")
    except RuntimeError as e:
        print(f"Error: {e}")
