import json, os, torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "thetrillioniar/Qwen3-8B-1Q"
PROMPTS = ["hi", "What is 2+2?", "Write a haiku about AI."]

os.makedirs("/tmp/qwen", exist_ok=True)
os.system(f"hf download {MODEL} --local-dir /tmp/qwen --quiet 2>/dev/null || huggingface-cli download {MODEL} --local-dir /tmp/qwen --quiet")

config = AutoConfig.from_pretrained("/tmp/qwen", trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

with open("/tmp/qwen/ternary_packed_info.json") as f:
    packed_info = json.load(f).get("packed_layers", {})

state = {}
with safe_open("/tmp/qwen/model.safetensors", framework="pt") as sf:
    for key in sf.keys():
        t = sf.get_tensor(key)
        if key.endswith(".ternary_packed"):
            base = key[:-16]
            meta = packed_info.get(base)
            if not meta: continue
            scale = sf.get_tensor(base + ".ternary_scale").item()
            idx = torch.stack([(t>>6)&3,(t>>4)&3,(t>>2)&3,t&3], dim=1).view(-1)
            idx = idx[:meta["shape"][0]*meta["shape"][1]]
            state[base] = (idx.to(torch.int8).sub_(1)).to(torch.bfloat16) * scale
        elif not key.endswith(".ternary_scale"):
            state[key] = t.to(torch.bfloat16)

model.load_state_dict(state, strict=False)
model.eval()
tok = AutoTokenizer.from_pretrained("/tmp/qwen", trust_remote_code=True)

for p in PROMPTS:
    out = model.generate(**tok(p, return_tensors="pt"), max_new_tokens=50, do_sample=True, temperature=0.7)
    print(f"\n=== {p} ===")
    print(tok.decode(out[0]))
