import json, os, torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-4-12B-it"
MODEL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "gemma-4-12B-it-bitnet"))
HF_REPO = "thetrillioniar/gemma-4-12b-bitnet"
PROMPTS = ["hi", "What is the capital of France?", "Write a haiku about AI."]

# load from local dir, or download from HF
if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
    import subprocess
    os.makedirs(MODEL_DIR, exist_ok=True)
    subprocess.run(["hf", "download", HF_REPO, "--local-dir", MODEL_DIR, "--quiet"], check=True)

with open(os.path.join(MODEL_DIR, "ternary_packed_info.json")) as f:
    packed_info = json.load(f).get("packed_layers", {})

class TernaryLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = torch.nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x):
        idx = torch.stack([(self.packed >> 6) & 3, (self.packed >> 4) & 3,
                           (self.packed >> 2) & 3, self.packed & 3], dim=1).view(-1)
        idx = idx[:self.in_features * self.out_features]
        w = (idx.to(torch.int8).sub_(1)).to(x.dtype) * self.scale
        return torch.nn.functional.linear(x, w.view(self.out_features, self.in_features), self.bias)

config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

def _replace_modules(module, path=""):
    for child_name, child in list(module.named_children()):
        full = f"{path}.{child_name}" if path else child_name
        if isinstance(child, torch.nn.Linear) and full in packed_info:
            tlin = TernaryLinear(child.in_features, child.out_features, child.bias is not None)
            setattr(module, child_name, tlin)
        else:
            _replace_modules(child, full)

_replace_modules(model)

sf_path = os.path.join(MODEL_DIR, "model.safetensors")
with safe_open(sf_path, framework="pt") as sf:
    for key in sf.keys():
        if key.endswith(".ternary_scale"):
            continue
        if key.endswith(".ternary_packed"):
            base = key[: -len(".ternary_packed")]
            mod = model.get_submodule(base)
            mod.register_buffer("packed", sf.get_tensor(key))
            mod.register_buffer("scale", sf.get_tensor(base + ".ternary_scale").to(torch.bfloat16))
        else:
            t = sf.get_tensor(key)
            *mod_path, param_name = key.split(".")
            mod = model.get_submodule(".".join(mod_path))
            if param_name in mod._parameters:
                mod._parameters[param_name] = torch.nn.Parameter(t.to(torch.bfloat16))
            elif param_name in mod._buffers:
                mod._buffers[param_name] = t.to(torch.bfloat16)

import gc
for param in model.parameters():
    if param.device.type == "meta":
        param.data = torch.zeros(param.shape, dtype=torch.bfloat16)
gc.collect()
model.eval()

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

for p in PROMPTS:
    out = model.generate(**tok(p, return_tensors="pt"), max_new_tokens=50, do_sample=True, temperature=0.7)
    print(f"\n=== {p} ===")
    print(tok.decode(out[0], skip_special_tokens=True))
