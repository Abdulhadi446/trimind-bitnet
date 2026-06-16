import json, os, torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "thetrillioniar/gemma-4-12b-bitnet"
CACHE = os.path.join(os.path.dirname(__file__), "..", "models", MODEL.replace("/", "_"))
PROMPTS = ["hi", "What is the capital of France?", "Write a haiku about AI."]

device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(CACHE):
    os.makedirs(CACHE, exist_ok=True)
    import subprocess
    subprocess.run(["hf", "download", MODEL, "--local-dir", CACHE, "--quiet"], check=True)

with open(os.path.join(CACHE, "ternary_packed_info.json")) as f:
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

def _replace_modules(module, path=""):
    for child_name, child in list(module.named_children()):
        full = f"{path}.{child_name}" if path else child_name
        if isinstance(child, torch.nn.Linear) and full in packed_info:
            tlin = TernaryLinear(child.in_features, child.out_features, child.bias is not None)
            setattr(module, child_name, tlin)
        else:
            _replace_modules(child, full)

config = AutoConfig.from_pretrained(CACHE, trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
_replace_modules(model)

sf_path = os.path.join(CACHE, "model.safetensors")
with safe_open(sf_path, framework="pt") as sf:
    for key in sf.keys():
        if key.endswith(".ternary_scale"):
            continue
        if key.endswith(".ternary_packed"):
            base = key[: -len(".ternary_packed")]
            scale_key = base + ".ternary_scale"
            mod = model.get_submodule(base)
            mod.register_buffer("packed", sf.get_tensor(key))
            mod.register_buffer("scale", sf.get_tensor(scale_key).to(torch.bfloat16))
        else:
            t = sf.get_tensor(key)
            *mod_path, param_name = key.split(".")
            mod = model.get_submodule(".".join(mod_path))
            p = mod._parameters.get(param_name)
            if p is not None:
                p.data.copy_(t.to(torch.bfloat16))
            else:
                b = mod._buffers.get(param_name)
                if b is not None:
                    mod._buffers[param_name] = t.to(torch.bfloat16)

model.to(device).eval()

tok = AutoTokenizer.from_pretrained(CACHE, trust_remote_code=True)

for p in PROMPTS:
    out = model.generate(**tok(p, return_tensors="pt").to(device), max_new_tokens=50, do_sample=True, temperature=0.7)
    print(f"\n=== {p} ===")
    print(tok.decode(out[0], skip_special_tokens=True))
