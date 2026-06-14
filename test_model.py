import json, os, torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

config = AutoConfig.from_pretrained('.', trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

with open('ternary_packed_info.json') as f:
    packed_info = json.load(f).get('packed_layers', {})

state = {}
with safe_open('model.safetensors', framework='pt') as sf:
    for key in sf.keys():
        t = sf.get_tensor(key)
        if key.endswith('.ternary_packed'):
            base = key[:-16]
            meta = packed_info.get(base)
            if not meta: continue
            scale = sf.get_tensor(base + '.ternary_scale').item()
            idx = torch.stack([(t>>6)&3,(t>>4)&3,(t>>2)&3,t&3], dim=1).view(-1)[:meta['shape'][0]*meta['shape'][1]]
            state[base] = (idx.to(torch.int8).sub_(1)).to(torch.bfloat16) * scale
        elif not key.endswith('.ternary_scale'):
            state[key] = t.to(torch.bfloat16)

model.load_state_dict(state, strict=False)
model.eval()
tok = AutoTokenizer.from_pretrained('.', trust_remote_code=True)
out = model.generate(**tok('hi', return_tensors='pt'), max_new_tokens=50, do_sample=True, temperature=0.7)
print(tok.decode(out[0]))
