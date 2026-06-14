#!/bin/bash
# Convert ternary_packed.bin + json -> model.safetensors
# Usage: ./convert_bin_to_safetensors.sh [model_dir] [hf_repo]
set -e
MODEL_DIR="${1:-.}"
REPO="${2:-thetrillioniar/Qwen3-8B-1Q}"

pip install transformers safetensors huggingface_hub -q

python3 -c "
import json, torch, os
from safetensors.torch import save_file
from transformers import AutoConfig

d = '$MODEL_DIR'
with open(f'{d}/ternary_metadata.json') as f: meta = json.load(f)
with open(f'{d}/ternary_offsets.json') as f: offs = json.load(f)
data = open(f'{d}/ternary_packed.bin', 'rb').read()

sd, pinfo = {}, {}
for key, m in meta.items():
    pos = offs[key]; n = m['shape'][0] * m['shape'][1]
    raw = data[pos : pos + (n + 3) // 4]
    sd[key + '.ternary_packed'] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    sd[key + '.ternary_scale'] = torch.tensor([m['scale']], dtype=torch.float16)
    pinfo[key] = {'shape': m['shape'], 'dtype': m['dtype']}

save_file(sd, f'{d}/model.safetensors')
json.dump({'packed_layers': pinfo, 'version': 1}, open(f'{d}/ternary_packed_info.json', 'w'))
cfg = AutoConfig.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True)
cfg.save_pretrained(d)
print('Converted: model.safetensors + config.json + ternary_packed_info.json')
"

echo "Uploading to https://huggingface.co/$REPO ..."
hf upload "$REPO" "$MODEL_DIR" .
echo "Done."

