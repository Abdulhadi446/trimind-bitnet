#!/bin/bash
# Upload model.safetensors to HF bypassing gitignore filters
# Usage: ./upload.sh [dir] [repo]
DIR="${1:-.}"
REPO="${2:-thetrillioniar/Qwen3-8B-1Q}"

python3 -c "
from huggingface_hub import HfApi
import os, sys
d = '$DIR'
r = '$REPO'
api = HfApi()
for f in os.listdir(d):
    fp = os.path.join(d, f)
    if os.path.isfile(fp) and f.endswith('.safetensors'):
        print(f'Uploading {f} ({os.path.getsize(fp)/1e9:.2f} GB)...')
        api.upload_file(path_or_fileobj=fp, path_in_repo=f, repo_id=r)
        print('Done.')
"
