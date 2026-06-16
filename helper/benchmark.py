import json, os, math, torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "google/gemma-4-12B-it"
MODEL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", MODEL_NAME))
SAVE_RESULTS = os.path.join(MODEL_DIR, "benchmark_results.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = [t for t in dataset["text"] if t.strip()]

config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

import json as _json
from safetensors import safe_open

packed_info_path = os.path.join(MODEL_DIR, "ternary_packed_info.json")
if os.path.exists(packed_info_path):
    with open(packed_info_path) as f:
        packed_info = _json.load(f).get("packed_layers", {})
    state_dict = {}
    with safe_open(os.path.join(MODEL_DIR, "model.safetensors"), framework="pt") as sf:
        for key in sf.keys():
            t = sf.get_tensor(key)
            if key.endswith(".ternary_packed"):
                base = key[: -len(".ternary_packed")]
                meta = packed_info.get(base)
                if meta is None:
                    continue
                scale = sf.get_tensor(base + ".ternary_scale").item()
                shape = meta["shape"]
                idx = torch.stack([(t >> 6) & 3, (t >> 4) & 3, (t >> 2) & 3, t & 3], dim=1).view(-1)
                idx = idx[: shape[0] * shape[1]]
                state_dict[base] = (idx.to(torch.int8).sub_(1)).to(torch.bfloat16) * scale
            elif not key.endswith(".ternary_scale"):
                state_dict[key] = t.to(torch.bfloat16)
    model.load_state_dict(state_dict, strict=False)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True)

model.to(device).eval()

@torch.no_grad()
def compute_ppl(model, tokenizer, texts, stride=512, max_len=2048):
    nlls = []
    for text in tqdm(texts, desc="Computing PPL"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc.input_ids.to(device)
        if input_ids.size(1) <= 1:
            continue
        seq_len = input_ids.size(1)
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + stride, seq_len)
            chunk = input_ids[:, begin:end]
            target_len = end - begin
            logits = model(chunk).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            nlls.append(loss.item() / target_len)
            prev_end = end
    ppl = math.exp(sum(nlls) / len(nlls))
    return ppl

ppl = compute_ppl(model, tokenizer, texts)
results = {"model": MODEL_NAME, "perplexity": round(ppl, 4), "dataset": "wikitext-2", "device": device}
os.makedirs(os.path.dirname(SAVE_RESULTS), exist_ok=True)
with open(SAVE_RESULTS, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
