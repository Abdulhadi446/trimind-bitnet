import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

DATASET_REPO = "thetrillioniar/Mythos-5-and-Fabel-5-Class-Model-Outputs"
MODEL_ID = "codys12/Qwen3-8B-BitNet"
OPENAI_FORMAT_FILES = [
    "armand0e_fable_5.jsonl",
    "norquinal_evol_210k.jsonl",
    "norquinal_evol_250k.jsonl",
    "roman_gemini31_code.jsonl",
    "roman_sonnet46.jsonl",
    "victor_fable_worldcup.jsonl",
    "within_us_ai_mythos_25k.jsonl",
    "within_us_ai_mythos_5k.jsonl",
    "within_us_ai_opus48_5k.jsonl",
]


def download_raw_jsonsl(data_dir: str):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fname in OPENAI_FORMAT_FILES:
        path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=f"openaiformat/{fname}",
            local_dir=data_dir / "raw",
            repo_type="dataset",
        )
        paths.append(path)
    return paths


def load_and_format(raw_paths: list[str], tokenizer, max_length: int = 4096):
    formatted = []
    skipped = 0
    for path in raw_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                messages = record.get("messages", [])
                if not messages:
                    skipped += 1
                    continue

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                tokens = tokenizer(text, truncation=True, max_length=max_length)
                if len(tokens["input_ids"]) < 10:
                    skipped += 1
                    continue
                formatted.append({"text": text, "len": len(tokens["input_ids"])})
    print(f"Formatted {len(formatted)} examples, skipped {skipped}")
    return formatted


def save_formatted(examples: list[dict], path: str):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def train_val_split(examples, val_ratio=0.05, seed=42):
    import random
    random.seed(seed)
    indices = list(range(len(examples)))
    random.shuffle(indices)
    n_val = int(len(indices) * val_ratio)
    val_idx = set(indices[:n_val])
    train = [ex for i, ex in enumerate(examples) if i not in val_idx]
    val = [ex for i, ex in enumerate(examples) if i in val_idx]
    return train, val


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    print("Downloading raw JSONL files...")
    raw_paths = download_raw_jsonsl(args.data_dir)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Formatting with chat template...")
    examples = load_and_format(raw_paths, tokenizer, max_length=args.max_length)

    train, val = train_val_split(examples, val_ratio=args.val_ratio)
    save_dir = Path(args.data_dir) / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_formatted(train, str(save_dir / "train.jsonl"))
    save_formatted(val, str(save_dir / "val.jsonl"))
    print(f"Saved {len(train)} train + {len(val)} val examples to {save_dir}/")
    print(f"Done! Ready for training.")
