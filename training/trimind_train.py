"""
Trimind v1 — Fine-tune Qwen3-8B-BitNet on one dataset file at a time.
Usage:  python trimind_train.py
"""

import json, os, sys, gc, subprocess, importlib

MODEL_ID = "codys12/Qwen3-8B-BitNet"
DATASET_REPO = "thetrillioniar/Mythos-5-and-Fabel-5-Class-Model-Outputs"

FILES = [
    "within_us_ai_mythos_5k.jsonl",
    "within_us_ai_mythos_25k.jsonl",
    "armand0e_fable_5.jsonl",
    "victor_fable_worldcup.jsonl",
    "roman_sonnet46.jsonl",
    "roman_gemini31_code.jsonl",
    "within_us_ai_opus48_5k.jsonl",
    "norquinal_evol_210k.jsonl",
    "norquinal_evol_250k.jsonl",
]


def ensure_deps():
    missing = []
    for pkg in ["torch", "transformers", "accelerate", "peft", "huggingface_hub"]:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing deps: {', '.join(missing)} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + missing +
            ["bitsandbytes", "datasets", "safetensors"]
        )
        print("Done.\n")
    try:
        import torchao
        from packaging import version
        if version.parse(torchao.__version__) < version.parse("0.16.0"):
            print(f"Upgrading torchao {torchao.__version__} -> >=0.16.0 ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao>=0.16.0"])
    except ImportError:
        pass


def patch_peft_for_bitnet():
    """Monkey-patch PEFT so LoRA works with BitLinear layers."""
    import torch.nn as nn
    from peft.tuners.lora.model import LoraModel
    from peft.tuners.lora import Linear as LoraLinear

    _orig_create = LoraModel._create_new_module

    def _patched_create(self, lora_config, adapter_name, target, **kwargs):
        if not isinstance(target, nn.Linear) and hasattr(target, 'in_features'):
            return LoraLinear(target, adapter_name, config=lora_config, **kwargs)
        return _orig_create(self, lora_config, adapter_name, target, **kwargs)

    LoraModel._create_new_module = _patched_create
    import warnings
    warnings.filterwarnings("ignore", message="Unsupported layer type")
    print("PEFT patched for BitLinear support.")


def pick_file():
    print("\nAvailable dataset files:\n")
    for i, f in enumerate(FILES, 1):
        print(f"  {i}. {f}")
    while True:
        try:
            choice = input(f"\nPick file to train on (1-{len(FILES)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(FILES):
                return FILES[idx]
        except ValueError:
            pass
        print(f"Enter a number 1-{len(FILES)}.")


def download_and_format(fname, tokenizer, max_length=4096, val_split=0.05):
    from huggingface_hub import hf_hub_download

    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"openaiformat/{fname}",
        local_dir=raw_dir,
        repo_type="dataset",
    )

    import random
    random.seed(42)

    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = record.get("messages", [])
            if not msgs:
                continue
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer(text, truncation=True, max_length=max_length)
            if len(tokens["input_ids"]) < 10:
                continue
            texts.append(text)

    random.shuffle(texts)
    n_val = max(1, int(len(texts) * val_split))
    print(f"  Loaded {len(texts)} examples ({n_val} for validation)")
    return texts[n_val:], texts[:n_val]


def main():
    ensure_deps()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset

    patch_peft_for_bitnet()

    class TextDataset(Dataset):
        def __init__(self, texts, tok, max_len):
            self.texts = texts
            self.tok = tok
            self.max_len = max_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tok(self.texts[idx], truncation=True, max_length=self.max_len, padding=False)
            return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask", [1]*len(enc["input_ids"])), "labels": enc["input_ids"].copy()}

    fname = pick_file()
    stem = fname.replace(".jsonl", "")

    print(f"\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model (this downloads ~3 GB on first run) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    target_modules = []
    for n, _ in model.named_modules():
        for s in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]:
            if n.endswith(s):
                target_modules.append(s)
    target_modules = sorted(set(target_modules))

    print(f"Applying LoRA (r=16, alpha=32) to {target_modules} ...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=target_modules, bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"\nDownloading & formatting {fname} ...")
    train_texts, val_texts = download_and_format(fname, tokenizer)

    train_ds = TextDataset(train_texts, tokenizer, 4096)
    val_ds = TextDataset(val_texts, tokenizer, 4096)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    out_dir = f"trimind-v1-{stem}"
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=50,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds,
        eval_dataset=val_ds, data_collator=collator,
    )

    print(f"\nStarting training on {fname} ...\n")
    result = trainer.train()

    adapter_dir = f"adapters-{stem}"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"\nDone! LoRA adapters saved to: {adapter_dir}/")
    print(f"Final train loss: {result.metrics.get('train_loss', 'N/A')}")

    if input("\nPush to Hub? (y/n): ").strip().lower() == "y":
        from huggingface_hub import HfApi, create_repo
        repo = "Abdulhadi446/Trimind-v1"
        create_repo(repo, exist_ok=True, private=True)
        HfApi().upload_folder(
            folder_path=adapter_dir, repo_id=repo, repo_type="model",
            path_in_repo=f"adapters/{stem}",
            commit_message=f"feat: LoRA adapters for {stem}",
        )
        print(f"Uploaded to https://huggingface.co/{repo}/tree/main/adapters/{stem}")


if __name__ == "__main__":
    main()
