"""Train on all 9 dataset files combined (one run, no reloads)."""
import json, os, sys, gc, subprocess, importlib

MODEL_ID = "codys12/Qwen3-8B-BitNet"
DATASET_REPO = "thetrillioniar/Mythos-5-and-Fabel-5-Class-Model-Outputs"
MAX_PER_FILE = 500  # cap each file to keep total manageable (~6h on T4)
MAX_LEN = 1024

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
    try:
        import torchao
        from packaging import version
        if version.parse(torchao.__version__) < version.parse("0.16.0"):
            print(f"Upgrading torchao {torchao.__version__} -> >=0.16.0 ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao>=0.16.0"])
    except ImportError:
        pass


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
    import warnings
    warnings.filterwarnings("ignore", message="Unsupported layer type")
    import transformers.trainer as _tr
    _tr.validate_quantization_for_training = lambda model: None
    print("PEFT patched for BitLinear support.")


def load_all_data(tokenizer, max_per_file=MAX_PER_FILE):
    from huggingface_hub import hf_hub_download
    import random
    random.seed(42)

    all_texts = []
    for fname in FILES:
        path = hf_hub_download(
            repo_id=DATASET_REPO, filename=f"openaiformat/{fname}",
            local_dir="data/raw", repo_type="dataset",
        )
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
                tokens = tokenizer(text, truncation=True, max_length=MAX_LEN)
                if len(tokens["input_ids"]) < 10:
                    continue
                texts.append(text)

        random.shuffle(texts)
        texts = texts[:max_per_file]
        print(f"  {fname}: {len(texts)} examples")
        all_texts.extend(texts)

    random.shuffle(all_texts)
    print(f"\n  Total: {len(all_texts)} examples across {len(FILES)} files")
    return all_texts


def main():
    ensure_deps()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset

    patch_peft_for_bitnet()

    class TextDataset(Dataset):
        def __init__(self, texts, tok, max_len):
            self.texts = texts
            self.tok = tok
            self.max_len = max_len
            self._cached = [None] * len(texts)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            if self._cached[idx] is not None:
                return self._cached[idx]
            enc = self.tok(self.texts[idx], truncation=True, max_length=self.max_len,
                           padding="max_length", return_tensors="pt")
            item = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0),
            }
            self._cached[idx] = item
            return item

    print(f"\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading & combining all datasets ...")
    all_texts = load_all_data(tokenizer)

    print(f"\nLoading model (this downloads ~3 GB on first run) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto", trust_remote_code=True,
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

    train_ds = TextDataset(all_texts, tokenizer, MAX_LEN)

    out_dir = "trimind-v1-all"
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=50,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        ddp_find_unused_parameters=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds,
    )

    print(f"\nStarting training on all {len(all_texts)} examples ...\n")
    result = trainer.train()

    adapter_dir = os.path.abspath(f"../model/trimind-v1-all")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"\nDone! LoRA adapters saved to: {adapter_dir}/")
    print(f"Final train loss: {result.metrics.get('train_loss', 'N/A')}")


if __name__ == "__main__":
    main()
