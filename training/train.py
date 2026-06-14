import json
import os
import math
import time
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from accelerate import Accelerator
from huggingface_hub import hf_hub_download, snapshot_download
import wandb


MODEL_ID = "codys12/Qwen3-8B-BitNet"
DATASET_REPO = "thetrillioniar/Mythos-5-and-Fabel-5-Class-Model-Outputs"
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


@dataclass
class TrainingConfig:
    output_dir: str = field(default="./trimind-v1-output")
    data_dir: str = field(default="./data")
    max_length: int = field(default=4096)
    batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=2e-4)
    num_epochs: int = field(default=1)
    max_steps: int = field(default=-1)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    eval_steps: int = field(default=200)
    val_ratio: float = field(default=0.05)
    test_run: bool = field(default=False)
    resume_from_checkpoint: Optional[str] = field(default=None)
    use_wandb: bool = field(default=True)
    wandb_project: str = field(default="trimind-bitnet")
    seed: int = field(default=42)


class TextDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                self.examples.append(ex["text"])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask", [1] * len(enc["input_ids"])),
            "labels": enc["input_ids"].copy(),
        }


def download_data(data_dir: str):
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname in OPENAI_FORMAT_FILES:
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=f"openaiformat/{fname}",
            local_dir=raw_dir,
            repo_type="dataset",
        )
    return raw_dir


def format_data(raw_dir: Path, tokenizer, max_length: int, val_ratio: float):
    import random
    random.seed(42)

    formatted = []
    skipped = 0
    for fname in OPENAI_FORMAT_FILES:
        path = raw_dir / fname
        if not path.exists():
            continue
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
                formatted.append(text)

    print(f"Formatted {len(formatted)} examples, skipped {skipped}")
    indices = list(range(len(formatted)))
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    val_idx = set(indices[:n_val])
    train_texts = [t for i, t in enumerate(formatted) if i not in val_idx]
    val_texts = [t for i, t in enumerate(formatted) if i in val_idx]

    proc_dir = Path(data_dir) / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    with open(proc_dir / "train.jsonl", "w") as f:
        for t in train_texts:
            f.write(json.dumps({"text": t}) + "\n")
    with open(proc_dir / "val.jsonl", "w") as f:
        for t in val_texts:
            f.write(json.dumps({"text": t}) + "\n")
    print(f"Saved {len(train_texts)} train + {len(val_texts)} val to {proc_dir}/")
    return proc_dir


def find_target_modules(model) -> list[str]:
    name_set = set()
    for name, _ in model.named_modules():
        for suffix in ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]:
            if name.endswith(suffix):
                name_set.add(suffix)
    print(f"Found target modules: {sorted(name_set)}")
    return sorted(name_set) if name_set else ["q_proj", "k_proj", "v_proj", "o_proj",
                                                "gate_proj", "up_proj", "down_proj"]


def setup_model_and_tokenizer(config: TrainingConfig):
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    target_modules = find_target_modules(model)

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = HfArgumentParser(TrainingConfig)
    config = parser.parse_args_into_dataclasses()[0]

    if config.test_run:
        print("=" * 60)
        print("TEST RUN MODE: 50 steps, no wandb, small subset")
        print("=" * 60)
        config.max_steps = 50
        config.use_wandb = False
        config.save_steps = 1000

    print(f"\n=== Trimind v1 Fine-Tuning ===")
    print(f"Model: {MODEL_ID}")
    print(f"Output: {config.output_dir}")
    print(f"Max length: {config.max_length}")
    print(f"Batch size: {config.batch_size}")
    print(f"Grad accum: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Test run: {config.test_run}")
    print()

    model, tokenizer = setup_model_and_tokenizer(config)

    print("\nPreparing dataset...")
    raw_dir = download_data(config.data_dir)
    proc_dir = format_data(raw_dir, tokenizer, config.max_length, config.val_ratio)

    train_dataset = TextDataset(
        proc_dir / "train.jsonl", tokenizer, config.max_length
    )
    eval_dataset = TextDataset(
        proc_dir / "val.jsonl", tokenizer, config.max_length
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    if config.use_wandb and not config.test_run:
        wandb.init(
            project=config.wandb_project,
            name=f"trimind-v1-lora-r{config.lora_r}",
            config={
                "model": MODEL_ID,
                "dataset": DATASET_REPO,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "max_length": config.max_length,
            },
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs if config.max_steps <= 0 else 1e9,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        seed=config.seed,
        report_to="wandb" if (config.use_wandb and not config.test_run) else "none",
        remove_unused_columns=True,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    print(f"\nSaving final model to {config.output_dir}...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metrics = train_result.metrics
    print(f"\n=== Training Complete ===")
    print(f"Train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Runtime: {metrics.get('train_runtime', 'N/A')}s")
    print(f"Samples/sec: {metrics.get('train_samples_per_second', 'N/A')}")

    if config.use_wandb and not config.test_run:
        wandb.finish()


if __name__ == "__main__":
    main()
