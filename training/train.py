import json
import os
import gc
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from huggingface_hub import hf_hub_download, HfApi, create_repo, snapshot_download
import wandb


MODEL_ID = "codys12/Qwen3-8B-BitNet"
DATASET_REPO = "thetrillioniar/Mythos-5-and-Fabel-5-Class-Model-Outputs"

AVAILABLE_FILES = [
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
    train_file: str = field(
        default="",
        metadata={"help": "Single JSONL file to train on (e.g. within_us_ai_mythos_5k.jsonl). "
                          "Leave empty to train on all files."}
    )
    max_length: int = field(default=4096)
    batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=2e-4)
    num_epochs: int = field(default=1)
    max_steps: int = field(default=-1)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    warmup_ratio: float = field(default=0.03)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    eval_steps: int = field(default=200)
    val_ratio: float = field(default=0.05)
    test_run: bool = field(default=False)
    resume_from_checkpoint: Optional[str] = field(default=None)
    use_wandb: bool = field(default=True)
    wandb_project: str = field(default="trimind-bitnet")
    hub_model_id: Optional[str] = field(
        default="Abdulhadi446/Trimind-v1",
        metadata={"help": "HF repo to push checkpoints to between sessions"}
    )
    push_to_hub: bool = field(default=False)
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


def download_and_format_single_file(
    data_dir: str, fname: str, tokenizer, max_length: int, val_ratio: float
):
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"openaiformat/{fname}",
        local_dir=raw_dir,
        repo_type="dataset",
    )

    import random
    random.seed(42)

    formatted = []
    skipped = 0
    path = raw_dir / fname
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

    name_stem = fname.replace(".jsonl", "")
    print(f"[{name_stem}] Formatted {len(formatted)} examples, skipped {skipped}")

    indices = list(range(len(formatted)))
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    val_idx = set(indices[:n_val])
    train_texts = [t for i, t in enumerate(formatted) if i not in val_idx]
    val_texts = [t for i, t in enumerate(formatted) if i in val_idx]

    proc_dir = Path(data_dir) / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    with open(proc_dir / f"train_{name_stem}.jsonl", "w") as f:
        for t in train_texts:
            f.write(json.dumps({"text": t}) + "\n")
    with open(proc_dir / f"val_{name_stem}.jsonl", "w") as f:
        for t in val_texts:
            f.write(json.dumps({"text": t}) + "\n")

    print(f"[{name_stem}] Saved {len(train_texts)} train + {len(val_texts)} val")
    return name_stem, len(train_texts), len(val_texts)


def find_target_modules(model) -> list[str]:
    name_set = set()
    for name, _ in model.named_modules():
        for suffix in ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]:
            if name.endswith(suffix):
                name_set.add(suffix)
    print(f"Found target modules: {sorted(name_set)}")
    return sorted(name_set) if name_set else [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]


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


def push_intermediate(api: HfApi, repo_id: str, local_dir: str, commit_msg: str):
    print(f"Pushing to {repo_id}: {commit_msg}")
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg,
            ignore_patterns=["*.safetensors", "pytorch_model*"],
        )
    except Exception as e:
        print(f"Push warning (non-fatal): {e}")


def main():
    parser = HfArgumentParser(TrainingConfig)
    config, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if config.test_run:
        print("=" * 60)
        print("TEST RUN MODE: 50 steps, no wandb")
        print("=" * 60)
        config.max_steps = 50
        config.use_wandb = False
        config.push_to_hub = False

    print(f"\n=== Trimind v1 — Single-File Fine-Tune ===")
    print(f"Model: {MODEL_ID}")
    print(f"Output: {config.output_dir}")
    print(f"Max length: {config.max_length}, Batch: {config.batch_size}, "
          f"Grad accum: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Push to hub: {config.push_to_hub} → {config.hub_model_id}")
    if config.resume_from_checkpoint:
        print(f"Resuming from: {config.resume_from_checkpoint}")
    print()

    model, tokenizer = setup_model_and_tokenizer(config)

    # Determine which file(s) to train on
    if config.train_file:
        fnames = [config.train_file]
    else:
        fnames = AVAILABLE_FILES

    # Track completed files in a JSON state file
    state_path = Path(config.output_dir) / ".trimind_state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
    else:
        state = {"completed_files": [], "current_file": None}

    # Download / process all requested files upfront
    file_info = []
    for fname in fnames:
        name_stem = fname.replace(".jsonl", "")
        proc_train = Path(config.data_dir) / "processed" / f"train_{name_stem}.jsonl"
        if proc_train.exists():
            with open(proc_train) as f:
                n_train = sum(1 for _ in f)
            with open(Path(config.data_dir) / "processed" / f"val_{name_stem}.jsonl") as f:
                n_val = sum(1 for _ in f)
            print(f"[{name_stem}] Using cached: {n_train} train, {n_val} val")
        else:
            name_stem, n_train, n_val = download_and_format_single_file(
                config.data_dir, fname, tokenizer, config.max_length, config.val_ratio
            )
        file_info.append((fname, name_stem, n_train, n_val))

    # LoRA adapters are loaded once; we train file-by-file, saving after each
    api = HfApi() if config.push_to_hub else None
    if config.push_to_hub and config.hub_model_id:
        create_repo(config.hub_model_id, exist_ok=True, private=True)

    total_steps_completed = 0

    for fname, name_stem, n_train, n_val in file_info:
        if name_stem in state.get("completed_files", []):
            print(f"\n=== Skipping {name_stem} (already completed) ===")
            continue

        print(f"\n{'='*60}")
        print(f"Training on: {name_stem}")
        print(f"  {n_train} train / {n_val} val examples")
        print(f"{'='*60}")

        train_dataset = TextDataset(
            Path(config.data_dir) / "processed" / f"train_{name_stem}.jsonl",
            tokenizer, config.max_length
        )
        eval_dataset = TextDataset(
            Path(config.data_dir) / "processed" / f"val_{name_stem}.jsonl",
            tokenizer, config.max_length
        )

        # Per-file output dir (all under main output_dir with subdirs)
        file_output_dir = Path(config.output_dir) / name_stem
        file_output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing checkpoints in this file's subdir
        resume_ckpt = config.resume_from_checkpoint
        if resume_ckpt is None and file_output_dir.exists():
            # auto-detect latest checkpoint
            ckpts = sorted(file_output_dir.glob("checkpoint-*"))
            if ckpts:
                resume_ckpt = str(ckpts[-1])
                print(f"Auto-resuming from: {resume_ckpt}")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )

        if config.use_wandb and not config.test_run:
            run_name = f"trimind-v1-{name_stem}-r{config.lora_r}"
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config={
                    "model": MODEL_ID,
                    "dataset_file": fname,
                    "lora_r": config.lora_r,
                    "lora_alpha": config.lora_alpha,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "max_length": config.max_length,
                    "n_train": n_train,
                    "n_val": n_val,
                },
                reinit=True,
            )
        else:
            os.environ["WANDB_DISABLED"] = "true"

        # Number of steps for this file
        steps_per_epoch = max(1, n_train // (config.batch_size * config.gradient_accumulation_steps))
        if config.test_run:
            max_steps = min(50, steps_per_epoch)
        elif config.max_steps > 0:
            max_steps = config.max_steps
        else:
            max_steps = steps_per_epoch * config.num_epochs

        training_args = TrainingArguments(
            output_dir=str(file_output_dir),
            overwrite_output_dir=True,
            max_steps=max_steps,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
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

        print(f"\nStarting training on {name_stem} ({max_steps} steps)...")
        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

        # Save adapters for this file
        adapter_path = Path(config.output_dir) / "adapters" / name_stem
        model.save_pretrained(str(adapter_path))
        print(f"Saved LoRA adapters to {adapter_path}")

        # Push intermediate adapters to Hub (small files only)
        if config.push_to_hub and api and not config.test_run:
            push_intermediate(
                api, config.hub_model_id, str(adapter_path),
                f"feat: adapters after {name_stem} (step {total_steps_completed + max_steps})"
            )

        # Update state
        state["completed_files"].append(name_stem)
        state["current_file"] = name_stem
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        total_steps_completed += max_steps

        metrics = train_result.metrics
        print(f"\n--- Results for {name_stem} ---")
        print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
        print(f"  Eval loss:  {metrics.get('eval_loss', 'N/A')}")
        print(f"  Runtime:    {metrics.get('train_runtime', 'N/A'):.1f}s")
        print(f"  Steps:      {metrics.get('global_step', max_steps)}")

        if config.use_wandb and not config.test_run:
            wandb.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"All requested files completed!")
    print(f"Completed: {state['completed_files']}")
    print(f"LoRA adapters saved in: {Path(config.output_dir) / 'adapters'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
