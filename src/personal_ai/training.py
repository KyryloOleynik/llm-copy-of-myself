from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from personal_ai.config import AppConfig


IGNORE_INDEX = -100
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return value


def _split_name(timestamp: str, boundaries: tuple[str, str]) -> str:
    if timestamp <= boundaries[0]:
        return "train"
    if timestamp <= boundaries[1]:
        return "validation"
    return "test"


def prepare_existing_dataset(config: AppConfig) -> dict[str, Any]:
    """Chronologically split examples per chat and write JSONL plus a manifest."""
    source = config.data.dataset
    examples = _read_json(source)
    by_chat: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        by_chat.setdefault(example["chat_id"], []).append(example)

    splits = {"train": [], "validation": [], "test": []}
    split_boundaries: dict[str, dict[str, str | None]] = {}
    for chat_id, rows in by_chat.items():
        rows.sort(key=lambda row: (row["timestamp"], row["example_id"]))
        count = len(rows)
        train_end = max(1, int(count * config.data.train_ratio))
        validation_end = max(train_end, int(count * (config.data.train_ratio + config.data.validation_ratio)))
        validation_end = min(validation_end, count)
        for index, row in enumerate(rows):
            split = "train" if index < train_end else "validation" if index < validation_end else "test"
            splits[split].append(row)
        split_boundaries[chat_id] = {
            "train_end": rows[train_end - 1]["timestamp"] if train_end else None,
            "validation_end": rows[validation_end - 1]["timestamp"] if validation_end > train_end else None,
        }

    output = config.data.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": digest,
        "redaction_version": "not-yet-implemented",
        "counts": {name: len(rows) for name, rows in splits.items()},
        "split_method": "chronological within chat",
        "split_boundaries": split_boundaries,
        "warning": "Existing dataset is not yet fully sanitized or deduplicated; review before real training.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@dataclass
class ReplyOnlyCollator:
    """Tokenize chat examples and compute loss only on the final assistant reply."""

    tokenizer: Any
    max_length: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        encoded_examples = []
        for example in examples:
            messages = example["messages"]
            if not messages or messages[-1]["role"] != "assistant":
                raise ValueError("Every example must end with the target assistant message")
            prompt_ids = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )
            full_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                enable_thinking=False,
            )
            full_ids = full_ids[: self.max_length]
            prompt_length = min(len(prompt_ids), len(full_ids))
            labels = [IGNORE_INDEX] * prompt_length + full_ids[prompt_length:]
            encoded_examples.append({"input_ids": full_ids, "labels": labels})

        max_len = max(len(row["input_ids"]) for row in encoded_examples)
        pad_id = self.tokenizer.pad_token_id
        input_ids, labels, attention_mask = [], [], []
        for row in encoded_examples:
            padding = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * padding)
            labels.append(row["labels"] + [IGNORE_INDEX] * padding)
            attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def train(config: AppConfig, smoke: bool = False, resume: str | None = None) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            Trainer, TrainingArguments, set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("Training dependencies are missing; install .[train]") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Training is intentionally disabled on CPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a CUDA GPU with BF16 support.")

    set_seed(config.training.seed)
    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_files = {
        "train": str(config.data.output_dir / "train.jsonl"),
        "validation": str(config.data.output_dir / "validation.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)
    if smoke:
        dataset["train"] = dataset["train"].select(range(min(20, len(dataset["train"]))))
        if len(dataset["validation"]):
            dataset["validation"] = dataset["validation"].select(range(min(20, len(dataset["validation"]))))

    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.training.gradient_checkpointing
    )
    model.add_adapter(LoraConfig(
        r=config.training.lora_rank,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    ))

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.micro_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.epochs,
        bf16=True,
        tf32=True,
        gradient_checkpointing=config.training.gradient_checkpointing,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_strategy="steps" if len(dataset["validation"]) else "no",
        eval_steps=config.training.save_steps,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=config.training.seed,
        data_seed=config.training.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if len(dataset["validation"]) else None,
        data_collator=ReplyOnlyCollator(tokenizer, config.model.sequence_length),
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "smoke": smoke,
        "seed": config.training.seed,
    }
    (output_dir / "reproducibility.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(output_dir / "adapter-final"))
    tokenizer.save_pretrained(str(output_dir / "adapter-final"))
