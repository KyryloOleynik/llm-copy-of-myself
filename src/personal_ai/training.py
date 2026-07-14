from __future__ import annotations

import platform
import re
import shutil
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from personal_ai.config import AppConfig
from personal_ai.modeling import load_quantized_base, load_tokenizer, select_language_lora_modules
from personal_ai.utils import assistant_target_ids, iter_jsonl, read_json, render_chat_ids, write_json


IGNORE_INDEX = -100


def truncate_prompt_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    prompt_ids: list[int],
    target_ids: list[int],
    max_length: int,
) -> tuple[list[int], int, str]:
    """Fit prompt+target while preserving the target and, when possible, the system prompt."""
    prompt_budget = max_length - len(target_ids)
    if prompt_budget < 1:
        raise ValueError(
            "Assistant target leaves no prompt tokens inside the configured "
            f"{max_length}-token sequence"
        )
    if len(prompt_ids) <= prompt_budget:
        return prompt_ids, 0, "none"

    if messages and messages[0]["role"] == "system":
        system_ids = render_chat_ids(tokenizer, messages[:1], generation=False)
        if prompt_ids[: len(system_ids)] == system_ids and len(system_ids) < prompt_budget:
            recent_budget = prompt_budget - len(system_ids)
            truncated = system_ids + prompt_ids[-recent_budget:]
            return truncated, len(prompt_ids) - len(truncated), "preserve_system_and_recent_tail"

    truncated = prompt_ids[-prompt_budget:]
    return truncated, len(prompt_ids) - len(truncated), "keep_recent_prompt_tail"


def _checkpoint_is_resumable(path: Path) -> bool:
    weight_names = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    )
    return (
        any((path / name).is_file() for name in weight_names)
        and (path / "trainer_state.json").is_file()
        and (path / "optimizer.pt").is_file()
        and (path / "scheduler.pt").is_file()
    )


def _latest_valid_checkpoint(output_dir: Path) -> Path | None:
    """Return the newest complete, resumable Trainer checkpoint."""
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))

    for _, path in sorted(candidates, reverse=True):
        if _checkpoint_is_resumable(path):
            return path
    return None


def _clear_previous_run(output_dir: Path, *, smoke: bool) -> None:
    """Remove stale adapters/checkpoints without deleting the full run's smoke gate."""
    if smoke:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path)
    for name in ("adapter-final", ".interrupt-checkpoint-staging"):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
    metadata = output_dir / "reproducibility.json"
    if metadata.is_file():
        metadata.unlink()


def _require_resume_dataset_match(output_dir: Path, dataset_sha256: str) -> None:
    metadata_path = output_dir / "reproducibility.json"
    if not metadata_path.is_file():
        raise RuntimeError(
            "A checkpoint exists without reproducibility metadata; use --fresh to avoid "
            "mixing training runs"
        )
    previous_hash = read_json(metadata_path).get("dataset_sha256")
    if previous_hash != dataset_sha256:
        raise RuntimeError(
            "Existing checkpoints were trained on a different dataset; rerun with --fresh"
        )


def _save_interrupted_checkpoint(trainer: Any, tokenizer: Any, output_dir: Path) -> Path:
    """Save into staging, verify it, and only then expose checkpoint-N."""
    step = trainer.state.global_step
    staging_root = output_dir / ".interrupt-checkpoint-staging"
    staging_checkpoint = staging_root / f"checkpoint-{step}"
    final_checkpoint = output_dir / f"checkpoint-{step}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    original_output_dir = trainer.args.output_dir
    try:
        trainer.args.output_dir = str(staging_root)
        trainer._save_checkpoint(trainer.model, trial=None)
        tokenizer.save_pretrained(str(staging_checkpoint))
    finally:
        trainer.args.output_dir = original_output_dir

    if not _checkpoint_is_resumable(staging_checkpoint):
        raise RuntimeError(
            f"Checkpoint save was incomplete; files were kept at {staging_checkpoint}"
        )
    if final_checkpoint.exists():
        shutil.rmtree(final_checkpoint)
    staging_checkpoint.replace(final_checkpoint)
    shutil.rmtree(staging_root)
    return final_checkpoint


@dataclass
class ReplyOnlyCollator:
    """Tokenize chat examples and compute loss only on the final assistant reply."""

    tokenizer: Any
    max_length: int
    max_target_tokens: int = 256
    capture_examples: bool = False
    audit: Counter[str] = field(default_factory=Counter)
    captured_examples: list[dict[str, Any]] = field(default_factory=list)
    backward_capture_ids: list[int] = field(default_factory=list)

    def mark_used_for_backward(self, capture_ids: list[int]) -> None:
        """Mark only batches whose Trainer training_step completed backward successfully."""
        for capture_id in capture_ids:
            record = self.captured_examples[capture_id]
            if "backward_order" in record:
                raise RuntimeError(f"Smoke audit capture {capture_id} was used more than once")
            record["backward_order"] = len(self.backward_capture_ids) + 1
            self.backward_capture_ids.append(capture_id)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        encoded_examples = []
        pending_captures: list[dict[str, Any]] = []
        for example in examples:
            messages = example["messages"]
            try:
                prompt_ids, full_ids, target_ids = assistant_target_ids(self.tokenizer, messages)
            except ValueError as exc:
                if "prefix" in str(exc):
                    self.audit["prompt_prefix_mismatch"] += 1
                else:
                    self.audit["zero_label_example"] += 1
                raise
            if len(target_ids) > self.max_target_tokens:
                self.audit["oversized_target"] += 1
                raise ValueError(
                    f"Assistant target has {len(target_ids)} tokens; "
                    f"maximum is {self.max_target_tokens}"
                )
            original_sequence_tokens = len(full_ids)
            truncation_strategy = "none"
            truncated_prompt_tokens = 0
            if len(full_ids) > self.max_length:
                try:
                    prompt_ids, truncated_prompt_tokens, truncation_strategy = (
                        truncate_prompt_tokens(
                            self.tokenizer,
                            messages,
                            prompt_ids,
                            target_ids,
                            self.max_length,
                        )
                    )
                except ValueError:
                    self.audit["target_cannot_fit"] += 1
                    raise
                full_ids = prompt_ids + target_ids
                self.audit["truncated_examples"] += 1
                self.audit["truncated_prompt_tokens"] += truncated_prompt_tokens
            labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
            self.audit["examples"] += 1
            self.audit["target_tokens"] += len(target_ids)
            if self.capture_examples:
                pending_captures.append(
                    {
                        "collation_order": len(self.captured_examples) + len(pending_captures) + 1,
                        "example_id": str(example.get("example_id", "")),
                        "chat_id": str(example.get("chat_id", "")),
                        "session_id": str(example.get("session_id", "")),
                        "source_type": str(example.get("source_type", "")),
                        "relationship": str(example.get("relationship", "")),
                        "split": str(example.get("split", "")),
                        "timestamp": str(example.get("timestamp", "")),
                        "masked_prompt_messages": messages[:-1],
                        "expected_assistant_reply": messages[-1]["content"],
                        "masked_prompt_tokens": len(prompt_ids),
                        "trained_target_tokens": len(target_ids),
                        "total_sequence_tokens": len(full_ids),
                        "original_sequence_tokens": original_sequence_tokens,
                        "prompt_tokens_truncated": truncated_prompt_tokens,
                        "prompt_truncation_strategy": truncation_strategy,
                    }
                )
            encoded_examples.append({"input_ids": full_ids, "labels": labels})

        max_len = max(len(row["input_ids"]) for row in encoded_examples)
        pad_id = self.tokenizer.pad_token_id
        input_ids, labels, attention_mask = [], [], []
        for row in encoded_examples:
            padding = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * padding)
            labels.append(row["labels"] + [IGNORE_INDEX] * padding)
            attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
        capture_ids: list[int] = []
        if self.capture_examples:
            for record, row_input_ids, row_labels, row_attention_mask in zip(
                pending_captures, input_ids, labels, attention_mask, strict=True
            ):
                # Capture the same fully padded arrays returned to Trainer below.
                # Nothing in this record is reconstructed after the smoke run.
                record["padding_tokens"] = row_attention_mask.count(0)
                record["input_ids"] = list(row_input_ids)
                record["training_labels"] = list(row_labels)
                record["attention_mask"] = list(row_attention_mask)
                record["capture_id"] = len(self.captured_examples)
                self.captured_examples.append(record)
                capture_ids.append(record["capture_id"])
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if self.capture_examples:
            batch["smoke_audit_capture_ids"] = torch.tensor(capture_ids, dtype=torch.long)
        return batch


def validate_prepared_dataset(config: AppConfig) -> dict[str, Any]:
    """Fail before training when dataset invariants or privacy acknowledgement drift."""
    manifest_path = config.data.output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Prepared manifest is missing; run personal-ai prepare-data")
    manifest = read_json(manifest_path)
    if manifest.get("model") != config.model.base_model:
        raise RuntimeError("Prepared dataset tokenizer model differs from training model")
    if manifest.get("sequence_length") != config.model.sequence_length:
        raise RuntimeError("Prepared dataset sequence length differs from training configuration")
    if not manifest.get("contains_unredacted_private_data"):
        raise RuntimeError("Unredacted private-data acknowledgement is missing")
    session_splits: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = config.data.output_dir / f"{split}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Prepared split is missing: {path}")
        for line_number, row in enumerate(iter_jsonl(path), 1):
            if row.get("split") != split:
                raise ValueError(f"{path}:{line_number} has the wrong split")
            session_id = row["session_id"]
            previous = session_splits.setdefault(session_id, split)
            if previous != split:
                raise ValueError(f"Session {session_id} crosses {previous}/{split}")
            if row["sequence_tokens"] > config.model.sequence_length:
                raise ValueError(f"{row['example_id']} exceeds the sequence limit")
            if not 0 < row["target_tokens"] <= config.data.max_target_tokens:
                raise ValueError(f"{row['example_id']} has invalid target length")
    return manifest


def require_successful_smoke(config: AppConfig, dataset_sha256: str) -> dict[str, Any]:
    """Block a full run until the same model/dataset passed the RTX VRAM smoke gate."""
    path = config.training.output_dir / "smoke-test.json"
    if not path.is_file():
        raise RuntimeError("Run personal-ai train --smoke --fresh before full training")
    metadata = read_json(path)
    if metadata.get("model") != config.model.base_model:
        raise RuntimeError("Smoke test used a different base model")
    if metadata.get("dataset_sha256") != dataset_sha256:
        raise RuntimeError("Smoke test used a different prepared dataset")
    peak = metadata.get("peak_vram_reserved_bytes")
    if peak is None or peak >= 12 * 1024**3:
        raise RuntimeError("Smoke test did not verify peak reserved VRAM below 12 GiB")
    return metadata


def longest_example_indices(dataset_split: Any, limit: int) -> list[int]:
    """Select the longest prepared examples so a smoke run exercises the VRAM ceiling."""
    lengths = list(dataset_split["sequence_tokens"])
    return sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))[:limit]


def _smoke_pool_record(example: dict[str, Any]) -> dict[str, Any]:
    """Return the readable prompt/target boundary for one selected smoke example."""
    messages = example["messages"]
    return {
        "example_id": str(example.get("example_id", "")),
        "chat_id": str(example.get("chat_id", "")),
        "session_id": str(example.get("session_id", "")),
        "source_type": str(example.get("source_type", "")),
        "relationship": str(example.get("relationship", "")),
        "split": str(example.get("split", "")),
        "timestamp": str(example.get("timestamp", "")),
        "sequence_tokens": int(example["sequence_tokens"]),
        "target_tokens": int(example["target_tokens"]),
        "masked_prompt_messages": messages[:-1],
        "expected_assistant_reply": messages[-1]["content"],
    }


def write_smoke_sample_audit(
    path: Path,
    config: AppConfig,
    manifest: dict[str, Any],
    selected_train: Any,
    selected_validation: Any,
    collator: ReplyOnlyCollator,
    status: str,
) -> None:
    """Write the selected pool and exact examples collated into a smoke training step."""
    used_for_backward = sorted(
        (record for record in collator.captured_examples if "backward_order" in record),
        key=lambda record: record["backward_order"],
    )
    collated_lookahead = [
        record for record in collator.captured_examples if "backward_order" not in record
    ]
    write_json(
        path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "contains_unredacted_private_data": True,
            "warning": "This audit contains private conversation text and must remain private.",
            "model": config.model.base_model,
            "dataset_sha256": manifest["dataset_sha256"],
            "selection_rule": "20 longest prepared examples per available split",
            "training_plan": {
                "optimizer_steps": 1,
                "micro_batch_size": config.training.micro_batch_size,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            },
            "loss_masking": {
                "algorithm_source": (
                    "ReplyOnlyCollator captures the exact tensors returned to Trainer. "
                    "SmokeAuditTrainer marks a capture as actually_used_for_backward only "
                    "after Trainer.training_step completes its backward call successfully."
                ),
                "ignored": (
                    "All tokens rendered from masked_prompt_messages have label -100 and "
                    "do not contribute to loss. If a sequence exceeds the configured limit, "
                    "old prompt context is removed while preserving the system prompt when "
                    "possible and retaining the newest prompt tail."
                ),
                "trained": (
                    "Every token belonging to expected_assistant_reply is preserved, retains "
                    "its token label, and contributes to loss."
                ),
                "verification": (
                    "Training rejects the example unless the rendered prompt is an exact "
                    "token prefix of the complete conversation."
                ),
                "raw_arrays": (
                    "For each actually collated sample, training_labels is the exact label "
                    "array sent to Trainer after batch padding: -100 entries are ignored; "
                    "non-padding entries after masked_prompt_tokens are target token IDs "
                    "and must equal input_ids at the same positions. attention_mask marks "
                    "real tokens with 1 and padding with 0."
                ),
            },
            "selected_training_pool": [
                _smoke_pool_record(example) for example in selected_train
            ],
            "selected_validation_pool_not_trained": [
                _smoke_pool_record(example) for example in selected_validation
            ],
            "actually_used_for_backward": used_for_backward,
            "collated_lookahead_not_used_for_backward": collated_lookahead,
        },
        sort_keys=False,
    )


def prepared_dataset_features() -> Any:
    """Return a stable schema instead of relying on JSON shard type inference."""
    from datasets import Features, List, Value

    return Features(
        {
            "chat_id": Value("string"),
            "example_id": Value("string"),
            "messages": List(
                {
                    "content": Value("string"),
                    "role": Value("string"),
                }
            ),
            "relationship": Value("string"),
            "sequence_tokens": Value("int64"),
            "session_id": Value("string"),
            "source_type": Value("string"),
            "split": Value("string"),
            "target_message_ids": List(Value("int64")),
            "target_tokens": Value("int64"),
            "timestamp": Value("string"),
        }
    )


def training_argument_overrides(smoke: bool) -> dict[str, Any]:
    """Keep smoke bounded and prevent Trainer's large default evaluation batch."""
    return {
        "per_device_eval_batch_size": 1,
        "eval_strategy": "no" if smoke else "steps",
        "save_strategy": "no" if smoke else "steps",
        "load_best_model_at_end": not smoke,
        "prediction_loss_only": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # Keep collation in the main process during smoke training so the audit
        # records the exact examples delivered to the one optimizer step.
        "dataloader_num_workers": 0 if smoke else 2,
        "dataloader_persistent_workers": not smoke,
        "dataloader_prefetch_factor": None if smoke else 4,
    }


def warmup_steps(config: AppConfig, train_examples: int, smoke: bool) -> int:
    """Convert the configured ratio to explicit optimizer steps for Transformers 5."""
    if smoke:
        return 0
    micro_batches = math.ceil(train_examples / config.training.micro_batch_size)
    optimizer_steps = math.ceil(
        micro_batches * config.training.epochs / config.training.gradient_accumulation_steps
    )
    return math.ceil(optimizer_steps * config.training.warmup_ratio)


def train(
    config: AppConfig,
    smoke: bool = False,
    resume: str | None = None,
    fresh: bool = False,
) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError("Training dependencies are missing; install .[train]") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Training is intentionally disabled on CPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a CUDA GPU with BF16 support.")

    manifest = validate_prepared_dataset(config)
    experiment_dir = config.training.output_dir
    if not smoke:
        require_successful_smoke(config, manifest["dataset_sha256"])
    set_seed(config.training.seed)
    output_dir = experiment_dir / "smoke" if smoke else experiment_dir
    if fresh and resume is not None:
        raise ValueError("Use either --fresh or --resume, not both")
    if fresh:
        _clear_previous_run(output_dir, smoke=smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = _latest_valid_checkpoint(output_dir)
    last_checkpoint = str(last_checkpoint_path) if last_checkpoint_path else None
    if resume == "last":
        resume = last_checkpoint
        if resume is None:
            raise RuntimeError(f"No checkpoint found in {output_dir}")
    elif resume is None and not fresh:
        resume = last_checkpoint
        if resume is not None:
            print(f"Automatically resuming from {resume}")
    if resume is not None:
        _require_resume_dataset_match(output_dir, manifest["dataset_sha256"])
    data_files = {
        "train": str(config.data.output_dir / "train.jsonl"),
        "validation": str(config.data.output_dir / "validation.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files, features=prepared_dataset_features())
    if smoke:
        dataset["train"] = dataset["train"].select(longest_example_indices(dataset["train"], 20))
        dataset["validation"] = dataset["validation"].select(
            longest_example_indices(dataset["validation"], 20)
        )

    tokenizer = load_tokenizer(config.model.base_model)
    model = load_quantized_base(config.model.base_model, torch, {"": 0})
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.training.gradient_checkpointing
    )
    target_modules = select_language_lora_modules(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.training.lora_rank,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        ),
    )

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(
        "vision" in name.casefold() or "visual" in name.casefold() for name in trainable_names
    ):
        raise RuntimeError("LoRA trainable parameters are empty or include the vision encoder")

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.micro_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.epochs,
        max_steps=1 if smoke else -1,
        warmup_steps=warmup_steps(config, len(dataset["train"]), smoke),
        lr_scheduler_type=config.training.lr_scheduler_type,
        bf16=True,
        tf32=True,
        gradient_checkpointing=config.training.gradient_checkpointing,
        dataloader_pin_memory=True,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        optim="adamw_torch_fused",
        report_to="none",
        seed=config.training.seed,
        data_seed=config.training.seed,
        remove_unused_columns=False,
        **training_argument_overrides(smoke),
    )
    collator = ReplyOnlyCollator(
        tokenizer,
        config.model.sequence_length,
        config.data.max_target_tokens,
        capture_examples=smoke,
    )
    class SmokeAuditTrainer(Trainer):
        def training_step(
            self,
            training_model: Any,
            inputs: dict[str, Any],
            num_items_in_batch: Any = None,
        ) -> Any:
            capture_ids = inputs.pop("smoke_audit_capture_ids", None)
            loss = super().training_step(training_model, inputs, num_items_in_batch)
            if capture_ids is not None:
                collator.mark_used_for_backward(capture_ids.detach().cpu().tolist())
            return loss

    trainer_class = SmokeAuditTrainer if smoke else Trainer
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=None if smoke else dataset["validation"],
        data_collator=collator,
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "smoke": smoke,
        "seed": config.training.seed,
        "training_method": config.training.method,
        "model": config.model.base_model,
        "dataset_sha256": manifest["dataset_sha256"],
        "lora_target_modules": target_modules,
        "trainable_parameter_names": trainable_names,
    }
    write_json(output_dir / "reproducibility.json", metadata)
    smoke_audit_path = experiment_dir / "smoke-samples.json"
    smoke_status = "prepared"
    if smoke:
        write_smoke_sample_audit(
            smoke_audit_path,
            config,
            manifest,
            dataset["train"],
            dataset["validation"],
            collator,
            smoke_status,
        )
    try:
        torch.cuda.reset_peak_memory_stats()
        trainer.train(resume_from_checkpoint=resume or None)
        smoke_status = "completed"
    except KeyboardInterrupt:
        smoke_status = "interrupted"
        # A normal Trainer save includes the adapter, optimizer, scheduler, RNG,
        # and trainer state, so this checkpoint can be resumed rather than merely
        # used for inference. Ctrl+C may take a moment to reach this handler while
        # a CUDA kernel is finishing.
        checkpoint = _save_interrupted_checkpoint(trainer, tokenizer, output_dir)
        print(
            f"\nInterrupted safely at optimizer step {trainer.state.global_step}: "
            f"{checkpoint}. "
            "Resume with: personal-ai train --resume last"
        )
        return
    except Exception:
        smoke_status = "failed"
        raise
    finally:
        if smoke:
            write_smoke_sample_audit(
                smoke_audit_path,
                config,
                manifest,
                dataset["train"],
                dataset["validation"],
                collator,
                smoke_status,
            )
    metadata["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated()
    metadata["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(output_dir / "reproducibility.json", metadata)
    if smoke:
        write_json(experiment_dir / "smoke-test.json", metadata)
    trainer.save_model(str(output_dir / "adapter-final"))
    tokenizer.save_pretrained(str(output_dir / "adapter-final"))
