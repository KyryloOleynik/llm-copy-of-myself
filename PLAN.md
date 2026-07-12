# QLoRA Dataset Repair and Qwen3.5 Migration

## Goal

Build a corrected personal-data baseline on `Qwen/Qwen3.5-4B` before adding retrieval,
Calendar access, GGUF export, or wider Telegram deployment. Test 4,096-token training
first on the Windows RTX 5070; fall back to 3,072, 2,048, then 1,024 only if VRAM requires it.

## Data pipeline

- Keep the raw Telegram export and all sensitive data, including credentials, unchanged
  and tracked in this private repository. Mark every manifest as containing unredacted
  private data; datasets, adapters, checkpoints, logs, and indexes must remain private.
- Use only `personal_chat` records and only owner messages as assistant targets.
- Preserve non-text media as typed placeholders and exclude a target when its immediately
  preceding user turn is media-only.
- Split complete sessions chronologically within each chat before example generation:
  - 10+ sessions: 80% train, 10% validation, remainder test.
  - 3–9 sessions: all but the newest two train, then one validation and one test.
  - Fewer than 3 sessions: train only.
- Add the shared relationship system prompt, remove leading assistant turns, and store the
  session ID plus actual target timestamp.
- Use the Qwen3.5 tokenizer to keep every complete example within 4,096 tokens. Targets
  may use at most 256 tokens; skip only an oversized target and remove oldest complete
  context turns when necessary.
- With seed 42, cap training examples at 1,000 per chat and identical targets of three
  tokens or fewer at 25 per relationship and split.
- Mix each split as 70% personal Telegram examples, 15% deterministic context-retention
  conversations, and 15% deterministic general-reasoning problems. Generate validation
  and test supplements from disjoint value ranges so the capabilities affect checkpoint
  selection without leaking exact training answers.
- Produce the combined tracked `dataset.json`, train/validation/test JSONL files, and a
  manifest containing hashes, counts, exclusions, split boundaries, token distributions,
  relationship distributions, and chat dominance.

## Training pipeline

- Use `Qwen/Qwen3.5-4B` in non-thinking mode with the exact pinned Transformers commit.
- Load the official multimodal model in NF4 with double quantization and BF16 compute, but
  freeze the vision encoder and select LoRA only on language token mixers:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`,
  `in_proj_b`, and `out_proj`.
- Fail before training when expected projections are absent, a vision parameter is
  trainable, prepared artifacts do not match the configuration, or a session crosses splits.
- Train rank 16, alpha 32, dropout 0.05, learning rate `3e-5`, cosine schedule, 3% warmup,
  micro-batch 1, accumulation 16, one epoch, gradient checkpointing, and paged AdamW 8-bit.
- Evaluate and save every 50 optimizer steps, retain the lowest-validation-loss checkpoint,
  and write into `artifacts/training/qwen3.5-4b-r16`.
- `train --smoke` uses the 20 longest train/validation examples, performs one optimizer step,
  and records peak allocated and reserved VRAM. A full run is forbidden until the smoke
  test succeeds below 12 GiB on the RTX 5070.

## Loss and evaluation

- Render prompt and full conversation separately, require an exact token prefix, and train
  only on the final assistant target. Never right-truncate a complete example.
- Reject empty targets, oversized targets, sequence overflow, and template-prefix mismatch.
- Compare the base model, every checkpoint, and `adapter-final` on delayed recall,
  corrected state, persistent instructions, general reasoning, multilingual questions,
  relationship conditioning, and reply style.
- Test 256/512/768/1K/2K/3K/4K training distances and 2K/4K/8K inference distances.
- Accept an adapter only when context and reasoning regress by no more than five percentage
  points versus base, verified training VRAM is below 12 GiB, and blind style review wins
  at least 60%. Keep the personal-data-only baseline until evaluation proves a need for a
  separately licensed general-reasoning replay set.

## Inference gate

- Do not migrate the Telegram bot to a new adapter until `personal-ai evaluate` marks it
  accepted.
- Training and inference use the same system prompt, chat template, relationship code, and
  `enable_thinking=False` behavior.
- Budget live prompts to 8K tokens by dropping oldest complete turns first.
- Prefer text-only serving for later deployment so the frozen vision encoder does not use
  serving memory.

## Required commands

```powershell
personal-ai prepare-data
personal-ai train --smoke --fresh
personal-ai train --fresh
personal-ai evaluate
```

Human reviewers complete `data/processed/evaluation/blind_style_review.json`, then record
adapter win totals in `data/processed/style_ratings.json` and rerun evaluation.

## Acceptance checklist

- Dataset regeneration is deterministic with seed 42.
- No session ID exists in more than one split.
- All examples begin with system then user and end with a non-empty assistant target.
- No sequence or target exceeds its configured token budget.
- No LoRA parameter belongs to the vision encoder.
- The RTX 5070 smoke run completes below 12 GiB without CPU offload.
- An adapter passes automatic context/reasoning gates and the 60% blind style gate.
