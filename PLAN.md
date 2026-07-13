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
- Add the shared relationship system prompt, preserve leading owner/assistant turns when
  they precede later user context, and store the session ID plus actual target timestamp.
- Keep at most the eight most recent personal-chat context messages, matching the earlier
  style-focused Qwen3-8B run. Use the separate context-retention replay set for longer
  dependencies. The Qwen3.5 tokenizer still enforces the 1,024-token hard limit and a
  256-token target limit.
- With seed 42, retain exactly 5,779 personal training examples and cap identical targets
  of three tokens or fewer at 25 per relationship and split. When the global quota removes
  examples, remove them from dominant chats first instead of sampling every chat equally.
- Mix each split as 90% personal Telegram examples, 8% unique context-retention
  conversations, and 2% unique general-reasoning problems written in the owner's concise
  chat style. Synthetic conversations and their final targets must both be unique; generate
  validation and test supplements from disjoint value ranges.
- Preserve media placeholders only in user context; never train an assistant target to
  emit `[sent image]`, `[sent video]`, or another unavailable media action.
- Produce the combined tracked `dataset.json`, train/validation/test JSONL files, and a
  manifest containing hashes, counts, exclusions, split boundaries, token distributions,
  relationship distributions, and chat dominance.

## Training pipeline

- Use `Qwen/Qwen3.5-4B` in non-thinking mode with the exact pinned Transformers commit.
- Use QLoRA, not full-precision LoRA: the official `4B` name describes parameter count,
  while the source checkpoint is BF16. Quantize the frozen source weights to NF4 only when
  loading them for training, then train the LoRA adapter weights in higher precision. Revisit
  ordinary LoRA only if a measured 4,096-token smoke run proves the BF16 base fits below the
  12 GiB limit with sufficient headroom.
- Load the official multimodal model in NF4 with double quantization and BF16 compute, but
  freeze the vision encoder and select LoRA only on language token mixers:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`,
  `in_proj_b`, and `out_proj`.
- Fail before training when expected projections are absent, a vision parameter is
  trainable, prepared artifacts do not match the configuration, or a session crosses splits.
- Train rank 16, alpha 32, dropout 0.05, learning rate `1e-4`, cosine schedule, 3% warmup,
  micro-batch 1, accumulation 16, one epoch, gradient checkpointing, and fused AdamW.
- Evaluate and save every 50 optimizer steps, retain the lowest-validation-loss checkpoint,
  and write into `artifacts/training/qwen3.5-4b-r16`.
- `train --smoke` uses the 20 longest train/validation examples, performs one optimizer step,
  and records peak allocated and reserved VRAM. A full run is forbidden until the smoke
  test succeeds below 12 GiB on the RTX 5070.

## Loss and evaluation

- Render prompt and full conversation separately, require an exact token prefix, and train
  only on the final assistant target. Never right-truncate a complete example.
- Reject empty targets, oversized targets, sequence overflow, and template-prefix mismatch.
- Compare the base model with `adapter-final`, or the Trainer-selected lowest-validation-loss
  checkpoint while training is incomplete, on delayed recall,
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
- Keep the official multimodal loader at inference so vision can be added later; this
  text-personality adapter changes language layers only and leaves the vision encoder intact.

## Required commands

```powershell
personal-ai prepare-data
personal-ai train --smoke --fresh
personal-ai train --fresh
personal-ai evaluate
```

Human reviewers complete `data/processed/evaluation/blind_style_review.json` and rerun
evaluation. Choices are retained and counted only when the generated comparison is unchanged.

## Acceptance checklist

- Dataset regeneration is deterministic with seed 42.
- No session ID exists in more than one split.
- All examples begin with system, contain at least one user turn, preserve available leading
  owner context, and end with a non-empty assistant target.
- No sequence or target exceeds its configured token budget.
- No LoRA parameter belongs to the vision encoder.
- The RTX 5070 smoke run completes below 12 GiB without CPU offload.
- An adapter passes automatic context/reasoning gates and the 60% blind style gate.
