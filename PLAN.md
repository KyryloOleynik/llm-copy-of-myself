# QLoRA Dataset Repair and Qwen3-4B-Instruct-2507 Migration

## Goal

Build a corrected personal-data baseline on `Qwen/Qwen3-4B-Instruct-2507` before adding
retrieval, Calendar access, GGUF export, or wider Telegram deployment. Train at the
validated 1,024-token limit on the Windows RTX 5070.

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
- Keep as much preceding personal-chat context as fits. Remove only the oldest complete
  turns when the Qwen3 tokenizer reaches the 1,024-token hard limit; targets retain a
  separate 256-token limit. The context-retention replay set supplements these real chats.
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

- Use the text-only `Qwen/Qwen3-4B-Instruct-2507` checkpoint with the exact pinned
  Transformers commit. This checkpoint is permanently non-thinking.
- Use QLoRA, not full-precision LoRA: the official `4B` name describes parameter count,
  while the source checkpoint is BF16. Quantize the frozen source weights to NF4 only when
  loading them for training, then train the LoRA adapter weights in higher precision. Revisit
  ordinary LoRA only if a measured 4,096-token smoke run proves the BF16 base fits below the
  12 GiB limit with sufficient headroom.
- Load the causal language model in NF4 with double quantization and BF16 compute. Apply
  LoRA to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- Fail before training when expected projections are absent, prepared artifacts do not
  match the configuration, or a session crosses splits.
- Train rank 16, alpha 32, dropout 0.05, learning rate `2e-4`, cosine schedule, 3% warmup,
  micro-batch 1, accumulation 16, one epoch, gradient checkpointing, and fused AdamW.
- Evaluate and save every 50 optimizer steps, retain the lowest-validation-loss checkpoint,
  and write into `artifacts/training/qwen3-4b-instruct-2507-r16`.
- `train --smoke` uses the 20 longest train/validation examples, performs one optimizer step,
  and records peak allocated and reserved VRAM. A full run is forbidden until the smoke
  test succeeds below 12 GiB on the RTX 5070.

## Loss and evaluation

- Render prompt and full conversation separately, require an exact token prefix, and train
  only on the final assistant target. Never right-truncate a complete example.
- Reject empty or oversized targets and template-prefix mismatches. When a complete
  sequence exceeds 1,024 tokens, remove old prompt context while preserving the entire
  final assistant target.
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
- Training and inference use the same system prompt, chat template, and relationship code.
- Budget live prompts to 8K tokens by dropping oldest complete turns first.
- Keep the same text-only causal-LM loader for training, evaluation, and bot inference.

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
- Every LoRA parameter belongs to a Qwen3 attention or MLP projection.
- The RTX 5070 smoke run completes below 12 GiB without CPU offload.
- An adapter passes automatic context/reasoning gates and the 60% blind style gate.
