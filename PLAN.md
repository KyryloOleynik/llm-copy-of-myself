# Personal Telegram AI Bot Project Plan

## Summary

Build a disclosed AI version of you as a separate Telegram bot, running on a native Windows RTX 5070 system. Train `Qwen/Qwen3-8B` using reply-only QLoRA on the cleaned Telegram export, then serve a merged 4-bit model locally.

The bot will:

- Automatically reply in your learned communication style.
- Admit that it is an AI modeled after you.
- Require manual approval of each new Telegram user.
- Use relationship-category conditioning.
- Retrieve both style examples and facts from sanitized historical conversations.
- Read Google Calendar event details and introduce relevant titles/times proactively.
- Never interact with or send messages through your personal Telegram account.

## Implementation

### 1. Project foundation

- Create a Python 3.11 project configured for native Windows, PowerShell, CUDA 12.8-compatible PyTorch, and locked dependencies.
- Use a `src/` package with commands for dataset preparation, training, evaluation, indexing, model serving, and bot execution.
- Store non-secret configuration in one validated YAML file.

### 2. Dataset and retrieval pipeline

- Parse `DataExport_2026-07-10/result.json` without modifying the source export.
- Keep personal chats; exclude Saved Messages, service records, bots, channels, empty media records, forwards without original conversational value, and duplicate/copied content.
- Treat only messages from `user624349412` as training targets. Other participants’ messages may appear only as context.
- Merge consecutive messages from the same sender into conversational turns while retaining emoji, slang, spelling, and Ukrainian/Russian/English code-switching.
- Replace direct identities with stable pseudonymous chat IDs and manually reviewed relationship tags such as `close_friend`, `friend`, `family`, `professional`, and `acquaintance`.
- Redact secrets, authentication codes, payment data, addresses, private links, phone numbers, and third-party sensitive data. Retain non-secret personal facts about you because the selected product intentionally learns facts from conversations.
- Create assistant-only-loss chat examples using recent turns plus relationship metadata, with your next turn as the sole completion.
- Split chronologically within each conversation: oldest 80% training, next 10% validation, newest 10% test. Deduplicate before splitting.
- Build a local retrieval index from sanitized historical turns:
  - Store text, chat pseudonym, relationship tag, timestamp, and source message IDs in SQLite.
  - Generate multilingual embeddings locally and store vectors for NumPy cosine search, avoiding a Windows-specific vector database dependency.
  - Retrieve both factual history and stylistically similar replies, restricted to sanitized records.
  - Never place raw Telegram contact data or secrets in the index.

### 3. QLoRA training and evaluation

- Train the post-trained `Qwen/Qwen3-8B` checkpoint using Transformers, PEFT, TRL, and bitsandbytes on native Windows.
- Load the frozen base in 4-bit NF4 with double quantization and BF16 compute; train LoRA adapters on:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `gate_proj`, `up_proj`, `down_proj`
- Initial run:
  - Sequence length: 1024
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
  - Learning rate: `1e-4`
  - Micro-batch: 1
  - Gradient accumulation: 16
  - Epochs: 1
  - Gradient checkpointing enabled
  - Thinking disabled in prompts and evaluation
- Save checkpoints, adapter weights, tokenizer, config, dataset manifest, redaction version, and reproducibility metadata.
- Compare three systems on the untouched chronological test set:
  - Base Qwen3-8B with persona prompt.
  - Base model with prompt and historical retrieval.
  - QLoRA adapter with retrieval.
- Measure reply appropriateness, style preference, language choice, length distribution, emoji/punctuation behavior, verbatim overlap, canary extraction, PII leakage, and assistant-like verbosity.
- Accept the adapter only if it wins at least 60% of blind style comparisons against prompt-plus-retrieval, does not reduce appropriateness, and passes all secret/canary extraction tests.
- If rank 16 underfits, run rank 32 with otherwise identical settings. Do not add epochs or DPO until this controlled comparison is complete.
- Do not use DPO in v1. Preference tuning becomes a later task only after collecting at least 500 genuine human-ranked comparisons.

### 4. Local model, Telegram bot, and Calendar

- Merge the accepted adapter into Qwen3-8B, convert it to GGUF, and quantize to `Q4_K_M`.
- Serve it through `llama-server` on localhost with an OpenAI-compatible API, 8K maximum context, and conservative concurrency for 12 GB VRAM.
- Implement the bot using Python and Telegram Bot API long polling, avoiding a public webhook requirement on Windows.
- Maintain state in SQLite:
  - Telegram user ID and approval status.
  - Relationship tag.
  - Recent bot conversation.
  - Calendar/RAG tool audit entries.
  - Generation metadata, without storing hidden reasoning.
- Access control:
  - Unknown users receive a disclosed AI introduction and a pending-approval response.
  - The owner receives an approval request.
  - Owner commands approve, reject, revoke, list, pause, resume, and inspect users.
  - Only approved IDs reach the model.
- Prompt assembly order:
  - AI disclosure and behavioral rules.
  - Relationship tag.
  - Sanitized learned persona instructions.
  - Retrieved historical facts and style examples.
  - Relevant Calendar results.
  - Recent live bot conversation.
  - Current user message.
- Connect Google Calendar with desktop OAuth and read-only Calendar scope.
- Calendar tools may read event titles and times and may be invoked proactively when the model considers them relevant, as explicitly selected.
- Calendar credentials remain encrypted or OS-protected locally; tokens, attendees, descriptions, meeting links, and authentication details never enter training data or logs.
- The bot may reveal titles and times to approved users, but must not expose attendee contact data, private links, notes, or credentials.
- Add timeouts and fallbacks:
  - If retrieval fails, answer without historical memory.
  - If Calendar fails, state that schedule information is temporarily unavailable.
  - If model generation fails, send a brief neutral error.
  - If the bot is paused, do not generate responses.
  - Truncate context by dropping the least relevant retrieval results first, then oldest live turns.

## Interfaces and Commands

- CLI commands:
  - `prepare-data` — build sanitized train/validation/test files and audit reports.
  - `build-index` — create the local historical retrieval index.
  - `train` — run QLoRA and resume checkpoints.
  - `evaluate` — run blind comparison and privacy suites.
  - `export-model` — merge, convert, and quantize the accepted adapter.
  - `run-bot` — start model client, Telegram polling, retrieval, and Calendar integration.
- Required environment variables:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_OWNER_ID`
  - Google OAuth client configuration path
  - Local model server URL
- Bot owner commands:
  - `/approve <user_id> <relationship_tag>`
  - `/reject <user_id>`
  - `/revoke <user_id>`
  - `/users`
  - `/pause`
  - `/resume`
  - `/status`
- Dataset artifacts must include a machine-readable manifest with source hash, filters, redaction version, split boundaries, example counts, and excluded-record counts.

## Test Plan

- Dataset tests:
  - Owner messages are the only completion targets.
  - No service records or disallowed chat types survive.
  - Chronological splits do not overlap.
  - PII and secret fixtures are redacted.
  - Consecutive-message grouping and multilingual text remain intact.
- Training tests:
  - A 20-example smoke run completes on the RTX 5070.
  - Loss masking excludes prompts and incoming messages.
  - Checkpoint resume produces consistent steps.
  - Peak VRAM remains below the 12 GB limit without CPU offload.
- Retrieval tests:
  - Results respect sanitization and relationship metadata.
  - Deleted or excluded records never appear.
  - Prompt construction stays inside the context budget.
- Bot tests:
  - Unknown users cannot query the model or Calendar.
  - Approval, rejection, revocation, pause, and restart persist correctly.
  - Duplicate Telegram updates do not cause duplicate replies.
  - AI disclosure is shown on first contact and available through `/about`.
- Calendar tests:
  - OAuth uses read-only scope.
  - Titles and times can appear for approved users.
  - Attendees, links, notes, tokens, and credentials are always suppressed.
  - Calendar outages degrade gracefully.
- Acceptance test:
  - Run a private pilot with approved users.
  - Confirm that the bot clearly identifies itself as AI.
  - Achieve the evaluation thresholds above.
  - Record pilot feedback and all remaining work in the project plan.

## Assumptions and Explicit Tradeoffs

- Training and deployment run on a native Windows RTX 5070 system; Mac packaging is out of scope.
- The bot is separate from the personal Telegram account and sends responses automatically to approved users.
- Anyone can find the bot link, but only manually approved Telegram IDs can use the model.
- Historical retrieval intentionally includes facts and style, not style alone.
- The adapter intentionally learns non-secret facts from conversations; this increases memorization risk and makes privacy testing mandatory.
- Calendar titles and times may be revealed proactively to approved users. This is a deliberate privacy tradeoff.
- V1 is text-only. Voice, images, groups, autonomous scheduling, calendar writes, and personal-account automation are out of scope.
