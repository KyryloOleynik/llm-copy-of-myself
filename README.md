<div align="center">

# Personal Telegram AI

### A privacy-first, end-to-end LLM personalization pipeline

Fine-tune **Qwen3-4B** with **4-bit QLoRA** on private Telegram conversations, verify that the adapter preserves context and reasoning, and serve it locally through a production-minded Telegram bot.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA-7C3AED)](https://huggingface.co/docs/peft/)
[![Telegram](https://img.shields.io/badge/Telegram-aiogram-26A5E4?logo=telegram&logoColor=white)](https://docs.aiogram.dev/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Code style](https://img.shields.io/badge/code%20style-Ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-not%20specified-lightgrey)](#license)

**Local-first · Reproducible · Leakage-aware · VRAM-constrained · Evaluation-gated**

</div>

---

## Why this project stands out

This repository is not just a fine-tuning script. It implements the full lifecycle of a local personalization system:

- cleans and sessionizes raw Telegram exports;
- builds a private multilingual hybrid vector/keyword RAG index;
- builds leakage-resistant chronological train, validation, and test splits;
- trains a 4B-parameter model using memory-efficient QLoRA;
- masks prompt tokens so loss is computed only on the target reply;
- runs a worst-case VRAM smoke test before full training;
- records dataset hashes, environment metadata, selected LoRA modules, and peak VRAM;
- compares the adapter against the base model on context retention, reasoning, and instructions;
- performs blind style evaluation before accepting the adapter;
- deploys the accepted adapter through a local Telegram bot with async inference control.

## What this demonstrates

For an internship or junior ML/software role, the project demonstrates practical experience with:

| Area | Evidence in this repository |
|---|---|
| LLM fine-tuning | QLoRA, PEFT, NF4 quantization, BF16 compute, gradient checkpointing |
| Data engineering | Telegram export cleaning, sessionization, deterministic dataset generation |
| ML evaluation | Base-vs-adapter comparisons, style review, context and reasoning checks |
| Reproducibility | Dataset hashes, environment capture, checkpoint validation, fixed seeds |
| GPU optimization | 4-bit loading, bounded sequence length, gradient accumulation, VRAM smoke gate |
| Software engineering | Typed configuration, CLI commands, tests, linting, modular package structure |
| Applied deployment | aiogram bot, SQLite state, bounded context, queued local inference |
| Privacy engineering | Local execution, ignored private artifacts, sanitized public configuration |

## Architecture

```mermaid
flowchart TD
    A[Telegram JSON export] --> B[Cleaning and sessionization]
    B --> C[Relationship-aware conversations]
    C --> D[Dataset builder]
    D --> E[Chronological train / validation / test split]
    D --> F[Supplemental context and reasoning examples]
    E --> G[Manifest, hashes, and statistics]
    F --> G
    G --> H[Worst-case VRAM smoke test]
    H --> I[Qwen3-4B QLoRA training]
    I --> J[Automated capability evaluation]
    J --> K[Blind style comparison]
    K --> L{Acceptance gate}
    L -->|pass| M[Local Telegram bot]
    L -->|fail| N[Reject adapter]
```

## Technology stack

<div align="center">

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="48" alt="Python" />
&nbsp;&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" height="48" alt="PyTorch" />
&nbsp;&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/sqlite/sqlite-original.svg" height="48" alt="SQLite" />
&nbsp;&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" height="48" alt="Git" />
&nbsp;&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="48" alt="GitHub" />

</div>

| Layer | Technologies |
|---|---|
| Language | Python 3.11 |
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| Training | PyTorch, Transformers, TRL, PEFT, Accelerate |
| Quantization | bitsandbytes, 4-bit NF4, double quantization, BF16 |
| Data | Hugging Face Datasets, JSONL, YAML, Jinja2 |
| Validation | Pydantic |
| CLI | Typer |
| Bot | aiogram |
| Storage | SQLite FTS5 plus dense embedding vectors |
| Quality | pytest, Ruff |

## Core engineering decisions

### Memory-efficient fine-tuning

The base model is loaded in 4-bit NF4 with double quantization and BF16 compute. LoRA adapters are attached to attention and MLP projections:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

Default adapter settings:

```yaml
rank: 16
alpha: 32
dropout: 0.05
learning_rate: 0.0002
micro_batch_size: 1
gradient_accumulation_steps: 16
sequence_length: 1024
```

### Leakage-resistant splitting

Complete conversation sessions are split chronologically rather than shuffling individual messages. This prevents fragments from the same conversation from appearing in both training and evaluation data.

### Target-only loss

Prompt tokens are assigned label `-100`, so the model is optimized only on the final owner reply instead of learning to reproduce the entire conversation prompt.

### Capability-preservation data

Personal style examples are supplemented with deterministic context-retention, reasoning, and instruction-following examples to reduce catastrophic behavioral regression.

### Hybrid semantic RAG

Knowledge files and every chat in the raw Telegram export are split into overlapping
chunks and embedded locally with `intfloat/multilingual-e5-small`. The embedding model
is loaded once after the main LLM and remains resident while the bot is running.
Retrieval combines cosine vector similarity with SQLite BM25 keyword ranking, so it
can find semantic matches as well as exact names, dates, and codes.

### Acceptance gate

An adapter is accepted only after it:

1. comes from a complete training run;
2. matches the prepared dataset hash;
3. has verified VRAM metadata;
4. does not regress below the base model on automated checks;
5. reaches the configured blind style preference threshold.

## Quick start

### 1. Prepare private data

```powershell
New-Item -ItemType Directory -Force private_data
Copy-Item .\DataExport_2026-07-17\result.json .\private_data\result.json
Copy-Item .\.env.example .\.env
Copy-Item .\config.example.yaml .\config.yaml
Copy-Item .\private_data\relationships.example.json .\private_data\relationships.json
```

Keep the original export. Before processing, set the correct `owner_from_id` and
paths in `config.yaml`, set `BOT_TOKEN` in `.env`, complete
`private_data/relationships.json`, verify free disk space, and obtain consent for
messages used in training. Put identity/story notes not present in Telegram under
`private_data/knowledge/`. Media files are not required.

### 2. Install and verify CUDA

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows-cuda.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
personal-ai authorize-calendar
```

### 3. Process, smoke-test, and train

```powershell
python clean_telegram_export.py
personal-ai prepare-data
personal-ai build-rag
personal-ai train --smoke --fresh
personal-ai train
personal-ai evaluate
```

Review the cleaning statistics and `data/processed/manifest.json` before training.
Use `personal-ai train --fresh` for a new run or `--resume last` to resume.

## Telegram bot behavior

The bot:

- loads the quantized base model and adapter once;
- loads and keeps the multilingual RAG embedding model resident;
- serializes GPU generation through an async queue;
- keeps bounded per-chat history;
- deterministically routes identity and people questions to personal-memory RAG;
- routes recent activity, plans, events, and availability to Google Calendar;
- routes arithmetic to the calculator, including arithmetic inside mixed questions;
- executes all required tools before asking the model for its styled final answer;
- stores tool calls and results in the same conversation turn for grounded follow-ups;
- includes the current date, weekday, timezone, and week boundaries in live context;
- logs every tool call and returned result to the console;
- batches rapidly arriving messages;
- converts unsupported media into stable text placeholders;
- conditions responses on the selected relationship type;
- stores lightweight state in SQLite;
- refuses an unaccepted adapter unless explicitly overridden.

Tool routing is enforced by live runtime code, not only by the system prompt. For
example, `Расскажи о себе` always searches personal memory, `Что делал на этой
неделе?` reads Calendar events from Monday through the current time, and a mixed
question such as `Где живёшь и сколько будет 2+2?` runs both RAG and the calculator
before producing one answer. This enforcement is currently limited to live chat and
does not alter training examples.

Supported commands:

```text
/start
/style
/delete
/about
```

## Testing and code quality

```powershell
pip install -e .[dev]
pytest
ruff check .
```

The test suite covers dataset determinism, session isolation, prompt truncation, target-only loss masking, checkpoint selection, generation settings, adapter-dataset matching, evaluation behavior, and bot media handling.

## Privacy and responsible use

Telegram exports can contain credentials, phone numbers, locations, private conversations, and information about third parties.

Never commit:

- raw Telegram exports;
- generated datasets;
- adapters or checkpoints;
- evaluation samples;
- SQLite databases;
- `.env` or `config.yaml`;
- Google OAuth credentials or tokens under `private_data/`;
- private relationship mappings.

Obtain appropriate consent before training on another person's messages. The bot should identify itself as an AI representation and should never be presented as the real person.

## License

No open-source license has been selected yet. Until a license is added, the source code remains available for viewing but is not automatically licensed for reuse.
