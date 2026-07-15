# Personal Telegram AI

A local, privacy-conscious pipeline for fine-tuning a language model on personal Telegram conversations and serving it through a Telegram bot.

The project covers the full ML lifecycle: Telegram export cleaning, leakage-resistant dataset construction, QLoRA training, reproducibility checks, automated evaluation, blind style comparison, and local bot inference.

## Highlights

- Fine-tunes `Qwen/Qwen3-4B-Instruct-2507` with 4-bit QLoRA.
- Trains only on the final assistant reply while masking prompt tokens from the loss.
- Splits complete conversation sessions chronologically to prevent train/test leakage.
- Conditions responses on relationship type, such as friend, family, or professional contact.
- Mixes personal conversations with context-retention, reasoning, and instruction-following examples.
- Runs a worst-case VRAM smoke test before allowing a full training run.
- Records dataset hashes, environment metadata, selected LoRA modules, and peak VRAM usage.
- Compares the adapter with the base model on context, reasoning, instructions, and blind style preference.
- Refuses to launch an unverified adapter unless explicitly confirmed in the terminal.
- Keeps inference local and stores lightweight bot state in SQLite.

## Architecture

```text
Telegram JSON export
        |
        v
clean_telegram_export.py
        |
        v
cleaned, sessionized conversations
        |
        v
personal-ai prepare-data
        |
        +--> train / validation / test JSONL
        +--> deterministic supplemental examples
        +--> manifest with hashes and statistics
        |
        v
personal-ai train --smoke --fresh
        |
        v
personal-ai train
        |
        v
personal-ai evaluate
        |
        +--> automated capability diagnostics
        +--> blind style comparison
        +--> adapter acceptance gate
        |
        v
bot.py
```

## Technology stack

- Python 3.11
- PyTorch and CUDA
- Hugging Face Transformers and Datasets
- PEFT / QLoRA
- bitsandbytes 4-bit NF4 quantization
- Typer CLI
- Pydantic configuration validation
- aiogram Telegram bot
- SQLite local state
- pytest and Ruff

## Repository structure

```text
.
├── bot.py                         # Local Telegram bot and inference queue
├── clean_telegram_export.py       # Export cleaning and sessionization
├── config.example.yaml            # Sanitized pipeline configuration
├── relationships.example.json     # Example relationship mapping
├── requirements-windows-cuda.txt  # Reproducible Windows/CUDA environment
├── WINDOWS_TRAINING.md            # Detailed Windows training guide
├── src/personal_ai/
│   ├── cli.py                     # prepare-data, train, and evaluate commands
│   ├── config.py                  # Strict validated configuration models
│   ├── data.py                    # Dataset construction and split logic
│   ├── evaluation.py              # Capability and style evaluation gates
│   ├── modeling.py                # Quantized loading and generation
│   ├── supplemental.py            # Synthetic capability-preservation data
│   ├── training.py                # QLoRA training and checkpoint safety
│   └── utils.py                   # Shared serialization and chat utilities
└── tests/                         # Dataset, modeling, training, and bot tests
```

## Setup

The training configuration targets native 64-bit Windows with an NVIDIA GPU supporting BF16. The current acceptance workflow is designed around a 12 GiB VRAM budget.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows-cuda.txt
```

Verify CUDA:

```powershell
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

For bot-only installation:

```powershell
pip install -r requirements.txt
```

## Private configuration

Copy the public examples into ignored local files:

```powershell
Copy-Item config.example.yaml config.yaml
Copy-Item relationships.example.json private_data/relationships.json
Copy-Item .env.example .env
```

Update:

- `project.owner_from_id` with the Telegram export identifier belonging to the model owner.
- Data paths and `personal_train_examples` in `config.yaml`.
- The private mapping between Telegram chat names and relationship categories.
- `BOT_TOKEN` in `.env`.

Private data, generated datasets, adapters, checkpoints, evaluation outputs, databases, and local configuration are excluded by `.gitignore`.

## Data preparation

Export Telegram data as machine-readable JSON, place it at `private_data/result.json`, and create a private relationship mapping.

```powershell
python clean_telegram_export.py
personal-ai prepare-data
```

The preparation pipeline:

1. removes service records and Telegram system chats;
2. converts stickers with emoji into textual emoji;
3. splits chats after gaps greater than 12 hours;
4. merges adjacent messages from the same speaker;
5. builds one supervised example per owner reply;
6. preserves the complete target reply while trimming old prompt context when required;
7. splits complete sessions chronologically;
8. adds deterministic capability-preservation examples;
9. writes hashes, distributions, exclusions, and split boundaries to `manifest.json`.

## Training

First run the bounded smoke test on the longest prepared examples:

```powershell
personal-ai train --smoke --fresh
```

A full run is blocked until the smoke test confirms the same model and dataset fit below the configured 12 GiB VRAM gate.

Start or automatically resume training:

```powershell
personal-ai train
```

Start a clean run while retaining the matching smoke-test record:

```powershell
personal-ai train --fresh
```

Resume explicitly:

```powershell
personal-ai train --resume last
```

The default QLoRA setup uses:

- rank 16, alpha 32, dropout 0.05;
- 4-bit NF4 with double quantization;
- BF16 compute;
- attention and MLP projection adapters;
- micro-batch size 1 with 16-step gradient accumulation;
- cosine learning-rate schedule with 3% warmup;
- gradient checkpointing and fused AdamW.

## Evaluation

```powershell
personal-ai evaluate
```

Evaluation compares the base model and available adapter on:

- delayed context recall at multiple token distances;
- corrected-state recall;
- persistent instruction following;
- basic reasoning and multilingual cases;
- held-out personal style examples.

The command creates a blind comparison file. After filling in the preferred outputs, rerun evaluation. An adapter is accepted only when it does not regress below the base model, receives at least a 60% blind style win rate, represents a complete training run, and has verified VRAM metadata.

## Telegram bot

After an adapter passes evaluation:

```powershell
python bot.py
```

The bot:

- loads the quantized base model and LoRA adapter once;
- serializes GPU generation through an async queue;
- maintains bounded per-chat context;
- batches rapidly arriving messages before replying;
- represents unsupported media with stable text markers;
- adapts its system prompt to the selected relationship;
- provides `/start`, `/style`, `/delete`, and `/about` commands.

## Testing and linting

```powershell
pip install -e .[dev]
pytest
ruff check .
```

The test suite covers deterministic dataset creation, session isolation, prompt truncation, target-only loss masking, checkpoint selection, evaluation behavior, generation, and bot media handling.

## Privacy and responsible use

This project is designed for local experimentation with personal data. Telegram exports can contain credentials, private conversations, phone numbers, locations, and information about third parties.

Do not commit or publish raw exports, generated datasets, model adapters, checkpoints, logs, evaluation samples, SQLite databases, `.env`, `config.yaml`, or relationship mappings. Obtain appropriate consent before training on or deploying a model that imitates another person's communication style. The bot identifies itself as an AI representation and should not be presented as the real person.

## Current status

The repository implements an end-to-end experimental pipeline and includes safety gates for data leakage, reproducibility, VRAM limits, checkpoint integrity, capability regression, and style validation. Retrieval configuration is present for future work, while the current bot uses bounded in-memory conversation history.
