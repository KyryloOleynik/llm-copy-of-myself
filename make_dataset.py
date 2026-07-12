#!/usr/bin/env python3
"""Build tokenizer-budgeted training artifacts from cleaned Telegram sessions."""

from pathlib import Path

from personal_ai.config import load_config
from personal_ai.data import prepare_dataset


def main() -> None:
    from transformers import AutoTokenizer

    project_dir = Path(__file__).resolve().parent
    config = load_config(project_dir / "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model, use_fast=True)
    manifest = prepare_dataset(config, tokenizer)
    print(f"Created prepared dataset: {manifest['counts']}")


if __name__ == "__main__":
    main()
