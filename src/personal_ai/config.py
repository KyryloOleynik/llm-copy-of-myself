from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    name: str
    owner_from_id: str


class DataConfig(StrictModel):
    source: Path
    cleaned: Path
    dataset: Path = Path("DataExport_2026-07-10/dataset.json")
    output_dir: Path = Path("data/processed")
    session_gap_hours: int = 12
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1

    @model_validator(mode="after")
    def validate_ratios(self) -> "DataConfig":
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("data split ratios must add up to 1.0")
        return self


class ModelConfig(StrictModel):
    base_model: str
    sequence_length: int = Field(1024, ge=128)
    thinking_enabled: bool = False


class TrainingConfig(StrictModel):
    output_dir: Path = Path("artifacts/training/qwen3-8b-r16")
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    epochs: float = 1
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    save_steps: int = 100
    seed: int = 42


class RetrievalConfig(StrictModel):
    database: Path
    context_limit: int = 8192


class BotConfig(StrictModel):
    polling: bool = True
    state_database: Path
    model_server_url: str


class AppConfig(StrictModel):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    retrieval: RetrievalConfig
    bot: BotConfig


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    return AppConfig.model_validate(raw)
