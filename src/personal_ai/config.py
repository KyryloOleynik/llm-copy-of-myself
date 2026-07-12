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
    train_ratio: float = Field(0.8, ge=0, le=1)
    validation_ratio: float = Field(0.1, ge=0, le=1)
    test_ratio: float = Field(0.1, ge=0, le=1)
    max_target_tokens: int = Field(256, ge=1)
    max_examples_per_chat: int = Field(1000, ge=1)
    max_identical_short_target: int = Field(25, ge=1)
    short_target_max_tokens: int = Field(3, ge=1)
    contains_unredacted_private_data: bool

    @model_validator(mode="after")
    def validate_ratios(self) -> "DataConfig":
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("data split ratios must add up to 1.0")
        if not self.contains_unredacted_private_data:
            raise ValueError(
                "This project intentionally contains unredacted private data; "
                "set data.contains_unredacted_private_data to true to acknowledge it"
            )
        return self


class ModelConfig(StrictModel):
    base_model: str
    sequence_length: int = Field(1024, ge=128)
    thinking_enabled: bool = False

    @model_validator(mode="after")
    def require_non_thinking(self) -> "ModelConfig":
        if self.thinking_enabled:
            raise ValueError("Personal conversation training requires thinking_enabled: false")
        return self


class TrainingConfig(StrictModel):
    output_dir: Path = Path("artifacts/training/qwen3.5-4b-r8")
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 3e-5
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    epochs: float = 1
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    lr_scheduler_type: str = "cosine"
    lora_target_policy: str = "language-token-mixers"
    seed: int = 42

    @model_validator(mode="after")
    def validate_policy(self) -> "TrainingConfig":
        if self.lora_target_policy != "language-token-mixers":
            raise ValueError("Only lora_target_policy: language-token-mixers is supported")
        if self.save_steps % self.eval_steps != 0:
            raise ValueError("save_steps must be a multiple of eval_steps")
        return self


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
