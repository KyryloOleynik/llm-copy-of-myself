import json
from types import SimpleNamespace

import pytest

from personal_ai.modeling import TOKEN_MIXER_SUFFIXES, select_language_lora_modules
from personal_ai.training import (
    _checkpoint_is_resumable,
    _latest_valid_checkpoint,
    longest_example_indices,
    require_successful_smoke,
)


def test_latest_valid_checkpoint_skips_incomplete_newer_directory(tmp_path):
    valid = tmp_path / "checkpoint-10"
    valid.mkdir()
    (valid / "adapter_model.safetensors").write_bytes(b"weights")
    (valid / "trainer_state.json").write_text("{}", encoding="utf-8")
    (valid / "optimizer.pt").write_bytes(b"optimizer")
    (valid / "scheduler.pt").write_bytes(b"scheduler")

    incomplete = tmp_path / "checkpoint-13"
    incomplete.mkdir()
    (incomplete / "generation_config.json").write_text("{}", encoding="utf-8")

    assert _latest_valid_checkpoint(tmp_path) == valid


def test_latest_valid_checkpoint_returns_none_without_resumable_checkpoint(tmp_path):
    incomplete = tmp_path / "checkpoint-13"
    incomplete.mkdir()
    (incomplete / "generation_config.json").write_text("{}", encoding="utf-8")

    assert _latest_valid_checkpoint(tmp_path) is None


def test_saved_peft_adapter_checkpoint_is_resumable(tmp_path):
    checkpoint = tmp_path / "checkpoint-130"
    checkpoint.mkdir()
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt"):
        (checkpoint / name).write_bytes(b"saved")
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    assert _checkpoint_is_resumable(checkpoint)


def test_full_training_requires_matching_smoke_gate(tmp_path):
    config = SimpleNamespace(
        training=SimpleNamespace(output_dir=tmp_path),
        model=SimpleNamespace(base_model="Qwen/Qwen3.5-4B"),
    )
    with pytest.raises(RuntimeError, match="--smoke"):
        require_successful_smoke(config, "dataset-hash")

    (tmp_path / "smoke-test.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3.5-4B",
                "dataset_sha256": "dataset-hash",
                "peak_vram_reserved_bytes": 11 * 1024**3,
            }
        ),
        encoding="utf-8",
    )
    assert require_successful_smoke(config, "dataset-hash")["model"] == "Qwen/Qwen3.5-4B"


def test_smoke_selection_uses_longest_examples():
    dataset = {"sequence_tokens": [100, 4096, 512, 2048]}
    assert longest_example_indices(dataset, 2) == [1, 3]


class WeightedModule:
    weight = object()


class FakeQwenModel:
    def named_modules(self):
        for suffix in sorted(TOKEN_MIXER_SUFFIXES):
            yield f"model.language_model.layers.0.{suffix}", WeightedModule()
        yield "model.visual.blocks.0.out_proj", WeightedModule()


def test_only_language_token_mixers_are_selected():
    selected = select_language_lora_modules(FakeQwenModel())
    assert len(selected) == len(TOKEN_MIXER_SUFFIXES)
    assert all("language_model" in name for name in selected)
    assert all("visual" not in name for name in selected)
