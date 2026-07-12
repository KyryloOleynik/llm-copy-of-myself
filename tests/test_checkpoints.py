import json
from types import SimpleNamespace

import pytest

from personal_ai.training import (
    _checkpoint_is_resumable,
    _latest_valid_checkpoint,
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

    (tmp_path / "smoke-test.json").write_text(json.dumps({
        "model": "Qwen/Qwen3.5-4B",
        "dataset_sha256": "dataset-hash",
        "peak_vram_reserved_bytes": 11 * 1024**3,
    }), encoding="utf-8")
    assert require_successful_smoke(config, "dataset-hash")["model"] == "Qwen/Qwen3.5-4B"
