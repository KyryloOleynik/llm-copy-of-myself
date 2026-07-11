from personal_ai.training import _checkpoint_is_resumable, _latest_valid_checkpoint


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
