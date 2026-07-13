from types import SimpleNamespace

import pytest

from bot import incoming_message_content, require_adapter_dataset_match, resolve_adapter_path


def _message(**values):
    defaults = {"text": None, "caption": None}
    defaults.update(values)
    return SimpleNamespace(**defaults)


def test_plain_text_is_preserved():
    assert incoming_message_content(_message(text="привет")) == "привет"


def test_photo_is_replaced_and_caption_is_preserved():
    message = _message(photo=[object()], caption="смотри")
    assert incoming_message_content(message) == "[sent image]\nсмотри"


def test_files_and_voice_messages_use_stable_placeholders():
    assert incoming_message_content(_message(document=object())) == "[sent document]"
    assert incoming_message_content(_message(voice=object())) == "[sent voice message]"


def _write_adapter(path, *, complete_checkpoint=False):
    path.mkdir()
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    if complete_checkpoint:
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
            (path / name).write_bytes(b"state")


def test_final_adapter_takes_precedence_over_checkpoints(tmp_path):
    final = tmp_path / "adapter-final"
    _write_adapter(final)
    _write_adapter(tmp_path / "checkpoint-100", complete_checkpoint=True)

    assert resolve_adapter_path(final) == final


def test_latest_complete_checkpoint_is_used_when_final_is_missing(tmp_path):
    _write_adapter(tmp_path / "checkpoint-50", complete_checkpoint=True)
    _write_adapter(tmp_path / "checkpoint-100", complete_checkpoint=True)
    _write_adapter(tmp_path / "checkpoint-150")

    assert resolve_adapter_path(tmp_path / "adapter-final") == tmp_path / "checkpoint-100"


def test_missing_custom_adapter_does_not_silently_fallback(tmp_path):
    _write_adapter(tmp_path / "checkpoint-100", complete_checkpoint=True)

    with pytest.raises(FileNotFoundError, match="not found or incomplete"):
        resolve_adapter_path(tmp_path / "custom-adapter")


def test_bot_rejects_adapter_from_different_dataset(tmp_path):
    checkpoint = tmp_path / "checkpoint-100"
    _write_adapter(checkpoint, complete_checkpoint=True)
    (tmp_path / "reproducibility.json").write_text(
        '{"dataset_sha256": "old"}', encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"dataset_sha256": "new"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="different dataset"):
        require_adapter_dataset_match(checkpoint, manifest)
