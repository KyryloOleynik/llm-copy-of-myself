from types import SimpleNamespace

import pytest

import bot as bot_module
from bot import (
    LocalModel,
    incoming_message_content,
    live_system_message,
    require_adapter_dataset_match,
    resolve_adapter_path,
)
from tests.helpers import FakeTokenizer


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


def test_live_system_prompt_explicitly_requires_native_tools():
    prompt = live_system_message("friend", "2026-07-18T12:00:00+03:00")

    assert "ОБЯЗАТЕЛЬНО" in prompt
    assert "нативный" in prompt
    assert "calculate" in prompt
    assert "search_personal_memory" in prompt
    assert "query_google_calendar" in prompt
    assert "2026-07-18T12:00:00+03:00" in prompt


def test_live_chat_runs_call_result_answer_loop(monkeypatch, caplog):
    caplog.set_level("INFO")
    generated_messages = []
    generated_tools = []
    replies = iter(
        [
            (
                '<tool_call>{"name":"calculate",'
                '"arguments":{"expression":"2+2"}}</tool_call>',
                10,
            ),
            ("получается 4", 20),
        ]
    )

    def fake_generate_reply(_torch, _tokenizer, _model, messages, *, tools, **_kwargs):
        generated_messages.append(list(messages))
        generated_tools.append(tools)
        return next(replies)

    executed_calls = []

    def fake_execute_tool_call(call, *_args):
        executed_calls.append(call)
        return '{"result": 4}'

    monkeypatch.setattr(bot_module, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(bot_module, "execute_tool_call", fake_execute_tool_call)

    local_model = object.__new__(LocalModel)
    local_model.torch = object()
    local_model.tokenizer = FakeTokenizer()
    local_model.model = object()
    local_model._fit_messages = lambda messages, _tools=None: list(messages)

    reply = local_model.generate(
        [
            {"role": "system", "content": "style"},
            {"role": "user", "content": "посчитай 2+2"},
        ]
    )

    assert reply == "получается 4"
    assert [message["role"] for message in generated_messages[1]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert generated_messages[1][-1] == {
        "role": "tool",
        "name": "calculate",
        "content": '{"result": 4}',
    }
    assert executed_calls[0].name == "calculate"
    assert executed_calls[0].arguments == {"expression": "2+2"}
    assert generated_tools == [bot_module.TOOL_SCHEMAS, bot_module.TOOL_SCHEMAS]
    assert "Using tool calculate with arguments" in caplog.text
    assert "Tool calculate completed successfully" in caplog.text
