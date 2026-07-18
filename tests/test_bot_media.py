from types import SimpleNamespace

import pytest

import bot as bot_module
from bot import (
    LocalModel,
    ModelGeneration,
    incoming_message_content,
    live_system_message,
    required_live_tool_calls,
    require_adapter_dataset_match,
    resolve_adapter_path,
    trim_chat_history,
)
from tests.helpers import FakeTokenizer
from personal_ai.tools import ToolCall


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
    assert "можешь свободно и активно использовать инструменты" in prompt
    assert "твой первый ответ должен содержать только вызов инструмента" in prompt
    assert "После сообщения role=tool сразу дай пользователю законченный ответ" in prompt
    assert "Для каждого нового фактического вопроса делай свежий" in prompt
    assert "расскажи о себе" in prompt
    assert "включи в запрос все имена без изменения" in prompt
    assert "что делал на этой неделе?" in prompt
    assert '<tool_call>{"name":"ИМЯ_ИНСТРУМЕНТА","arguments":{...}}</tool_call>' in prompt
    assert '{"expression":"арифметическое выражение"}' in prompt
    assert '"query":"конкретный поисковый запрос","limit":5' in prompt
    assert "action=events" in prompt
    assert "action=free_time" in prompt
    assert "ISO-8601" in prompt
    assert "calculate" in prompt
    assert "search_personal_memory" in prompt
    assert "query_google_calendar" in prompt
    assert "текущая дата: 2026-07-18." in prompt
    assert "Текущий день недели: суббота." in prompt
    assert "2026-07-18T12:00:00+03:00" in prompt
    assert (
        "Текущая неделя: от 2026-07-13T00:00:00+03:00 "
        "до 2026-07-20T00:00:00+03:00."
    ) in prompt


def test_live_system_prompt_forbids_observed_tool_failures():
    prompt = live_system_message("friend", "2026-07-18T02:30:00+03:00")

    assert "Запрещено вместо вызова писать «посмотрю»" in prompt
    assert "а не поиск в памяти" in prompt
    assert "не повторяй старый результат из истории" in prompt
    assert "Если results не пуст, нельзя говорить, что поиск ничего не нашёл" in prompt
    assert "Не придумывай ссылки, репозитории" in prompt
    assert "реплика другого человека не является фактом о Родионе" in prompt
    assert "Пустой events означает только" in prompt
    assert "для доступности используй free_time" in prompt


def test_live_router_forces_memory_for_identity_and_explicit_search():
    history = [
        {"role": "user", "content": "Расскажи о себе"},
        {"role": "assistant", "content": "Поиск ничего не нашел"},
        {"role": "user", "content": "Ты ничего не искал, поищи"},
    ]

    identity_calls = required_live_tool_calls(
        "Расскажи о себе",
        history[:1],
        "2026-07-18T02:32:00+03:00",
    )
    retry_calls = required_live_tool_calls(
        "Ты ничего не искал, поищи",
        history,
        "2026-07-18T02:32:00+03:00",
    )

    assert [call.name for call in identity_calls] == ["search_personal_memory"]
    assert identity_calls[0].arguments["limit"] == 8
    assert "identity biography" in identity_calls[0].arguments["query"]
    assert [call.name for call in retry_calls] == ["search_personal_memory"]
    assert "Расскажи о себе" in retry_calls[0].arguments["query"]


def test_live_router_uses_calendar_for_this_weeks_past_activity():
    calls = required_live_tool_calls(
        "Что на этой неделе делал?",
        [{"role": "user", "content": "Что на этой неделе делал?"}],
        "2026-07-18T02:33:00+03:00",
    )

    assert [call.name for call in calls] == ["query_google_calendar"]
    assert calls[0].arguments == {
        "action": "events",
        "start": "2026-07-13T00:00:00+03:00",
        "end": "2026-07-18T02:33:00+03:00",
        "limit": 50,
    }


def test_live_router_executes_every_tool_needed_by_a_mixed_question():
    calls = required_live_tool_calls(
        "Где живёшь? И сколько будет 2+2+200/13",
        [{"role": "user", "content": "Где живёшь? И сколько будет 2+2+200/13"}],
        "2026-07-18T02:37:00+03:00",
    )

    assert [call.name for call in calls] == ["search_personal_memory", "calculate"]
    assert calls[1].arguments == {"expression": "2+2+200/13"}


def test_required_live_tools_run_before_the_models_first_reply(monkeypatch):
    generated_messages = []

    def fake_generate_reply(_torch, _tokenizer, _model, messages, **_kwargs):
        generated_messages.append(list(messages))
        return "Живу в Киеве, а результат 19.3846153846", 20

    executed_calls = []

    def fake_execute_tool_call(call, *_args):
        executed_calls.append(call)
        if call.name == "calculate":
            return '{"result": 19.3846153846}'
        return '{"results":[{"content":"Rodion lives in Kyiv"}]}'

    monkeypatch.setattr(bot_module, "generate_reply", fake_generate_reply)
    monkeypatch.setattr(bot_module, "execute_tool_call", fake_execute_tool_call)

    local_model = object.__new__(LocalModel)
    local_model.torch = object()
    local_model.tokenizer = FakeTokenizer()
    local_model.model = object()
    local_model._fit_messages = lambda messages, _tools=None: list(messages)
    required = (
        ToolCall("search_personal_memory", {"query": "where Rodion lives", "limit": 5}),
        ToolCall("calculate", {"expression": "2+2+200/13"}),
    )

    generation = local_model.generate(
        [
            {"role": "system", "content": "style"},
            {"role": "user", "content": "Где живёшь? И сколько будет 2+2+200/13"},
        ],
        required,
    )

    assert generation.text == "Живу в Киеве, а результат 19.3846153846"
    assert [call.name for call in executed_calls] == [
        "search_personal_memory",
        "calculate",
    ]
    assert [message["role"] for message in generated_messages[0]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]


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

    generation = local_model.generate(
        [
            {"role": "system", "content": "style"},
            {"role": "user", "content": "посчитай 2+2"},
        ]
    )

    assert generation.text == "получается 4"
    assert [message["role"] for message in generation.tool_trace] == [
        "assistant",
        "tool",
    ]
    assert generation.tool_trace[-1]["content"] == '{"result": 4}'
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
    assert 'Tool calculate result: {"result": 4}' in caplog.text


def test_tool_trace_stays_with_its_user_turn_in_chat_history():
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "what did memory return?"},
    ]
    generation = ModelGeneration(
        "the result said family",
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_personal_memory",
                            "arguments": {"query": "family"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "search_personal_memory",
                "content": '{"results":[{"content":"family fact"}]}',
            },
        ),
    )
    history.extend(generation.tool_trace)
    history.append({"role": "assistant", "content": generation.text})

    trim_chat_history(history, max_user_turns=1)

    assert [message["role"] for message in history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history[2]["content"] == '{"results":[{"content":"family fact"}]}'
