import json
from types import SimpleNamespace

import pytest

from personal_ai.training import IGNORE_INDEX, ReplyOnlyCollator, write_smoke_sample_audit
from personal_ai.utils import assistant_target_ids, assistant_target_spans
from tests.helpers import FakeTokenizer


def test_only_final_reply_has_labels():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "style"},
        {"role": "user", "content": "abc"},
        {"role": "assistant", "content": "xy"},
    ]
    prompt = tokenizer.apply_chat_template(messages[:-1], True, True)
    full = tokenizer.apply_chat_template(messages, True, False)
    collator = ReplyOnlyCollator(tokenizer, max_length=100)
    batch = collator([{"messages": messages}])
    labels = batch["labels"][0].tolist()
    assert labels[: len(prompt)] == [IGNORE_INDEX] * len(prompt)
    assert labels[len(prompt) :] == full[len(prompt) :]
    assert collator.audit["zero_label_example"] == 0


def test_one_tool_chat_trains_call_and_post_tool_answer():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "style"},
        {"role": "user", "content": "посчитай 2+2"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "2+2"},
                    },
                }
            ],
        },
        {"role": "tool", "name": "calculate", "content": '{"result": 4}'},
        {"role": "assistant", "content": "получается 4"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "parameters": {"type": "object"},
            },
        }
    ]
    full, spans = assistant_target_spans(tokenizer, messages, tools)
    collator = ReplyOnlyCollator(tokenizer, max_length=1000)

    batch = collator(
        [
            {
                "messages": messages,
                "tools": json.dumps(tools),
                "supervise_all_assistant_turns": True,
            }
        ]
    )

    labels = batch["labels"][0].tolist()
    expected = [IGNORE_INDEX] * len(full)
    for start, end in spans:
        expected[start:end] = full[start:end]
    assert labels == expected
    assert len(spans) == 2
    assert all(labels[start:end] == full[start:end] for start, end in spans)
    tool_result_start = tokenizer.apply_chat_template(
        messages[:3], True, False, tools=tools
    )
    final_prompt_end = tokenizer.apply_chat_template(
        messages[:4], True, True, tools=tools
    )
    assert labels[len(tool_result_start) : len(final_prompt_end)] == [
        IGNORE_INDEX
    ] * (len(final_prompt_end) - len(tool_result_start))


def test_smoke_audit_identifies_masked_prompt_and_expected_reply():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "style"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "expected reply"},
    ]
    collator = ReplyOnlyCollator(tokenizer, max_length=200, capture_examples=True)

    batch = collator(
        [
            {
                "example_id": "example-1",
                "chat_id": "chat-1",
                "session_id": "session-1",
                "source_type": "personal_telegram",
                "relationship": "friend",
                "split": "train",
                "timestamp": "2026-01-01T00:00:00",
                "messages": messages,
            }
        ]
    )
    capture_ids = batch.pop("smoke_audit_capture_ids").tolist()
    collator.mark_used_for_backward(capture_ids)

    record = collator.captured_examples[0]
    assert record["example_id"] == "example-1"
    assert record["masked_prompt_messages"] == messages[:-1]
    assert record["expected_assistant_reply"] == "expected reply"
    assert record["masked_prompt_tokens"] > 0
    assert record["trained_target_tokens"] > 0
    assert record["input_ids"] == batch["input_ids"][0].tolist()
    assert record["training_labels"] == batch["labels"][0].tolist()
    assert record["attention_mask"] == batch["attention_mask"][0].tolist()
    boundary = record["masked_prompt_tokens"]
    sequence_end = record["total_sequence_tokens"]
    assert record["training_labels"][:boundary] == [IGNORE_INDEX] * boundary
    assert record["training_labels"][boundary:sequence_end] == record["input_ids"][
        boundary:sequence_end
    ]
    assert record["training_labels"][sequence_end:] == [IGNORE_INDEX] * record["padding_tokens"]


def test_smoke_audit_file_persists_the_exact_collator_arrays(tmp_path):
    tokenizer = FakeTokenizer()
    example = {
        "example_id": "example-1",
        "chat_id": "chat-1",
        "session_id": "session-1",
        "source_type": "personal_telegram",
        "relationship": "friend",
        "split": "train",
        "timestamp": "2026-01-01T00:00:00",
        "sequence_tokens": 42,
        "target_tokens": 4,
        "messages": [
            {"role": "system", "content": "style"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "reply"},
        ],
    }
    collator = ReplyOnlyCollator(tokenizer, max_length=200, capture_examples=True)
    batch = collator([example])
    capture_ids = batch.pop("smoke_audit_capture_ids").tolist()
    collator.mark_used_for_backward(capture_ids)
    path = tmp_path / "smoke-samples.json"
    config = SimpleNamespace(
        model=SimpleNamespace(base_model="fake/model"),
        training=SimpleNamespace(micro_batch_size=1, gradient_accumulation_steps=16),
    )

    write_smoke_sample_audit(
        path,
        config,
        {"dataset_sha256": "dataset-hash"},
        [example],
        [],
        collator,
        "completed",
    )

    audit = json.loads(path.read_text(encoding="utf-8"))
    persisted = audit["actually_used_for_backward"][0]
    assert audit["status"] == "completed"
    assert audit["collated_lookahead_not_used_for_backward"] == []
    assert persisted["expected_assistant_reply"] == "reply"
    assert persisted["input_ids"] == batch["input_ids"][0].tolist()
    assert persisted["training_labels"] == batch["labels"][0].tolist()
    assert persisted["attention_mask"] == batch["attention_mask"][0].tolist()


def test_smoke_audit_separates_accelerate_lookahead_from_backward(tmp_path):
    tokenizer = FakeTokenizer()
    base = {
        "chat_id": "chat-1",
        "session_id": "session-1",
        "source_type": "personal_telegram",
        "relationship": "friend",
        "split": "train",
        "timestamp": "2026-01-01T00:00:00",
        "sequence_tokens": 42,
        "target_tokens": 4,
    }
    first = {
        **base,
        "example_id": "used",
        "messages": [
            {"role": "system", "content": "style"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "reply one"},
        ],
    }
    lookahead = {
        **base,
        "example_id": "lookahead",
        "messages": [
            {"role": "system", "content": "style"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "reply two"},
        ],
    }
    collator = ReplyOnlyCollator(tokenizer, max_length=200, capture_examples=True)
    used_batch = collator([first])
    collator([lookahead])
    collator.mark_used_for_backward(used_batch["smoke_audit_capture_ids"].tolist())
    path = tmp_path / "smoke-samples.json"
    config = SimpleNamespace(
        model=SimpleNamespace(base_model="fake/model"),
        training=SimpleNamespace(micro_batch_size=1, gradient_accumulation_steps=16),
    )

    write_smoke_sample_audit(
        path,
        config,
        {"dataset_sha256": "dataset-hash"},
        [first, lookahead],
        [],
        collator,
        "completed",
    )

    audit = json.loads(path.read_text(encoding="utf-8"))
    assert [row["example_id"] for row in audit["actually_used_for_backward"]] == ["used"]
    assert [
        row["example_id"] for row in audit["collated_lookahead_not_used_for_backward"]
    ] == ["lookahead"]


def test_overflow_truncates_prompt_without_truncating_target():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "abcdefghij"},
        {"role": "assistant", "content": "xy"},
    ]
    _, original, target = assistant_target_ids(tokenizer, messages)
    collator = ReplyOnlyCollator(tokenizer, max_length=8, capture_examples=True)

    batch = collator([{"messages": messages}])

    record = collator.captured_examples[0]
    assert batch["input_ids"].shape[1] == 8
    assert record["original_sequence_tokens"] == len(original)
    assert record["prompt_tokens_truncated"] == len(original) - 8
    assert record["prompt_truncation_strategy"] == "preserve_system_and_recent_tail"
    assert record["trained_target_tokens"] == len(target)
    assert record["training_labels"][-len(target) :] == target
    assert collator.audit["truncated_examples"] == 1


def test_overflow_preserves_system_prompt_and_recent_context_tail():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "style"},
        {"role": "user", "content": "x" * 100},
        {"role": "assistant", "content": "ok"},
    ]
    system_ids = tokenizer.apply_chat_template(messages[:1], True, False)
    _, _, target = assistant_target_ids(tokenizer, messages)
    collator = ReplyOnlyCollator(tokenizer, max_length=32, capture_examples=True)

    batch = collator([{"messages": messages}])

    record = collator.captured_examples[0]
    actual = batch["input_ids"][0].tolist()
    assert actual[: len(system_ids)] == system_ids
    assert actual[-len(target) :] == target
    assert record["prompt_truncation_strategy"] == "preserve_system_and_recent_tail"


def test_target_that_cannot_leave_prompt_space_is_rejected():
    collator = ReplyOnlyCollator(FakeTokenizer(), max_length=8, max_target_tokens=100)
    with pytest.raises(ValueError, match="leaves no prompt tokens"):
        collator(
            [
                    {
                        "messages": [
                            {"role": "system", "content": "style"},
                            {"role": "user", "content": "a"},
                            {"role": "assistant", "content": "abcdefghij"},
                    ]
                }
            ]
        )
    assert collator.audit["target_cannot_fit"] == 1


def test_oversized_target_is_rejected():
    collator = ReplyOnlyCollator(FakeTokenizer(), max_length=100, max_target_tokens=2)
    with pytest.raises(ValueError, match="Assistant target"):
        collator(
            [
                    {
                        "messages": [
                            {"role": "system", "content": "style"},
                            {"role": "user", "content": "a"},
                            {"role": "assistant", "content": "long"},
                    ]
                }
            ]
        )
    assert collator.audit["oversized_target"] == 1


class NonPrefixTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        if not add_generation_prompt:
            ids[0] = 999
        return ids


def test_prompt_prefix_mismatch_is_rejected():
    collator = ReplyOnlyCollator(NonPrefixTokenizer(), max_length=100)
    with pytest.raises(ValueError, match="not a prefix"):
        collator(
            [
                    {
                        "messages": [
                            {"role": "system", "content": "style"},
                            {"role": "user", "content": "a"},
                            {"role": "assistant", "content": "b"},
                    ]
                }
            ]
        )
