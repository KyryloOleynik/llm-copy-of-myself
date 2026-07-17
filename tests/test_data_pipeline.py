import json
from pathlib import Path
from types import SimpleNamespace

from personal_ai.data import (
    _balanced_chat_limit,
    _fit_example,
    merge_turns,
    prepare_dataset,
    split_dataset_sessions,
)
from personal_ai.supplemental import build_supplemental_examples
from personal_ai.utils import assistant_target_text
from tests.helpers import FakeTokenizer


def _message(message_id, role, text, date, **extra):
    return {
        "id": message_id,
        "from_id": "owner" if role == "assistant" else "other",
        "text": text,
        "date": date,
        **extra,
    }


def _config(tmp_path: Path):
    return SimpleNamespace(
        project=SimpleNamespace(owner_from_id="owner"),
        model=SimpleNamespace(base_model="fake/model", sequence_length=4096),
        data=SimpleNamespace(
            source=tmp_path / "raw.json",
            cleaned=tmp_path / "cleaned.json",
            dataset=tmp_path / "dataset.json",
            output_dir=tmp_path / "processed",
            train_ratio=0.8,
            validation_ratio=0.1,
            max_target_tokens=256,
            personal_train_examples=None,
            personal_data_ratio=1.0,
            context_retention_ratio=0.0,
            general_reasoning_ratio=0.0,
            instruction_following_ratio=0.0,
            tool_calling_ratio=0.0,
            rag_retrieval_ratio=0.0,
        ),
        training=SimpleNamespace(seed=42),
    )


def test_long_assistant_message_does_not_drop_session():
    turns = merge_turns(
        [
            _message(1, "user", "hello", "2026-01-01T00:00:00"),
            _message(2, "assistant", "x" * 1000, "2026-01-01T00:01:00"),
            _message(3, "user", "next", "2026-01-01T00:02:00"),
        ],
        "owner",
    )
    assert len(turns) == 3
    assert len(turns[1]["content"]) == 1000


def test_context_that_cannot_fit_is_marked_for_token_level_truncation():
    tokenizer = FakeTokenizer()
    system = {"role": "system", "content": "style"}
    context = [
        {
            "role": "user",
            "content": "x" * 100,
            "source_message_ids": [1],
            "last_message_date": "2026-01-01T00:00:00",
            "media_only": False,
        }
    ]
    target = {
        "role": "assistant",
        "content": "complete reply",
        "source_message_ids": [2],
        "last_message_date": "2026-01-01T00:01:00",
        "media_only": False,
    }

    fitted = _fit_example(
        tokenizer,
        system,
        context,
        target,
        max_length=64,
        max_target_tokens=32,
    )

    assert fitted is not None
    messages, sequence_tokens, target_tokens, truncated_prompt_tokens = fitted
    assert sequence_tokens == 64
    assert target_tokens > 0
    assert truncated_prompt_tokens > 0
    assert messages[-1] == {"role": "assistant", "content": "complete reply"}


def test_media_is_preserved_and_marked_media_only():
    turns = merge_turns(
        [
            _message(1, "user", "", "2026-01-01T00:00:00", photo="photo.jpg"),
        ],
        "owner",
    )
    assert turns[0]["content"] == "[sent image]"
    assert turns[0]["media_only"] is True


def test_assistant_media_only_messages_are_omitted():
    turns = merge_turns(
        [
            _message(1, "user", "покажи", "2026-01-01T00:00:00"),
            _message(2, "assistant", "", "2026-01-01T00:01:00", photo="photo.jpg"),
            _message(3, "assistant", "вот", "2026-01-01T00:02:00"),
        ],
        "owner",
    )
    assert turns[-1]["content"] == "вот"
    assert "[sent " not in turns[-1]["content"]


def test_session_split_rules():
    sessions = [
        {"session_id": f"s{i:02d}", "messages": [{"date": f"2026-01-{i + 1:02d}"}]}
        for i in range(12)
    ]
    splits = split_dataset_sessions(sessions)
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [9, 1, 2]
    assert not (
        set(s["session_id"] for s in splits["train"]) & set(s["session_id"] for s in splits["test"])
    )


def test_prepare_dataset_is_deterministic_and_session_isolated(tmp_path):
    config = _config(tmp_path)
    sessions = []
    for index in range(12):
        date = f"2026-01-{index + 1:02d}T00:00:00"
        messages = [
            _message(index * 3 + 1, "user", f"question {index}", date),
            _message(index * 3 + 2, "assistant", f"answer {index}", date),
        ]
        if index == 0:
            messages.insert(
                0,
                _message(index * 3, "assistant", "meaningful opening owner context", date),
            )
        sessions.append(
            {
                "session_id": f"session-{index:02d}",
                "messages": messages,
            }
        )
    cleaned = {
        "chats": [
            {"id": 1, "type": "personal_chat", "relationship": "family", "sessions": sessions},
            {"id": 2, "type": "private_group", "relationship": "friend", "sessions": sessions},
        ]
    }
    config.data.cleaned.write_text(json.dumps(cleaned), encoding="utf-8")
    config.data.source.write_text(json.dumps(cleaned), encoding="utf-8")
    manifest = prepare_dataset(config, FakeTokenizer())
    first_bytes = config.data.dataset.read_bytes()
    second_manifest = prepare_dataset(config, FakeTokenizer())
    assert config.data.dataset.read_bytes() == first_bytes
    assert (
        manifest["counts"]
        == second_manifest["counts"]
        == {
            "train": 9,
            "validation": 1,
            "test": 2,
        }
    )
    assert manifest["exclusions"]["non_personal_chat"] == 1
    seen = {}
    for split in ("train", "validation", "test"):
        for line in (config.data.output_dir / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["messages"][0]["role"] == "system"
            assert any(message["role"] == "user" for message in row["messages"][1:-1])
            assert row["messages"][-1]["role"] == "assistant"
            assert row["timestamp"].startswith("2026-01-")
            assert row["session_id"] not in seen
            seen[row["session_id"]] = split
    leading = next(
        row
        for line in (config.data.output_dir / "train.jsonl").read_text().splitlines()
        if (row := json.loads(line))["session_id"] == "session-00"
    )
    assert leading["messages"][1]["role"] == "assistant"
    assert leading["messages"][1]["content"] == "meaningful opening owner context"


def test_context_and_reasoning_supplements_are_deterministic_and_disjoint():
    tokenizer = FakeTokenizer()
    kwargs = {
        "tokenizer": tokenizer,
        "count": 10,
        "max_length": 4096,
        "max_target_tokens": 256,
        "seed": 42,
    }
    context_train = build_supplemental_examples(
        split="train", category="context_retention", **kwargs
    )
    context_test = build_supplemental_examples(split="test", category="context_retention", **kwargs)
    reasoning = build_supplemental_examples(split="train", category="general_reasoning", **kwargs)
    reasoning_test = build_supplemental_examples(
        split="test", category="general_reasoning", **kwargs
    )
    instructions = build_supplemental_examples(
        split="train", category="instruction_following", **kwargs
    )
    tool_rows = build_supplemental_examples(
        split="train", category="tool_calling", **kwargs
    )
    rag_rows = build_supplemental_examples(
        split="train", category="rag_retrieval", **kwargs
    )
    assert context_train == build_supplemental_examples(
        split="train", category="context_retention", **kwargs
    )
    assert {row["example_id"] for row in context_train}.isdisjoint(
        row["example_id"] for row in context_test
    )
    assert all(row["source_type"] == "context_retention" for row in context_train)
    assert all(row["source_type"] == "general_reasoning" for row in reasoning)
    assert all(row["source_type"] == "instruction_following" for row in instructions)
    assert any(row["messages"][-1]["content"].startswith("{") for row in instructions)
    assert all(row["tools"] != "[]" for row in tool_rows + rag_rows)
    assert all(row["supervise_all_assistant_turns"] for row in tool_rows + rag_rows)
    assert all(
        [message["role"] for message in row["messages"]]
        == ["system", "user", "assistant", "tool", "assistant"]
        for row in tool_rows + rag_rows
    )
    assert all(
        json.loads(row["messages"][2]["tool_calls"])
        and row["messages"][4]["content"]
        for row in tool_rows + rag_rows
    )
    all_rows = (
        context_train
        + context_test
        + reasoning
        + reasoning_test
        + instructions
        + tool_rows
        + rag_rows
    )
    assert all(row["messages"][-1]["role"] == "assistant" for row in all_rows)
    assert all(
        row["messages"][0]["content"].startswith("Отвечай в стиле Родиона.")
        for row in all_rows
    )
    fingerprints = {
        json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        for row in all_rows
    }
    assert len(fingerprints) == len(all_rows)
    targets = [
        assistant_target_text(row["messages"]).strip().casefold()
        for row in all_rows
    ]
    assert len(targets) == len(set(targets))


def test_personal_limit_removes_excess_from_dominant_chats():
    rows = [
        {
            "chat_id": chat_id,
            "timestamp": f"2026-01-01T00:{index:02d}:00",
            "example_id": f"{chat_id}-{index}",
        }
        for chat_id, count in (("large", 10), ("medium", 4), ("small", 2))
        for index in range(count)
    ]

    selected = _balanced_chat_limit(rows, 12, seed=42)
    counts = {
        chat_id: sum(row["chat_id"] == chat_id for row in selected)
        for chat_id in ("large", "medium", "small")
    }

    assert len(selected) == 12
    assert counts == {"large": 6, "medium": 4, "small": 2}
