import json
from pathlib import Path
from types import SimpleNamespace

from personal_ai.data import merge_turns, prepare_dataset, split_sessions


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and enable_thinking is False
        ids = []
        for message in messages:
            ids.append({"system": 10, "user": 20, "assistant": 30}[message["role"]])
            ids.extend(100 + ord(char) % 50 for char in message["content"])
            ids.append(2)
        if add_generation_prompt:
            ids.append(30)
        return ids

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


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
        model=SimpleNamespace(base_model="fake/model", sequence_length=1024),
        data=SimpleNamespace(
            source=tmp_path / "raw.json",
            cleaned=tmp_path / "cleaned.json",
            dataset=tmp_path / "dataset.json",
            output_dir=tmp_path / "processed",
            train_ratio=0.8,
            validation_ratio=0.1,
            max_target_tokens=256,
            max_examples_per_chat=1000,
            max_identical_short_target=25,
            short_target_max_tokens=3,
        ),
        training=SimpleNamespace(seed=42),
    )


def test_long_assistant_message_does_not_drop_session():
    turns = merge_turns([
        _message(1, "user", "hello", "2026-01-01T00:00:00"),
        _message(2, "assistant", "x" * 1000, "2026-01-01T00:01:00"),
        _message(3, "user", "next", "2026-01-01T00:02:00"),
    ], "owner")
    assert len(turns) == 3
    assert len(turns[1]["content"]) == 1000


def test_media_is_preserved_and_marked_media_only():
    turns = merge_turns([
        _message(1, "user", "", "2026-01-01T00:00:00", photo="photo.jpg"),
    ], "owner")
    assert turns[0]["content"] == "[sent image]"
    assert turns[0]["media_only"] is True


def test_session_split_rules():
    sessions = [
        {"session_id": f"s{i:02d}", "messages": [{"date": f"2026-01-{i + 1:02d}"}]}
        for i in range(12)
    ]
    splits = split_sessions(sessions)
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [9, 1, 2]
    assert not (set(s["session_id"] for s in splits["train"]) & set(
        s["session_id"] for s in splits["test"]
    ))


def test_prepare_dataset_is_deterministic_and_session_isolated(tmp_path):
    config = _config(tmp_path)
    sessions = []
    for index in range(12):
        date = f"2026-01-{index + 1:02d}T00:00:00"
        sessions.append({
            "session_id": f"session-{index:02d}",
            "messages": [
                _message(index * 2, "user", f"question {index}", date),
                _message(index * 2 + 1, "assistant", f"answer {index}", date),
            ],
        })
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
    assert manifest["counts"] == second_manifest["counts"] == {
        "train": 9,
        "validation": 1,
        "test": 2,
    }
    assert manifest["exclusions"]["non_personal_chat"] == 1
    seen = {}
    for split in ("train", "validation", "test"):
        for line in (config.data.output_dir / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["messages"][0]["role"] == "system"
            assert row["messages"][1]["role"] == "user"
            assert row["messages"][-1]["role"] == "assistant"
            assert row["timestamp"].startswith("2026-01-")
            assert row["session_id"] not in seen
            seen[row["session_id"]] = split
