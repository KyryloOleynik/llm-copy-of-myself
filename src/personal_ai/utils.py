from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def load_dotenv(path: Path) -> None:
    """Load a simple KEY=VALUE file without overwriting existing environment values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, value: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        target.write("\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_ids(rendered: Any) -> list[int]:
    """Normalize Transformers 4/5 chat-template results to one token-id list."""
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], (list, tuple)):
        if len(rendered) != 1:
            raise ValueError("Expected one rendered conversation")
        rendered = rendered[0]
    return [int(value) for value in rendered]


def normalize_messages_for_storage(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Store flexible tool messages in one stable Arrow/JSON schema."""
    return [
        {
            "content": str(message.get("content") or ""),
            "name": str(message.get("name") or ""),
            "role": str(message["role"]),
            "tool_calls": json.dumps(
                message.get("tool_calls", []),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        for message in messages
    ]


def materialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert stored string fields back to Transformers' native tool-message format."""
    materialized: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {
            "role": str(message["role"]),
            "content": str(message.get("content") or ""),
        }
        name = message.get("name")
        if name:
            item["name"] = str(name)
        calls = message.get("tool_calls")
        if isinstance(calls, str):
            calls = json.loads(calls) if calls else []
        if calls:
            item["tool_calls"] = calls
        materialized.append(item)
    return materialized


def decode_tools(tools: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if isinstance(tools, str):
        return json.loads(tools) if tools else []
    return list(tools or [])


def assistant_target_text(messages: list[dict[str, Any]]) -> str:
    """Return a readable, unique representation of the final supervised target."""
    target = materialize_messages(messages)[-1]
    calls = target.get("tool_calls")
    if calls:
        return json.dumps(calls, ensure_ascii=False, sort_keys=True)
    return str(target.get("content") or "")


def render_chat_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    generation: bool,
    tools: str | list[dict[str, Any]] | None = None,
) -> list[int]:
    materialized = materialize_messages(messages)
    decoded_tools = decode_tools(tools)
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": generation,
    }
    if decoded_tools:
        template_kwargs["tools"] = decoded_tools
    return token_ids(
        tokenizer.apply_chat_template(
            materialized,
            **template_kwargs,
        )
    )


def assistant_target_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: str | list[dict[str, Any]] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Render one chat and return verified prompt, complete, and final-target IDs."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Every example must end with an assistant target")
    prompt_ids = render_chat_ids(tokenizer, messages[:-1], generation=True, tools=tools)
    full_ids = render_chat_ids(tokenizer, messages, generation=False, tools=tools)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt is not a prefix of the complete example")
    target_ids = full_ids[len(prompt_ids) :]
    if not target_ids:
        raise ValueError("Example has no trainable assistant target tokens")
    return prompt_ids, full_ids, target_ids


def assistant_target_spans(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: str | list[dict[str, Any]] | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return the complete chat and exact token spans for every assistant turn."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Every example must end with an assistant target")
    full_ids = render_chat_ids(tokenizer, messages, generation=False, tools=tools)
    spans: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt_ids = render_chat_ids(
            tokenizer,
            messages[:index],
            generation=True,
            tools=tools,
        )
        through_target_ids = render_chat_ids(
            tokenizer,
            messages[: index + 1],
            generation=False,
            tools=tools,
        )
        if through_target_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Assistant prompt is not a prefix of its completed turn")
        if full_ids[: len(through_target_ids)] != through_target_ids:
            raise ValueError("Assistant turn is not a prefix of the complete conversation")
        if len(through_target_ids) == len(prompt_ids):
            raise ValueError("Assistant turn has no trainable target tokens")
        spans.append((len(prompt_ids), len(through_target_ids)))
    if not spans:
        raise ValueError("Example has no assistant target")
    return full_ids, spans


def relationship_system_message(relationship: str) -> str:
    """Return the persona prompt shared by preparation and inference."""
    return (
        "Отвечай в стиле Родиона. Отношения с собеседником: "
        f"{relationship}. Выбирай естественную для контекста длину ответа. "
        "Не утверждай, что ты настоящий Родион."
    )
