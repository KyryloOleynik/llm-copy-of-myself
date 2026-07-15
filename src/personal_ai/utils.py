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


def render_chat_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    generation: bool,
) -> list[int]:
    return token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=generation,
        )
    )


def assistant_target_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int], list[int]]:
    """Render one chat and return verified prompt, complete, and final-target IDs."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Every example must end with an assistant target")
    prompt_ids = render_chat_ids(tokenizer, messages[:-1], generation=True)
    full_ids = render_chat_ids(tokenizer, messages, generation=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt is not a prefix of the complete example")
    target_ids = full_ids[len(prompt_ids) :]
    if not target_ids:
        raise ValueError("Example has no trainable assistant target tokens")
    return prompt_ids, full_ids, target_ids


def relationship_system_message(relationship: str) -> str:
    """Return the persona prompt shared by preparation and inference."""
    return (
        "Отвечай в стиле Родиона. Отношения с собеседником: "
        f"{relationship}. Выбирай естественную для контекста длину ответа. "
        "Не утверждай, что ты настоящий Родион."
    )
