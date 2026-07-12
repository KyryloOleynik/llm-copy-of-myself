from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from personal_ai.config import AppConfig
from personal_ai.prompts import relationship_system_message
from personal_ai.supplemental import build_supplemental_examples
from personal_ai.tokenization import token_ids


def message_text(message: dict[str, Any]) -> str:
    """Normalize Telegram string/rich text without altering its content."""
    text = message.get("text", "")
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, list):
        parts: list[str] = []
        for part in text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "".join(parts).strip()
    return ""


def media_placeholder(message: dict[str, Any]) -> str:
    """Represent unsupported Telegram media with a stable text placeholder."""
    media_type = message.get("media_type")
    if message.get("photo"):
        kind = "image"
    elif media_type == "voice_message":
        kind = "voice message"
    elif media_type in {"video_message", "video_file"}:
        kind = "video"
    elif media_type == "audio_file":
        kind = "audio file"
    elif media_type == "animation":
        kind = "animation"
    elif message.get("location_information"):
        kind = "location"
    elif message.get("contact_information"):
        kind = "contact"
    elif message.get("poll"):
        kind = "poll"
    elif message.get("file"):
        kind = "document"
    else:
        kind = "media"
    return f"[sent {kind}]"


def merge_turns(messages: Iterable[dict[str, Any]], owner_id: str) -> list[dict[str, Any]]:
    """Merge consecutive messages from the same side and preserve media markers."""
    turns: list[dict[str, Any]] = []
    for message in messages:
        text = message_text(message)
        is_media_only = not bool(text)
        content = text or media_placeholder(message)
        role = "assistant" if message.get("from_id") == owner_id else "user"
        item = {
            "role": role,
            "content": content,
            "source_message_ids": [message.get("id")],
            "last_message_date": message.get("date"),
            "media_only": is_media_only,
        }
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + content
            turns[-1]["source_message_ids"].append(message.get("id"))
            turns[-1]["last_message_date"] = message.get("date")
            turns[-1]["media_only"] = turns[-1]["media_only"] and is_media_only
        else:
            turns.append(item)
    return turns


def split_sessions(
    sessions: list[dict[str, Any]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Split complete sessions chronologically without cross-split leakage."""
    ordered = sorted(
        sessions,
        key=lambda session: (
            session["messages"][0].get("date", "") if session.get("messages") else "",
            session.get("session_id", ""),
        ),
    )
    count = len(ordered)
    if count < 3:
        return {"train": ordered, "validation": [], "test": []}
    if count < 10:
        return {
            "train": ordered[:-2],
            "validation": ordered[-2:-1],
            "test": ordered[-1:],
        }
    train_end = int(count * train_ratio)
    validation_count = max(1, int(count * validation_ratio))
    validation_end = train_end + validation_count
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def _render_ids(tokenizer: Any, messages: list[dict[str, str]], generation: bool) -> list[int]:
    return token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=generation,
            enable_thinking=False,
        )
    )


def _fit_example(
    tokenizer: Any,
    system: dict[str, str],
    context: list[dict[str, Any]],
    target: dict[str, Any],
    max_length: int,
    max_target_tokens: int,
) -> tuple[list[dict[str, str]], int, int] | None:
    context = list(context)
    while context and context[0]["role"] == "assistant":
        context.pop(0)
    # Avoid rendering an arbitrarily long Telegram session only to discard its
    # oldest turns. This conservative content-only estimate bounds the first
    # real chat-template render; the exact loop below remains authoritative.
    fixed_tokens = len(tokenizer.encode(system["content"], add_special_tokens=False))
    fixed_tokens += len(tokenizer.encode(target["content"], add_special_tokens=False))
    approximate_budget = max(1, max_length - fixed_tokens - 32)
    approximate_tokens = 0
    start = len(context)
    for candidate_index in range(len(context) - 1, -1, -1):
        turn_tokens = len(
            tokenizer.encode(context[candidate_index]["content"], add_special_tokens=False)
        ) + 4
        if approximate_tokens and approximate_tokens + turn_tokens > approximate_budget:
            break
        approximate_tokens += turn_tokens
        start = candidate_index
    context = context[start:]
    while context and context[0]["role"] == "assistant":
        context.pop(0)
    while context and any(turn["role"] == "user" for turn in context):
        prompt = [system] + [
            {"role": turn["role"], "content": turn["content"]} for turn in context
        ]
        messages = prompt + [{"role": "assistant", "content": target["content"]}]
        prompt_ids = _render_ids(tokenizer, prompt, generation=True)
        full_ids = _render_ids(tokenizer, messages, generation=False)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Prompt is not a prefix of the complete chat template")
        target_count = len(full_ids) - len(prompt_ids)
        if target_count <= 0:
            raise ValueError("Chat template produced an empty assistant target")
        if target_count > max_target_tokens:
            return None
        if len(full_ids) <= max_length:
            return messages, len(full_ids), target_count
        context.pop(0)
        while context and context[0]["role"] == "assistant":
            context.pop(0)
    return None


def _build_session_examples(
    tokenizer: Any,
    config: AppConfig,
    chat: dict[str, Any],
    session: dict[str, Any],
    split: str,
    exclusions: Counter[str],
) -> list[dict[str, Any]]:
    turns = merge_turns(session.get("messages", []), config.project.owner_from_id)
    relationship = chat.get("relationship", "unknown")
    system = {"role": "system", "content": relationship_system_message(relationship)}
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(turns):
        if target["role"] != "assistant" or index == 0:
            continue
        context = turns[:index]
        while context and context[0]["role"] == "assistant":
            context = context[1:]
        if not context or not any(turn["role"] == "user" for turn in context):
            exclusions["missing_user_context"] += 1
            continue
        if context[-1]["role"] == "user" and context[-1]["media_only"]:
            exclusions["immediately_preceded_by_media_only_user_turn"] += 1
            continue
        fitted = _fit_example(
            tokenizer,
            system,
            context,
            target,
            config.model.sequence_length,
            config.data.max_target_tokens,
        )
        if fitted is None:
            exclusions["target_too_long_or_context_cannot_fit"] += 1
            continue
        messages, sequence_tokens, target_tokens = fitted
        rows.append(
            {
                "example_id": f"{session['session_id']}_reply_{index:04d}",
                "chat_id": f"chat_{chat['id']}",
                "session_id": session["session_id"],
                "relationship": relationship,
                "source_type": "personal_telegram",
                "split": split,
                "timestamp": target["last_message_date"],
                "target_message_ids": target["source_message_ids"],
                "sequence_tokens": sequence_tokens,
                "target_tokens": target_tokens,
                "messages": messages,
            }
        )
    return rows


def _deterministic_limit(
    rows: list[dict[str, Any]], limit: int, seed: int, namespace: str
) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    rng = random.Random(f"{seed}:{namespace}")
    selected = rng.sample(rows, limit)
    return sorted(selected, key=lambda row: (row["timestamp"], row["example_id"]))


def _balance_split(
    rows: list[dict[str, Any]], tokenizer: Any, config: AppConfig, split: str
) -> tuple[list[dict[str, Any]], Counter[str]]:
    exclusions: Counter[str] = Counter()
    short_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    retained: list[dict[str, Any]] = []
    for row in rows:
        target = row["messages"][-1]["content"]
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if len(target_ids) <= config.data.short_target_max_tokens:
            short_groups[(row["relationship"], target.strip().casefold())].append(row)
        else:
            retained.append(row)
    for key, group in sorted(short_groups.items()):
        kept = _deterministic_limit(
            group,
            config.data.max_identical_short_target,
            config.training.seed,
            f"{split}:short:{key[0]}:{key[1]}",
        )
        exclusions["duplicate_short_target_cap"] += len(group) - len(kept)
        retained.extend(kept)

    if split == "train":
        by_chat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in retained:
            by_chat[row["chat_id"]].append(row)
        retained = []
        for chat_id, group in sorted(by_chat.items()):
            kept = _deterministic_limit(
                group,
                config.data.max_examples_per_chat,
                config.training.seed,
                f"{split}:chat:{chat_id}",
            )
            exclusions["per_chat_cap"] += len(group) - len(kept)
            retained.extend(kept)
    retained.sort(key=lambda row: (row["timestamp"], row["example_id"]))
    return retained, exclusions


def _percentiles(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_dataset(config: AppConfig, tokenizer: Any) -> dict[str, Any]:
    """Build deterministic tokenizer-budgeted examples from cleaned sessions."""
    with config.data.cleaned.open("r", encoding="utf-8") as source_file:
        cleaned = json.load(source_file)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    split_boundaries: dict[str, dict[str, str | None]] = {}
    exclusions: Counter[str] = Counter()
    session_owners: dict[str, str] = {}
    for chat in cleaned.get("chats", []):
        if chat.get("type") != "personal_chat":
            exclusions["non_personal_chat"] += 1
            continue
        allocated = split_sessions(
            chat.get("sessions", []),
            config.data.train_ratio,
            config.data.validation_ratio,
        )
        chat_id = f"chat_{chat['id']}"
        split_boundaries[chat_id] = {}
        for split, sessions in allocated.items():
            split_boundaries[chat_id][f"{split}_last_session"] = (
                sessions[-1]["session_id"] if sessions else None
            )
            for session in sessions:
                session_id = session["session_id"]
                previous = session_owners.setdefault(session_id, split)
                if previous != split:
                    raise ValueError(f"Session {session_id} appears in multiple splits")
                splits[split].extend(
                    _build_session_examples(tokenizer, config, chat, session, split, exclusions)
                )

    for split in splits:
        splits[split], balancing = _balance_split(splits[split], tokenizer, config, split)
        exclusions.update(balancing)
        personal_count = len(splits[split])
        for category, ratio in (
            ("context_retention", config.data.context_retention_ratio),
            ("general_reasoning", config.data.general_reasoning_ratio),
        ):
            supplemental_count = round(
                personal_count * ratio / config.data.personal_data_ratio
            )
            splits[split].extend(build_supplemental_examples(
                tokenizer=tokenizer,
                split=split,
                category=category,
                count=supplemental_count,
                max_length=config.model.sequence_length,
                max_target_tokens=config.data.max_target_tokens,
                seed=config.training.seed,
            ))
        splits[split].sort(key=lambda row: (row["timestamp"], row["example_id"]))

    output = config.data.output_dir
    output.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        artifact_paths[split] = path

    combined = [row for split in ("train", "validation", "test") for row in splits[split]]
    with config.data.dataset.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(combined, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")

    relationship_counts = {
        split: dict(sorted(Counter(row["relationship"] for row in rows).items()))
        for split, rows in splits.items()
    }
    source_type_counts = {
        split: dict(sorted(Counter(row["source_type"] for row in rows).items()))
        for split, rows in splits.items()
    }
    chat_counts = {
        split: Counter(row["chat_id"] for row in rows) for split, rows in splits.items()
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contains_unredacted_private_data": True,
        "warning": (
            "Raw data, generated datasets, adapters, checkpoints, logs, and indexes "
            "contain intentionally unredacted private data and must remain private."
        ),
        "model": config.model.base_model,
        "sequence_length": config.model.sequence_length,
        "max_target_tokens": config.data.max_target_tokens,
        "seed": config.training.seed,
        "raw_source": str(config.data.source),
        "raw_source_sha256": _sha256(config.data.source),
        "source": str(config.data.cleaned),
        "source_sha256": _sha256(config.data.cleaned),
        "dataset_sha256": _sha256(config.data.dataset),
        "artifact_sha256": {name: _sha256(path) for name, path in artifact_paths.items()},
        "counts": {name: len(rows) for name, rows in splits.items()},
        "exclusions": dict(sorted(exclusions.items())),
        "split_method": "complete sessions, chronological within each chat",
        "split_boundaries": split_boundaries,
        "relationship_counts": relationship_counts,
        "source_type_counts": source_type_counts,
        "configured_mixture": {
            "personal_telegram": config.data.personal_data_ratio,
            "context_retention": config.data.context_retention_ratio,
            "general_reasoning": config.data.general_reasoning_ratio,
        },
        "sequence_token_distribution": {
            name: _percentiles([row["sequence_tokens"] for row in rows])
            for name, rows in splits.items()
        },
        "target_token_distribution": {
            name: _percentiles([row["target_tokens"] for row in rows])
            for name, rows in splits.items()
        },
        "chat_dominance": {
            name: [
                {"chat_id": chat_id, "examples": count}
                for chat_id, count in counts.most_common(10)
            ]
            for name, counts in chat_counts.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
