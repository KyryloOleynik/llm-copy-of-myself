from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from personal_ai.config import AppConfig
from personal_ai.supplemental import (
    build_additional_tool_example,
    build_calendar_tool_examples,
    build_supplemental_examples,
)
from personal_ai.utils import (
    assistant_target_ids,
    assistant_target_text,
    normalize_messages_for_storage,
    read_json,
    relationship_system_message,
    sha256_file,
    write_json,
    write_jsonl,
)


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
    """Merge turns while never teaching the assistant to emit media placeholders."""
    turns: list[dict[str, Any]] = []
    for message in messages:
        text = message_text(message)
        role = "assistant" if message.get("from_id") == owner_id else "user"
        is_media_only = not bool(text)
        if role == "assistant" and is_media_only:
            continue
        content = text if role == "assistant" else (text or media_placeholder(message))
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


def split_dataset_sessions(
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


def _fit_example(
    tokenizer: Any,
    system: dict[str, str],
    context: list[dict[str, Any]],
    target: dict[str, Any],
    max_length: int,
    max_target_tokens: int,
) -> tuple[list[dict[str, str]], int, int, int] | None:
    context = list(context)
    # Avoid rendering an arbitrarily long Telegram session only to discard its
    # oldest turns. This conservative content-only estimate bounds the first
    # real chat-template render; the exact loop below remains authoritative.
    fixed_tokens = len(tokenizer.encode(system["content"], add_special_tokens=False))
    fixed_tokens += len(tokenizer.encode(target["content"], add_special_tokens=False))
    approximate_budget = max(1, max_length - fixed_tokens - 32)
    approximate_tokens = 0
    start = len(context)
    for candidate_index in range(len(context) - 1, -1, -1):
        turn_tokens = (
            len(tokenizer.encode(context[candidate_index]["content"], add_special_tokens=False)) + 4
        )
        if approximate_tokens and approximate_tokens + turn_tokens > approximate_budget:
            break
        approximate_tokens += turn_tokens
        start = candidate_index
    context = context[start:]
    while context and any(turn["role"] == "user" for turn in context):
        prompt = [system] + [{"role": turn["role"], "content": turn["content"]} for turn in context]
        messages = prompt + [{"role": "assistant", "content": target["content"]}]
        prompt_ids, full_ids, target_ids = assistant_target_ids(tokenizer, messages)
        target_count = len(target_ids)
        if target_count > max_target_tokens:
            return None
        if len(full_ids) <= max_length:
            return messages, len(full_ids), target_count, 0
        remaining_context = context[1:]
        if not any(turn["role"] == "user" for turn in remaining_context):
            overflow = len(full_ids) - max_length
            if overflow >= len(prompt_ids):
                return None
            # Keep the readable messages intact in the dataset. ReplyOnlyCollator
            # applies this exact left-side token truncation during training while
            # preserving every final assistant target token.
            return messages, max_length, target_count, overflow
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
        messages, sequence_tokens, target_tokens, truncated_prompt_tokens = fitted
        if truncated_prompt_tokens:
            exclusions["examples_with_token_level_prompt_truncation"] += 1
            exclusions["prompt_tokens_truncated"] += truncated_prompt_tokens
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
                "messages": normalize_messages_for_storage(messages),
                "tools": "[]",
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


def _balanced_chat_limit(
    rows: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    """Keep the requested total while removing excess primarily from dominant chats."""
    if len(rows) <= limit:
        return rows
    by_chat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chat[row["chat_id"]].append(row)

    low, high = 1, max(len(group) for group in by_chat.values())
    while low < high:
        cap = (low + high) // 2
        if sum(min(len(group), cap) for group in by_chat.values()) >= limit:
            high = cap
        else:
            low = cap + 1
    cap = low
    allocations = {chat_id: min(len(group), cap - 1) for chat_id, group in by_chat.items()}
    remaining = limit - sum(allocations.values())
    eligible = sorted(
        (chat_id for chat_id, group in by_chat.items() if len(group) >= cap),
        key=lambda chat_id: random.Random(f"{seed}:chat-allocation:{chat_id}").random(),
    )
    for chat_id in eligible[:remaining]:
        allocations[chat_id] += 1

    selected: list[dict[str, Any]] = []
    for chat_id, group in sorted(by_chat.items()):
        selected.extend(
            _deterministic_limit(
                group,
                allocations[chat_id],
                seed,
                f"train:personal-chat:{chat_id}",
            )
        )
    if len(selected) != limit:
        raise RuntimeError(f"Balanced personal selection produced {len(selected)} rows, not {limit}")
    return sorted(selected, key=lambda row: (row["timestamp"], row["example_id"]))


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


def _personal_style_samples(rows: list[dict[str, Any]]) -> list[str]:
    samples: list[str] = []
    seen: set[str] = set()
    for row in rows:
        content = str(row["messages"][-1].get("content") or "").strip()
        if (
            content
            and len(content) <= 80
            and len(content.split()) <= 10
            and content not in seen
        ):
            seen.add(content)
            samples.append(content)
    return samples


def _validate_synthetic_examples(rows: list[dict[str, Any]]) -> dict[str, int]:
    synthetic = [row for row in rows if row["source_type"] != "personal_telegram"]
    fingerprints: set[str] = set()
    targets: set[str] = set()
    persona_prefix = "Отвечай в стиле Родиона."
    for row in synthetic:
        messages = row["messages"]
        if not messages or not messages[0]["content"].startswith(persona_prefix):
            raise ValueError(f"{row['example_id']} does not preserve the owner persona prompt")
        fingerprint = json.dumps(
            {"messages": messages, "tools": row.get("tools", "[]")},
            ensure_ascii=False,
            sort_keys=True,
        )
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate synthetic conversation: {row['example_id']}")
        fingerprints.add(fingerprint)
        target = assistant_target_text(messages).strip().casefold()
        if not target or target in targets:
            raise ValueError(f"Duplicate or empty synthetic target: {row['example_id']}")
        targets.add(target)
        if row["source_type"] in {"tool_calling", "rag_retrieval"}:
            if not json.loads(row.get("tools", "[]")):
                raise ValueError(f"{row['example_id']} is missing native tool definitions")
            target_calls = json.loads(messages[-1].get("tool_calls", "[]"))
            if not target_calls:
                tool_messages = [message for message in messages if message["role"] == "tool"]
                prior_calls = [
                    message
                    for message in messages[:-1]
                    if message["role"] == "assistant"
                    and json.loads(message.get("tool_calls", "[]"))
                ]
                if not tool_messages or not prior_calls:
                    raise ValueError(
                        f"{row['example_id']} must contain a tool call and returned result"
                    )
                returned_text = " ".join(
                    message["content"] for message in tool_messages
                ).casefold()
                answer_text = messages[-1]["content"].casefold()
                returned_terms = set(re.findall(r"[\w-]{3,}", returned_text)) | set(
                    re.findall(r"\d{1,2}:\d{2}", returned_text)
                )
                answer_terms = set(re.findall(r"[\w-]{3,}", answer_text)) | set(
                    re.findall(r"\d{1,2}:\d{2}", answer_text)
                )
                empty_result = all(
                    not json.loads(message["content"]).get("results")
                    for message in tool_messages
                    if message.get("name") == "search_personal_memory"
                ) if all(
                    message.get("name") == "search_personal_memory"
                    for message in tool_messages
                ) else False
                if empty_result:
                    if not {"ничего", "нашел", "нашёл"} & answer_terms:
                        raise ValueError(
                            f"{row['example_id']} does not answer an empty tool result"
                        )
                elif not returned_terms & answer_terms:
                    raise ValueError(
                        f"{row['example_id']} final answer is not grounded in its tool result"
                    )
    return {
        "examples": len(synthetic),
        "unique_conversations": len(fingerprints),
        "unique_targets": len(targets),
    }


def prepare_dataset(config: AppConfig, tokenizer: Any) -> dict[str, Any]:
    """Build deterministic tokenizer-budgeted examples from cleaned sessions."""
    cleaned = read_json(config.data.cleaned)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    split_boundaries: dict[str, dict[str, str | None]] = {}
    exclusions: Counter[str] = Counter()
    session_owners: dict[str, str] = {}
    for chat in cleaned.get("chats", []):
        if chat.get("type") != "personal_chat":
            exclusions["non_personal_chat"] += 1
            continue
        if chat.get("relationship", "unknown") == "unknown":
            exclusions["unknown_relationship_chat"] += 1
            continue
        allocated = split_dataset_sessions(
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
        if split == "train" and config.data.personal_train_examples is not None:
            available = len(splits[split])
            requested = config.data.personal_train_examples
            if available < requested:
                raise ValueError(
                    f"Only {available} personal training examples are available; "
                    f"{requested} were requested"
                )
            splits[split] = _balanced_chat_limit(
                splits[split], requested, config.training.seed
            )
            exclusions["personal_train_global_limit"] += available - requested
        personal_count = len(splits[split])
        style_samples = _personal_style_samples(splits[split])
        for category, ratio in (
            ("context_retention", config.data.context_retention_ratio),
            ("general_reasoning", config.data.general_reasoning_ratio),
            ("instruction_following", config.data.instruction_following_ratio),
            ("tool_calling", config.data.tool_calling_ratio),
            ("rag_retrieval", config.data.rag_retrieval_ratio),
        ):
            supplemental_count = round(personal_count * ratio / config.data.personal_data_ratio)
            splits[split].extend(
                build_supplemental_examples(
                    tokenizer=tokenizer,
                    split=split,
                    category=category,
                    count=supplemental_count,
                    max_length=config.model.sequence_length,
                    max_target_tokens=config.data.max_target_tokens,
                    seed=config.training.seed,
                    style_samples=style_samples,
                )
            )
        if split == "train" and config.data.tool_calling_ratio > 0:
            splits[split].append(
                build_additional_tool_example(
                    tokenizer,
                    config.model.sequence_length,
                    config.data.max_target_tokens,
                )
            )
            splits[split].extend(
                build_calendar_tool_examples(
                    tokenizer,
                    config.model.sequence_length,
                    config.data.max_target_tokens,
                )
            )
        splits[split].sort(key=lambda row: (row["timestamp"], row["example_id"]))

    output = config.data.output_dir
    output.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        write_jsonl(path, rows)
        artifact_paths[split] = path

    combined = [row for split in ("train", "validation", "test") for row in splits[split]]
    synthetic_uniqueness = _validate_synthetic_examples(combined)
    write_json(config.data.dataset, combined)

    relationship_counts = {
        split: dict(sorted(Counter(row["relationship"] for row in rows).items()))
        for split, rows in splits.items()
    }
    source_type_counts = {
        split: dict(sorted(Counter(row["source_type"] for row in rows).items()))
        for split, rows in splits.items()
    }
    chat_counts = {split: Counter(row["chat_id"] for row in rows) for split, rows in splits.items()}
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
        "raw_source_sha256": sha256_file(config.data.source),
        "source": str(config.data.cleaned),
        "source_sha256": sha256_file(config.data.cleaned),
        "dataset_sha256": sha256_file(config.data.dataset),
        "artifact_sha256": {name: sha256_file(path) for name, path in artifact_paths.items()},
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
            "instruction_following": config.data.instruction_following_ratio,
            "tool_calling": config.data.tool_calling_ratio,
            "rag_retrieval": config.data.rag_retrieval_ratio,
        },
        "synthetic_uniqueness": synthetic_uniqueness,
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
                {"chat_id": chat_id, "examples": count} for chat_id, count in counts.most_common(10)
            ]
            for name, counts in chat_counts.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest
