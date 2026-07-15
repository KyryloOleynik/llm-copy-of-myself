#!/usr/bin/env python3
"""Clean a Telegram JSON export and split personal chats into training sessions."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from personal_ai.utils import read_json, write_json

SESSION_GAP_SECONDS = 12 * 60 * 60


def clean_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Return a cleaned message, or ``None`` when it should be excluded."""
    if message.get("type") == "service":
        return None

    if message.get("media_type") == "sticker":
        emoji = message.get("sticker_emoji")
        if not emoji:
            return None
        return {
            key: value
            for key, value in message.items()
            if key
            in {
                "id",
                "type",
                "date",
                "date_unixtime",
                "edited",
                "edited_unixtime",
                "from",
                "from_id",
                "reply_to_message_id",
            }
        } | {"text": emoji, "text_entities": [{"type": "plain", "text": emoji}]}

    return deepcopy(message)


def split_messages_into_sessions(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a chat whenever the gap between messages is greater than 12 hours."""
    sessions: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_timestamp: int | None = None

    for message in messages:
        timestamp = int(message["date_unixtime"])
        if previous_timestamp is not None and timestamp - previous_timestamp > SESSION_GAP_SECONDS:
            if current:
                sessions.append(current)
            current = []
        current.append(message)
        previous_timestamp = timestamp

    if current:
        sessions.append(current)
    return sessions


def clean_export(
    source: dict[str, Any], relationships: dict[str, str]
) -> dict[str, Any]:
    """Clean a Telegram desktop export without embedding personal labels in source code."""
    output_chats: list[dict[str, Any]] = []
    stats = {
        "source_chats": len(source["chats"]["list"]),
        "saved_messages_chats_removed": 0,
        "telegram_chats_removed": 0,
        "chats_with_one_or_fewer_messages_removed": 0,
        "service_records_removed": 0,
        "stickers_replaced_with_emoji": 0,
        "stickers_without_emoji_removed": 0,
        "messages_kept": 0,
        "sessions_created": 0,
    }

    for chat in source["chats"]["list"]:
        if chat.get("type") == "saved_messages":
            stats["saved_messages_chats_removed"] += 1
            continue
        if chat.get("name") == "Telegram":
            stats["telegram_chats_removed"] += 1
            continue

        cleaned_messages: list[dict[str, Any]] = []
        for original in chat.get("messages", []):
            if original.get("type") == "service":
                stats["service_records_removed"] += 1
            elif original.get("media_type") == "sticker":
                if original.get("sticker_emoji"):
                    stats["stickers_replaced_with_emoji"] += 1
                else:
                    stats["stickers_without_emoji_removed"] += 1

            cleaned = clean_message(original)
            if cleaned is not None:
                cleaned_messages.append(cleaned)

        if len(cleaned_messages) <= 1:
            stats["chats_with_one_or_fewer_messages_removed"] += 1
            continue

        sessions = split_messages_into_sessions(cleaned_messages)
        output_chats.append(
            {
                "name": chat.get("name"),
                "type": chat.get("type"),
                "id": chat.get("id"),
                "relationship": (
                    relationships.get(str(chat.get("name")), "unknown")
                    if chat.get("type") == "personal_chat"
                    else "not_applicable"
                ),
                "sessions": [
                    {
                        "session_id": f"chat_{chat.get('id')}_session_{index:04d}",
                        "started_at": messages[0]["date"],
                        "ended_at": messages[-1]["date"],
                        "messages": messages,
                    }
                    for index, messages in enumerate(sessions, start=1)
                ],
            }
        )
        stats["messages_kept"] += len(cleaned_messages)
        stats["sessions_created"] += len(sessions)

    return {
        "cleaning_rules": {
            "session_gap_hours": 12,
            "session_split_condition": "gap strictly greater than 12 hours",
            "service_records": "removed",
            "saved_messages": "removed",
            "telegram_chat": "removed",
            "stickers_with_emoji": "replaced by emoji text",
            "stickers_without_emoji": "removed",
            "other_conversation_content": "preserved",
        },
        "stats": stats,
        "chats": output_chats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("private_data/result.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("private_data/cleaned_sessions.json")
    )
    parser.add_argument(
        "--relationships",
        type=Path,
        default=Path("private_data/relationships.json"),
        help="Private JSON mapping of Telegram chat names to relationship categories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relationships = read_json(args.relationships) if args.relationships.is_file() else {}
    cleaned = clean_export(read_json(args.source), relationships)
    write_json(args.output, cleaned, sort_keys=False)
    print(json.dumps(cleaned["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
