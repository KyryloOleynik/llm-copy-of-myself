#!/usr/bin/env python3
"""Clean a Telegram JSON export and split chats into training sessions."""

import json
from copy import deepcopy
from pathlib import Path


SESSION_GAP_SECONDS = 12 * 60 * 60

# Provisional English relationship labels inferred from the supplied category
# list and conversation content. These are intentionally easy to review/edit.
RELATIONSHIPS = {
    "Папа": "father",
    "Svik": "mother",
    "Mark": "close_friend",
    "Дятел 2": "close_friend",
    "Илья Крипта": "friend",
    "Петя": "friend",
    "Абсолютный": "friend",
    "Воздух": "friend",
    "Маруся": "friend",
    "Конь": "acquaintance",
    "Бонданка": "acquaintance",
    "Is Is": "acquaintance",
    "рыжий": "acquaintance",
    "Саня": "acquaintance",
    "Лёша": "acquaintance",
    "Вадик": "acquaintance",
    "Ди": "acquaintance",
    "Саша": "acquaintance",
    "#суета": "acquaintance",
    "Даря": "acquaintance",
    "Вика": "acquaintance",
    "Timmurrka": "acquaintance",
    "Евгений": "friend",
    "Yurii": "friend",
    "Женя": "friend",
    "улег": "acquaintance",
    "Павлик Морозов": "friend",
    "Миша": "friend",
    "тим": "friend",
    "Xlebuhek11": "acquaintance",
    "Вася": "acquaintance",
    "машка": "acquaintance",
    "Алина": "acquaintance",
    "c": "acquaintance",
    "_Vika_": "acquaintance",
    "Husher_Vladislava": "acquaintance",
    "Людмила": "school_acquaintance",
    "milana": "acquaintance",
    "###": "acquaintance",
    "Аля<3": "acquaintance",
    "Артур": "acquaintance",
    "Dima Zhelezniak": "acquaintance",
    "Dasha": "unknown",
    "Англ": "unknown",
    "hyoka": "unknown",
    "хуй": "unknown",
    "Наталія Миронюк": "professional_contact",
    "+380 67 630 3742": "professional_contact",
    "Саня айтишечка": "acquaintance",
    "Кирюша сайт": "acquaintance",
    "Сайт кент Красного": "professional_contact",
    "продвижение Biokmedical": "professional_contact",
    "Фотопик": "professional_contact",
    "Перевод": "professional_contact",
    "Кравченко Оксана": "professional_contact",
    "Ольга Черниш": "professional_contact",
    "Вікторія": "professional_contact",
    "Ansty": "professional_contact",
    "Vadym": "professional_contact",
    "IQ200 Дніпро": "professional_contact",
    "Шепотенко": "professional_contact",
    "Secure Shop Support": "professional_contact",
    "Incredible Store": "professional_contact",
    "Friendly": "professional_contact",
    "EntertainSubs": "professional_contact",
    "🇸 🇮 🇲 🇨 🇦 🇷 🇩": "professional_contact",
    "CompX": "professional_contact",
    "ремонт шлем": "professional_contact",
    "Олх": "professional_contact",
    "RoM4iK": "professional_contact",
    "A": "professional_contact",
    "Incredible Store": "professional_contact",
    "𝙰𝚗𝚍𝚛𝚎𝚎𝚟𝚊 𝙳𝚊𝚗𝚊": "acquaintance",
}


def clean_message(message):
    """Return a cleaned message, or None when it should be excluded."""
    if message.get("type") == "service":
        return None

    if message.get("media_type") == "sticker":
        emoji = message.get("sticker_emoji")
        if not emoji:
            return None

        # Preserve conversational timing and authorship, but discard sticker files.
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


def split_sessions(messages):
    sessions = []
    current = []
    previous_timestamp = None

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


def clean_export(source):
    output_chats = []
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

        cleaned_messages = []
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

        sessions = split_sessions(cleaned_messages)
        output_chats.append(
            {
                "name": chat.get("name"),
                "type": chat.get("type"),
                "id": chat.get("id"),
                "relationship": (
                    RELATIONSHIPS.get(chat.get("name"), "unknown")
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
            "self_name": "Родион",
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


def main():
    project_dir = Path(__file__).resolve().parent
    source_path = project_dir / "DataExport_2026-07-10" / "result.json"
    output_path = project_dir / "DataExport_2026-07-10" / "cleaned_sessions.json"

    with source_path.open("r", encoding="utf-8") as source_file:
        source = json.load(source_file)

    cleaned = clean_export(source)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(cleaned, output_file, ensure_ascii=False, indent=2)

    print(json.dumps(cleaned["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
