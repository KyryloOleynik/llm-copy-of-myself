#!/usr/bin/env python3
"""Convert cleaned Telegram sessions into model-ready chat examples."""

import json
from pathlib import Path


OWNER_ID = "user624349412"
MAX_CONTEXT_TURNS = 8


def message_text(message):
    """Return plain text from Telegram's string or rich-text representation."""
    text = message.get("text", "")
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, list):
        parts = []
        for part in text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "".join(parts).strip()
    return ""


def merge_turns(messages):
    """Merge consecutive non-empty messages sent by the same side."""
    turns = []
    for message in messages:
        content = message_text(message)
        if not content:
            continue

        role = "assistant" if message.get("from_id") == OWNER_ID else "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + content
            turns[-1]["source_message_ids"].append(message.get("id"))
        else:
            turns.append(
                {
                    "role": role,
                    "content": content,
                    "source_message_ids": [message.get("id")],
                }
            )
    return turns


def build_examples(cleaned):
    examples = []
    for chat in cleaned["chats"]:
        for session in chat["sessions"]:
            turns = merge_turns(session["messages"])
            for index, target in enumerate(turns):
                if target["role"] != "assistant" or index == 0:
                    continue

                context = turns[max(0, index - MAX_CONTEXT_TURNS) : index]
                if not any(turn["role"] == "user" for turn in context):
                    continue

                messages = [
                    {"role": turn["role"], "content": turn["content"]}
                    for turn in context
                ]
                messages.append(
                    {"role": "assistant", "content": target["content"]}
                )
                examples.append(
                    {
                        "example_id": f"{session['session_id']}_reply_{index:04d}",
                        "chat_id": f"chat_{chat['id']}",
                        "relationship": chat["relationship"],
                        "timestamp": session["messages"][0]["date"],
                        "target_message_ids": target["source_message_ids"],
                        "messages": messages,
                    }
                )
    return examples


def main():
    project_dir = Path(__file__).resolve().parent
    source_path = project_dir / "DataExport_2026-07-10" / "cleaned_sessions.json"
    output_path = project_dir / "DataExport_2026-07-10" / "dataset.json"

    with source_path.open("r", encoding="utf-8") as source_file:
        cleaned = json.load(source_file)

    examples = build_examples(cleaned)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(examples, output_file, ensure_ascii=False, indent=2)

    print(f"Created {len(examples)} examples at {output_path}")


if __name__ == "__main__":
    main()
