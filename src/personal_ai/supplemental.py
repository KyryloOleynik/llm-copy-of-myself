from __future__ import annotations

import json
import random
from typing import Any

from personal_ai.utils import assistant_target_ids, relationship_system_message


def _row(
    tokenizer: Any,
    split: str,
    category: str,
    index: int,
    messages: list[dict[str, str]],
    max_length: int,
    max_target_tokens: int,
) -> dict[str, Any]:
    _, full_ids, target_ids = assistant_target_ids(tokenizer, messages)
    if not 0 < len(target_ids) <= max_target_tokens:
        raise ValueError("Supplemental target violates the configured token budget")
    if len(full_ids) > max_length:
        raise ValueError("Supplemental example violates the configured sequence budget")
    example_id = f"supplemental_{category}_{split}_{index:05d}"
    return {
        "example_id": example_id,
        "chat_id": f"supplemental_{category}",
        "session_id": example_id,
        "relationship": "friend",
        "source_type": category,
        "split": split,
        "timestamp": f"synthetic-{split}-{index:05d}",
        "target_message_ids": [],
        "sequence_tokens": len(full_ids),
        "target_tokens": len(target_ids),
        "messages": messages,
    }


def _filler_turns(rng: random.Random, count: int) -> list[dict[str, str]]:
    topics = ("тетрадей", "треков", "фоток", "игр", "файлов", "уроков", "видосов", "книг")
    acknowledgements = ("ага", "понял", "ок", "ясно")
    turns: list[dict[str, str]] = []
    for filler_index in range(count):
        topic = rng.choice(topics)
        value = rng.randint(10, 99)
        turns.extend(
            [
                {
                    "role": "user",
                    "content": f"кстати я там насчитал {value} {topic}, запись {filler_index + 1}",
                },
                {"role": "assistant", "content": rng.choice(acknowledgements)},
            ]
        )
    return turns


def _context_messages(index: int, rng: random.Random) -> list[dict[str, str]]:
    variant = index % 5
    system = {"role": "system", "content": relationship_system_message("friend")}
    fillers = _filler_turns(rng, 2 + index % 4)
    if variant == 0:
        code = f"COBALT-{index:05d}"
        return [
            system,
            {"role": "user", "content": f"запомни код проекта {code}"},
            {"role": "assistant", "content": "ок запомнил"},
            *fillers,
            {"role": "user", "content": "какой я код проекта говорил?"},
            {"role": "assistant", "content": code},
        ]
    if variant == 1:
        old_day, new_day = (
            ("вторник", "понедельник") if index % 2 else ("пятницу", "четверг")
        )
        return [
            system,
            {"role": "user", "content": f"встреча вроде во {old_day}"},
            {"role": "assistant", "content": "понял"},
            {"role": "user", "content": f"не, перенесли на {new_day}, номер {index}"},
            {"role": "assistant", "content": "ок"},
            *fillers,
            {"role": "user", "content": "так когда теперь встреча?"},
            {"role": "assistant", "content": f"{new_day}, номер {index}"},
        ]
    if variant == 2:
        first_value, second_value = f"amber-{index}", f"violet-{index}"
        return [
            system,
            {"role": "user", "content": f"у маши код {first_value}, а у бори {second_value}"},
            {"role": "assistant", "content": "ага"},
            *fillers,
            {"role": "user", "content": "какой там код у бори был?"},
            {"role": "assistant", "content": second_value},
        ]
    if variant == 3:
        amount = 1000 + index
        return [
            system,
            {"role": "user", "content": f"я тебе должен {amount} грн, запомни"},
            {"role": "assistant", "content": "ок"},
            *fillers,
            {"role": "user", "content": "сколько я тебе там должен?"},
            {"role": "assistant", "content": f"{amount} грн"},
        ]
    city = ("Киев", "Львов", "Одессу", "Днепр")[index % 4]
    return [
        system,
        {"role": "user", "content": f"я решил ехать в {city}, если что, поездка {index}"},
        {"role": "assistant", "content": "понял"},
        *fillers,
        {"role": "user", "content": "куда я в итоге ехать хотел?"},
        {"role": "assistant", "content": f"{city}, поездка {index}"},
    ]


def _reasoning_messages(index: int) -> list[dict[str, str]]:
    variant = index % 5
    domain = index // 100_000
    serial = (index % 100_000) // 5
    system = {"role": "system", "content": relationship_system_message("friend")}
    if variant == 0:
        a, b, c = 20 + domain * 10_000 + serial * 10, 3, 2
        question, answer = f"сколько будет ({a} + {b}) * {c}?", str((a + b) * c)
    elif variant == 1:
        boxes = 3 + domain * 10_000 + serial * 10
        each, removed = 5, 1
        question = (
            f"короче {boxes} коробок по {each} штук и {removed} убрали, "
            "сколько осталось?"
        )
        answer = f"{boxes * each - removed} штук"
    elif variant == 2:
        speed, hours = 30 + domain * 10_000 + serial * 10, 2
        question = f"если ехать {speed} км в час {hours} часа, сколько км получится?"
        answer = f"{speed * hours} км"
    elif variant == 3:
        start = 10 + domain * 10_000 + serial
        values = [start, start + 13, start + 31, start + 57]
        question = f"расставь от большего к меньшему: {', '.join(map(str, values))}"
        answer = ", ".join(map(str, reversed(values)))
    else:
        start = 8 + domain
        duration = 20 + serial
        day = ("сегодня", "завтра", "в пятницу")[domain]
        question = f"{day} начало в {start}:00, идёт {duration} минут. во сколько конец?"
        end_minutes = start * 60 + duration
        answer = f"{end_minutes // 60}:{end_minutes % 60:02d}"
    return [system, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def build_supplemental_examples(
    tokenizer: Any,
    split: str,
    category: str,
    count: int,
    max_length: int,
    max_target_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate unique deterministic replay examples in the owner's concise chat style."""
    if category not in {"context_retention", "general_reasoning"}:
        raise ValueError(f"Unsupported supplemental category: {category}")
    split_offset = {"train": 0, "validation": 100_000, "test": 200_000}[split]
    rows: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    targets: set[str] = set()
    for local_index in range(count):
        index = split_offset + local_index
        rng = random.Random(f"{seed}:{category}:{split}:{index}")
        messages = (
            _context_messages(index, rng)
            if category == "context_retention"
            else _reasoning_messages(index)
        )
        fingerprint = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate synthetic conversation generated for {category}")
        fingerprints.add(fingerprint)
        target = messages[-1]["content"].strip().casefold()
        if target in targets:
            raise ValueError(f"Duplicate synthetic target generated for {category}: {target}")
        targets.add(target)
        rows.append(
            _row(
                tokenizer,
                split,
                category,
                local_index,
                messages,
                max_length,
                max_target_tokens,
            )
        )
    return rows
