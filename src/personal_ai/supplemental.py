from __future__ import annotations

import json
import random
from typing import Any

from personal_ai.tools import TOOL_SCHEMAS, calculate
from personal_ai.utils import (
    assistant_target_ids,
    assistant_target_text,
    normalize_messages_for_storage,
    relationship_system_message,
)


def _style_ack(rng: random.Random, style_samples: list[str]) -> str:
    """Use only replies that are semantically safe as acknowledgements."""
    allowed = {
        "ага",
        "да",
        "договорились",
        "ок",
        "пон",
        "понял",
        "хорошо",
        "угу",
        "ясно",
    }
    candidates = [
        sample.strip()
        for sample in style_samples
        if sample.strip().casefold() in allowed
    ]
    return rng.choice(candidates) if candidates else rng.choice(("ага", "понял", "ок", "ясно"))


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _row(
    tokenizer: Any,
    split: str,
    category: str,
    index: int,
    messages: list[dict[str, Any]],
    max_length: int,
    max_target_tokens: int,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tools = tools or []
    _, full_ids, target_ids = assistant_target_ids(tokenizer, messages, tools)
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
        "messages": normalize_messages_for_storage(messages),
        "tools": json.dumps(tools, ensure_ascii=False, sort_keys=True),
    }


def _context_messages(
    index: int, rng: random.Random, style_samples: list[str]
) -> list[dict[str, Any]]:
    variant = index % 5
    system = {"role": "system", "content": relationship_system_message("friend")}
    acknowledgement = _style_ack(rng, style_samples)
    if variant == 0:
        code = f"COBALT-{index:05d}"
        return [
            system,
            {"role": "user", "content": f"для нового бота код проекта {code}"},
            {"role": "assistant", "content": f"{acknowledgement}, для бота {code}"},
            {"role": "user", "content": "репозиторий уже создал, вечером начну авторизацию"},
            {"role": "assistant", "content": "тогда сначала проверь вход, потом уже подключай api"},
            {"role": "user", "content": "напомни код этого проекта"},
            {"role": "assistant", "content": f"для бота код {code}"},
        ]
    if variant == 1:
        old_day, new_day = ("вторник", "среду") if index % 2 else ("пятницу", "четверг")
        hour = 10 + (index % 8)
        room = f"B-{index:06d}"
        return [
            system,
            {"role": "user", "content": f"созвон планировали на {old_day} в {hour}:00"},
            {"role": "assistant", "content": f"{acknowledgement}, {old_day} в {hour}:00"},
            {
                "role": "user",
                "content": f"перенесли на {new_day}, время то же, комната {room}",
            },
            {"role": "assistant", "content": f"ок, уже в {new_day} в {hour}:00, комната {room}"},
            {"role": "user", "content": "когда и где теперь созвон?"},
            {
                "role": "assistant",
                "content": f"в {new_day} в {hour}:00, комната {room}",
            },
        ]
    if variant == 2:
        first_value, second_value = f"amber-{index}", f"violet-{index}"
        return [
            system,
            {"role": "user", "content": f"у маши код {first_value}, а у бори {second_value}"},
            {
                "role": "assistant",
                "content": f"{acknowledgement}, у маши {first_value}, у бори {second_value}",
            },
            {"role": "user", "content": "машин код уже использовал, теперь нужен второй"},
            {"role": "assistant", "content": "тогда бери борин"},
            {"role": "user", "content": "какой там код у бори был?"},
            {"role": "assistant", "content": f"у бори {second_value}"},
        ]
    if variant == 3:
        domain, serial = divmod(index, 100_000)
        amount = 1200 + domain * 600 + serial * 37
        paid = 100 + serial * 3
        remaining = amount - paid
        return [
            system,
            {"role": "user", "content": f"я тебе должен {amount} грн, запомни"},
            {"role": "assistant", "content": f"{acknowledgement}, долг {amount} грн"},
            {"role": "user", "content": f"{paid} грн только что перевёл"},
            {"role": "assistant", "content": f"вижу, тогда осталось {remaining} грн"},
            {"role": "user", "content": "сколько ещё осталось?"},
            {"role": "assistant", "content": f"осталось {remaining} грн"},
        ]
    city = ("Киев", "Львов", "Одессу", "Днепр")[index % 4]
    train = 700 + index
    return [
        system,
        {"role": "user", "content": f"в субботу еду в {city}, поезд {train}"},
        {"role": "assistant", "content": f"{acknowledgement}, в субботу в {city}"},
        {"role": "user", "content": "билет уже взял, выезд рано утром"},
        {"role": "assistant", "content": "тогда лучше всё собрать с вечера"},
        {"role": "user", "content": "куда и на каком поезде я еду?"},
        {"role": "assistant", "content": f"в {city}, поезд {train}"},
    ]


def _reasoning_messages(index: int) -> list[dict[str, Any]]:
    variant = index % 5
    domain = index // 100_000
    serial = index % 100_000
    system = {"role": "system", "content": relationship_system_message("friend")}
    if variant == 0:
        a, b, c = 20 + domain * 10_000 + serial * 3, 3 + domain, 2
        question, answer = f"сколько будет ({a} + {b}) * {c}?", str((a + b) * c)
    elif variant == 1:
        boxes = 3 + domain * 10_000 + serial
        question = f"короче {boxes} коробок по 5 штук и одну убрали, сколько осталось?"
        answer = f"{boxes * 5 - 1} штук"
    elif variant == 2:
        rate = 30 + domain * 10_000 + serial
        question = f"если делать {rate} деталей в час 2 часа, сколько получится?"
        answer = f"{rate * 2} деталей"
    elif variant == 3:
        start = 10 + domain * 10_000 + serial * 4
        values = [start, start + 13, start + 31, start + 57]
        question = f"расставь от большего к меньшему: {', '.join(map(str, values))}"
        answer = ", ".join(map(str, reversed(values)))
    else:
        start = 8 + domain
        duration = 20 + serial
        day = ("сегодня", "завтра", "в пятницу")[domain]
        question = f"{day} начало в {start}:00, идёт {duration} минут. во сколько конец?"
        end_minutes = start * 60 + duration
        answer = f"{day} в {end_minutes // 60}:{end_minutes % 60:02d}"
    return [system, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def _instruction_messages(
    index: int, rng: random.Random, style_samples: list[str]
) -> list[dict[str, Any]]:
    system = {"role": "system", "content": relationship_system_message("friend")}
    marker = f"задача-{index:06d}"
    variant = index % 3
    if variant == 0:
        left, right = 10 + index, 2
        return [
            system,
            {
                "role": "user",
                "content": (
                    f"короче {marker}, сколько {left} плюс {right}? "
                    "ответь только json с ключом answer"
                ),
            },
            {"role": "assistant", "content": json.dumps({"answer": left + right})},
        ]
    if variant == 1:
        code = f"САПФИР-{index:06d}"
        return [
            system,
            {
                "role": "user",
                "content": f"{marker}, код {code}. потом скажи только код без объяснений",
            },
            {"role": "assistant", "content": _style_ack(rng, style_samples)},
            {"role": "user", "content": "какой код?"},
            {"role": "assistant", "content": code},
        ]
    value = 1000 + index
    return [
        system,
        {
            "role": "user",
            "content": (
                f"{marker}, сумма {value} гривен. ответь только json с answer и unit"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps({"answer": value, "unit": "грн"}, ensure_ascii=False),
        },
    ]


def _tool_messages(
    index: int, rng: random.Random, style_samples: list[str]
) -> list[dict[str, Any]]:
    system = {"role": "system", "content": relationship_system_message("friend")}
    serial = index
    variant = index % 4
    if variant == 0:
        expression = f"({20 + serial} + 3) * 2"
    elif variant == 1:
        expression = f"{40 + serial} * 5 - 7"
    elif variant == 2:
        expression = f"({90 + serial} - 6) / 3"
    else:
        expression = f"{8 + serial} ** 2 + 1"
    result = calculate(expression)
    user = {
        "role": "user",
        "content": f"посчитай точно {expression}, не угадывай, расчёт {index}",
    }
    call = _tool_call("calculate", {"expression": expression})
    if index % 2 == 0:
        return [system, user, call]
    return [
        system,
        user,
        call,
        {
            "role": "tool",
            "name": "calculate",
            "content": json.dumps({"result": result}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": f"{_style_ack(rng, style_samples)}, получается {result}",
        },
    ]


def _rag_messages(
    index: int, rng: random.Random, style_samples: list[str]
) -> list[dict[str, Any]]:
    system = {"role": "system", "content": relationship_system_message("friend")}
    marker = f"MEM-{index:06d}"
    serial = index % 100_000
    records = (
        (
            f"проект Atlas-{serial}",
            f"проект Atlas-{serial} сейчас на паузе до понедельника",
        ),
        (
            f"трек Neon-{serial}",
            f"трек Neon-{serial} добавлен в плейлист для дороги",
        ),
        (
            f"поездку номер {serial}",
            f"поездка номер {serial} запланирована в Черновцы на 24 июля",
        ),
        (
            f"книгу из списка {serial}",
            f"следующая книга в списке {serial} — «Человек в поисках смысла»",
        ),
        (
            f"идею приложения Pulse-{serial}",
            f"идея Pulse-{serial} — трекер коротких ежедневных заметок",
        ),
        (
            f"задачу Focus-{serial}",
            f"задачу Focus-{serial} нужно закончить до пятницы",
        ),
    )
    topic, detail = records[index % len(records)]
    query = f"{marker} {topic}"
    user = {
        "role": "user",
        "content": f"что написано в учебной записи {marker} про {topic}? проверь память",
    }
    call = _tool_call("search_personal_memory", {"query": query, "limit": 3})
    if index % 3 == 0:
        return [system, user, call]
    if index % 5 == 0:
        tool_result = {"results": []}
        answer = f"по записи {marker} ничего не нашел, выдумывать не буду"
    else:
        fact = f"Учебная запись {marker}: {detail}."
        tool_result = {
            "results": [
                {
                    "source": f"training/tool-use-{index}.md#0",
                    "content": fact,
                    "score": -1.0 - index / 100_000,
                }
            ]
        }
        answer = f"{_style_ack(rng, style_samples)}, по записи {marker}: {detail}"
    return [
        system,
        user,
        call,
        {
            "role": "tool",
            "name": "search_personal_memory",
            "content": json.dumps(tool_result, ensure_ascii=False),
        },
        {"role": "assistant", "content": answer},
    ]


def build_additional_tool_example(
    tokenizer: Any,
    max_length: int,
    max_target_tokens: int,
) -> dict[str, Any]:
    """Teach the model to combine two returned tool results into one answer."""
    messages = [
        {"role": "system", "content": relationship_system_message("friend")},
        {
            "role": "user",
            "content": (
                "посчитай отдельно (91 + 17) * 2 и 144 / 6, "
                "мне нужны оба точных результата"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "(91 + 17) * 2"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "144 / 6"},
                    },
                },
            ],
        },
        {
            "role": "tool",
            "name": "calculate",
            "content": json.dumps({"result": 216}, ensure_ascii=False),
        },
        {
            "role": "tool",
            "name": "calculate",
            "content": json.dumps({"result": 24}, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": "получается 216 и 24 соответственно",
        },
    ]
    return _row(
        tokenizer,
        "train",
        "tool_calling",
        90_000,
        messages,
        max_length,
        max_target_tokens,
        TOOL_SCHEMAS,
    )


def build_calendar_tool_examples(
    tokenizer: Any,
    max_length: int,
    max_target_tokens: int,
) -> list[dict[str, Any]]:
    """Teach both Google Calendar calls and grounded answers from returned data."""
    system = {
        "role": "system",
        "content": (
            relationship_system_message("friend")
            + " Текущая дата и время: 2026-07-17T12:00:00+03:00. "
            "Для относительных дат используй это время."
        ),
    }
    event_call = _tool_call(
        "query_google_calendar",
        {
            "action": "events",
            "start": "2026-07-18T12:00:00+03:00",
            "end": "2026-07-18T20:00:00+03:00",
        },
    )
    event_answer_messages = [
        system,
        {"role": "user", "content": "что у меня завтра после обеда по календарю?"},
        event_call,
        {
            "role": "tool",
            "name": "query_google_calendar",
            "content": json.dumps(
                {
                    "action": "events",
                    "time_zone": "Europe/Kyiv",
                    "events": [
                        {
                            "summary": "Созвон по проекту",
                            "start": "2026-07-18T15:00:00+03:00",
                            "end": "2026-07-18T15:45:00+03:00",
                            "location": "Google Meet",
                            "description": "",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "assistant",
            "content": "завтра в 15:00 созвон по проекту, до 15:45, в Google Meet",
        },
    ]
    free_call = _tool_call(
        "query_google_calendar",
        {
            "action": "free_time",
            "start": "2026-07-19T10:00:00+03:00",
            "end": "2026-07-19T18:00:00+03:00",
            "minimum_free_minutes": 60,
        },
    )
    free_answer_messages = [
        system,
        {"role": "user", "content": "когда я свободен в воскресенье с 10 до 18 хотя бы на час?"},
        free_call,
        {
            "role": "tool",
            "name": "query_google_calendar",
            "content": json.dumps(
                {
                    "action": "free_time",
                    "time_zone": "Europe/Kyiv",
                    "busy": [
                        {
                            "start": "2026-07-19T12:30:00+03:00",
                            "end": "2026-07-19T15:00:00+03:00",
                        }
                    ],
                    "free": [
                        {
                            "start": "2026-07-19T10:00:00+03:00",
                            "end": "2026-07-19T12:30:00+03:00",
                        },
                        {
                            "start": "2026-07-19T15:00:00+03:00",
                            "end": "2026-07-19T18:00:00+03:00",
                        },
                    ],
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "assistant",
            "content": "свободен с 10:00 до 12:30 и потом с 15:00 до 18:00",
        },
    ]
    conversations = [
        [
            system,
            {"role": "user", "content": "какие у меня планы завтра с 12 до 20?"},
            event_call,
        ],
        event_answer_messages,
        [
            system,
            {"role": "user", "content": "найди свободное окно в воскресенье с 10 до 18"},
            free_call,
        ],
        free_answer_messages,
    ]
    return [
        _row(
            tokenizer,
            "train",
            "tool_calling",
            91_000 + index,
            messages,
            max_length,
            max_target_tokens,
            TOOL_SCHEMAS,
        )
        for index, messages in enumerate(conversations)
    ]


def build_supplemental_examples(
    tokenizer: Any,
    split: str,
    category: str,
    count: int,
    max_length: int,
    max_target_tokens: int,
    seed: int,
    style_samples: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate deterministic, globally distinguishable examples in the owner's style."""
    categories = {
        "context_retention",
        "general_reasoning",
        "instruction_following",
        "tool_calling",
        "rag_retrieval",
    }
    if category not in categories:
        raise ValueError(f"Unsupported supplemental category: {category}")
    split_offset = {"train": 0, "validation": 100_000, "test": 200_000}[split]
    style_samples = style_samples or []
    rows: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    targets: set[str] = set()
    for local_index in range(count):
        index = split_offset + local_index
        rng = random.Random(f"{seed}:{category}:{split}:{index}")
        if category == "context_retention":
            messages = _context_messages(index, rng, style_samples)
        elif category == "general_reasoning":
            messages = _reasoning_messages(index)
        elif category == "instruction_following":
            messages = _instruction_messages(index, rng, style_samples)
        elif category == "tool_calling":
            messages = _tool_messages(index, rng, style_samples)
        else:
            messages = _rag_messages(index, rng, style_samples)
        tools = TOOL_SCHEMAS if category in {"tool_calling", "rag_retrieval"} else []
        fingerprint = json.dumps(
            {"messages": messages, "tools": tools},
            ensure_ascii=False,
            sort_keys=True,
        )
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate synthetic conversation generated for {category}")
        fingerprints.add(fingerprint)
        target = assistant_target_text(messages).strip().casefold()
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
                tools,
            )
        )
    return rows
