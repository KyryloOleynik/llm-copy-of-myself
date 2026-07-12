from __future__ import annotations

import random
from typing import Any

from personal_ai.tokenization import token_ids


SYSTEM_PROMPT = (
    "You are a careful multilingual assistant. Use earlier conversation state, follow "
    "persistent instructions, and solve the user's task accurately. Return only the final "
    "answer unless an output format is explicitly requested."
)


def _render(tokenizer: Any, messages: list[dict[str, str]], generation: bool) -> list[int]:
    return token_ids(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation,
        enable_thinking=False,
    ))


def _row(
    tokenizer: Any,
    split: str,
    category: str,
    index: int,
    messages: list[dict[str, str]],
    max_length: int,
    max_target_tokens: int,
) -> dict[str, Any]:
    prompt_ids = _render(tokenizer, messages[:-1], generation=True)
    full_ids = _render(tokenizer, messages, generation=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Supplemental prompt is not a prefix of the full example")
    target_tokens = len(full_ids) - len(prompt_ids)
    if not 0 < target_tokens <= max_target_tokens:
        raise ValueError("Supplemental target violates the configured token budget")
    if len(full_ids) > max_length:
        raise ValueError("Supplemental example violates the configured sequence budget")
    example_id = f"supplemental_{category}_{split}_{index:05d}"
    return {
        "example_id": example_id,
        "chat_id": f"supplemental_{category}",
        "session_id": example_id,
        "relationship": "general_instruction",
        "source_type": category,
        "split": split,
        "timestamp": f"synthetic-{split}-{index:05d}",
        "target_message_ids": [],
        "sequence_tokens": len(full_ids),
        "target_tokens": target_tokens,
        "messages": messages,
    }


def _filler_turns(rng: random.Random, count: int) -> list[dict[str, str]]:
    topics = (
        "notebooks", "coffee", "weather", "keyboards", "music", "trains",
        "gardening", "photography", "books", "bicycles",
    )
    turns: list[dict[str, str]] = []
    for filler_index in range(count):
        topic = topics[rng.randrange(len(topics))]
        value = rng.randint(10, 99)
        turns.extend([
            {
                "role": "user",
                "content": f"Unrelated note {filler_index + 1}: I counted {value} {topic} items.",
            },
            {"role": "assistant", "content": "Understood."},
        ])
    return turns


def _context_messages(index: int, rng: random.Random) -> list[dict[str, str]]:
    variant = index % 5
    system = {"role": "system", "content": SYSTEM_PROMPT}
    fillers = _filler_turns(rng, 3 + index % 4)
    if variant == 0:
        code = f"COBALT-{index:04d}"
        return [
            system,
            {"role": "user", "content": f"Remember that the project codename is {code}."},
            {"role": "assistant", "content": "I will remember it."},
            *fillers,
            {"role": "user", "content": "What project codename did I give you?"},
            {"role": "assistant", "content": code},
        ]
    if variant == 1:
        old_day, new_day = ("Tuesday", "Monday") if index % 2 else ("Friday", "Thursday")
        return [
            system,
            {"role": "user", "content": f"The appointment is on {old_day}."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": f"Correction: it is on {new_day}, not {old_day}."},
            {"role": "assistant", "content": "Updated."},
            *fillers,
            {"role": "user", "content": "Which day is the appointment now?"},
            {"role": "assistant", "content": new_day},
        ]
    if variant == 2:
        alice_value = f"amber-{index}"
        boris_value = f"violet-{index}"
        return [
            system,
            {
                "role": "user",
                "content": f"Alice selected {alice_value}; Boris selected {boris_value}.",
            },
            {"role": "assistant", "content": "I have both selections."},
            *fillers,
            {"role": "user", "content": "What did Boris select?"},
            {"role": "assistant", "content": boris_value},
        ]
    if variant == 3:
        answer = rng.randint(20, 90)
        return [
            system,
            {
                "role": "user",
                "content": "For my final question, reply as JSON with exactly one key named result.",
            },
            {"role": "assistant", "content": "Understood."},
            *fillers,
            {"role": "user", "content": f"What number did I choose: {answer}?"},
            {"role": "assistant", "content": f'{{"result": {answer}}}'},
        ]
    city = ("Kyiv", "Lviv", "Odesa", "Dnipro")[index % 4]
    return [
        system,
        {"role": "user", "content": f"My preferred destination is {city}."},
        {"role": "assistant", "content": "Noted."},
        *fillers,
        {"role": "user", "content": "Returning to the travel topic, where did I prefer to go?"},
        {"role": "assistant", "content": city},
    ]


def _reasoning_messages(index: int, rng: random.Random) -> list[dict[str, str]]:
    variant = index % 8
    system = {"role": "system", "content": SYSTEM_PROMPT}
    if variant == 0:
        a, b, c = rng.randint(10, 80), rng.randint(2, 15), rng.randint(1, 9)
        question, answer = f"Calculate ({a} + {b}) × {c}.", str((a + b) * c)
    elif variant == 1:
        boxes, each, removed = rng.randint(3, 12), rng.randint(4, 20), rng.randint(1, 10)
        question = f"There are {boxes} boxes with {each} items each. {removed} items are removed. How many remain?"
        answer = str(boxes * each - removed)
    elif variant == 2:
        speed, hours = rng.randint(20, 90), rng.randint(2, 8)
        question, answer = f"A vehicle travels {speed} km per hour for {hours} hours. What distance does it cover?", f"{speed * hours} km"
    elif variant == 3:
        values = sorted(rng.sample(range(10, 100), 4))
        question = f"Order these numbers from largest to smallest: {', '.join(map(str, values))}."
        answer = ", ".join(map(str, reversed(values)))
    elif variant == 4:
        start, duration = rng.randint(8, 15), rng.randint(2, 6)
        question = f"A meeting starts at {start}:00 and lasts {duration} hours. At what hour does it end?"
        answer = f"{start + duration}:00"
    elif variant == 5:
        word = f"logic{index}"
        question, answer = f"Reverse the exact string `{word}`.", word[::-1]
    elif variant == 6:
        start, step = rng.randint(1, 20), rng.randint(2, 9)
        sequence = [start + step * offset for offset in range(4)]
        question = f"Continue the arithmetic sequence: {', '.join(map(str, sequence))}, ?"
        answer = str(start + step * 4)
    else:
        red, blue = rng.randint(2, 12), rng.randint(2, 12)
        question = (
            f"A bag contains {red} red and {blue} blue tokens. How many tokens are in the bag altogether?"
        )
        answer = str(red + blue)
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
    """Generate deterministic, split-specific context or reasoning replay examples."""
    if category not in {"context_retention", "general_reasoning"}:
        raise ValueError(f"Unsupported supplemental category: {category}")
    split_offset = {"train": 0, "validation": 100_000, "test": 200_000}[split]
    rows = []
    for local_index in range(count):
        index = split_offset + local_index
        rng = random.Random(f"{seed}:{category}:{split}:{index}")
        messages = (
            _context_messages(index, rng)
            if category == "context_retention"
            else _reasoning_messages(index, rng)
        )
        rows.append(_row(
            tokenizer,
            split,
            category,
            local_index,
            messages,
            max_length,
            max_target_tokens,
        ))
    return rows
