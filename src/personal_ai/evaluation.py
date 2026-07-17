from __future__ import annotations

import gc
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from personal_ai.config import AppConfig
from personal_ai.modeling import (
    generate_reply,
    generate_replies,
    load_inference_model,
    personal_style_generation_options,
)
from personal_ai.tools import TOOL_SCHEMAS, parse_tool_calls
from personal_ai.training import validate_prepared_dataset
from personal_ai.utils import (
    iter_jsonl,
    read_json,
    relationship_system_message,
    render_chat_ids,
    write_json,
)


DISTANCES = (256, 512, 768, 1024, 2048, 4096, 8192)
STYLE_SAMPLE_SIZE = 25
EVALUATION_BATCH_SIZE = 4


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _case_batches(cases: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Keep equal-distance diagnostics together to avoid expensive cross-length padding."""
    context = [case for case in cases if case["category"] in {"context", "instruction"}]
    other = [case for case in cases if case["category"] not in {"context", "instruction"}]
    return [*_batches(context, 3), *_batches(other, EVALUATION_BATCH_SIZE)]


def _diagnostic_cases(distance: int) -> list[dict[str, Any]]:
    filler = "{FILLER}"
    return [
        {
            "id": f"delayed-recall-{distance}",
            "category": "context",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Remember that the project codename is COBALT."
                        + filler
                        + " What is the project codename? Answer with the codename only."
                    ),
                }
            ],
            "expected": ["cobalt"],
            "target_distance": distance,
        },
        {
            "id": f"corrected-state-{distance}",
            "category": "context",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "The meeting was first planned for Tuesday, but it was changed to Monday."
                        + filler
                        + " Which day is the meeting now? Answer with the day only."
                    ),
                }
            ],
            "expected": ["monday"],
            "target_distance": distance,
        },
        {
            "id": f"instruction-persistence-{distance}",
            "category": "instruction",
            "messages": [
                {
                    "role": "system",
                    "content": "Every answer must be valid JSON with a key named answer.",
                },
                {"role": "user", "content": filler + " What is two plus two?"},
            ],
            "expected": ['"answer"', "4"],
            "target_distance": distance,
        },
    ]


def _reasoning_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "reasoning-arithmetic",
            "category": "reasoning",
            "messages": [
                {"role": "user", "content": "A box has 7 rows of 8 items. How many items?"}
            ],
            "expected": ["56"],
        },
        {
            "id": "reasoning-latest-value",
            "category": "reasoning",
            "messages": [
                {
                    "role": "user",
                    "content": "I had 12 files, deleted 3, then added 5. How many files do I have?",
                }
            ],
            "expected": ["14"],
        },
        {
            "id": "multilingual-ukrainian",
            "category": "reasoning",
            "messages": [
                {
                    "role": "user",
                    "content": "В Олени було 9 книг, вона віддала 4. Скільки залишилось?",
                }
            ],
            "expected": ["5"],
        },
        {
            "id": "relationship-conditioning",
            "category": "style",
            "messages": [
                {"role": "system", "content": relationship_system_message("professional_contact")},
                {"role": "user", "content": "Can we move our meeting to tomorrow?"},
            ],
            "expected": [],
        },
    ]


def _tool_cases() -> list[dict[str, Any]]:
    system = {"role": "system", "content": relationship_system_message("friend")}
    calendar_system = {
        "role": "system",
        "content": (
            relationship_system_message("friend")
            + " Текущая дата и время: 2026-07-17T12:00:00+03:00. "
            "Для относительных дат используй это время."
        ),
    }
    memory_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "search_personal_memory",
                    "arguments": {"query": "первый проект", "limit": 3},
                },
            }
        ],
    }
    calendar_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "query_google_calendar",
                    "arguments": {
                        "action": "events",
                        "start": "2026-07-18T12:00:00+03:00",
                        "end": "2026-07-18T20:00:00+03:00",
                    },
                },
            }
        ],
    }
    return [
        {
            "id": "tool-calculator",
            "category": "tool_calling",
            "messages": [
                system,
                {"role": "user", "content": "посчитай точно (47 + 5) * 3, не угадывай"},
            ],
            "expected_tool": "calculate",
        },
        {
            "id": "tool-personal-memory",
            "category": "tool_calling",
            "messages": [
                system,
                {"role": "user", "content": "какой у меня был первый проект? проверь память"},
            ],
            "expected_tool": "search_personal_memory",
        },
        {
            "id": "tool-google-calendar",
            "category": "tool_calling",
            "messages": [
                calendar_system,
                {"role": "user", "content": "что у меня завтра после обеда?"},
            ],
            "expected_tool": "query_google_calendar",
        },
        {
            "id": "rag-grounded-answer",
            "category": "rag_grounding",
            "messages": [
                system,
                {"role": "user", "content": "какой у меня был первый проект? проверь память"},
                memory_call,
                {
                    "role": "tool",
                    "name": "search_personal_memory",
                    "content": (
                        '{"results":[{"source":"identity.md#0",'
                        '"content":"Первый проект: сайт с меткой COBALT-741."}]}'
                    ),
                },
            ],
            "expected": ["COBALT-741"],
        },
        {
            "id": "calendar-grounded-answer",
            "category": "tool_grounding",
            "messages": [
                calendar_system,
                {"role": "user", "content": "что у меня завтра после обеда?"},
                calendar_call,
                {
                    "role": "tool",
                    "name": "query_google_calendar",
                    "content": (
                        '{"action":"events","time_zone":"Europe/Kyiv","events":['
                        '{"summary":"Созвон по проекту",'
                        '"start":"2026-07-18T15:00:00+03:00",'
                        '"end":"2026-07-18T15:45:00+03:00"}]}'
                    ),
                },
            ],
            "expected": ["15:00", "созвон"],
        },
    ]


def _adapter_is_loadable(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def _candidate_paths(output_dir: Path) -> list[Path | None]:
    checkpoints = sorted(
        (
            path
            for path in output_dir.glob("checkpoint-*")
            if path.is_dir() and _adapter_is_loadable(path)
        ),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    final = output_dir / "adapter-final"
    if final.is_dir() and _adapter_is_loadable(final):
        return [None, final]
    if checkpoints:
        latest = checkpoints[-1]
        state_path = latest / "trainer_state.json"
        if state_path.is_file():
            best_value = read_json(state_path).get("best_model_checkpoint")
            if best_value:
                best = Path(best_value)
                if not best.is_absolute():
                    best = output_dir / best.name
                if best.is_dir() and _adapter_is_loadable(best):
                    return [None, best]
        return [None, latest]
    raise FileNotFoundError(
        f"No adapter-final or valid checkpoint containing adapter_config.json in {output_dir}"
    )


def _messages_at_distance(
    tokenizer: Any, messages: list[dict[str, str]], target_tokens: int | None
) -> list[dict[str, str]]:
    """Expand the filler marker so the rendered question ends near the requested distance."""
    copied = [dict(message) for message in messages]
    if target_tokens is None:
        return copied
    filler = " We briefly discussed an unrelated ordinary topic."
    base = [
        {**message, "content": message["content"].replace("{FILLER}", "")} for message in copied
    ]
    base_ids = render_chat_ids(tokenizer, base, generation=True)
    filler_tokens = max(1, len(tokenizer.encode(filler, add_special_tokens=False)))
    repeats = max(0, (target_tokens - len(base_ids)) // filler_tokens)
    while True:
        materialized = [
            {**message, "content": message["content"].replace("{FILLER}", filler * repeats)}
            for message in copied
        ]
        rendered = render_chat_ids(tokenizer, materialized, generation=True)
        if len(rendered) > target_tokens and repeats:
            repeats -= 1
            continue
        next_messages = [
            {**message, "content": message["content"].replace("{FILLER}", filler * (repeats + 1))}
            for message in copied
        ]
        next_ids = render_chat_ids(tokenizer, next_messages, generation=True)
        return next_messages if len(next_ids) <= target_tokens else materialized


def _score_case(output: str, expected: list[str]) -> bool | None:
    if not expected:
        return None
    normalized = output.casefold()
    return all(value.casefold() in normalized for value in expected)


def _training_vram_ok(config: AppConfig) -> tuple[bool, int | None]:
    path = config.training.output_dir / "reproducibility.json"
    if not path.is_file():
        return False, None
    metadata = read_json(path)
    peak = metadata.get("peak_vram_reserved_bytes")
    return bool(peak is not None and peak < 12 * 1024**3), peak


def _training_progress(adapter: Path | None) -> dict[str, int | float | bool] | None:
    if adapter is None or not adapter.name.startswith("checkpoint-"):
        return None
    state_path = adapter / "trainer_state.json"
    if not state_path.is_file():
        return None
    state = read_json(state_path)
    step = int(state.get("global_step", int(adapter.name.rsplit("-", 1)[-1])))
    max_steps = int(state.get("max_steps", 0))
    return {
        "global_step": step,
        "max_steps": max_steps,
        "epoch": float(state.get("epoch", 0.0)),
        "complete": bool(max_steps and step >= max_steps),
    }


def evaluate_checkpoints(config: AppConfig) -> Path:
    """Run deterministic diagnostics and create a machine-readable acceptance report."""
    manifest = validate_prepared_dataset(config)
    training_metadata_path = config.training.output_dir / "reproducibility.json"
    if not training_metadata_path.is_file() or read_json(training_metadata_path).get(
        "dataset_sha256"
    ) != manifest["dataset_sha256"]:
        raise RuntimeError(
            "Available adapter/checkpoints do not match the prepared dataset; "
            "train a new run with --fresh before evaluation"
        )
    cases = [case for distance in DISTANCES for case in _diagnostic_cases(distance)]
    cases.extend(_reasoning_cases())
    tool_cases = _tool_cases()
    test_rows = [
        row
        for row in iter_jsonl(config.data.output_dir / "test.jsonl")
        if row.get("source_type") == "personal_telegram"
    ]
    sample_rng = random.Random(config.training.seed)
    style_sample = sample_rng.sample(test_rows, min(STYLE_SAMPLE_SIZE, len(test_rows)))
    style_outputs: dict[str, dict[str, str]] = {}
    results: dict[str, Any] = {}
    candidates = _candidate_paths(config.training.output_dir)
    total_generations = len(candidates) * (len(cases) + len(tool_cases) + len(style_sample))
    with tqdm(total=total_generations, desc="Evaluating", unit="reply") as progress:
        for adapter in candidates:
            candidate = "base" if adapter is None else adapter.name
            progress.set_postfix_str(candidate)
            torch, tokenizer, model = load_inference_model(
                config.model.base_model, adapter, {"": 0}
            )
            try:
                rows = []
                for case_batch in _case_batches(cases):
                    messages_batch = [
                        _messages_at_distance(
                            tokenizer, case["messages"], case.get("target_distance")
                        )
                        for case in case_batch
                    ]
                    outputs = generate_replies(
                        torch,
                        tokenizer,
                        model,
                        messages_batch,
                        max_new_tokens=128,
                        do_sample=False,
                    )
                    for case, (output, actual_tokens) in zip(
                        case_batch, outputs, strict=True
                    ):
                        rows.append(
                            {
                                "id": case["id"],
                                "category": case["category"],
                                "target_distance": case.get("target_distance"),
                                "actual_prompt_tokens": actual_tokens,
                                "output": output,
                                "passed": _score_case(output, case["expected"]),
                            }
                        )
                    progress.update(len(case_batch))
                for case in tool_cases:
                    output, actual_tokens = generate_reply(
                        torch,
                        tokenizer,
                        model,
                        case["messages"],
                        tools=TOOL_SCHEMAS,
                        max_new_tokens=128,
                        do_sample=False,
                    )
                    if "expected_tool" in case:
                        try:
                            calls = parse_tool_calls(output)
                        except (json.JSONDecodeError, ValueError):
                            calls = []
                        passed = any(call.name == case["expected_tool"] for call in calls)
                    else:
                        passed = _score_case(output, case["expected"])
                    rows.append(
                        {
                            "id": case["id"],
                            "category": case["category"],
                            "target_distance": None,
                            "actual_prompt_tokens": actual_tokens,
                            "output": output,
                            "passed": passed,
                        }
                    )
                    progress.update(1)
                scores: dict[str, float] = {}
                for category in (
                    "context",
                    "instruction",
                    "reasoning",
                    "tool_calling",
                    "rag_grounding",
                ):
                    scored = [
                        row
                        for row in rows
                        if row["category"] == category and row["passed"] is not None
                    ]
                    scores[category] = sum(bool(row["passed"]) for row in scored) / len(scored)
                style_outputs[candidate] = {}
                style_sample_by_length = sorted(
                    style_sample,
                    key=lambda row: len(
                        render_chat_ids(tokenizer, row["messages"][:-1], generation=True)
                    ),
                )
                for batch_index, style_batch in enumerate(
                    _batches(style_sample_by_length, EVALUATION_BATCH_SIZE)
                ):
                    torch.manual_seed(config.training.seed + batch_index)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(config.training.seed + batch_index)
                    outputs = generate_replies(
                        torch,
                        tokenizer,
                        model,
                        [row["messages"][:-1] for row in style_batch],
                        max_new_tokens=128,
                        **personal_style_generation_options(),
                    )
                    for row, (output, _) in zip(style_batch, outputs, strict=True):
                        style_outputs[candidate][row["example_id"]] = output
                    progress.update(len(style_batch))
                results[candidate] = {
                    "adapter": str(adapter) if adapter else None,
                    "training_progress": _training_progress(adapter),
                    "scores": scores,
                    "style_wins": 0,
                    "style_total": 0,
                    "style_win_rate": None,
                    "cases": rows,
                }
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()

    output_dir = config.data.output_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    style_ratings = _write_blind_style_template(
        config, output_dir, style_sample, style_outputs
    )
    for candidate, rating in style_ratings.items():
        wins, total = rating
        results[candidate]["style_wins"] = wins
        results[candidate]["style_total"] = total
        results[candidate]["style_win_rate"] = wins / total if total else None

    base_scores = results["base"]["scores"]
    vram_ok, peak_vram = _training_vram_ok(config)
    for candidate, result in results.items():
        if candidate == "base":
            result["accepted"] = False
            result["acceptance_reasons"] = ["base model is a comparison baseline"]
            continue
        reasons = []
        progress_state = result.get("training_progress")
        if progress_state and not progress_state["complete"]:
            reasons.append(
                "training is incomplete: "
                f"step {progress_state['global_step']}/{progress_state['max_steps']}"
            )
        for category in ("context", "instruction", "reasoning"):
            if result["scores"][category] < base_scores[category]:
                reasons.append(f"{category} regressed below the base model")
        for category in ("tool_calling", "rag_grounding"):
            if result["scores"][category] < 0.80:
                reasons.append(f"{category} score is below 80%")
        if not style_sample:
            reasons.append("held-out style sample is empty")
        elif result["style_total"] < len(style_sample):
            reasons.append(
                f"blind style ratings are incomplete: {result['style_total']}/{len(style_sample)}"
            )
        elif result["style_win_rate"] < 0.60:
            reasons.append("blind style win rate is below 60%")
        if not vram_ok:
            reasons.append("verified peak training VRAM below 12 GiB is missing")
        result["accepted"] = not reasons
        result["acceptance_reasons"] = reasons

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model.base_model,
        "dataset_sha256": manifest["dataset_sha256"],
        "peak_training_vram_reserved_bytes": peak_vram,
        "results": results,
    }
    report_path = output_dir / "evaluation.json"
    write_json(report_path, report)
    return report_path


def _write_blind_style_template(
    config: AppConfig,
    output_dir: Path,
    sample: list[dict[str, Any]],
    outputs: dict[str, dict[str, str]],
) -> dict[str, tuple[int, int]]:
    """Write blind comparisons and preserve choices only for byte-identical reviews."""
    rng = random.Random(config.training.seed)
    reviews = []
    answer_key: dict[str, dict[str, str]] = {}
    for candidate in sorted(name for name in outputs if name != "base"):
        answer_key[candidate] = {}
        for row in sample:
            review_id = f"{candidate}:{row['example_id']}"
            base_output = outputs["base"][row["example_id"]]
            adapter_output = outputs[candidate][row["example_id"]]
            if rng.random() < 0.5:
                output_a, output_b, adapter_label = base_output, adapter_output, "B"
            else:
                output_a, output_b, adapter_label = adapter_output, base_output, "A"
            reviews.append(
                {
                    "review_id": review_id,
                    "relationship": row["relationship"],
                    "messages": row["messages"][:-1],
                    "reference_reply": row["messages"][-1]["content"],
                    "output_a": output_a,
                    "output_b": output_b,
                    "preferred": None,
                }
            )
            answer_key[candidate][review_id] = adapter_label
    existing_by_id: dict[str, dict[str, Any]] = {}
    review_path = output_dir / "blind_style_review.json"
    if review_path.is_file():
        existing_by_id = {
            review.get("review_id", ""): review
            for review in read_json(review_path).get("reviews", [])
        }
    for review in reviews:
        existing = existing_by_id.get(review["review_id"])
        if existing is None or existing.get("preferred") not in {"A", "B"}:
            continue
        comparable_existing = {key: value for key, value in existing.items() if key != "preferred"}
        comparable_new = {key: value for key, value in review.items() if key != "preferred"}
        if comparable_existing == comparable_new:
            review["preferred"] = existing["preferred"]

    template = {
        "instructions": (
            "Choose A or B for each prompt without opening blind_style_key.json. "
            "Save this file and rerun personal-ai evaluate. Matching choices are counted "
            "automatically; changed prompts or outputs reset their choices."
        ),
        "reviews": reviews,
    }
    ratings: dict[str, tuple[int, int]] = {}
    for candidate, candidate_key in answer_key.items():
        candidate_reviews = [
            review for review in reviews if review["review_id"] in candidate_key
        ]
        rated = [review for review in candidate_reviews if review["preferred"] in {"A", "B"}]
        wins = sum(
            review["preferred"] == candidate_key[review["review_id"]] for review in rated
        )
        ratings[candidate] = wins, len(rated)
    write_json(review_path, template)
    write_json(output_dir / "blind_style_key.json", answer_key)
    return ratings
