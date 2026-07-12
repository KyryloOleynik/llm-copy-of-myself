from __future__ import annotations

import gc
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from personal_ai.config import AppConfig
from personal_ai.modeling import generate_reply, load_inference_model
from personal_ai.training import validate_prepared_dataset
from personal_ai.utils import (
    iter_jsonl,
    read_json,
    relationship_system_message,
    render_chat_ids,
    write_json,
)


DISTANCES = (256, 512, 768, 1024, 2048, 4096, 8192)


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
            "category": "context",
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


def _candidate_paths(output_dir: Path) -> list[Path | None]:
    candidates: list[Path | None] = [None]
    checkpoints = sorted(
        (
            path
            for path in output_dir.glob("checkpoint-*")
            if path.is_dir() and (path / "adapter_config.json").is_file()
        ),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    candidates.extend(checkpoints)
    final = output_dir / "adapter-final"
    if final.is_dir() and (final / "adapter_config.json").is_file():
        candidates.append(final)
    return candidates


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


def _style_rating(config: AppConfig, candidate: str) -> tuple[int, int]:
    path = config.data.output_dir / "style_ratings.json"
    if not path.is_file():
        return 0, 0
    ratings = read_json(path).get(candidate, {})
    wins, total = int(ratings.get("wins", 0)), int(ratings.get("total", 0))
    if wins < 0 or total < 0 or wins > total:
        raise ValueError(f"Invalid blind style rating for {candidate}: {wins}/{total}")
    return wins, total


def _training_vram_ok(config: AppConfig) -> tuple[bool, int | None]:
    path = config.training.output_dir / "reproducibility.json"
    if not path.is_file():
        return False, None
    metadata = read_json(path)
    peak = metadata.get("peak_vram_reserved_bytes")
    return bool(peak is not None and peak < 12 * 1024**3), peak


def evaluate_checkpoints(config: AppConfig) -> Path:
    """Run deterministic diagnostics and create a machine-readable acceptance report."""
    manifest = validate_prepared_dataset(config)
    cases = [case for distance in DISTANCES for case in _diagnostic_cases(distance)]
    cases.extend(_reasoning_cases())
    test_rows = list(iter_jsonl(config.data.output_dir / "test.jsonl"))
    sample_rng = random.Random(config.training.seed)
    style_sample = sample_rng.sample(test_rows, min(50, len(test_rows)))
    style_outputs: dict[str, dict[str, str]] = {}
    results: dict[str, Any] = {}
    for adapter in _candidate_paths(config.training.output_dir):
        candidate = "base" if adapter is None else adapter.name
        torch, tokenizer, model = load_inference_model(config.model.base_model, adapter, {"": 0})
        rows = []
        for case in cases:
            messages = _messages_at_distance(
                tokenizer, case["messages"], case.get("target_distance")
            )
            output, actual_tokens = generate_reply(
                torch,
                tokenizer,
                model,
                messages,
                max_new_tokens=128,
                do_sample=False,
            )
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
        scores: dict[str, float] = {}
        for category in ("context", "reasoning"):
            scored = [
                row for row in rows if row["category"] == category and row["passed"] is not None
            ]
            scores[category] = sum(bool(row["passed"]) for row in scored) / len(scored)
        wins, total = _style_rating(config, candidate)
        style_outputs[candidate] = {}
        for row in style_sample:
            output, _ = generate_reply(
                torch,
                tokenizer,
                model,
                row["messages"][:-1],
                max_new_tokens=128,
                do_sample=False,
            )
            style_outputs[candidate][row["example_id"]] = output
        results[candidate] = {
            "adapter": str(adapter) if adapter else None,
            "scores": scores,
            "style_wins": wins,
            "style_total": total,
            "style_win_rate": wins / total if total else None,
            "cases": rows,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

    base_scores = results["base"]["scores"]
    vram_ok, peak_vram = _training_vram_ok(config)
    for candidate, result in results.items():
        if candidate == "base":
            result["accepted"] = False
            result["acceptance_reasons"] = ["base model is a comparison baseline"]
            continue
        reasons = []
        for category in ("context", "reasoning"):
            if result["scores"][category] + 0.05 < base_scores[category]:
                reasons.append(f"{category} regressed by more than five percentage points")
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

    output_dir = config.data.output_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model.base_model,
        "dataset_sha256": manifest["dataset_sha256"],
        "peak_training_vram_reserved_bytes": peak_vram,
        "results": results,
    }
    report_path = output_dir / "evaluation.json"
    write_json(report_path, report)
    _write_blind_style_template(config, output_dir, style_sample, style_outputs)
    return report_path


def _write_blind_style_template(
    config: AppConfig,
    output_dir: Path,
    sample: list[dict[str, Any]],
    outputs: dict[str, dict[str, str]],
) -> None:
    """Create anonymized base/adapter comparisons plus a separate answer key."""
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
    template = {
        "instructions": (
            "Choose A or B for each prompt without opening blind_style_key.json. "
            "After review, use the key to count adapter wins and write style_ratings.json "
            "as {candidate: {wins: integer, total: integer}}."
        ),
        "reviews": reviews,
    }
    write_json(output_dir / "blind_style_review.json", template)
    write_json(output_dir / "blind_style_key.json", answer_key)
