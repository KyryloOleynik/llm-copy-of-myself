import json
from types import SimpleNamespace

from personal_ai.evaluation import (
    _candidate_paths,
    _diagnostic_cases,
    _training_progress,
    _write_blind_style_template,
)


def test_instruction_persistence_is_scored_separately_from_context_recall():
    cases = _diagnostic_cases(256)
    assert [case["category"] for case in cases] == ["context", "context", "instruction"]


def _adapter(path):
    path.mkdir()
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"weights")


def test_evaluation_prefers_best_checkpoint_from_trainer_state(tmp_path):
    best = tmp_path / "checkpoint-50"
    latest = tmp_path / "checkpoint-60"
    _adapter(best)
    _adapter(latest)
    (latest / "trainer_state.json").write_text(
        json.dumps(
            {
                "best_model_checkpoint": str(best),
                "global_step": 60,
                "max_steps": 402,
            }
        ),
        encoding="utf-8",
    )

    assert _candidate_paths(tmp_path) == [None, best]


def test_evaluation_prefers_final_adapter(tmp_path):
    _adapter(tmp_path / "checkpoint-60")
    final = tmp_path / "adapter-final"
    _adapter(final)

    assert _candidate_paths(tmp_path) == [None, final]


def test_checkpoint_progress_reports_incomplete_training(tmp_path):
    checkpoint = tmp_path / "checkpoint-60"
    _adapter(checkpoint)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 60, "max_steps": 402, "epoch": 0.149}),
        encoding="utf-8",
    )

    assert _training_progress(checkpoint) == {
        "global_step": 60,
        "max_steps": 402,
        "epoch": 0.149,
        "complete": False,
    }


def test_blind_choices_are_reused_only_for_identical_outputs(tmp_path):
    config = SimpleNamespace(training=SimpleNamespace(seed=42))
    sample = [
        {
            "example_id": "example-1",
            "relationship": "friend",
            "messages": [
                {"role": "user", "content": "привет"},
                {"role": "assistant", "content": "салам"},
            ],
        }
    ]
    outputs = {"base": {"example-1": "Здравствуйте"}, "checkpoint-50": {"example-1": "салам"}}

    assert _write_blind_style_template(config, tmp_path, sample, outputs) == {
        "checkpoint-50": (0, 0)
    }
    review_path = tmp_path / "blind_style_review.json"
    key = json.loads((tmp_path / "blind_style_key.json").read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review_id = review["reviews"][0]["review_id"]
    review["reviews"][0]["preferred"] = key["checkpoint-50"][review_id]
    review_path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")

    assert _write_blind_style_template(config, tmp_path, sample, outputs) == {
        "checkpoint-50": (1, 1)
    }

    outputs["checkpoint-50"]["example-1"] = "привет"
    assert _write_blind_style_template(config, tmp_path, sample, outputs) == {
        "checkpoint-50": (0, 0)
    }
