import pytest

from personal_ai.training import IGNORE_INDEX, ReplyOnlyCollator
from tests.helpers import FakeTokenizer


def test_only_final_reply_has_labels():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "abc"},
        {"role": "assistant", "content": "xy"},
    ]
    prompt = tokenizer.apply_chat_template(messages[:-1], True, True, False)
    full = tokenizer.apply_chat_template(messages, True, False, False)
    collator = ReplyOnlyCollator(tokenizer, max_length=100)
    batch = collator([{"messages": messages}])
    labels = batch["labels"][0].tolist()
    assert labels[: len(prompt)] == [IGNORE_INDEX] * len(prompt)
    assert labels[len(prompt) :] == full[len(prompt) :]
    assert collator.audit["zero_label_example"] == 0


def test_overflow_is_rejected_instead_of_truncating_target():
    collator = ReplyOnlyCollator(FakeTokenizer(), max_length=8)
    with pytest.raises(ValueError, match="Prepared example"):
        collator(
            [
                {
                    "messages": [
                        {"role": "user", "content": "abc"},
                        {"role": "assistant", "content": "xy"},
                    ]
                }
            ]
        )
    assert collator.audit["sequence_overflow"] == 1


def test_oversized_target_is_rejected():
    collator = ReplyOnlyCollator(FakeTokenizer(), max_length=100, max_target_tokens=2)
    with pytest.raises(ValueError, match="Assistant target"):
        collator(
            [
                {
                    "messages": [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "long"},
                    ]
                }
            ]
        )
    assert collator.audit["oversized_target"] == 1


class NonPrefixTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        ids = super().apply_chat_template(
            messages, tokenize, add_generation_prompt, enable_thinking
        )
        if not add_generation_prompt:
            ids[0] = 999
        return ids


def test_prompt_prefix_mismatch_is_rejected():
    collator = ReplyOnlyCollator(NonPrefixTokenizer(), max_length=100)
    with pytest.raises(ValueError, match="not a prefix"):
        collator(
            [
                {
                    "messages": [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                    ]
                }
            ]
        )
