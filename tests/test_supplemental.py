from personal_ai.supplemental import build_supplemental_examples


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and enable_thinking is False
        ids = []
        for message in messages:
            ids.append({"system": 10, "user": 20, "assistant": 30}[message["role"]])
            ids.extend(100 + ord(char) % 50 for char in message["content"])
            ids.append(2)
        if add_generation_prompt:
            ids.append(30)
        return ids


def test_context_and_reasoning_supplements_are_deterministic_and_disjoint():
    tokenizer = FakeTokenizer()
    kwargs = {
        "tokenizer": tokenizer,
        "count": 10,
        "max_length": 4096,
        "max_target_tokens": 256,
        "seed": 42,
    }
    context_train = build_supplemental_examples(
        split="train", category="context_retention", **kwargs
    )
    context_test = build_supplemental_examples(
        split="test", category="context_retention", **kwargs
    )
    reasoning = build_supplemental_examples(
        split="train", category="general_reasoning", **kwargs
    )
    assert context_train == build_supplemental_examples(
        split="train", category="context_retention", **kwargs
    )
    assert {row["example_id"] for row in context_train}.isdisjoint(
        row["example_id"] for row in context_test
    )
    assert all(row["source_type"] == "context_retention" for row in context_train)
    assert all(row["source_type"] == "general_reasoning" for row in reasoning)
    assert all(row["messages"][-1]["role"] == "assistant" for row in context_train + reasoning)
