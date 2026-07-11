from personal_ai.training import IGNORE_INDEX, ReplyOnlyCollator


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and enable_thinking is False
        size = sum(len(message["content"]) + 2 for message in messages)
        if add_generation_prompt:
            size += 2
        return list(range(1, size + 1))


def test_only_final_reply_has_labels():
    collator = ReplyOnlyCollator(FakeTokenizer(), max_length=100)
    batch = collator([{"messages": [
        {"role": "user", "content": "abc"},
        {"role": "assistant", "content": "xy"},
    ]}])
    labels = batch["labels"][0].tolist()
    assert labels[:7] == [IGNORE_INDEX] * 7
    assert labels[7:] != []
    assert all(label != IGNORE_INDEX for label in labels[7:])
