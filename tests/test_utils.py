from personal_ai.utils import assistant_target_ids, relationship_system_message, token_ids
from tests.helpers import FakeTokenizer


def test_token_ids_normalizes_mapping_and_single_batch():
    assert token_ids({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == [1, 2, 3]
    assert token_ids([[4, 5]]) == [4, 5]


def test_shared_chat_helpers_render_verified_target_and_relationship():
    messages = [
        {"role": "system", "content": relationship_system_message("family")},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "reply"},
    ]
    prompt, full, target = assistant_target_ids(FakeTokenizer(), messages)
    assert full[: len(prompt)] == prompt
    assert target
    assert "family" in messages[0]["content"]
