from personal_ai.tokenization import token_ids


def test_chat_template_batch_encoding_is_normalized():
    assert token_ids({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == [1, 2, 3]


def test_single_batched_token_list_is_normalized():
    assert token_ids([[4, 5]]) == [4, 5]
