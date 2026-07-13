import torch

from personal_ai.modeling import (
    generate_replies,
    generate_reply,
    personal_style_generation_options,
)


class Batch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"

    def __call__(self, prompts, **kwargs):
        if isinstance(prompts, list):
            assert kwargs.get("padding") is True
            return Batch(
                input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
                attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
            )
        return Batch(input_ids=torch.tensor([[1, 2]]))

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "reply"


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.generation_kwargs = None

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        generated = torch.tensor([[7, 9]] * input_ids.shape[0])
        return torch.cat((input_ids, generated), dim=1)


def _assert_chat_stop_tokens(model):
    assert model.generation_kwargs["eos_token_id"] == 9
    assert model.generation_kwargs["pad_token_id"] == 0


def test_single_reply_stops_at_chat_end_token():
    tokenizer = FakeTokenizer()
    model = FakeModel()

    reply, input_tokens = generate_reply(
        torch, tokenizer, model, [{"role": "user", "content": "hi"}], max_new_tokens=8
    )

    assert (reply, input_tokens) == ("reply", 2)
    _assert_chat_stop_tokens(model)


def test_batched_replies_stop_at_chat_end_token():
    tokenizer = FakeTokenizer()
    model = FakeModel()

    replies = generate_replies(
        torch,
        tokenizer,
        model,
        [[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]],
        max_new_tokens=8,
    )

    assert replies == [("reply", 2), ("reply", 3)]
    _assert_chat_stop_tokens(model)


def test_personal_style_generation_uses_qwen_non_thinking_sampling():
    assert personal_style_generation_options() == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.1,
    }
