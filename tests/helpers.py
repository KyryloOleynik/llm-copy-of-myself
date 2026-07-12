class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize and enable_thinking is False
        ids = []
        for message in messages:
            ids.append({"system": 10, "user": 20, "assistant": 30}[message["role"]])
            ids.extend(100 + ord(char) % 50 for char in message["content"])
            ids.append(2)
        if add_generation_prompt:
            ids.append(30)
        return ids

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()
