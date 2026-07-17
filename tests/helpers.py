import json


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        tools=None,
    ):
        assert tokenize
        ids = []
        if tools:
            ids.extend(100 + ord(char) % 50 for char in json.dumps(tools, sort_keys=True))
        for message in messages:
            ids.append({"system": 10, "user": 20, "assistant": 30, "tool": 40}[message["role"]])
            ids.extend(100 + ord(char) % 50 for char in message["content"])
            if message.get("tool_calls"):
                ids.extend(
                    100 + ord(char) % 50
                    for char in json.dumps(message["tool_calls"], sort_keys=True)
                )
            ids.append(2)
        if add_generation_prompt:
            ids.append(30)
        return ids

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()
