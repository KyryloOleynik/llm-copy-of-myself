from personal_ai.training import TOKEN_MIXER_SUFFIXES, select_language_lora_modules


class WeightedModule:
    weight = object()


class FakeModel:
    def named_modules(self):
        for suffix in sorted(TOKEN_MIXER_SUFFIXES):
            yield f"model.language_model.layers.0.{suffix}", WeightedModule()
        yield "model.visual.blocks.0.out_proj", WeightedModule()


def test_only_language_token_mixers_are_selected():
    selected = select_language_lora_modules(FakeModel())
    assert len(selected) == len(TOKEN_MIXER_SUFFIXES)
    assert all("language_model" in name for name in selected)
    assert all("visual" not in name for name in selected)
