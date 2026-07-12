from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_MIXER_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
}


def load_tokenizer(source: str | Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_quantized_base(model_name: str, torch: Any, device_map: Any) -> Any:
    from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return AutoModelForMultimodalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )


def load_inference_model(
    model_name: str,
    adapter_path: Path | None,
    device_map: Any,
) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel

    tokenizer = load_tokenizer(adapter_path or model_name)
    model = load_quantized_base(model_name, torch, device_map)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return torch, tokenizer, model


def generate_reply(
    torch: Any,
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    **generation_options: Any,
) -> tuple[str, int]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **generation_options,
        )
    input_tokens = inputs["input_ids"].shape[1]
    generated = output[0, input_tokens:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), input_tokens


def select_language_lora_modules(model: Any) -> list[str]:
    """Select exact Qwen3.5 language token-mixer paths, never vision modules."""
    selected: list[str] = []
    observed: Counter[str] = Counter()
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in TOKEN_MIXER_SUFFIXES or not hasattr(module, "weight"):
            continue
        lowered = name.casefold()
        if "vision" in lowered or "visual" in lowered or "language_model" not in lowered:
            continue
        selected.append(name)
        observed[suffix] += 1
    missing = sorted(TOKEN_MIXER_SUFFIXES - set(observed))
    if missing:
        raise RuntimeError(f"Qwen3.5 language projections are missing: {missing}")
    if not selected:
        raise RuntimeError("No Qwen3.5 language-model LoRA modules were selected")
    return selected
