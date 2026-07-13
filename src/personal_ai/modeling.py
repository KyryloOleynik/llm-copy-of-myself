from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


LANGUAGE_LORA_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def personal_style_generation_options() -> dict[str, Any]:
    """Return the non-thinking sampling settings used for personal chat replies."""
    return {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
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
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_options,
        )
    input_tokens = inputs["input_ids"].shape[1]
    generated = output[0, input_tokens:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), input_tokens


def generate_replies(
    torch: Any,
    tokenizer: Any,
    model: Any,
    conversations: list[list[dict[str, str]]],
    *,
    max_new_tokens: int,
    **generation_options: Any,
) -> list[tuple[str, int]]:
    """Generate a padded batch of replies to keep CUDA inference efficiently occupied."""
    if not conversations:
        return []
    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in conversations
    ]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    finally:
        tokenizer.padding_side = original_padding_side
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_options,
        )
    padded_tokens = inputs["input_ids"].shape[1]
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    return [
        (
            tokenizer.decode(row[padded_tokens:], skip_special_tokens=True).strip(),
            int(prompt_length),
        )
        for row, prompt_length in zip(output, prompt_lengths, strict=True)
    ]


def select_language_lora_modules(model: Any) -> list[str]:
    """Select Qwen3.5 language token-mixer and MLP paths, never vision modules."""
    selected: list[str] = []
    observed: Counter[str] = Counter()
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in LANGUAGE_LORA_SUFFIXES or not hasattr(module, "weight"):
            continue
        lowered = name.casefold()
        if "vision" in lowered or "visual" in lowered or "language_model" not in lowered:
            continue
        selected.append(name)
        observed[suffix] += 1
    missing = sorted(LANGUAGE_LORA_SUFFIXES - set(observed))
    if missing:
        raise RuntimeError(f"Qwen3.5 language LoRA projections are missing: {missing}")
    if not selected:
        raise RuntimeError("No Qwen3.5 language-model LoRA modules were selected")
    return selected
