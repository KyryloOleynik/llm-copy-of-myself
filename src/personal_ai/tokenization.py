from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def token_ids(rendered: Any) -> list[int]:
    """Normalize Transformers 4/5 chat-template results to one token-id list."""
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], (list, tuple)):
        if len(rendered) != 1:
            raise ValueError("Expected one rendered conversation")
        rendered = rendered[0]
    return [int(value) for value in rendered]
