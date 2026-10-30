from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol

import torch

PROMPTFN = Callable[[str], str]
COLLATEFN = Callable[[list[dict]], Dict[str, torch.Tensor]]


class GenerateFn(Protocol):
    def __call__(
        self,
        model: Any,
        processor: Any,
        examples: list[dict],
        *,
        max_new_tokens: int,
    ) -> list[str]: ...


@dataclass(frozen=True)
class ModelSpec:
    prompt_fn: PROMPTFN
    build_collator: Callable[[Any], COLLATEFN]
    generate_fn: GenerateFn
    default_max_length: int = 256
    default_max_new_tokens: int = 16
