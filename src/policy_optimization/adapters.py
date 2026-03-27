from __future__ import annotations

import torch
from transformers.pytorch_utils import Conv1D

COMMON_LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
FALLBACK_LORA_TARGET_SUFFIXES = ("c_attn", "c_proj", "c_fc")


def _linear_module_suffixes(model: torch.nn.Module) -> list[str]:
    return sorted(
        {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, (torch.nn.Linear, Conv1D))
        }
    )


def detect_lora_target_modules(model: torch.nn.Module) -> list[str]:
    suffixes = _linear_module_suffixes(model)
    preferred = [suffix for suffix in COMMON_LORA_TARGET_SUFFIXES if suffix in suffixes]
    if preferred:
        return preferred
    fallback = [suffix for suffix in FALLBACK_LORA_TARGET_SUFFIXES if suffix in suffixes]
    if fallback:
        return fallback
    generic = [suffix for suffix in suffixes if suffix != "lm_head"]
    if not generic:
        raise ValueError("Could not find any linear modules suitable for LoRA targeting.")
    return generic


def apply_lora_adapters(
    model: torch.nn.Module,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> tuple[torch.nn.Module, list[str]]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - exercised via caller environments
        raise ImportError(
            "LoRA support requires `peft`. Install project dependencies, for example with "
            '`python -m pip install -e ".[dev]"`.'
        ) from exc

    resolved_targets = detect_lora_target_modules(model) if target_modules is None else list(target_modules)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=resolved_targets,
    )
    wrapped = get_peft_model(model, config)
    if hasattr(wrapped, "config") and wrapped.config is not None:
        wrapped.config.use_cache = False
    return wrapped, resolved_targets
