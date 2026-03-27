import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from policy_optimization.adapters import apply_lora_adapters, detect_lora_target_modules

pytest.importorskip("peft")


def test_detect_lora_target_modules_prefers_common_decoder_suffixes() -> None:
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2))
    detected = detect_lora_target_modules(model)
    assert "c_attn" in detected


def test_apply_lora_adapters_makes_subset_of_parameters_trainable() -> None:
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2))
    wrapped, target_modules = apply_lora_adapters(model, rank=4, alpha=8, dropout=0.0)
    trainable = sum(parameter.numel() for parameter in wrapped.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in wrapped.parameters())
    assert "c_attn" in target_modules
    assert 0 < trainable < total
