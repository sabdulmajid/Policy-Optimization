from __future__ import annotations

import torch

from policy_optimization.precision import as_float32, stable_log_softmax


def masked_sum(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask_f = mask.to(dtype=torch.float32)
    return (as_float32(values) * mask_f).sum(dim=dim)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    numerator = masked_sum(values, mask, dim=dim)
    denominator = mask.to(dtype=torch.float32).sum(dim=dim).clamp_min(eps)
    return numerator / denominator


def masked_var(values: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    mean = masked_mean(values, mask, dim=dim, eps=eps)
    centered = as_float32(values) - mean.unsqueeze(dim)
    return masked_mean(centered.square(), mask, dim=dim, eps=eps)


def gather_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = stable_log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1))
    return gathered.squeeze(-1)


def sequence_logprob(
    token_logprobs: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    length_normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    sequence_lp = masked_sum(token_logprobs, completion_mask, dim=-1)
    if not length_normalize:
        return sequence_lp
    lengths = completion_mask.to(dtype=torch.float32).sum(dim=-1).clamp_min(eps)
    return sequence_lp / lengths


def importance_ratio(current_logprobs: torch.Tensor, old_logprobs: torch.Tensor) -> torch.Tensor:
    log_ratio = as_float32(current_logprobs) - as_float32(old_logprobs)
    return torch.exp(log_ratio.clamp(min=-60.0, max=60.0))
