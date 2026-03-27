from __future__ import annotations

import torch

from policy_optimization.precision import as_float32


def _group_mask(group_ids: torch.Tensor, group_id: int) -> torch.Tensor:
    return group_ids == group_id


def group_mean(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    values = as_float32(values)
    output = torch.empty_like(values)
    for group_id in torch.unique(group_ids):
        mask = _group_mask(group_ids, int(group_id))
        output[mask] = values[mask].mean()
    return output


def group_var(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    values = as_float32(values)
    output = torch.empty_like(values)
    for group_id in torch.unique(group_ids):
        mask = _group_mask(group_ids, int(group_id))
        output[mask] = values[mask].var(unbiased=False)
    return output


def group_count(group_ids: torch.Tensor) -> torch.Tensor:
    counts = torch.empty_like(group_ids, dtype=torch.float32)
    for group_id in torch.unique(group_ids):
        mask = _group_mask(group_ids, int(group_id))
        counts[mask] = float(mask.sum())
    return counts


def leave_one_out_baseline(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    values = as_float32(values)
    baseline = torch.zeros_like(values)
    counts = group_count(group_ids)
    means = group_mean(values, group_ids)
    totals = means * counts
    multi_sample = counts > 1
    baseline[multi_sample] = (totals[multi_sample] - values[multi_sample]) / (counts[multi_sample] - 1.0)
    return baseline


def group_centered_advantages(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    return as_float32(values) - group_mean(values, group_ids)


def group_zscore_advantages(values: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = group_centered_advantages(values, group_ids)
    std = torch.sqrt(group_var(values, group_ids) + eps)
    return centered / std


def rloo_advantages(values: torch.Tensor, group_ids: torch.Tensor, *, normalize: bool = False) -> torch.Tensor:
    advantages = as_float32(values) - leave_one_out_baseline(values, group_ids)
    if normalize:
        std = advantages.std(unbiased=False).clamp_min(1e-6)
        advantages = (advantages - advantages.mean()) / std
    return advantages


def maxrl_weights(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    use_control_variate: bool = False,
    success_threshold: float = 0.5,
) -> torch.Tensor:
    rewards = as_float32(rewards)
    successes = (rewards >= success_threshold).to(dtype=torch.float32)
    weights = torch.zeros_like(successes)
    counts = group_count(group_ids)
    success_counts = group_mean(successes, group_ids) * counts
    for group_id in torch.unique(group_ids):
        mask = _group_mask(group_ids, int(group_id))
        k = float(success_counts[mask][0].item())
        n = float(counts[mask][0].item())
        if k >= 1.0:
            weights[mask] = successes[mask] / k
            if use_control_variate:
                weights[mask] = weights[mask] - (1.0 / n)
        elif use_control_variate:
            weights[mask] = -1.0 / n
    return weights


def maxrl_compute_index_weight(
    success_probability: torch.Tensor,
    truncation_level: int | torch.Tensor,
) -> torch.Tensor:
    success_probability = as_float32(success_probability).clamp(min=1e-6, max=1.0 - 1e-6)
    if isinstance(truncation_level, torch.Tensor):
        counts = truncation_level.to(device=success_probability.device, dtype=torch.long)
    else:
        counts = torch.full_like(success_probability, int(truncation_level), dtype=torch.long)
    counts = counts.clamp_min(0)
    max_count = int(counts.max().item()) if counts.numel() else 0
    if max_count == 0:
        return torch.zeros_like(success_probability)
    powers = torch.arange(max_count, device=success_probability.device, dtype=torch.float32)
    series = torch.pow(1.0 - success_probability.unsqueeze(-1), powers)
    valid = powers.unsqueeze(0) < counts.unsqueeze(-1)
    return (series * valid.to(dtype=series.dtype)).sum(dim=-1)
