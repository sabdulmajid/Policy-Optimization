from __future__ import annotations

import torch

from policy_optimization.advantages import group_centered_advantages
from policy_optimization.ops import importance_ratio, masked_mean
from policy_optimization.types import ObjectiveOutput, RolloutBatch


def cispo_loss(
    batch: RolloutBatch,
    *,
    advantages: torch.Tensor | None = None,
    min_weight: float | None = None,
    max_weight: float | None = 5.0,
) -> ObjectiveOutput:
    advantages = group_centered_advantages(batch.rewards, batch.group_ids) if advantages is None else advantages.float()
    weights = importance_ratio(batch.token_logprobs, batch.old_token_logprobs)
    if min_weight is not None:
        weights = torch.maximum(weights, torch.tensor(min_weight, device=weights.device, dtype=weights.dtype))
    if max_weight is not None:
        weights = torch.minimum(weights, torch.tensor(max_weight, device=weights.device, dtype=weights.dtype))
    weighted_logprobs = weights.detach() * advantages.detach().unsqueeze(-1) * batch.token_logprobs.float()
    loss = -masked_mean(weighted_logprobs, batch.completion_mask).mean()
    metrics = {
        "advantage_mean": float(advantages.mean().item()),
        "weight_mean": float(weights.mean().item()),
        "weight_max": float(weights.max().item()),
    }
    return ObjectiveOutput(loss=loss, metrics=metrics)
