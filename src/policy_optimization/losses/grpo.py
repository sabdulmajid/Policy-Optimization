from __future__ import annotations

import torch

from policy_optimization.advantages import group_zscore_advantages
from policy_optimization.ops import importance_ratio, masked_mean
from policy_optimization.types import ObjectiveOutput, RolloutBatch


def grpo_loss(
    batch: RolloutBatch,
    *,
    advantages: torch.Tensor | None = None,
    clip_epsilon: float = 0.2,
    normalize_by_group_std: bool = True,
) -> ObjectiveOutput:
    advantages = (
        group_zscore_advantages(batch.rewards, batch.group_ids)
        if advantages is None and normalize_by_group_std
        else batch.rewards.float() if advantages is None else advantages.float()
    )
    ratio = importance_ratio(batch.token_logprobs, batch.old_token_logprobs)
    clipped_ratio = ratio.clamp(min=1.0 - clip_epsilon, max=1.0 + clip_epsilon)
    token_advantages = advantages.detach().unsqueeze(-1)
    surrogate = torch.minimum(ratio * token_advantages, clipped_ratio * token_advantages)
    loss = -masked_mean(surrogate, batch.completion_mask).mean()
    metrics = {
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()),
        "clip_fraction": float((((ratio != clipped_ratio) & batch.completion_mask).float().mean().item())),
        "reward_mean": float(batch.rewards.float().mean().item()),
    }
    return ObjectiveOutput(loss=loss, metrics=metrics)
