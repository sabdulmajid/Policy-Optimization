from __future__ import annotations

import torch

from policy_optimization.advantages import group_centered_advantages
from policy_optimization.ops import importance_ratio, masked_mean
from policy_optimization.types import ObjectiveOutput, RolloutBatch


def dapo_loss(
    batch: RolloutBatch,
    *,
    advantages: torch.Tensor | None = None,
    lower_clip: float = 0.2,
    upper_clip: float = 0.2,
) -> ObjectiveOutput:
    advantages = group_centered_advantages(batch.rewards, batch.group_ids) if advantages is None else advantages.float()
    ratio = importance_ratio(batch.token_logprobs, batch.old_token_logprobs)
    clipped_ratio = ratio.clamp(min=1.0 - lower_clip, max=1.0 + upper_clip)
    token_advantages = advantages.detach().unsqueeze(-1)
    unclipped = ratio * token_advantages
    clipped = clipped_ratio * token_advantages
    surrogate = torch.minimum(unclipped, clipped)
    loss = -masked_mean(surrogate, batch.completion_mask).mean()
    clip_fraction = ((ratio < (1.0 - lower_clip)) | (ratio > (1.0 + upper_clip))) & batch.completion_mask
    metrics = {
        "advantage_mean": float(advantages.mean().item()),
        "clip_fraction": float(clip_fraction.float().mean().item()),
        "reward_mean": float(batch.rewards.float().mean().item()),
    }
    return ObjectiveOutput(loss=loss, metrics=metrics)
