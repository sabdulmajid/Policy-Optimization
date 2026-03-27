from __future__ import annotations

"""DAPO: Distribution-Aware Policy Optimization.

Intuition
---------
DAPO is a PPO-style clipped surrogate over token ratios with group-centered
advantages. In practice this objective often pairs with variance-aware group
filtering (done upstream) to avoid wasting updates on flat-reward groups.

When this is useful
-------------------
- You want familiar clipped-ratio behavior with rollout groups.
- You want robust updates when reward variance across groups is uneven.
- You track clip behavior as a health signal (`clip_fraction`).

What outcome to expect
----------------------
- Controlled policy steps via lower/upper clip limits.
- Reduced over-updating on samples with very large policy shifts.

Mini example
------------
With advantage +0.3, ratio 0.6, and clip band [0.8, 1.2],
the clipped contribution uses 0.8*0.3 instead of 0.6*0.3,
preventing overly pessimistic or unstable updates.
"""

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
    """Compute DAPO clipped surrogate loss.

    Code map
    --------
    1) Build group-centered advantages.
    2) Compute token importance ratio and clipped ratio.
    3) Use `min(unclipped, clipped)` surrogate (PPO-style).
    4) Return objective value plus clipping diagnostics.
    """
    advantages = group_centered_advantages(batch.rewards, batch.group_ids) if advantages is None else advantages.float()
    ratio = importance_ratio(batch.token_logprobs, batch.old_token_logprobs)
    clipped_ratio = ratio.clamp(min=1.0 - lower_clip, max=1.0 + upper_clip)
    token_advantages = advantages.detach().unsqueeze(-1)
    unclipped = ratio * token_advantages
    clipped = clipped_ratio * token_advantages
    surrogate = torch.minimum(unclipped, clipped)
    loss = -masked_mean(surrogate, batch.completion_mask).mean()
    clipped_tokens = ((ratio < (1.0 - lower_clip)) | (ratio > (1.0 + upper_clip))) & batch.completion_mask
    clip_fraction = float(clipped_tokens[batch.completion_mask].float().mean().item()) if batch.completion_mask.any() else 0.0
    metrics = {
        "advantage_mean": float(advantages.mean().item()),
        "clip_fraction": clip_fraction,
        "reward_mean": float(batch.rewards.float().mean().item()),
    }
    return ObjectiveOutput(loss=loss, metrics=metrics)
