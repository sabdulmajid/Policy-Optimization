from __future__ import annotations

"""GRPO: Group Relative Policy Optimization.

Intuition
---------
GRPO normalizes rewards within each group (z-score by default) and applies a
clipped-ratio surrogate. This makes updates depend on *relative* performance
inside prompt groups, helping stabilize gradients when absolute reward scales
shift between prompts.

When this is useful
-------------------
- Reward scale differs significantly across prompts/batches.
- You want value-head-free relative optimization.
- You want PPO-style clipping plus group normalization.

What outcome to expect
----------------------
- More comparable update magnitudes across heterogeneous prompts.
- Better resilience to reward-scale drift.

Mini example
------------
Group rewards [0.1, 0.9, 0.2, 0.8] become roughly z-scored
[-1.0, +1.0, -0.8, +0.8], so updates reinforce winners relative to peers,
not raw global reward magnitude.
"""

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
    """Compute GRPO objective with optional group-std normalization.

    Code map
    --------
    1) Build group-relative advantages (z-score by default).
    2) Compute token ratio between new and old policies.
    3) Clip ratios and apply PPO-style surrogate.
    4) Return loss plus clipping and reward diagnostics.
    """
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
    clipped_tokens = ((ratio != clipped_ratio) & batch.completion_mask)
    clip_fraction = float(clipped_tokens[batch.completion_mask].float().mean().item()) if batch.completion_mask.any() else 0.0
    metrics = {
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()),
        "clip_fraction": clip_fraction,
        "reward_mean": float(batch.rewards.float().mean().item()),
    }
    return ObjectiveOutput(loss=loss, metrics=metrics)
