from __future__ import annotations

"""CISPO: Clipped Importance-Scaled Policy Optimization.

Intuition
---------
CISPO weights token logprobs by importance ratios, but detaches those weights
and applies clipping bounds. The detached weighting keeps optimization focused
on policy scores while reducing instability from extreme off-policy ratios.

When this is useful
-------------------
- You want importance-weighted correction with guardrails.
- You observe occasional ratio explosions and need bounded updates.
- You prefer conservative behavior in mixed-distribution rollouts.

What outcome to expect
----------------------
- More numerically stable training under distribution shift.
- Better-behaved gradients when old/new policy mismatch is large.

Mini example
------------
If ratio is 12.0 and `max_weight=5.0`, CISPO clamps it to 5.0.
The update still favors high-importance samples, but avoids catastrophic scale.
"""

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
    """Compute CISPO loss with clipped, detached importance weights.

    Code map
    --------
    1) Compute group-centered advantages from rewards.
    2) Compute token-level importance ratios (new/old policy).
    3) Clamp ratio-derived weights to optional min/max.
    4) Form weighted logprob objective with detached weights/advantages.
    """
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
