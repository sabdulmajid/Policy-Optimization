from __future__ import annotations

"""MaxRL: maximum-style reward-weighted sequence optimization.

Intuition
---------
MaxRL builds per-sample weights from success/reward statistics and optimizes
sequence logprob under those weights. Instead of explicit advantages, it uses
weight design to emphasize likely-successful trajectories while controlling
compute through an index scaling term.

When this is useful
-------------------
- You want direct success-aware weighting behavior.
- You need objective knobs around success threshold and control variates.
- You want sequence-level optimization with explicit compute-index diagnostics.

What outcome to expect
----------------------
- Stronger reinforcement on high-success samples.
- Useful introspection from `success_rate_mean`, `weight_mean`, and
    `compute_index_scale_mean`.

Mini example
------------
If success threshold is 0.5 and a sample reward is 0.9, it receives higher
weight than a 0.2-reward sample; sequence logprob updates therefore prioritize
high-success behavior under the chosen weighting rule.
"""

import torch

from policy_optimization.advantages import group_mean, maxrl_compute_index_weight, maxrl_weights
from policy_optimization.ops import sequence_logprob
from policy_optimization.types import ObjectiveOutput, RolloutBatch


def maxrl_loss(
    batch: RolloutBatch,
    *,
    use_control_variate: bool = False,
    success_threshold: float = 0.5,
    length_normalize: bool = False,
) -> ObjectiveOutput:
    """Compute MaxRL weighted sequence-logprob objective.

    Code map
    --------
    1) Convert rewards to success-aware weights.
    2) Compute sequence logprob for each completion.
    3) Optimize weighted sequence objective.
    4) Emit success/weight/index-scale diagnostics.
    """
    weights = maxrl_weights(
        batch.rewards,
        batch.group_ids,
        use_control_variate=use_control_variate,
        success_threshold=success_threshold,
    )
    seq_logprobs = sequence_logprob(batch.token_logprobs, batch.completion_mask, length_normalize=length_normalize)
    loss = -(seq_logprobs * weights.detach()).mean()
    successes = (batch.rewards.float() >= success_threshold).float()
    estimated_success_rate = group_mean(successes, batch.group_ids)
    truncation_level = int((batch.group_ids == batch.group_ids[0]).sum().item())
    scaling = maxrl_compute_index_weight(estimated_success_rate, truncation_level=truncation_level)
    metrics = {
        "success_rate_mean": float(estimated_success_rate.mean().item()),
        "weight_mean": float(weights.mean().item()),
        "compute_index_scale_mean": float(scaling.mean().item()),
    }
    return ObjectiveOutput(loss=loss, metrics=metrics)
