from __future__ import annotations

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
