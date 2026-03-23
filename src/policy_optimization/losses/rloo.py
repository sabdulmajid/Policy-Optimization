from __future__ import annotations

import torch

from policy_optimization.advantages import leave_one_out_baseline, rloo_advantages
from policy_optimization.ops import sequence_logprob
from policy_optimization.types import ObjectiveOutput, RolloutBatch


def rloo_loss(
    batch: RolloutBatch,
    *,
    advantages: torch.Tensor | None = None,
    normalize_advantages: bool = False,
    length_normalize: bool = False,
) -> ObjectiveOutput:
    advantages = rloo_advantages(batch.rewards, batch.group_ids, normalize=normalize_advantages) if advantages is None else advantages.float()
    baselines = leave_one_out_baseline(batch.rewards, batch.group_ids)
    seq_logprobs = sequence_logprob(batch.token_logprobs, batch.completion_mask, length_normalize=length_normalize)
    loss = -(seq_logprobs * advantages.detach()).mean()
    metrics = {
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()),
        "baseline_mean": float(baselines.mean().item()),
        "reward_mean": float(batch.rewards.float().mean().item()),
    }
    return ObjectiveOutput(loss=loss, metrics=metrics)
