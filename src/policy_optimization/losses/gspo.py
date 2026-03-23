from __future__ import annotations

import torch

from policy_optimization.advantages import group_centered_advantages
from policy_optimization.ops import sequence_logprob
from policy_optimization.precision import as_float32
from policy_optimization.types import ObjectiveOutput, RolloutBatch


def gspo_loss(
    batch: RolloutBatch,
    *,
    advantages: torch.Tensor | None = None,
    clip_epsilon: float = 0.2,
    length_normalize: bool = True,
) -> ObjectiveOutput:
    advantages = group_centered_advantages(batch.rewards, batch.group_ids) if advantages is None else advantages.float()
    log_ratio = sequence_logprob(
        batch.token_logprobs - batch.old_token_logprobs,
        batch.completion_mask,
        length_normalize=length_normalize,
    )
    ratio = torch.exp(as_float32(log_ratio).clamp(min=-60.0, max=60.0))
    clipped_ratio = ratio.clamp(min=1.0 - clip_epsilon, max=1.0 + clip_epsilon)
    unclipped = ratio * advantages.detach()
    clipped = clipped_ratio * advantages.detach()
    surrogate = torch.minimum(unclipped, clipped)
    loss = -surrogate.mean()
    metrics = {
        "advantage_mean": float(advantages.mean().item()),
        "ratio_mean": float(ratio.mean().item()),
        "sequence_clip_fraction": float(((ratio != clipped_ratio).float().mean().item())),
    }
    return ObjectiveOutput(loss=loss, metrics=metrics)
