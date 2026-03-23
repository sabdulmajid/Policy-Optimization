from __future__ import annotations

import torch

from policy_optimization.ops import masked_sum
from policy_optimization.precision import as_float32


def sequence_kl_penalty(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    return masked_sum(policy_logprobs - reference_logprobs, completion_mask, dim=-1)


def apply_kl_reward_penalty(
    rewards: torch.Tensor,
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor | None,
    completion_mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if reference_logprobs is None or beta == 0.0:
        return as_float32(rewards)
    kl = sequence_kl_penalty(policy_logprobs, reference_logprobs, completion_mask)
    return as_float32(rewards) - beta * kl


def apply_overlong_reward_penalty(
    rewards: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    max_length: int,
    penalty_weight: float,
) -> torch.Tensor:
    if penalty_weight == 0.0:
        return as_float32(rewards)
    overflow = (as_float32(sequence_lengths) - float(max_length)).clamp_min(0.0)
    return as_float32(rewards) - penalty_weight * overflow
