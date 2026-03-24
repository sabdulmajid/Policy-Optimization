from __future__ import annotations

"""GSPO: Group Sequence Policy Optimization.

Intuition
---------
GSPO applies PPO-style clipping at the sequence level instead of token level.
Each completion gets a single sequence ratio, then receives a clipped update
against a group-centered advantage. This emphasizes completion-level quality
and often behaves smoothly when token-level noise is high.

When this is useful
-------------------
- You care about whole-completion quality more than per-token fluctuations.
- You want PPO-like trust-region behavior with group-relative rewards.
- You need a stable sequence-level objective for long responses.

What outcome to expect
----------------------
- Controlled update size through `sequence_clip_fraction`.
- Cleaner objective signal when completion-level outcomes dominate.

Mini example
------------
If a completion has advantage +0.4 and sequence ratio 1.35 with epsilon 0.2,
the clipped ratio is 1.2. Surrogate uses min(1.35*0.4, 1.2*0.4)=0.48,
preventing overly aggressive policy jumps.
"""

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
    """Compute GSPO sequence-level clipped surrogate loss.

    Code map
    --------
    1) Compute group-centered advantages from rewards.
    2) Convert token logprob deltas into one sequence log-ratio.
    3) Exponentiate and clip ratio for trust-region style updates.
    4) Use clipped surrogate objective and emit diagnostics.
    """
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
