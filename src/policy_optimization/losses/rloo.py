from __future__ import annotations

"""RLOO: REINFORCE with Leave-One-Out baselines.

Intuition
---------
RLOO treats each completion as one policy sample and asks:
"Was this sample better or worse than its peers from the same prompt group?"
The leave-one-out baseline removes shared prompt difficulty and reduces variance,
so gradients focus on relative quality inside each sampled group.

When this is useful
-------------------
- You want a simple, robust rollout objective without a learned value head.
- You want lower-memory RL updates compared with PPO-style actor-critic loops.
- You want group-based variance reduction while keeping implementation compact.

What outcome to expect
----------------------
- More stable policy-gradient updates than plain REINFORCE on noisy rewards.
- Clear reward-linked signal through `advantage_*` and `baseline_mean` metrics.

Mini example
------------
Suppose one prompt has 4 sampled completions with rewards [1.0, 0.0, 1.0, 0.0].
For the first sample, the leave-one-out baseline is mean([0.0, 1.0, 0.0]) = 0.333.
Its advantage becomes 1.0 - 0.333 = +0.667, so its logprob is reinforced.
"""

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
    """Compute RLOO objective on a rollout batch.

    Code map
    --------
    1) Build per-sample advantages with leave-one-out baselines by group.
    2) Collapse token logprobs into one sequence score per completion.
    3) Weight sequence logprobs by detached advantages.
    4) Return scalar loss + diagnostics for debugging/tracking.
    """
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
