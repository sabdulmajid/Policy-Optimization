from __future__ import annotations

"""DPO: Direct Preference Optimization.

Intuition
---------
DPO turns preference learning into a binary logistic objective on the margin
between chosen and rejected responses. If a reference model is provided,
optimization is anchored to improvement over that reference margin.

When this is useful
-------------------
- You have pairwise preference data (chosen vs rejected outputs).
- You want RLHF-style alignment without explicit reward-model rollouts.
- You need a strong baseline objective for preference tuning.

What outcome to expect
----------------------
- Increased probability of chosen responses relative to rejected ones.
- Clear interpretability through margin and preference-accuracy metrics.

Mini example
------------
If chosen logprob is -2.1 and rejected is -2.8, policy margin is +0.7.
Positive margin increases `logsigmoid(beta * margin)`, reducing loss.
"""

import torch
import torch.nn.functional as F

from policy_optimization.precision import as_float32
from policy_optimization.types import ObjectiveOutput, PreferenceBatch


def dpo_loss(
    batch: PreferenceBatch,
    *,
    beta: float = 0.1,
) -> ObjectiveOutput:
    """Compute DPO logistic preference objective.

    Code map
    --------
    1) Compute policy margin = chosen - rejected.
    2) Optionally subtract reference margin for anchored improvement.
    3) Apply `-logsigmoid(beta * adjusted_margin)`.
    4) Return loss + margin/accuracy diagnostics.
    """
    chosen = as_float32(batch.chosen_logprobs)
    rejected = as_float32(batch.rejected_logprobs)
    reference_margin = 0.0
    if batch.ref_chosen_logprobs is not None and batch.ref_rejected_logprobs is not None:
        reference_margin = as_float32(batch.ref_chosen_logprobs) - as_float32(batch.ref_rejected_logprobs)
    policy_margin = chosen - rejected
    logits = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(logits).mean()
    metrics = {
        "policy_margin_mean": float(policy_margin.mean().item()),
        "reference_margin_mean": float(reference_margin.mean().item()) if isinstance(reference_margin, torch.Tensor) else float(reference_margin),
        "preference_accuracy": float((policy_margin > 0.0).float().mean().item()),
    }
    return ObjectiveOutput(loss=loss, metrics=metrics)
