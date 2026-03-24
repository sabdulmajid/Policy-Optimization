from __future__ import annotations

"""MDPO: multimodal/conditional Direct Preference Optimization.

Intuition
---------
MDPO extends DPO with conditional grounding and anchor penalties:
- preference term: chosen should beat rejected,
- conditional term: policy margin should align with context-grounded margin,
- anchor term: chosen reward should not fall below a target margin.

When this is useful
-------------------
- You have preference pairs plus contextual signals (e.g., image/context).
- You want DPO-style learning with additional grounding constraints.
- You need protection against reward collapse via an anchor term.

What outcome to expect
----------------------
- Better preference alignment while respecting contextual consistency.
- Tunable balance between pure preference fit and grounding fidelity.

Mini example
------------
If policy prefers chosen over rejected but contradicts context-derived margin,
the conditional term increases loss, nudging the model toward context-consistent
preferences.
"""

import torch
import torch.nn.functional as F

from policy_optimization.precision import as_float32
from policy_optimization.types import ObjectiveOutput, PreferenceBatch


def mdpo_loss(
    batch: PreferenceBatch,
    *,
    beta: float = 0.1,
    image_beta: float = 0.1,
    image_weight: float = 1.0,
    anchor_weight: float = 0.1,
    anchor_margin: float = 0.0,
) -> ObjectiveOutput:
    """Compute MDPO objective with preference, conditional, and anchor terms.

    Code map
    --------
    1) Compute policy and (optional) reference margins.
    2) Build base DPO-like preference term.
    3) Add context-grounding term when context logprobs are provided.
    4) Add anchor penalty to discourage weak chosen rewards.
    """
    chosen = as_float32(batch.chosen_logprobs)
    rejected = as_float32(batch.rejected_logprobs)
    reference_margin = 0.0
    chosen_reward = chosen
    if batch.ref_chosen_logprobs is not None and batch.ref_rejected_logprobs is not None:
        ref_chosen = as_float32(batch.ref_chosen_logprobs)
        ref_rejected = as_float32(batch.ref_rejected_logprobs)
        reference_margin = ref_chosen - ref_rejected
        chosen_reward = chosen - ref_chosen

    policy_margin = chosen - rejected
    preference_term = -F.logsigmoid(beta * (policy_margin - reference_margin))

    conditional_term = torch.zeros_like(preference_term)
    grounding_gap = torch.zeros_like(preference_term)
    if batch.context_chosen_logprobs is not None and batch.context_rejected_logprobs is not None:
        context_margin = as_float32(batch.context_chosen_logprobs) - as_float32(batch.context_rejected_logprobs)
        grounding_gap = policy_margin - context_margin
        conditional_term = -F.logsigmoid(image_beta * grounding_gap)

    anchor_term = torch.relu(torch.tensor(anchor_margin, device=chosen.device, dtype=chosen.dtype) - chosen_reward)
    loss = preference_term.mean() + image_weight * conditional_term.mean() + anchor_weight * anchor_term.mean()
    metrics = {
        "policy_margin_mean": float(policy_margin.mean().item()),
        "grounding_gap_mean": float(grounding_gap.mean().item()),
        "anchor_violation_mean": float(anchor_term.mean().item()),
        "preference_accuracy": float((policy_margin > 0.0).float().mean().item()),
    }
    return ObjectiveOutput(loss=loss, metrics=metrics)
