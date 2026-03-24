from __future__ import annotations

"""DGPO: grounded and risk-aware preference optimization.

Intuition
---------
DGPO extends preference optimization with:
- dynamic sample weighting from rewards/risk signals,
- grounding regularization against context margins,
- anchor pressure to avoid low-quality chosen behavior.

This is useful when pairwise preference data alone is insufficient and you want
additional robustness/control signals in the objective.

When this is useful
-------------------
- You have reward side-signals and want risk-aware weighting.
- You have context-conditioned margins and want grounded preferences.
- You need stronger control over failure-prone or low-quality regions.

What outcome to expect
----------------------
- Preference learning biased toward higher-value samples.
- Improved grounding consistency when context terms are available.
- Better diagnostics on margin quality and weighting behavior.

Mini example
------------
If two pairs have identical preference margin but one has higher reward/risk
weight, DGPO gives that pair larger optimization influence via sample weights.
"""

import torch
import torch.nn.functional as F

from policy_optimization.advantages import group_zscore_advantages
from policy_optimization.precision import as_float32
from policy_optimization.types import ObjectiveOutput, PreferenceBatch


def dgpo_loss(
    batch: PreferenceBatch,
    *,
    beta: float = 0.1,
    grounding_beta: float = 0.1,
    grounding_weight: float = 1.0,
    risk_weight: float = 0.5,
    anchor_weight: float = 0.05,
) -> ObjectiveOutput:
    """Compute DGPO objective with weighted preference + grounding + anchor.

    Code map
    --------
    1) Compute policy/reference margins and base preference term.
    2) Build sample weights from rewards and group-relative variance signals.
    3) Add grounding term from policy-context margin gap.
    4) Add anchor penalty and combine all components.
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
    preference_logits = beta * (policy_margin - reference_margin)
    preference_term = -F.logsigmoid(preference_logits)

    sample_weights = torch.ones_like(preference_term)
    if batch.rewards is not None:
        normalized_rewards = as_float32(batch.rewards) / as_float32(batch.rewards).mean().clamp_min(1e-6)
        sample_weights = sample_weights + risk_weight * normalized_rewards
    if batch.group_ids is not None and batch.rewards is not None:
        sample_weights = sample_weights + 0.25 * torch.abs(group_zscore_advantages(batch.rewards, batch.group_ids))

    grounding_term = torch.zeros_like(preference_term)
    grounding_gap = torch.zeros_like(preference_term)
    if batch.context_chosen_logprobs is not None and batch.context_rejected_logprobs is not None:
        context_margin = as_float32(batch.context_chosen_logprobs) - as_float32(batch.context_rejected_logprobs)
        grounding_gap = policy_margin - context_margin
        grounding_term = -F.logsigmoid(grounding_beta * grounding_gap)

    anchor_term = torch.relu(-chosen_reward)
    loss = (sample_weights.detach() * preference_term).mean() + grounding_weight * grounding_term.mean() + anchor_weight * anchor_term.mean()
    metrics = {
        "policy_margin_mean": float(policy_margin.mean().item()),
        "grounding_gap_mean": float(grounding_gap.mean().item()),
        "sample_weight_mean": float(sample_weights.mean().item()),
        "preference_accuracy": float((policy_margin > 0.0).float().mean().item()),
    }
    return ObjectiveOutput(loss=loss, metrics=metrics)
