from __future__ import annotations

import torch
import torch.nn.functional as F

from policy_optimization.precision import as_float32
from policy_optimization.types import ObjectiveOutput, PreferenceBatch


def dpo_loss(
    batch: PreferenceBatch,
    *,
    beta: float = 0.1,
) -> ObjectiveOutput:
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
