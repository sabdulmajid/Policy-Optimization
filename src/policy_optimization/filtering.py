from __future__ import annotations

from dataclasses import dataclass

import torch

from policy_optimization.advantages import group_var
from policy_optimization.types import RolloutBatch


@dataclass(slots=True)
class FilterResult:
    keep_mask: torch.Tensor
    kept_group_count: int
    dropped_group_count: int


def zero_variance_group_filter(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-8) -> FilterResult:
    keep_mask = torch.zeros_like(group_ids, dtype=torch.bool)
    kept = 0
    dropped = 0
    group_variances = group_var(rewards, group_ids)
    for group_id in torch.unique(group_ids):
        group_mask = group_ids == group_id
        keep_group = bool(group_variances[group_mask][0].item() > eps)
        keep_mask[group_mask] = keep_group
        if keep_group:
            kept += 1
        else:
            dropped += 1
    return FilterResult(keep_mask=keep_mask, kept_group_count=kept, dropped_group_count=dropped)


def drop_zero_variance_groups(batch: RolloutBatch, eps: float = 1e-8) -> tuple[RolloutBatch, FilterResult]:
    result = zero_variance_group_filter(batch.rewards, batch.group_ids, eps=eps)
    return batch.subset(result.keep_mask), result
