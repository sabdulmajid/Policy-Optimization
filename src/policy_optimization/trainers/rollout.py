from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from policy_optimization.trainers.step import run_policy_optimization_step
from policy_optimization.types import RolloutBatch


@dataclass(slots=True)
class RolloutOptimizationResult:
    optimizer_steps: int
    metrics: dict[str, float]


def _group_minibatch_masks(
    group_ids: torch.Tensor,
    *,
    groups_per_minibatch: int,
    seed: int,
) -> list[torch.Tensor]:
    unique_groups = [int(group_id) for group_id in torch.unique(group_ids).tolist()]
    if not unique_groups:
        return []
    if groups_per_minibatch <= 0 or groups_per_minibatch >= len(unique_groups):
        return [torch.ones_like(group_ids, dtype=torch.bool)]
    rng = random.Random(seed)
    shuffled_groups = list(unique_groups)
    rng.shuffle(shuffled_groups)
    masks: list[torch.Tensor] = []
    for start in range(0, len(shuffled_groups), groups_per_minibatch):
        keep_groups = shuffled_groups[start : start + groups_per_minibatch]
        mask = torch.zeros_like(group_ids, dtype=torch.bool)
        for group_id in keep_groups:
            mask |= group_ids == group_id
        masks.append(mask)
    return masks


def optimize_rollout_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    *,
    objective_name: str,
    objective_kwargs: dict[str, object] | None = None,
    max_grad_norm: float | None = None,
    epochs_per_rollout: int,
    groups_per_minibatch: int,
    seed: int,
) -> RolloutOptimizationResult:
    if epochs_per_rollout <= 0:
        raise ValueError("epochs_per_rollout must be positive.")

    metric_totals: dict[str, float] = {}
    total_examples = 0.0
    optimizer_steps = 0

    for epoch in range(epochs_per_rollout):
        masks = _group_minibatch_masks(batch.group_ids, groups_per_minibatch=groups_per_minibatch, seed=seed + epoch)
        for mask in masks:
            minibatch = batch.subset(mask)
            if minibatch.batch_size == 0:
                continue
            result = run_policy_optimization_step(
                model,
                optimizer,
                minibatch,
                objective_name=objective_name,
                objective_kwargs=objective_kwargs,
                max_grad_norm=max_grad_norm,
            )
            weight = float(minibatch.batch_size)
            total_examples += weight
            optimizer_steps += 1
            for key, value in result.metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + weight * float(value)

    if optimizer_steps == 0 or total_examples == 0.0:
        raise ValueError("Rollout optimization produced zero optimizer steps.")

    metrics = {key: value / total_examples for key, value in metric_totals.items()}
    metrics["optimizer_steps"] = float(optimizer_steps)
    metrics["epochs_per_rollout"] = float(epochs_per_rollout)
    metrics["groups_per_minibatch"] = float(groups_per_minibatch)
    return RolloutOptimizationResult(optimizer_steps=optimizer_steps, metrics=metrics)
