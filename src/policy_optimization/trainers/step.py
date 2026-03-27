from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from policy_optimization.hf import compute_token_logprobs
from policy_optimization.losses import compute_objective
from policy_optimization.types import RolloutBatch


@dataclass(slots=True)
class OptimizationStepResult:
    loss: float
    grad_norm: float
    metrics: dict[str, float]


def _global_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().item())
    return total**0.5


def run_policy_optimization_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    *,
    objective_name: str,
    objective_kwargs: dict[str, object] | None = None,
    max_grad_norm: float | None = None,
) -> OptimizationStepResult:
    objective_kwargs = {} if objective_kwargs is None else dict(objective_kwargs)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters found on the model.")

    optimizer.zero_grad(set_to_none=True)
    objective_batch = batch
    if batch.input_ids is not None and batch.attention_mask is not None:
        current_token_logprobs = compute_token_logprobs(model, batch.input_ids, batch.attention_mask)
        objective_batch = replace(batch, token_logprobs=current_token_logprobs)
    output = compute_objective(objective_name, objective_batch, **objective_kwargs)
    output.loss.backward()

    if max_grad_norm is not None:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm).item())
    else:
        grad_norm = _global_grad_norm(trainable_parameters)

    optimizer.step()
    metrics = dict(output.metrics)
    metrics["recomputed_logprobs"] = 1.0 if objective_batch is not batch else 0.0
    metrics["loss"] = float(output.loss.detach().item())
    metrics["grad_norm"] = grad_norm
    return OptimizationStepResult(loss=metrics["loss"], grad_norm=grad_norm, metrics=metrics)
