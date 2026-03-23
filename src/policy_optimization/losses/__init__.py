from __future__ import annotations

from policy_optimization.losses.cispo import cispo_loss
from policy_optimization.losses.dapo import dapo_loss
from policy_optimization.losses.gspo import gspo_loss
from policy_optimization.losses.maxrl import maxrl_loss
from policy_optimization.losses.rloo import rloo_loss
from policy_optimization.types import ObjectiveOutput, RolloutBatch

OBJECTIVE_REGISTRY = {
    "cispo": cispo_loss,
    "dapo": dapo_loss,
    "gspo": gspo_loss,
    "maxrl": maxrl_loss,
    "rloo": rloo_loss,
}


def compute_objective(name: str, batch: RolloutBatch, **kwargs: object) -> ObjectiveOutput:
    try:
        objective_fn = OBJECTIVE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(OBJECTIVE_REGISTRY))
        raise ValueError(f"Unknown objective '{name}'. Available: {available}") from exc
    return objective_fn(batch, **kwargs)


__all__ = ["OBJECTIVE_REGISTRY", "compute_objective", "cispo_loss", "dapo_loss", "gspo_loss", "maxrl_loss", "rloo_loss"]
