from __future__ import annotations

from policy_optimization.losses.dgpo import dgpo_loss
from policy_optimization.losses.cispo import cispo_loss
from policy_optimization.losses.dapo import dapo_loss
from policy_optimization.losses.dpo import dpo_loss
from policy_optimization.losses.gspo import gspo_loss
from policy_optimization.losses.grpo import grpo_loss
from policy_optimization.losses.maxrl import maxrl_loss
from policy_optimization.losses.mdpo import mdpo_loss
from policy_optimization.losses.rloo import rloo_loss
from policy_optimization.types import ObjectiveOutput

OBJECTIVE_REGISTRY = {
    "cispo": cispo_loss,
    "dapo": dapo_loss,
    "dgpo": dgpo_loss,
    "dpo": dpo_loss,
    "gspo": gspo_loss,
    "grpo": grpo_loss,
    "maxrl": maxrl_loss,
    "mdpo": mdpo_loss,
    "rloo": rloo_loss,
}


def compute_objective(name: str, batch: object, **kwargs: object) -> ObjectiveOutput:
    try:
        objective_fn = OBJECTIVE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(OBJECTIVE_REGISTRY))
        raise ValueError(f"Unknown objective '{name}'. Available: {available}") from exc
    return objective_fn(batch, **kwargs)


__all__ = [
    "OBJECTIVE_REGISTRY",
    "compute_objective",
    "cispo_loss",
    "dapo_loss",
    "dgpo_loss",
    "dpo_loss",
    "gspo_loss",
    "grpo_loss",
    "maxrl_loss",
    "mdpo_loss",
    "rloo_loss",
]
