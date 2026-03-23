import torch

from policy_optimization.losses import compute_objective
from policy_optimization.types import PreferenceBatch, RolloutBatch


def _batch() -> RolloutBatch:
    return RolloutBatch(
        token_logprobs=torch.tensor(
            [
                [-0.3, -0.2, -0.5],
                [-0.8, -0.6, -0.4],
                [-0.5, -0.4, -0.3],
                [-0.2, -0.1, -0.2],
            ]
        ),
        old_token_logprobs=torch.tensor(
            [
                [-0.35, -0.25, -0.55],
                [-0.75, -0.55, -0.35],
                [-0.45, -0.35, -0.25],
                [-0.25, -0.15, -0.25],
            ]
        ),
        completion_mask=torch.tensor(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ]
        ),
        rewards=torch.tensor([1.0, 0.0, 0.5, 1.5]),
        group_ids=torch.tensor([0, 0, 1, 1]),
    )


def test_all_objectives_produce_scalar_losses() -> None:
    batch = _batch()
    for name in ["rloo", "dapo", "gspo", "cispo", "maxrl"]:
        output = compute_objective(name, batch)
        assert output.loss.ndim == 0
        assert output.loss.isfinite()
        assert output.metrics


def test_gspo_length_normalization_changes_ratio_statistics() -> None:
    batch = _batch()
    normalized = compute_objective("gspo", batch, length_normalize=True)
    unnormalized = compute_objective("gspo", batch, length_normalize=False)
    assert normalized.metrics["ratio_mean"] != unnormalized.metrics["ratio_mean"]


def test_cispo_clipping_caps_importance_weights() -> None:
    batch = _batch()
    output = compute_objective("cispo", batch, max_weight=1.0)
    assert output.metrics["weight_max"] <= 1.0 + 1e-6


def test_preference_objectives_produce_scalar_losses() -> None:
    batch = PreferenceBatch(
        chosen_logprobs=torch.tensor([2.0, 1.5, 1.2]),
        rejected_logprobs=torch.tensor([1.0, 1.1, 1.0]),
        ref_chosen_logprobs=torch.tensor([1.7, 1.4, 1.1]),
        ref_rejected_logprobs=torch.tensor([1.1, 1.2, 1.0]),
        context_chosen_logprobs=torch.tensor([1.4, 1.1, 1.0]),
        context_rejected_logprobs=torch.tensor([1.2, 1.0, 0.95]),
        rewards=torch.tensor([2.0, 1.0, 3.0]),
        group_ids=torch.tensor([0, 0, 1]),
    )
    for name in ["dpo", "mdpo", "dgpo"]:
        output = compute_objective(name, batch)
        assert output.loss.ndim == 0
        assert output.loss.isfinite()
        assert output.metrics
