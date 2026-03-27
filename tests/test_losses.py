import torch
import pytest

from policy_optimization.advantages import group_count, group_mean, maxrl_compute_index_weight
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
    for name in ["rloo", "dapo", "gspo", "cispo", "maxrl", "grpo"]:
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


def test_dapo_clip_fraction_ignores_padding_tokens() -> None:
    batch = RolloutBatch(
        token_logprobs=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        old_token_logprobs=torch.zeros((2, 3)),
        completion_mask=torch.tensor([[True, False, False], [True, True, False]]),
        rewards=torch.tensor([1.0, 0.0]),
        group_ids=torch.tensor([0, 0]),
    )
    output = compute_objective("dapo", batch)
    assert output.metrics["clip_fraction"] == pytest.approx(1.0 / 3.0)


def test_grpo_clip_fraction_ignores_padding_tokens() -> None:
    batch = RolloutBatch(
        token_logprobs=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        old_token_logprobs=torch.zeros((2, 3)),
        completion_mask=torch.tensor([[True, False, False], [True, True, False]]),
        rewards=torch.tensor([1.0, 0.0]),
        group_ids=torch.tensor([0, 0]),
    )
    output = compute_objective("grpo", batch)
    assert output.metrics["clip_fraction"] == pytest.approx(1.0 / 3.0)


def test_maxrl_loss_uses_compute_index_scaling() -> None:
    batch = RolloutBatch(
        token_logprobs=torch.tensor([[0.2], [0.4], [0.6], [0.8], [1.0]]),
        old_token_logprobs=torch.zeros((5, 1)),
        completion_mask=torch.tensor([[True], [True], [True], [True], [True]]),
        rewards=torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0]),
        group_ids=torch.tensor([0, 0, 1, 1, 1]),
    )
    output = compute_objective("maxrl", batch)
    successes = (batch.rewards >= 0.5).float()
    weights = torch.tensor([1.0, 0.0, 0.5, 0.5, 0.0])
    success_rate = group_mean(successes, batch.group_ids)
    scaling = maxrl_compute_index_weight(success_rate, group_count(batch.group_ids))
    expected = -((batch.token_logprobs.squeeze(-1) * weights * scaling).mean())
    assert torch.allclose(output.loss, expected)


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


def test_dgpo_keeps_sample_weights_nonnegative_with_zero_mean_rewards() -> None:
    batch = PreferenceBatch(
        chosen_logprobs=torch.tensor([2.0, 1.5]),
        rejected_logprobs=torch.tensor([1.0, 1.0]),
        rewards=torch.tensor([-1.0, 1.0]),
        group_ids=torch.tensor([0, 0]),
    )
    output = compute_objective("dgpo", batch)
    assert output.loss.isfinite()
    assert output.metrics["sample_weight_min"] >= 0.0
