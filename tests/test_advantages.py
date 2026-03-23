import torch

from policy_optimization.advantages import (
    group_centered_advantages,
    leave_one_out_baseline,
    maxrl_compute_index_weight,
    maxrl_weights,
    rloo_advantages,
)


def test_leave_one_out_baseline_matches_manual_values() -> None:
    rewards = torch.tensor([1.0, 3.0, 2.0, 4.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    baseline = leave_one_out_baseline(rewards, group_ids)
    assert torch.allclose(baseline, torch.tensor([3.0, 1.0, 4.0, 2.0]))


def test_rloo_advantages_subtract_leave_one_out_baseline() -> None:
    rewards = torch.tensor([1.0, 3.0, 2.0, 4.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    advantages = rloo_advantages(rewards, group_ids)
    assert torch.allclose(advantages, torch.tensor([-2.0, 2.0, -2.0, 2.0]))


def test_group_centered_advantages_zero_center_per_group() -> None:
    rewards = torch.tensor([1.0, 3.0, 2.0, 4.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    advantages = group_centered_advantages(rewards, group_ids)
    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0, -1.0, 1.0]))


def test_maxrl_weights_use_success_count_normalization() -> None:
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    weights = maxrl_weights(rewards, group_ids)
    assert torch.allclose(weights, torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_maxrl_compute_index_weight_matches_finite_series() -> None:
    success_prob = torch.tensor([0.25])
    weight = maxrl_compute_index_weight(success_prob, truncation_level=3)
    expected = 1.0 + 0.75 + 0.75**2
    assert torch.allclose(weight, torch.tensor([expected]))
