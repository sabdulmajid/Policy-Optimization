import torch

from policy_optimization.filtering import zero_variance_group_filter


def test_zero_variance_group_filter_drops_flat_reward_groups() -> None:
    rewards = torch.tensor([1.0, 1.0, 0.0, 1.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    result = zero_variance_group_filter(rewards, group_ids)
    assert torch.equal(result.keep_mask, torch.tensor([False, False, True, True]))
    assert result.kept_group_count == 1
    assert result.dropped_group_count == 1
