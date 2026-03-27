import torch

from policy_optimization.rewards import apply_kl_reward_penalty, sequence_kl_penalty


def test_sequence_kl_penalty_respects_mask() -> None:
    policy = torch.tensor([[0.5, 0.0, 2.0]])
    reference = torch.tensor([[0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, False, True]])
    penalty = sequence_kl_penalty(policy, reference, mask)
    assert torch.allclose(penalty, torch.tensor([2.5]))


def test_apply_kl_reward_penalty_subtracts_beta_scaled_kl() -> None:
    rewards = torch.tensor([1.0])
    policy = torch.tensor([[0.5, 0.0]])
    reference = torch.tensor([[0.0, -0.5]])
    mask = torch.tensor([[True, True]])
    shaped = apply_kl_reward_penalty(rewards, policy, reference, mask, beta=0.1)
    assert torch.allclose(shaped, torch.tensor([0.9]))
