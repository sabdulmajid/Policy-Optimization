import torch

from policy_optimization.trainers.rollout import optimize_rollout_batch
from policy_optimization.types import RolloutBatch


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[-0.5], [-0.2], [-0.1], [-0.3]], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.logits


def test_optimize_rollout_batch_uses_group_minibatches() -> None:
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batch = RolloutBatch(
        token_logprobs=model().clone(),
        old_token_logprobs=model().detach().clone(),
        completion_mask=torch.tensor([[True], [True], [True], [True]]),
        rewards=torch.tensor([1.0, 0.0, 0.0, 1.0]),
        group_ids=torch.tensor([0, 0, 1, 1]),
    )
    result = optimize_rollout_batch(
        model,
        optimizer,
        batch,
        objective_name="rloo",
        epochs_per_rollout=2,
        groups_per_minibatch=1,
        seed=7,
    )
    assert result.optimizer_steps == 4
    assert result.metrics["optimizer_steps"] == 4.0
    assert result.metrics["epochs_per_rollout"] == 2.0
    assert result.metrics["groups_per_minibatch"] == 1.0
