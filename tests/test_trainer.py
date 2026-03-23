import torch

from policy_optimization.trainers.step import run_policy_optimization_step
from policy_optimization.types import RolloutBatch


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[-0.5, -0.1], [-0.3, -0.2]], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.logits


def test_run_policy_optimization_step_updates_parameters() -> None:
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batch = RolloutBatch(
        token_logprobs=model().clone(),
        old_token_logprobs=model().detach().clone(),
        completion_mask=torch.tensor([[True, True], [True, True]]),
        rewards=torch.tensor([1.0, 0.0]),
        group_ids=torch.tensor([0, 0]),
    )
    before = model.logits.detach().clone()
    result = run_policy_optimization_step(model, optimizer, batch, objective_name="rloo")
    after = model.logits.detach()
    assert result.loss != 0.0
    assert result.grad_norm > 0.0
    assert not torch.allclose(before, after)
