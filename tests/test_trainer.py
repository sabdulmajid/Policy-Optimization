import torch

from policy_optimization.trainers.step import run_policy_optimization_step
from policy_optimization.types import RolloutBatch


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[-0.5, -0.1], [-0.3, -0.2]], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.logits


class _ForwardOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_logits = torch.nn.Parameter(
            torch.tensor(
                [
                    [2.0, 0.0, -1.0],
                    [0.5, 1.5, -0.5],
                    [1.0, -0.2, 0.8],
                ],
                dtype=torch.float32,
            )
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False) -> _ForwardOutput:
        del input_ids, attention_mask, use_cache
        logits = self.base_logits.unsqueeze(0).expand(2, -1, -1)
        return _ForwardOutput(logits=logits)


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


def test_run_policy_optimization_step_recomputes_logprobs_from_model_inputs() -> None:
    model = TinyCausalLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batch = RolloutBatch(
        token_logprobs=torch.zeros((2, 2)),
        old_token_logprobs=torch.zeros((2, 2)),
        completion_mask=torch.tensor([[True, True], [True, True]]),
        rewards=torch.tensor([1.0, 0.0]),
        group_ids=torch.tensor([0, 0]),
        input_ids=torch.tensor([[0, 1, 2], [0, 2, 1]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 1]]),
    )
    before = model.base_logits.detach().clone()
    result = run_policy_optimization_step(model, optimizer, batch, objective_name="rloo")
    after = model.base_logits.detach()
    assert result.metrics["recomputed_logprobs"] == 1.0
    assert result.grad_norm > 0.0
    assert not torch.allclose(before, after)
