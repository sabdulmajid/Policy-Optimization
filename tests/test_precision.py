import torch

from policy_optimization.ops import gather_logprobs, sequence_logprob
from policy_optimization.precision import stable_log_softmax


def test_stable_log_softmax_upcasts_to_float32() -> None:
    logits = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    result = stable_log_softmax(logits)
    assert result.dtype == torch.float32


def test_gather_logprobs_matches_manual_gather() -> None:
    logits = torch.tensor([[[1.0, 3.0], [2.0, 0.0]]])
    targets = torch.tensor([[1, 0]])
    gathered = gather_logprobs(logits, targets)
    expected = torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(gathered, expected)


def test_sequence_logprob_respects_length_normalization() -> None:
    token_logprobs = torch.tensor([[-1.0, -2.0, -3.0]])
    mask = torch.tensor([[True, True, False]])
    assert torch.allclose(sequence_logprob(token_logprobs, mask, length_normalize=False), torch.tensor([-3.0]))
    assert torch.allclose(sequence_logprob(token_logprobs, mask, length_normalize=True), torch.tensor([-1.5]))
