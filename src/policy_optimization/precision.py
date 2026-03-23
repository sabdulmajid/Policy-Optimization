from __future__ import annotations

import torch


def as_float32(tensor: torch.Tensor) -> torch.Tensor:
    """Upcast to FP32 for numerically sensitive operations."""

    if tensor.dtype == torch.float32:
        return tensor
    return tensor.float()


def stable_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable FP32 log-softmax."""

    return torch.log_softmax(as_float32(logits), dim=dim)


def stable_logsumexp(values: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Numerically stable FP32 log-sum-exp."""

    return torch.logsumexp(as_float32(values), dim=dim, keepdim=keepdim)
