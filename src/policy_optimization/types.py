from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch


@dataclass(slots=True)
class ObjectiveOutput:
    """Container returned by every objective implementation."""

    loss: torch.Tensor
    metrics: dict[str, float]


@dataclass(slots=True)
class RolloutBatch:
    """Grouped rollout tensors for sequence-level RL objectives."""

    token_logprobs: torch.Tensor
    old_token_logprobs: torch.Tensor
    completion_mask: torch.Tensor
    rewards: torch.Tensor
    group_ids: torch.Tensor
    ref_token_logprobs: torch.Tensor | None = None
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def sequence_lengths(self) -> torch.Tensor:
        return self.completion_mask.to(dtype=torch.float32).sum(dim=-1)

    def clone_with_detached_old(self) -> "RolloutBatch":
        return replace(self, old_token_logprobs=self.token_logprobs.detach())

    def subset(self, keep_mask: torch.Tensor) -> "RolloutBatch":
        keep_mask = keep_mask.to(dtype=torch.bool)
        extras = {key: value[keep_mask] for key, value in self.extras.items()}
        return RolloutBatch(
            token_logprobs=self.token_logprobs[keep_mask],
            old_token_logprobs=self.old_token_logprobs[keep_mask],
            completion_mask=self.completion_mask[keep_mask],
            rewards=self.rewards[keep_mask],
            group_ids=self.group_ids[keep_mask],
            ref_token_logprobs=None if self.ref_token_logprobs is None else self.ref_token_logprobs[keep_mask],
            input_ids=None if self.input_ids is None else self.input_ids[keep_mask],
            attention_mask=None if self.attention_mask is None else self.attention_mask[keep_mask],
            extras=extras,
        )

    def to(self, device: torch.device | str) -> "RolloutBatch":
        extras = {key: value.to(device) for key, value in self.extras.items()}
        return RolloutBatch(
            token_logprobs=self.token_logprobs.to(device),
            old_token_logprobs=self.old_token_logprobs.to(device),
            completion_mask=self.completion_mask.to(device),
            rewards=self.rewards.to(device),
            group_ids=self.group_ids.to(device),
            ref_token_logprobs=None if self.ref_token_logprobs is None else self.ref_token_logprobs.to(device),
            input_ids=None if self.input_ids is None else self.input_ids.to(device),
            attention_mask=None if self.attention_mask is None else self.attention_mask.to(device),
            extras=extras,
        )

    def validate(self) -> None:
        if self.token_logprobs.ndim != 2:
            raise ValueError("token_logprobs must be rank-2 [batch, seq].")
        if self.old_token_logprobs.shape != self.token_logprobs.shape:
            raise ValueError("old_token_logprobs must match token_logprobs shape.")
        if self.completion_mask.shape != self.token_logprobs.shape:
            raise ValueError("completion_mask must match token_logprobs shape.")
        if self.rewards.ndim != 1:
            raise ValueError("rewards must be rank-1 [batch].")
        if self.group_ids.ndim != 1:
            raise ValueError("group_ids must be rank-1 [batch].")
        if self.rewards.shape[0] != self.token_logprobs.shape[0]:
            raise ValueError("rewards batch dimension must match token_logprobs.")
        if self.group_ids.shape[0] != self.token_logprobs.shape[0]:
            raise ValueError("group_ids batch dimension must match token_logprobs.")
        if self.ref_token_logprobs is not None and self.ref_token_logprobs.shape != self.token_logprobs.shape:
            raise ValueError("ref_token_logprobs must match token_logprobs shape when provided.")


@dataclass(slots=True)
class PreferenceBatch:
    """Pairwise preferences for DPO-style objectives."""

    chosen_logprobs: torch.Tensor
    rejected_logprobs: torch.Tensor
    ref_chosen_logprobs: torch.Tensor | None = None
    ref_rejected_logprobs: torch.Tensor | None = None
    context_chosen_logprobs: torch.Tensor | None = None
    context_rejected_logprobs: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    group_ids: torch.Tensor | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.chosen_logprobs.shape[0])

    def to(self, device: torch.device | str) -> "PreferenceBatch":
        extras = {key: value.to(device) for key, value in self.extras.items()}
        return PreferenceBatch(
            chosen_logprobs=self.chosen_logprobs.to(device),
            rejected_logprobs=self.rejected_logprobs.to(device),
            ref_chosen_logprobs=None if self.ref_chosen_logprobs is None else self.ref_chosen_logprobs.to(device),
            ref_rejected_logprobs=None if self.ref_rejected_logprobs is None else self.ref_rejected_logprobs.to(device),
            context_chosen_logprobs=None if self.context_chosen_logprobs is None else self.context_chosen_logprobs.to(device),
            context_rejected_logprobs=None if self.context_rejected_logprobs is None else self.context_rejected_logprobs.to(device),
            rewards=None if self.rewards is None else self.rewards.to(device),
            group_ids=None if self.group_ids is None else self.group_ids.to(device),
            extras=extras,
        )

    def validate(self) -> None:
        if self.chosen_logprobs.ndim != 1 or self.rejected_logprobs.ndim != 1:
            raise ValueError("chosen_logprobs and rejected_logprobs must be rank-1 [batch].")
        if self.chosen_logprobs.shape != self.rejected_logprobs.shape:
            raise ValueError("chosen_logprobs and rejected_logprobs must match shape.")
        for name, tensor in [
            ("ref_chosen_logprobs", self.ref_chosen_logprobs),
            ("ref_rejected_logprobs", self.ref_rejected_logprobs),
            ("context_chosen_logprobs", self.context_chosen_logprobs),
            ("context_rejected_logprobs", self.context_rejected_logprobs),
            ("rewards", self.rewards),
            ("group_ids", self.group_ids),
        ]:
            if tensor is None:
                continue
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be rank-1 [batch] when provided.")
            if tensor.shape != self.chosen_logprobs.shape:
                raise ValueError(f"{name} must match chosen_logprobs shape when provided.")
