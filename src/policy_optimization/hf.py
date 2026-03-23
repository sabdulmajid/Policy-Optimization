from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from policy_optimization.ops import gather_logprobs
from policy_optimization.types import RolloutBatch


@dataclass(slots=True)
class SampledRollouts:
    prompts: list[str]
    completions: list[str]
    prompt_token_ids: list[list[int]]
    completion_token_ids: list[list[int]]
    group_ids: torch.Tensor


def _torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.") from exc


def load_causal_lm(
    model_id: str,
    *,
    device: torch.device | str,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    device = torch.device(device)
    torch_dtype = _torch_dtype(dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    model.to(device)
    model.config.use_cache = False
    return model, tokenizer


def sample_group_rollouts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> SampledRollouts:
    prompts_out: list[str] = []
    completions_out: list[str] = []
    prompt_token_ids_out: list[list[int]] = []
    completion_token_ids_out: list[list[int]] = []
    group_ids_out: list[int] = []

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for group_id, prompt in enumerate(prompts):
            prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_ids = prompt_inputs["input_ids"].to(model.device)
            prompt_attention_mask = prompt_inputs["attention_mask"].to(model.device)
            prompt_len = prompt_ids.shape[-1]
            sequences = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                do_sample=True,
                num_return_sequences=group_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for sequence in sequences:
                completion_token_ids = sequence[prompt_len:].tolist()
                completion = tokenizer.decode(completion_token_ids, skip_special_tokens=True)
                prompts_out.append(prompt)
                completions_out.append(completion)
                prompt_token_ids_out.append(prompt_ids[0].tolist())
                completion_token_ids_out.append(completion_token_ids)
                group_ids_out.append(group_id)
    if was_training:
        model.train()

    return SampledRollouts(
        prompts=prompts_out,
        completions=completions_out,
        prompt_token_ids=prompt_token_ids_out,
        completion_token_ids=completion_token_ids_out,
        group_ids=torch.tensor(group_ids_out, dtype=torch.long, device=model.device),
    )


def _pad_sequences(sequences: list[list[int]], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        seq_tensor = torch.tensor(sequence, dtype=torch.long, device=device)
        input_ids[row, : len(sequence)] = seq_tensor
        attention_mask[row, : len(sequence)] = 1
    return input_ids, attention_mask


def build_rollout_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    rollouts: SampledRollouts,
    rewards: torch.Tensor,
) -> RolloutBatch:
    device = model.device
    full_sequences = [
        prompt_ids + completion_ids
        for prompt_ids, completion_ids in zip(rollouts.prompt_token_ids, rollouts.completion_token_ids, strict=True)
    ]
    input_ids, attention_mask = _pad_sequences(full_sequences, tokenizer.pad_token_id, device=device)
    prompt_lengths = torch.tensor([len(ids) for ids in rollouts.prompt_token_ids], dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    shifted_attention = attention_mask[:, 1:].bool()
    positions = torch.arange(target_ids.shape[1], device=device).unsqueeze(0)
    completion_mask = shifted_attention & (positions >= (prompt_lengths - 1).unsqueeze(1))
    token_logprobs = gather_logprobs(logits, target_ids)

    return RolloutBatch(
        token_logprobs=token_logprobs,
        old_token_logprobs=token_logprobs.detach(),
        completion_mask=completion_mask,
        rewards=rewards.to(device=device, dtype=torch.float32),
        group_ids=rollouts.group_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
