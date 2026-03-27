from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import replace
from typing import Any

import torch

from policy_optimization.adapters import apply_lora_adapters
from policy_optimization.filtering import drop_zero_variance_groups
from policy_optimization.gpu import inspect_gpu_environment
from policy_optimization.hf import build_rollout_batch, load_causal_lm, sample_group_rollouts
from policy_optimization.rewards import apply_kl_reward_penalty, apply_overlong_reward_penalty, sequence_kl_penalty
from policy_optimization.trainers.rollout import optimize_rollout_batch

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-model smoke training step for an RL objective.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--objective", choices=["rloo", "dapo", "gspo", "cispo", "maxrl"], required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--updates-per-rollout", type=int, default=4)
    parser.add_argument("--minibatch-groups", type=int, default=0)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--eval-prompts", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=424242)
    parser.add_argument("--eval-max-new-tokens", type=int, default=0)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--overlong-penalty", type=float, default=0.0)
    parser.add_argument("--max-completion-length", type=int, default=24)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--reference-model-id", default="")
    parser.add_argument("--reference-device", default="")
    parser.add_argument("--trainable-scope", choices=["full", "lm_head", "lora"], default="lm_head")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dataset", choices=["synthetic", "gsm8k"], default="synthetic")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-dir")
    return parser.parse_args()


def build_arithmetic_prompts(prompt_count: int, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    prompts: list[str] = []
    answers: list[str] = []
    for _ in range(prompt_count):
        task_kind = rng.choices(["add", "sub", "mul", "nested"], weights=[0.3, 0.2, 0.3, 0.2], k=1)[0]
        if task_kind == "add":
            left = rng.randint(25, 350)
            right = rng.randint(25, 350)
            problem = f"{left} + {right}"
            answer = left + right
        elif task_kind == "sub":
            left = rng.randint(75, 500)
            right = rng.randint(10, left)
            problem = f"{left} - {right}"
            answer = left - right
        elif task_kind == "mul":
            left = rng.randint(11, 35)
            right = rng.randint(11, 35)
            problem = f"{left} * {right}"
            answer = left * right
        else:
            left = rng.randint(10, 40)
            middle = rng.randint(10, 35)
            right = rng.randint(3, 12)
            problem = f"({left} + {middle}) * {right}"
            answer = (left + middle) * right
        prompt = (
            "Solve the arithmetic problem carefully. "
            "Return only the final integer with no explanation.\n"
            f"Problem: {problem}\nAnswer:"
        )
        prompts.append(prompt)
        answers.append(str(answer))
    return prompts, answers


def normalize_number_text(text: str) -> str:
    normalized = text.replace(",", "").strip()
    if normalized.startswith("+"):
        normalized = normalized[1:]
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def parse_gsm8k_final_answer(answer_text: str) -> str | None:
    marker = "####"
    if marker in answer_text:
        candidate = answer_text.split(marker)[-1].strip()
        numbers = NUMBER_RE.findall(candidate)
        if numbers:
            return normalize_number_text(numbers[-1])
    numbers = NUMBER_RE.findall(answer_text)
    if not numbers:
        return None
    return normalize_number_text(numbers[-1])


def load_gsm8k_examples(split: str) -> list[tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package is required for --dataset gsm8k. Install with: pip install datasets") from exc

    dataset = load_dataset("gsm8k", "main", split=split)
    examples: list[tuple[str, str]] = []
    for row in dataset:
        question = str(row.get("question", "")).strip()
        answer = parse_gsm8k_final_answer(str(row.get("answer", "")))
        if not question or answer is None:
            continue
        prompt = (
            "Solve the following real-world math word problem. "
            "Return only the final numeric answer with no explanation.\n"
            f"Problem: {question}\n"
            "Answer:"
        )
        examples.append((prompt, answer))

    if not examples:
        raise RuntimeError(f"No valid GSM8K examples found for split={split}")
    return examples


def sample_gsm8k_prompts(examples: list[tuple[str, str]], prompt_count: int, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    prompts: list[str] = []
    answers: list[str] = []
    for _ in range(prompt_count):
        prompt, answer = examples[rng.randrange(len(examples))]
        prompts.append(prompt)
        answers.append(answer)
    return prompts, answers


def extract_last_number(text: str) -> str | None:
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return normalize_number_text(matches[-1])


def exact_match_rewards(completions: list[str], answers: list[str], group_ids: torch.Tensor) -> torch.Tensor:
    rewards = []
    for completion, group_id in zip(completions, group_ids.tolist(), strict=True):
        predicted = extract_last_number(completion)
        expected = normalize_number_text(answers[group_id])
        rewards.append(1.0 if predicted == expected else 0.0)
    return torch.tensor(rewards, dtype=torch.float32)


def build_eval_prompts(
    *,
    dataset: str,
    prompt_count: int,
    eval_seed: int,
    gsm8k_examples: list[tuple[str, str]] | None,
    dataset_split: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    if dataset == "gsm8k":
        if gsm8k_examples is None:
            gsm8k_examples = load_gsm8k_examples(dataset_split)
        prompts, answers = sample_gsm8k_prompts(gsm8k_examples, prompt_count, seed=eval_seed)
        metadata: dict[str, Any] = {"dataset": dataset, "split": dataset_split}
        return prompts, answers, metadata
    prompts, answers = build_arithmetic_prompts(prompt_count, seed=eval_seed)
    metadata = {"dataset": dataset, "split": dataset_split}
    return prompts, answers, metadata


def evaluate_policy(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    answers: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, float]:
    rollouts = sample_group_rollouts(
        model,
        tokenizer,
        prompts,
        group_size=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0.0,
    )
    rewards = exact_match_rewards(rollouts.completions, answers, rollouts.group_ids)
    completion_lengths = torch.tensor([len(ids) for ids in rollouts.completion_token_ids], dtype=torch.float32)
    return {
        "eval_reward_mean": float(rewards.mean().item()),
        "eval_success_rate": float((rewards >= 0.5).float().mean().item()),
        "eval_completion_length_mean": float(completion_lengths.mean().item()) if len(completion_lengths) else 0.0,
        "eval_examples": float(len(prompts)),
    }


def set_trainable_scope(model: torch.nn.Module, trainable_scope: str) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = trainable_scope == "full"

    if trainable_scope == "lm_head":
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None:
            raise ValueError("Model does not expose output embeddings for lm_head-only training.")
        for parameter in output_embeddings.parameters():
            parameter.requires_grad = True

    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_count == 0:
        raise ValueError("Selected trainable scope produced zero trainable parameters.")
    return int(trainable_count)


def freeze_model(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


def main() -> None:
    args = parse_args()
    eval_max_new_tokens = args.max_new_tokens if args.eval_max_new_tokens <= 0 else args.eval_max_new_tokens
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(args.device)
    print(json.dumps({"event": "gpu_preflight", **inspect_gpu_environment(device)}))
    model, tokenizer = load_causal_lm(
        args.model_id,
        device=device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )
    lora_target_modules: list[str] | None = None
    if args.trainable_scope == "lora":
        model, lora_target_modules = apply_lora_adapters(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    reference_model = None
    reference_model_id = args.reference_model_id.strip()
    reference_device = args.reference_device.strip() or str(device)
    if args.kl_beta > 0.0:
        if not reference_model_id:
            reference_model_id = args.model_id
        if reference_device != str(device):
            print(json.dumps({"event": "gpu_preflight", **inspect_gpu_environment(reference_device)}))
        reference_model, _ = load_causal_lm(
            reference_model_id,
            device=reference_device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
        )
        freeze_model(reference_model)

    model.train()
    if args.trainable_scope == "lora":
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    else:
        trainable_parameters = set_trainable_scope(model, args.trainable_scope)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )

    print(
        json.dumps(
            {
                "event": "init",
                "device": str(device),
                "dtype": args.dtype,
                "model_id": args.model_id,
                "objective": args.objective,
                "trainable_parameters": trainable_parameters,
                "updates_per_rollout": args.updates_per_rollout,
                "minibatch_groups": args.minibatch_groups,
                "cache_dir": args.cache_dir,
                "eval_prompts": args.eval_prompts,
                "eval_seed": args.eval_seed,
                "eval_max_new_tokens": eval_max_new_tokens,
                "eval_temperature": args.eval_temperature,
                "eval_top_p": args.eval_top_p,
                "kl_beta": args.kl_beta,
                "reference_model_id": reference_model_id or None,
                "reference_device": reference_device if args.kl_beta > 0.0 else None,
                "lora_rank": args.lora_rank if args.trainable_scope == "lora" else None,
                "lora_alpha": args.lora_alpha if args.trainable_scope == "lora" else None,
                "lora_dropout": args.lora_dropout if args.trainable_scope == "lora" else None,
                "lora_target_modules": lora_target_modules,
            }
        )
    )

    gsm8k_examples: list[tuple[str, str]] | None = None
    if args.dataset == "gsm8k":
        gsm8k_examples = load_gsm8k_examples(args.dataset_split)
        print(
            json.dumps(
                {
                    "event": "dataset_loaded",
                    "dataset": "gsm8k",
                    "split": args.dataset_split,
                    "examples": len(gsm8k_examples),
                }
            )
        )

    eval_prompts, eval_answers, eval_metadata = build_eval_prompts(
        dataset=args.dataset,
        prompt_count=args.eval_prompts,
        eval_seed=args.eval_seed,
        gsm8k_examples=gsm8k_examples,
        dataset_split=args.dataset_split,
    )
    print(json.dumps({"event": "eval_dataset", "stage": "setup", **eval_metadata, "examples": len(eval_prompts)}))
    baseline_eval = evaluate_policy(
        model,
        tokenizer,
        eval_prompts,
        eval_answers,
        max_new_tokens=eval_max_new_tokens,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
    )
    print(json.dumps({"event": "eval", "stage": "before_training", **baseline_eval}))

    for step in range(args.steps):
        if args.dataset == "gsm8k":
            assert gsm8k_examples is not None
            prompts, answers = sample_gsm8k_prompts(gsm8k_examples, args.prompts_per_step, seed=args.seed + step)
        else:
            prompts, answers = build_arithmetic_prompts(args.prompts_per_step, seed=args.seed + step)
        rollouts = sample_group_rollouts(
            model,
            tokenizer,
            prompts,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        rewards = exact_match_rewards(rollouts.completions, answers, rollouts.group_ids)
        batch = build_rollout_batch(model, tokenizer, rollouts, rewards, reference_model=reference_model)
        kl_mean = 0.0
        if args.kl_beta:
            if batch.ref_token_logprobs is None:
                raise RuntimeError("KL penalty requested, but no reference logprobs are available.")
            kl_penalty = sequence_kl_penalty(batch.old_token_logprobs, batch.ref_token_logprobs, batch.completion_mask)
            kl_mean = float(kl_penalty.mean().item())
            shaped_rewards = apply_kl_reward_penalty(
                batch.rewards,
                batch.old_token_logprobs,
                batch.ref_token_logprobs,
                batch.completion_mask,
                beta=args.kl_beta,
            )
            batch = replace(batch, rewards=shaped_rewards)
        if args.overlong_penalty:
            shaped_rewards = apply_overlong_reward_penalty(
                batch.rewards,
                batch.sequence_lengths,
                max_length=args.max_completion_length,
                penalty_weight=args.overlong_penalty,
            )
            batch = replace(batch, rewards=shaped_rewards)

        filtering = None
        if args.objective == "dapo":
            batch, filtering = drop_zero_variance_groups(batch)
            if batch.batch_size == 0:
                print(json.dumps({"event": "skip_step", "reason": "all groups had zero reward variance", "step": step}))
                continue

        result = optimize_rollout_batch(
            model,
            optimizer,
            batch,
            objective_name=args.objective,
            max_grad_norm=args.max_grad_norm,
            epochs_per_rollout=args.updates_per_rollout,
            groups_per_minibatch=args.minibatch_groups,
            seed=args.seed + step,
        )
        payload = {
            "event": "train_step",
            "step": step,
            "updates_per_rollout": args.updates_per_rollout,
            "minibatch_groups": args.minibatch_groups,
            "reward_mean": float(batch.rewards.mean().item()),
            "success_rate": float((batch.rewards >= 0.5).float().mean().item()),
            "examples": batch.batch_size,
            "kl_mean": kl_mean,
            **result.metrics,
        }
        if filtering is not None:
            payload["kept_groups"] = filtering.kept_group_count
            payload["dropped_groups"] = filtering.dropped_group_count
        print(json.dumps(payload))

    final_eval = evaluate_policy(
        model,
        tokenizer,
        eval_prompts,
        eval_answers,
        max_new_tokens=eval_max_new_tokens,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
    )
    print(json.dumps({"event": "eval", "stage": "after_training", **final_eval}))


if __name__ == "__main__":
    main()
