from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import replace

import torch

from policy_optimization.filtering import drop_zero_variance_groups
from policy_optimization.hf import build_rollout_batch, load_causal_lm, sample_group_rollouts
from policy_optimization.rewards import apply_overlong_reward_penalty
from policy_optimization.trainers.step import run_policy_optimization_step

INTEGER_RE = re.compile(r"-?\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-model smoke training step for an RL objective.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--objective", choices=["rloo", "dapo", "gspo", "cispo", "maxrl"], required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--overlong-penalty", type=float, default=0.0)
    parser.add_argument("--max-completion-length", type=int, default=24)
    parser.add_argument("--trainable-scope", choices=["full", "lm_head"], default="lm_head")
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


def extract_last_integer(text: str) -> str | None:
    matches = INTEGER_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def exact_match_rewards(completions: list[str], answers: list[str], group_ids: torch.Tensor) -> torch.Tensor:
    rewards = []
    for completion, group_id in zip(completions, group_ids.tolist(), strict=True):
        predicted = extract_last_integer(completion)
        rewards.append(1.0 if predicted == answers[group_id] else 0.0)
    return torch.tensor(rewards, dtype=torch.float32)


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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(args.device)
    model, tokenizer = load_causal_lm(
        args.model_id,
        device=device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )
    model.train()
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
                "cache_dir": args.cache_dir,
            }
        )
    )

    for step in range(args.steps):
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
        batch = build_rollout_batch(model, tokenizer, rollouts, rewards)
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

        result = run_policy_optimization_step(
            model,
            optimizer,
            batch,
            objective_name=args.objective,
            max_grad_norm=args.max_grad_norm,
        )
        payload = {
            "event": "train_step",
            "step": step,
            "reward_mean": float(batch.rewards.mean().item()),
            "success_rate": float((batch.rewards >= 0.5).float().mean().item()),
            "examples": batch.batch_size,
            **result.metrics,
        }
        if filtering is not None:
            payload["kept_groups"] = filtering.kept_group_count
            payload["dropped_groups"] = filtering.dropped_group_count
        print(json.dumps(payload))


if __name__ == "__main__":
    main()
