# Policy-Optimization

`Policy-Optimization` is a PyTorch-native research repository for large language model post-training with modern reinforcement learning objectives implemented from scratch.

The codebase is built around a simple rule: keep advantage estimation, rollout filtering, reward shaping, and policy losses fully decoupled so new objectives can be assembled without rewriting the training stack.

## Included Objectives

- `RLOO`: REINFORCE Leave-One-Out with a memory-efficient leave-one-out baseline and no value network.
- `DAPO`: decoupled lower/upper clipping, zero-variance group filtering, and overlong reward shaping.
- `GSPO`: sequence-level importance ratios with length normalization for numerically stable optimization.
- `CISPO`: direct clipping of importance-sampling weights rather than PPO-style ratio clipping.
- `MaxRL`: a compute-indexed maximum-likelihood RL estimator that targets a truncated pass@k objective.

## Design Principles

- `Numerical stability first`: log-softmax, sequence log-probability accumulation, and ratio construction are explicitly upcast to FP32.
- `Reference-free when possible`: KL can be folded directly into shaped rewards instead of forcing a reference-policy loss term.
- `Modular data flow`: reward shaping, advantage construction, filtering, and loss computation are separate modules.
- `Real-model friendly`: the package includes Hugging Face helpers and smoke-training scripts for open models.

## Project Layout

```text
src/policy_optimization/
  advantages.py
  filtering.py
  hf.py
  ops.py
  precision.py
  rewards.py
  types.py
  losses/
  trainers/
tests/
tasks/
```

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

If your home directory cache is space-constrained, point Hugging Face caching at a larger volume before smoke runs:

```bash
export HF_HOME=$PWD/.hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME
```

## Minimal Example

```python
import torch

from policy_optimization.losses import compute_objective
from policy_optimization.types import RolloutBatch

batch = RolloutBatch(
    token_logprobs=torch.tensor([[-0.4, -0.3], [-0.8, -0.2]]),
    old_token_logprobs=torch.tensor([[-0.5, -0.4], [-0.7, -0.1]]),
    completion_mask=torch.tensor([[True, True], [True, True]]),
    rewards=torch.tensor([1.0, 0.0]),
    group_ids=torch.tensor([0, 0]),
)

output = compute_objective("rloo", batch)
print(output.loss)
print(output.metrics)
```

## Smoke Runs

The repository ships with `po-smoke`, a small CLI that:

1. loads a Hugging Face causal LM,
2. samples grouped rollouts for arithmetic prompts,
3. computes exact-match rewards,
4. applies one of the policy objectives, and
5. runs a real backward/optimizer step.

Example:

```bash
export HF_HOME=$PWD/.hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME
po-smoke --model-id Qwen/Qwen2.5-0.5B --objective gspo --steps 2 --device cuda:0
```

## Status

This repository is under active construction and currently targets a clean, well-tested v0.1 foundation for open-source LLM RL experimentation.

### Verified Locally

- `pytest`: `13 passed`
- GPU smoke matrix on `Qwen/Qwen2.5-0.5B`: `rloo`, `dapo`, `gspo`, `cispo`, and `maxrl`
- Larger-checkpoint GPU smoke on `Qwen/Qwen2.5-7B`: `rloo`
- Moonshot remote-code preflight on `moonshotai/Moonlight-16B-A3B`: `AutoConfig` and `AutoTokenizer`
