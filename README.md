# Policy-Optimization

Train LLMs with modern RL objectives beyond PPO, using a modular PyTorch stack built for research speed and engineering reliability.

## In one sentence

Give this library your model, prompts, and reward signal; it gives you objective-specific policy updates plus reproducible benchmark artifacts.

## Why an MLE team would use this

- You need to compare RL objectives quickly (`rloo`/`gspo`/`cispo`/`dapo`/`maxrl`) without rewriting training code.
- You need reproducible benchmark outputs for PR reviews and experiment gates (JSONL + markdown artifacts).
- You need to tune larger open models with lower-memory objective variants before committing to expensive full-scale training.

## Current trainer capabilities

- GPU preflight is emitted before every smoke run and benchmark run.
- Rollout objectives recompute current-policy logprobs during optimization instead of reusing cached rollout values.
- Training and evaluation are separated: sampled prompts drive training, while a fixed held-out prompt set drives before/after comparison.
- Frozen-rollout optimization supports repeated epochs plus group-aware minibatching.
- Optional KL reward shaping can use a separately loaded frozen reference model, including on a second GPU.

## Operational guarantees in this repo

- Benchmarks only count as comparative evidence when they include fixed before/after eval events.
- Experiment CLIs emit machine-readable JSONL plus markdown summaries instead of relying on terminal output.
- The default workflow is single-node and PyTorch-native, so objective bugs are easier to inspect than in a large distributed trainer.

## What this project is

`Policy-Optimization` is a library for LLM post-training objectives, including:

- rollout-based: `RLOO`, `DAPO`, `GSPO`, `CISPO`, `MaxRL`, `GRPO`
- preference-based: `DPO`, `MDPO`, `DGPO`

It is designed so you can swap objective logic without rewriting your whole training loop.

## What goes in -> what comes out

- Input:
  - a causal LM (Hugging Face model)
  - prompts / rollout generation setup
  - reward function or preference pairs
  - objective choice (`rloo`, `dapo`, `gspo`, `cispo`, `maxrl`, etc.)
- Output:
  - updated policy parameters (via real optimizer steps)
  - objective metrics (`loss`, `reward_mean`, `success_rate`, grad norms)
  - benchmark artifacts (`.jsonl` + markdown reports) for auditability

## Three concrete production workflows

1. **Objective selection gate**
  - Run the same model + seed set across multiple objectives.
  - Pick objective by stability and reward delta before expensive long runs.
2. **Regression protection in PRs**
  - Keep benchmark outputs in `reports/` and compare deltas between commits.
  - Catch objective regressions before they reach training pipelines.
3. **Memory-aware rollout tuning**
  - Use value-free objectives (`rloo`, `grpo`) to increase batch/context under fixed VRAM.
  - Convert saved memory into more samples or longer trajectories.

## Why use this instead of a PPO-only stack?

- no value-network dependency for key methods (`RLOO`, `GRPO`) -> lower memory pressure
- objective-level modularity -> easier ablations and faster research iteration
- precision-safe math in critical paths -> fewer long-horizon instability failures
- real-model runnable CLIs (`po-smoke`, `po-bench`) -> reproducible evidence, not toy pseudocode

## Benchmark note

The repository includes archived real-model artifacts in `reports/`, but comparative claims should only be made from runs produced by the corrected fixed-eval benchmark path.

Current benchmark standard in this repo:

- run training on sampled prompts,
- evaluate before and after training on a fixed held-out prompt set,
- compare objectives using those fixed eval deltas, not changing training rewards.

Archived artifacts remain useful for debugging and reproducibility, but they should be treated as historical traces unless regenerated with the corrected benchmark flow.

## Path to industry-standard post-training

This repo is now a correct small-node trainer and objective workbench. The next bar is not cosmetic tuning; it is standardization:

1. Reproduce the same tasks with accepted baselines such as `TRL` and compare on identical evals.
2. Replace synthetic arithmetic as the headline result with accepted math, code, and instruction-following eval suites.
3. Add resumable training state, richer reference-policy management, and stronger long-run logging.
4. Move large-scale rollout serving to a distributed stack such as `verl` or `NVIDIA NeMo RL` once the single-node path is exhausted.

That progression keeps this repository focused on objective quality and experiment rigor, while scaling only after the local training loop is trustworthy.

## Examples

### 1) Compute one objective loss directly

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

### 2) Run a real-model smoke step

```bash
export HF_HOME=$PWD/.hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME
po-smoke \
  --model-id Qwen/Qwen2.5-0.5B \
  --objective gspo \
  --device cuda:0 \
  --trainable-scope lm_head \
  --updates-per-rollout 4 \
  --minibatch-groups 2 \
  --steps 2
```

### 3) Run a full benchmark matrix (resumable)

```bash
po-bench \
  --model-id Qwen/Qwen2.5-0.5B \
  --device cuda:0 \
  --trainable-scope lm_head \
  --updates-per-rollout 4 \
  --minibatch-groups 2 \
  --eval-prompts 16 \
  --steps 10 \
  --seeds 23 24 25 \
  --objectives rloo dapo gspo cispo maxrl \
  --output-prefix qwen_0.5b_benchmark_long_h10_2026-03-23

# If interrupted:
po-bench --resume \
  --model-id Qwen/Qwen2.5-0.5B \
  --device cuda:0 \
  --trainable-scope lm_head \
  --updates-per-rollout 4 \
  --minibatch-groups 2 \
  --eval-prompts 16 \
  --steps 10 \
  --seeds 23 24 25 \
  --objectives rloo dapo gspo cispo maxrl \
  --output-prefix qwen_0.5b_benchmark_long_h10_2026-03-23
```

### 4) Run with a frozen KL reference model on a second GPU

```bash
po-smoke \
  --model-id Qwen/Qwen2.5-0.5B \
  --reference-model-id Qwen/Qwen2.5-0.5B \
  --reference-device cuda:1 \
  --objective rloo \
  --device cuda:0 \
  --trainable-scope lm_head \
  --kl-beta 0.01 \
  --updates-per-rollout 2 \
  --minibatch-groups 1 \
  --steps 2
```

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

## Verified now

- tests: `29 passed, 1 skipped`
- objectives smoke-tested on real model: `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
- benchmark pipeline: resumable, GPU-preflighted, and fixed-eval artifact-backed
- trainer supports grouped rollout minibatching, repeated frozen-rollout epochs, and KL/reference penalties

The skipped test is LoRA-only and is skipped cleanly when `peft` is not installed in the current shell.

## Who this is for

- research engineers comparing RL objectives quickly
- teams that need reproducible objective benchmarks, not one-off scripts
- practitioners who want to move beyond PPO without rewriting infrastructure

## Why choose this

- You need faster objective iteration than monolithic PPO trainers.
- You care about reproducibility (runs produce auditable files, not just terminal logs).
- You want to benchmark multiple RL objectives on the same setup with minimal glue code.

## Current limit (and why)

This repo currently demonstrates **objective-engineering value** (modularity, reproducibility, stability checks) rather than claiming leaderboard SOTA.

What is still needed for SOTA-style claims:

- task-specific benchmarks (e.g., coding/math/instruction datasets with accepted eval suites)
- larger model matrix beyond one 7B family
- stronger baseline comparisons against standard stacks such as TRL or distributed RL runners
- checkpoint/resume infrastructure for longer training jobs

The point today: this is a practical objective-engineering platform MLEs can use immediately, and extend toward SOTA evaluations without rewriting the stack.
