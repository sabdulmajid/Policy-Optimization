# Policy-Optimization

Train LLMs with modern RL objectives beyond PPO, using a modular PyTorch stack built for research speed and engineering reliability.

## In one sentence

Give this library your model, prompts, and reward signal; it gives you objective-specific policy updates plus reproducible benchmark artifacts.

## Why an MLE team would use this

- You need to compare RL objectives quickly (`rloo`/`gspo`/`cispo`/`dapo`/`maxrl`) without rewriting training code.
- You need reproducible benchmark outputs for PR reviews and experiment gates (JSONL + markdown artifacts).
- You need to tune larger open models with lower-memory objective variants before committing to expensive full-scale training.

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

## Latest benchmark results (easy to read)

Real long-horizon objective matrix (`24` steps, `3` seeds, `15` runs/model) on two widely-used models:

- `Qwen/Qwen2.5-7B`
- `NousResearch/Hermes-3-Llama-3.1-8B`

### What improved most

| Model | Baseline (`rloo`) final reward | Best objective final reward | Improvement vs `rloo` |
|---|---:|---:|---:|
| `Qwen/Qwen2.5-7B` | `0.8438` | `0.8854` (`maxrl`) | `+0.0416` |
| `Hermes-3-Llama-3.1-8B` | `0.3229` | `0.3738` (`dapo`) | `+0.0509` |

How to interpret:

- reward/success are in `[0, 1]`.
- `+0.01` = `+1` absolute percentage point improvement.
- This means the best objective is giving about `+4.16` points on Qwen-7B and `+5.09` points on Hermes-8B versus the `rloo` baseline.

### Full objective deltas (final-step reward vs `rloo`)

| Objective | Qwen-7B Δ vs `rloo` | Hermes-8B Δ vs `rloo` |
|---|---:|---:|
| `rloo` | `+0.0000` | `+0.0000` |
| `dapo` | `-0.1771` | `+0.0509` |
| `gspo` | `+0.0312` | `+0.0000` |
| `cispo` | `+0.0312` | `+0.0000` |
| `maxrl` | `+0.0416` | `-0.0104` |

Data quality (both models):

- parsed runs: `15/15`
- invalid runs: `0`
- tracebacks: `False`
- JSON parse errors: `0`

Raw artifacts:

- `reports/qwen_7b_sota_track_h24_2026-03-23.jsonl`
- `reports/qwen_7b_sota_track_h24_2026-03-23.md`
- `reports/hermes3_8b_sota_track_h24_2026-03-23.jsonl`
- `reports/hermes3_8b_sota_track_h24_2026-03-23.md`

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
po-smoke --model-id Qwen/Qwen2.5-0.5B --objective gspo --steps 2 --device cuda:0
```

### 3) Run a full benchmark matrix (resumable)

```bash
po-bench \
  --model-id Qwen/Qwen2.5-0.5B \
  --device cuda:0 \
  --steps 10 \
  --seeds 23 24 25 \
  --objectives rloo dapo gspo cispo maxrl \
  --output-prefix qwen_0.5b_benchmark_long_h10_2026-03-23

# If interrupted:
po-bench --resume \
  --model-id Qwen/Qwen2.5-0.5B \
  --device cuda:0 \
  --steps 10 \
  --seeds 23 24 25 \
  --objectives rloo dapo gspo cispo maxrl \
  --output-prefix qwen_0.5b_benchmark_long_h10_2026-03-23
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

- tests: `16 passed`
- objectives smoke-tested on real model: `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
- benchmark pipeline: resumable and artifact-backed

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
- longer-horizon training and stronger baseline comparisons

The point today: this is a practical objective-engineering platform MLEs can use immediately, and extend toward SOTA evaluations without rewriting the stack.
