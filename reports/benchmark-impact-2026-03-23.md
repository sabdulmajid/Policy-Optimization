# Benchmark Impact Report: 2026-03-23

## What Was Actually Tested

- Model: `Qwen/Qwen2.5-0.5B`
- Device: `cuda:0`
- Training scope: `lm_head` only
- Objectives benchmarked: `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
- Run shape per objective: `2` optimizer steps, `6` prompts per step, group size `4` (up to `24` sampled completions per step before filtering)
- Artifact with raw machine-readable outputs: `reports/qwen_0.5b_benchmark_2026-03-23_v2.jsonl`

## Benchmark Results (First Step -> Last Step)

| Objective | Reward Mean | Delta | Success Rate | Last-Step Examples | Grad Norm (first -> last) |
|---|---:|---:|---:|---:|---:|
| `rloo` | `0.5833 -> 0.6667` | `+0.0833` | `0.5833 -> 0.6667` | `24` | `328.0 -> 520.0` |
| `dapo` | `0.7500 -> 0.5000` | `-0.2500` | `0.7500 -> 0.5000` | `8` | `45.5 -> 65.0` |
| `gspo` | `0.5833 -> 0.6667` | `+0.0833` | `0.5833 -> 0.6667` | `24` | `15.1875 -> 21.75` |
| `cispo` | `0.5833 -> 0.6667` | `+0.0833` | `0.5833 -> 0.6667` | `24` | `15.1875 -> 21.75` |
| `maxrl` | `0.5833 -> 0.6667` | `+0.0833` | `0.5833 -> 0.6667` | `24` | `324.0 -> 478.0` |

## Practical Impact (Current Evidence)

- All five objectives run real forward + backward optimization steps on a real model with nontrivial rewards (not mocked).
- On this short run, `rloo`, `gspo`, `cispo`, and `maxrl` all improved reward and success rate by `+0.0833`.
- `gspo`/`cispo` achieved the same reward movement with substantially lower gradient norms than `rloo`/`maxrl` in this setup.
- `dapo` dropped to lower reward in this particular seed/config because group filtering retained only `8` examples at each step (`kept_groups=2`, `dropped_groups=4`).

## What This Does NOT Prove Yet

- This is a short smoke benchmark, not a final convergence or SOTA comparison.
- It does not yet include multi-seed confidence intervals, wall-clock throughput tracking, or larger-model scaling curves.
- It does not yet benchmark preference objectives (`dpo`, `mdpo`, `dgpo`) on preference datasets.

## Reproduction Command Pattern

```bash
HF_HOME=/pub7/hf-cache HUGGINGFACE_HUB_CACHE=/pub7/hf-cache ./.venv/bin/po-smoke \
  --model-id Qwen/Qwen2.5-0.5B \
  --objective <objective> \
  --device cuda:0 \
  --cache-dir /pub7/hf-cache \
  --trainable-scope lm_head \
  --steps 2 \
  --prompts-per-step 6 \
  --group-size 4 \
  --max-new-tokens 18 \
  --temperature 1.0 \
  --seed 23
```
