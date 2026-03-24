# Long-Horizon Benchmark Impact Report

## Setup

- Model: `Qwen/Qwen2.5-7B`
- Device: `cuda:0`
- Trainable scope: `lm_head`
- Objectives: `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
- Seeds: `23`, `24`, `25`
- Per-run config: `steps=8`, `prompts-per-step=6`, `group-size=4`, `max-new-tokens=18`, `temperature=1.0`
- Raw parsed artifact: `reports/qwen_7b_benchmark_real_h8_2026-03-23.jsonl`

## Aggregate Results (Mean ± Std across seeds)

| Objective | Valid runs | Reward step0 | Reward stepN | Reward Δ | Success step0 | Success stepN | Success Δ | GradNorm step0 | GradNorm stepN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rloo` | 3/3 | 0.7778 ± 0.0393 | 0.8472 ± 0.1571 | +0.0694 ± 0.1416 | 0.7778 ± 0.0393 | 0.8472 ± 0.1571 | +0.0694 ± 0.1416 | 146.667 ± 53.990 | 91.708 ± 73.496 |
| `dapo` | 3/3 | 0.7083 ± 0.0589 | 0.7500 ± 0.0000 | +0.0417 ± 0.0589 | 0.7083 ± 0.0589 | 0.7500 ± 0.0000 | +0.0417 ± 0.0589 | 12.021 ± 2.072 | 40.792 ± 15.759 |
| `gspo` | 3/3 | 0.7778 ± 0.0393 | 0.8472 ± 0.1571 | +0.0694 ± 0.1416 | 0.7778 ± 0.0393 | 0.8472 ± 0.1571 | +0.0694 ± 0.1416 | 6.802 ± 2.590 | 6.797 ± 2.626 |
| `cispo` | 3/3 | 0.7778 ± 0.0393 | 0.8472 ± 0.1571 | +0.0694 ± 0.1416 | 0.7778 ± 0.0393 | 0.8472 ± 0.1571 | +0.0694 ± 0.1416 | 6.802 ± 2.590 | 6.797 ± 2.626 |
| `maxrl` | 3/3 | 0.7778 ± 0.0393 | 0.8472 ± 0.1571 | +0.0694 ± 0.1416 | 0.7778 ± 0.0393 | 0.8472 ± 0.1571 | +0.0694 ± 0.1416 | 381.667 ± 135.377 | 152.333 ± 99.379 |

## Data Quality Checks

- Parsed runs: `15`
- Invalid runs: `0`
- Any traceback in logs: `False`
- Any JSON parse errors: `0`

## Interpretation

- This benchmark is designed as strong engineering evidence for reproducibility and objective behavior under consistent conditions.
- It is not a final SOTA or convergence claim; use longer horizons, larger models, and task-specific datasets for that bar.

