# Long-Horizon Benchmark Impact Report

## Setup

- Model: `Qwen/Qwen2.5-7B`
- Device: `cuda:0`
- Trainable scope: `lm_head`
- Objectives: `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
- Seeds: `23`, `24`, `25`
- Per-run config: `steps=24`, `prompts-per-step=8`, `group-size=4`, `max-new-tokens=24`, `temperature=1.0`
- Raw parsed artifact: `reports/qwen_7b_sota_track_h24_2026-03-23.jsonl`

## Aggregate Results (Mean ± Std across seeds)

| Objective | Valid runs | Reward step0 | Reward stepN | Reward Δ | Success step0 | Success stepN | Success Δ | GradNorm step0 | GradNorm stepN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rloo` | 3/3 | 0.7396 ± 0.0390 | 0.8438 ± 0.1112 | +0.1042 ± 0.1451 | 0.7396 ± 0.0390 | 0.8438 ± 0.1112 | +0.1042 ± 0.1451 | 142.083 ± 83.542 | 302.042 ± 202.452 |
| `dapo` | 3/3 | 0.6042 ± 0.1062 | 0.6667 ± 0.1179 | +0.0625 ± 0.1350 | 0.6042 ± 0.1062 | 0.6667 ± 0.1179 | +0.0625 ± 0.1350 | 14.812 ± 4.133 | 33.708 ± 15.087 |
| `gspo` | 3/3 | 0.7396 ± 0.0390 | 0.8750 ± 0.1326 | +0.1354 ± 0.1640 | 0.7396 ± 0.0390 | 0.8750 ± 0.1326 | +0.1354 ± 0.1640 | 4.740 ± 1.409 | 4.214 ± 1.886 |
| `cispo` | 3/3 | 0.7396 ± 0.0390 | 0.8750 ± 0.1326 | +0.1354 ± 0.1640 | 0.7396 ± 0.0390 | 0.8750 ± 0.1326 | +0.1354 ± 0.1640 | 4.740 ± 1.409 | 4.214 ± 1.886 |
| `maxrl` | 3/3 | 0.7396 ± 0.0390 | 0.8854 ± 0.1031 | +0.1458 ± 0.1405 | 0.7396 ± 0.0390 | 0.8854 ± 0.1031 | +0.1458 ± 0.1405 | 315.667 ± 80.085 | 106.917 ± 67.324 |

## Data Quality Checks

- Parsed runs: `15`
- Invalid runs: `0`
- Any traceback in logs: `False`
- Any JSON parse errors: `0`

## Interpretation

- This benchmark is designed as strong engineering evidence for reproducibility and objective behavior under consistent conditions.
- It is not a final SOTA or convergence claim; use longer horizons, larger models, and task-specific datasets for that bar.

