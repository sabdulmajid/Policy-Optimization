# Long-Horizon Benchmark Impact Report

## Setup

- Model: `NousResearch/Hermes-3-Llama-3.1-8B`
- Device: `cuda:1`
- Trainable scope: `lm_head`
- Objectives: `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
- Seeds: `23`, `24`, `25`
- Per-run config: `steps=24`, `prompts-per-step=8`, `group-size=4`, `max-new-tokens=24`, `temperature=1.0`
- Raw parsed artifact: `reports/hermes3_8b_sota_track_h24_2026-03-23.jsonl`

## Aggregate Results (Mean ± Std across seeds)

| Objective | Valid runs | Reward step0 | Reward stepN | Reward Δ | Success step0 | Success stepN | Success Δ | GradNorm step0 | GradNorm stepN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rloo` | 3/3 | 0.1979 ± 0.0295 | 0.3229 ± 0.0966 | +0.1250 ± 0.1112 | 0.1979 ± 0.0295 | 0.3229 ± 0.0966 | +0.1250 ± 0.1112 | 33.917 ± 7.484 | 42.250 ± 2.880 |
| `dapo` | 3/3 | 0.4014 ± 0.0373 | 0.3738 ± 0.0897 | -0.0276 ± 0.0829 | 0.4014 ± 0.0373 | 0.3738 ± 0.0897 | -0.0276 ± 0.0829 | 2.135 ± 0.246 | 1.784 ± 0.212 |
| `gspo` | 3/3 | 0.1979 ± 0.0295 | 0.3229 ± 0.0966 | +0.1250 ± 0.1112 | 0.1979 ± 0.0295 | 0.3229 ± 0.0966 | +0.1250 ± 0.1112 | 1.060 ± 0.234 | 1.320 ± 0.090 |
| `cispo` | 3/3 | 0.1979 ± 0.0295 | 0.3229 ± 0.0966 | +0.1250 ± 0.1112 | 0.1979 ± 0.0295 | 0.3229 ± 0.0966 | +0.1250 ± 0.1112 | 1.060 ± 0.234 | 1.320 ± 0.090 |
| `maxrl` | 3/3 | 0.1979 ± 0.0295 | 0.3125 ± 0.1112 | +0.1146 ± 0.1259 | 0.1979 ± 0.0295 | 0.3125 ± 0.1112 | +0.1146 ± 0.1259 | 25.583 ± 5.864 | 31.125 ± 3.127 |

## Data Quality Checks

- Parsed runs: `15`
- Invalid runs: `0`
- Any traceback in logs: `False`
- Any JSON parse errors: `0`

## Interpretation

- This benchmark is designed as strong engineering evidence for reproducibility and objective behavior under consistent conditions.
- It is not a final SOTA or convergence claim; use longer horizons, larger models, and task-specific datasets for that bar.

