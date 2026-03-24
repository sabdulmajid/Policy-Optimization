# Long-Horizon Benchmark Impact Report: 2026-03-23

## Setup

- Model: `Qwen/Qwen2.5-0.5B`
- Device: `cuda:0`
- Trainable scope: `lm_head`
- Objectives: `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
- Seeds: `23`, `24`, `25` (3 runs/objective)
- Per-run config: `steps=10`, `prompts-per-step=6`, `group-size=4`, `max-new-tokens=18`, `temperature=1.0`
- Raw artifact: `reports/qwen_0.5b_benchmark_long_h10_2026-03-23.jsonl`

## Aggregate Results (Mean ± Std across seeds)

| Objective | Valid runs | Reward step0 | Reward step9 | Reward Δ | Success step0 | Success step9 | Success Δ | GradNorm step0 | GradNorm step9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rloo` | 3/3 | 0.5972 ± 0.0520 | 0.6389 ± 0.0708 | +0.0417 ± 0.1179 | 0.5972 ± 0.0520 | 0.6389 ± 0.0708 | +0.0417 ± 0.1179 | 361.333 ± 64.855 | 25.458 ± 20.141 |
| `dapo` | 3/3 | 0.6111 ± 0.1416 | 0.5625 ± 0.0884 | -0.0486 ± 0.1025 | 0.6111 ± 0.1416 | 0.5625 ± 0.0884 | -0.0486 ± 0.1025 | 36.208 ± 8.288 | 45.750 ± 17.150 |
| `gspo` | 3/3 | 0.5972 ± 0.0520 | 0.5556 ± 0.1195 | -0.0417 ± 0.1559 | 0.5972 ± 0.0520 | 0.5556 ± 0.1195 | -0.0417 ± 0.1559 | 15.583 ± 2.541 | 16.156 ± 9.125 |
| `cispo` | 3/3 | 0.5972 ± 0.0520 | 0.5556 ± 0.1195 | -0.0417 ± 0.1559 | 0.5972 ± 0.0520 | 0.5556 ± 0.1195 | -0.0417 ± 0.1559 | 15.583 ± 2.541 | 16.156 ± 9.125 |
| `maxrl` | 3/3 | 0.5972 ± 0.0520 | 0.6111 ± 0.0856 | +0.0139 ± 0.1288 | 0.5972 ± 0.0520 | 0.6111 ± 0.0856 | +0.0139 ± 0.1288 | 494.667 ± 123.476 | 449.333 ± 137.466 |

## Data Quality Checks

- Parsed runs: `15`
- Invalid runs: `0`
- Any traceback in logs: `False`
- Any JSON parse errors: `0`

## Interpretation

- This longer-horizon benchmark is stronger than short smoke runs for measuring stability trends under repeated optimization updates.
- It is still not a convergence or SOTA benchmark; that requires larger scales and task-specific evaluations.
