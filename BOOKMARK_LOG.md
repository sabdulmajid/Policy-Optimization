# Bookmark Log

## 2026-03-24 00:56 America/Toronto

### GSM8K Real-Dataset Matrix Completion (Qwen-7B + Qwen-3B)

- Completed full GSM8K benchmark matrices for:
  - `Qwen/Qwen2.5-7B`
  - `Qwen/Qwen2.5-3B-Instruct`
- Both models ran with:
  - objectives: `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
  - seeds: `23`, `24`, `25`
  - config: `steps=24`, `prompts-per-step=8`, `group-size=4`, `max-new-tokens=24`, `temperature=1.0`, `top-p=0.95`

### Reliability/Robustness Fixes Applied

- Hardened `po-bench` aggregation in `src/policy_optimization/scripts/benchmark_matrix.py`:
  - dedupe by latest `(objective, seed)` row,
  - compute markdown/summary metrics from valid `ok` rows only,
  - avoid crash when resumed runs include intermediate bad rows.
- Resolved restart instability from inherited `CUDA_VISIBLE_DEVICES` causing `invalid device ordinal` on `cuda:1` relaunches.

### Final Artifacts

- Qwen-7B GSM8K:
  - `reports/qwen_7b_gsm8k_h24_2026-03-23.jsonl`
  - `reports/qwen_7b_gsm8k_h24_2026-03-23.summary.json`
  - `reports/qwen_7b_gsm8k_h24_2026-03-23.md`
- Qwen-3B GSM8K:
  - `reports/qwen_3b_gsm8k_h24_2026-03-23.jsonl`
  - `reports/qwen_3b_gsm8k_h24_2026-03-23.summary.json`
  - `reports/qwen_3b_gsm8k_h24_2026-03-23.md`

### Data Quality Summary

- Qwen-7B run completion reported: `parsed_runs=15`, `invalid_runs=0`.
- Qwen-3B run completion reported: `parsed_runs=15`, `invalid_runs=0`.

## 2026-03-23 18:55 America/Toronto

### Newer Llama Baseline Benchmark (Completed)

- Completed full objective matrix on `unsloth/Llama-3.2-3B-Instruct`:
  - objectives: `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
  - seeds: `23`, `24`, `25`
  - per run config: `steps=8`, `prompts-per-step=6`, `group-size=4`, `max-new-tokens=18`
- Built clean, auditable artifacts directly from per-objective logs after JSONL corruption in an interrupted run:
  - `reports/llama32_3b_real_h8_2026-03-23_clean.jsonl`
  - `reports/llama32_3b_real_h8_2026-03-23_clean.summary.json`
  - `reports/llama32_3b_real_h8_2026-03-23_clean.md`

### Baseline Comparison (Final Step Mean, 3 Seeds)

- `rloo`: `0.2639`
- `dapo`: `0.3722` (`+0.1083` vs `rloo`)
- `gspo`: `0.2500` (`-0.0139` vs `rloo`)
- `cispo`: `0.2500` (`-0.0139` vs `rloo`)
- `maxrl`: `0.2361` (`-0.0278` vs `rloo`)

### Reliability Notes

- Valid run coverage: `15/15`, invalid runs: `0`.
- `/pub3` cache path is space-constrained; keep `HF_HOME` and `TRANSFORMERS_CACHE` on `/pub7` during smoke/bench runs.
- Updated `README.md` to include newer Llama baseline table and plain-language interpretation (`+0.01` = `+1` absolute point).

## 2026-03-23 07:52 America/Toronto

### Real 7B Benchmark Checkpoint

- Completed real-model benchmark on `Qwen/Qwen2.5-7B`:
  - objectives: `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
  - seeds: `23`, `24`
  - per run config: `steps=8`, `prompts-per-step=6`, `group-size=4`, `max-new-tokens=18`, `temperature=1.0`
- Artifact files:
  - `reports/qwen_7b_benchmark_real_h8_2026-03-23.jsonl`
  - `reports/qwen_7b_benchmark_real_h8_2026-03-23.md`

### Aggregate Reward Delta (Step7 - Step0)

- `rloo`: `+0.0417 ± 0.1667`
- `dapo`: `+0.0625 ± 0.0625`
- `gspo`: `+0.0417 ± 0.1667`
- `cispo`: `+0.0417 ± 0.1667`
- `maxrl`: `+0.0417 ± 0.1667`

### Data Quality

- parsed runs: `10`
- invalid runs: `0`
- tracebacks in logs: `False`
- JSON parse errors: `0`

## 2026-03-23 07:35 America/Toronto

### Long-Horizon Benchmark + Reliability Upgrade

- Added a resumable benchmark pipeline CLI:
  - `po-bench` (`src/policy_optimization/scripts/benchmark_matrix.py`)
  - supports `--resume` so interrupted sessions can continue without losing completed runs.
- Completed long-horizon matrix artifacts:
  - `reports/qwen_0.5b_benchmark_long_h10_2026-03-23.jsonl`
  - `reports/qwen_0.5b_benchmark_long_h10_2026-03-23.summary.json`
  - `reports/qwen_0.5b_benchmark_long_h10_2026-03-23.md`

### Long-Horizon Data Quality

- parsed runs: `15`
- invalid runs: `0`
- tracebacks in logs: `False`
- JSON parse errors: `0`

### Key Aggregate Signal (Reward Delta, Step9 - Step0)

- `rloo`: `+0.0417 ± 0.1179`
- `dapo`: `-0.0486 ± 0.1025`
- `gspo`: `-0.0417 ± 0.1559`
- `cispo`: `-0.0417 ± 0.1559`
- `maxrl`: `+0.0139 ± 0.1288`

### Resume Next

1. Re-run `po-bench --resume` with larger horizons (`steps=20` or `40`) to tighten confidence intervals.
2. Add preference-objective benchmark matrix (`dpo`, `mdpo`, `dgpo`) on preference datasets.
3. Keep commit cadence tight: benchmark artifacts first, then docs/logs.

## 2026-03-23 07:16 America/Toronto

### Full Matrix Benchmark Checkpoint

- Completed full multi-seed benchmark matrix on `Qwen/Qwen2.5-0.5B`:
  - objectives: `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
  - seeds: `23`, `24`, `25`
  - run config: `steps=3`, `prompts-per-step=6`, `group-size=4`, `max-new-tokens=18`, `temperature=1.0`
- Parsed run coverage: `15/15` valid runs, `0` parse errors, `0` tracebacks.

### Artifacts

- Raw parsed records: `reports/qwen_0.5b_benchmark_multiseed_2026-03-23.jsonl`
- Human report: `reports/benchmark-impact-multiseed-2026-03-23.md`

### Aggregate Reward Delta (Step2 - Step0, Mean ± Std)

- `rloo`: `-0.0694 ± 0.1039`
- `dapo`: `-0.0278 ± 0.1571`
- `gspo`: `-0.0972 ± 0.1375`
- `cispo`: `-0.0972 ± 0.1375`
- `maxrl`: `-0.0556 ± 0.1094`

### Interpretation Note

- This is a strong smoke-style reliability benchmark (real model, real optimizer steps) and should not be treated as final convergence evidence.
- Next quality gate is longer-horizon runs and/or larger model scale with the same reporting format.

## 2026-03-23 07:06 America/Toronto

### Benchmark Checkpoint (Concrete Impact)

- Completed comparable real-model benchmark sweep on `Qwen/Qwen2.5-0.5B` across:
  - `rloo`, `dapo`, `gspo`, `cispo`, `maxrl`
- Benchmark artifacts:
  - `reports/qwen_0.5b_benchmark_2026-03-23_v2.jsonl` (structured per-objective first/last step metrics)
  - `reports/benchmark-impact-2026-03-23.md` (human-readable impact summary)
- `pytest` re-run after report generation: all tests passed (`16` tests).

### Key Measured Outcome

- On this fixed seed/config smoke benchmark:
  - `rloo`, `gspo`, `cispo`, `maxrl`: reward mean `0.5833 -> 0.6667` (`+0.0833`)
  - `dapo`: reward mean `0.7500 -> 0.5000` (`-0.2500`) with aggressive group filtering (`kept_groups=2`, `dropped_groups=4`)
- `gspo`/`cispo` reached the same reward delta with lower grad norms than `rloo`/`maxrl` in this run.

### Resume Next

1. If extending benchmark quality, run multi-seed repeats (`seed`: 23/24/25) and aggregate means/std.
2. Add preference-objective benchmark (`dpo`, `mdpo`, `dgpo`) on a preference-formatted dataset.
3. Commit after each benchmark chunk to avoid data loss on disconnect.

## 2026-03-23 06:56 America/Toronto

### Session Checkpoint

- Created commit `52bcc63` on `main`:
  - README positioning updated to emphasize advanced RL for LLM post-training.
  - Added `GRPO` coverage to the scalar-loss objective test loop.
- Created follow-on commits `3d91670` and `b33ca32`:
  - persisted session handoff notes,
  - reconciled README objective coverage with implemented registry.
- Focused verification passed:
  - `./.venv/bin/pytest -q tests/test_losses.py`
- Full suite verification passed:
  - `./.venv/bin/pytest -q`
- Working tree was clean immediately after this checkpoint.

### Fast Resume Path

1. `git log --oneline -n 5` and start from `52bcc63`.
2. Re-run `./.venv/bin/pytest -q` to re-validate the full suite in-session.
3. Continue objective/documentation reconciliation and commit in small chunks.

### Next Suggested Work Chunk

- Reconcile README objective coverage against `OBJECTIVE_REGISTRY` (`cispo`, `dapo`, `dgpo`, `dpo`, `gspo`, `grpo`, `maxrl`, `mdpo`, `rloo`) while keeping the top-level narrative focused.
- Add one brief smoke/report note after next real run, then checkpoint commit again.

## 2026-03-23 06:33 America/Toronto

### Current State

- The repo is no longer an empty scaffold. It now contains:
  - the core `policy_optimization` package,
  - modular RL losses for `rloo`, `dapo`, `gspo`, `cispo`, and `maxrl`,
  - additional in-progress losses/modules from another interrupted session: `dgpo`, `dpo`, `grpo`, `mdpo`, `vlm`, and `driving/*`,
  - smoke/eval scripts, tests, and reports under `reports/`.
- Local env exists at `./.venv` and the package is installed editable.
- Git remote `origin` is set to `https://github.com/sabdulmajid/Policy-Optimization.git`.
- Local git identity is configured as:
  - `user.name = Neel Abdul-Majid`
  - `user.email = sabdulmajid@users.noreply.github.com`

### Verified So Far

- `./.venv/bin/pytest` passed locally.
  - `tasks/todo.md` reports `16 passed`.
- Direct CUDA probe is healthy now:
  - `torch.cuda.is_available() == True`
  - device `0`: `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`
- GPUs are currently idle.
- Existing smoke artifacts indicate real-model success on `Qwen/Qwen2.5-0.5B`.
  - See `reports/smoke-2026-03-23.md`
  - See `reports/qwen_0.5b_smoke.jsonl`
- Existing Moonshot-family preflight succeeded for `moonshotai/Moonlight-16B-A3B`.

### Important Resume Files

- `tasks/todo.md`
- `reports/smoke-2026-03-23.md`
- `reports/qwen_0.5b_smoke.jsonl`
- `src/policy_optimization/scripts/smoke_train.py`
- `src/policy_optimization/hf.py`
- `src/policy_optimization/losses/__init__.py`
- `src/policy_optimization/losses/dgpo.py`
- `src/policy_optimization/losses/dpo.py`
- `src/policy_optimization/losses/grpo.py`
- `src/policy_optimization/losses/mdpo.py`
- `src/policy_optimization/vlm.py`
- `src/policy_optimization/driving/`
- `tests/test_losses.py`
- `tests/test_driving.py`

### Important Environment Notes

- Redirect Hugging Face cache to `/pub7/hf-cache` on this machine.
  - `/pub3` is full.
- Example command pattern from the existing smoke report:

```bash
HF_HOME=/pub7/hf-cache ./.venv/bin/po-smoke \
  --model-id Qwen/Qwen2.5-0.5B \
  --device cuda:0 \
  --trainable-scope lm_head
```

### Git / Persistence Notes

- There was no successful GitHub push yet.
- `ssh -T git@github.com` currently fails with `Permission denied (publickey)`.
- No git credential helper is configured locally.
- Because of the disconnect risk, commit after every meaningful chunk before starting long downloads or GPU runs.

### Clean Resume Checklist

1. Run `git status` and inspect the current staged/untracked state.
2. Open `tasks/todo.md` and `reports/smoke-2026-03-23.md`.
3. Re-run `./.venv/bin/pytest` before making behavioral changes.
4. Preserve the extra interrupted-session files:
   - `reports/*`
   - `src/policy_optimization/driving/*`
   - `src/policy_optimization/losses/{dgpo,dpo,grpo,mdpo}.py`
   - `src/policy_optimization/vlm.py`
   - `src/policy_optimization/scripts/drivingvqa_eval.py`
   - `tests/test_driving.py`
5. Make a checkpoint commit before any new smoke run or refactor.
6. When ready to push, use HTTPS auth or configure GitHub credentials/SSH first.

### Immediate Next Steps

1. Commit the entire current workspace as a checkpoint so nothing is lost.
2. Reconcile the interrupted-session files with the main package exports and docs.
3. Run/extend larger smoke experiments after the checkpoint is safely committed.
