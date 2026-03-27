# Task Plan

## Next Small-Node Review

- [x] Re-audit the rollout, optimization, and benchmark paths after the recent fixes.
- [x] Rank the next three highest-leverage, low-risk implementation improvements for single-node training.
- [ ] Implement batched rollout generation so prompt groups stop serializing generation calls.
- [ ] Implement token-budget-aware optimization with gradient accumulation so long completions stop dictating unstable memory use.
- [ ] Implement single-run checkpoint/resume so interrupted training jobs do not waste GPU time.

## Review Session

- [x] Inspect repository state, recent history, and local instructions.
- [x] Review core objective implementations and training code for correctness.
- [x] Verify local test suite and inspect shipped benchmark artifacts.
- [x] Check available GPU resources on this machine.
- [x] Research current post-training stacks and candidate open models to target next.

## Immediate Repair Work

- [x] Fix the old-policy/current-policy bug in the rollout trainer.
  Store rollout tokens plus frozen old logprobs, then recompute current logprobs inside the optimization step before forming ratios.
- [x] Separate training rollouts from evaluation.
  Use a fixed held-out eval set and report eval metrics there instead of comparing step-0 vs step-N rewards on changing prompts.
- [x] Make objective implementations match their claims where the mismatch was clear.
  Wire `MaxRL` pass@k scaling into the actual loss, harden `DGPO` weighting against zero or negative reward means, and decide whether `DAPO`/`GSPO`/`CISPO` remain simplified variants or are brought to paper-faithful formulations.
- [x] Add algorithm-defining tests.
  Cover ratio recomputation, clipped metrics on variable-length sequences, `MaxRL` scaling behavior, `DGPO` bounded weights, and benchmark status handling.
- [x] Tighten benchmark validity and failure handling.
  Include process return code in validity, preserve eval prompts per run, and emit confidence intervals plus seed-level summaries.
- [x] Enforce GPU preflight before experiments and benchmarks.
  Add a reusable GPU inspection utility and emit preflight metadata from `po-smoke` and `po-bench`.

## Final Polish And Publish

- [x] Align the task plan, README, and implementation status.
  Make sure the repo docs describe what is implemented now, what is historical, and what is still missing.
- [x] Re-run final verification before publishing.
  Confirm unit tests still pass and inspect the final diff for accidental scope drift.
- [x] Publish from a dedicated branch instead of `main`.
  Create a `codex/...` branch, commit intentionally, and push with upstream tracking.

## Scale-Up Roadmap

- [ ] Phase 1: make this repo a correct small-node trainer.
  Next sequence:
  1. Batch rollout generation.
  2. Add token-budget-aware minibatching / accumulation.
  3. Add checkpoint resume.
  4. Add config files, structured logging, and reproducible eval jobs.
- [ ] Phase 2: establish industry-standard baselines before custom research.
  Reproduce equivalent GRPO/DPO/RLOO baselines with `TRL`, then compare this repo against them on the same data and evals.
- [ ] Phase 3: move online RL to a distributed stack built for rollouts.
  Use `verl` or `NVIDIA NeMo RL` with vLLM/SGLang-backed generation and FSDP/ZeRO-style training once single-node correctness is proven.
- [ ] Phase 4: promote the library into a research layer, not a monolith.
  Keep this repo for objective kernels, experiment orchestration, and validation; delegate large-scale rollout serving and distributed scheduling to the upstream training stack.

## Model Ladder For This Machine

- [ ] Tier 1: dense 7B to 8B instruction models for fast iteration.
  Start with `Qwen/Qwen3-8B` and one strong 8B baseline from the Llama family.
- [ ] Tier 2: reasoning-focused 27B to 32B models that fit this node well with PEFT.
  Prioritize `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`; pair it with one modern 27B to 30B class baseline.
- [ ] Tier 3: MoE experiments that stress rollout and serving infrastructure.
  Add a Qwen3 MoE checkpoint after the trainer is stable so you can validate sequence-level objectives under expert routing.
- [ ] Tier 4: heavyweight control baseline.
  Use a 70B-class model with QLoRA or selective tuning as a scorecard model, not as the first online-RL target on this two-GPU node.

## Evaluation Standard

- [ ] Replace synthetic arithmetic as the headline benchmark.
  Keep it only as a smoke test.
- [ ] Add accepted task suites by domain.
  Math: GSM8K, MATH-500, AIME-style evals.
  Code: LiveCodeBench and SWE-style agentic tasks only after the text trainer is trustworthy.
  General alignment/instruction-following: preference datasets plus judge-backed evals.
- [ ] Track both training metrics and real eval metrics.
  Reward, KL, entropy, clip stats, response length, pass@k, exact match, judge win rate, and cost per successful sample.

# Review

## What Is Good

- The package layout is clean and the objective registry is easy to extend.
- Numerical-safety helpers are simple and consistently used for sensitive reductions.
- The repo already has CI, unit tests, and artifact-backed reporting instead of ad hoc terminal logs.
- The benchmark scripts are readable, resumable, and easy to operate on one machine.
- Local verification is healthy:
  - `pytest`: `29 passed, 1 skipped in 5.09s`
  - GPU inventory: `2 x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`, about `97.9 GiB` each

## Historical Critical Findings Addressed

- The online-RL trainer previously did not compare an old policy to a recomputed current policy.
  It now recomputes current logprobs from model weights during each optimizer step.
  Files:
  - `src/policy_optimization/hf.py`
  - `src/policy_optimization/trainers/step.py`
- The headline benchmark previously compared changing training prompts and presented them like comparable before/after results.
  It now emits fixed held-out eval events before and after training and aggregates those deltas.
  Files:
  - `src/policy_optimization/scripts/smoke_train.py`
  - `src/policy_optimization/scripts/benchmark_matrix.py`
  - `README.md`
- `MaxRL` previously computed a pass@k-style scaling signal but never used it in the loss.
  File:
  - `src/policy_optimization/losses/maxrl.py`
- `DGPO` previously could produce unstable or negative sample weights when reward means were small, zero, or negative.
  File:
  - `src/policy_optimization/losses/dgpo.py`

## Remaining Gaps

- The trainer still uses full-group minibatching rather than token-count-aware packing, so throughput is not yet optimized for long-sequence imbalance.
- KL/reference-policy support exists for reward shaping, but there is not yet a richer reference-policy management story with checkpoints and resumable training state.
- `DAPO` / `GSPO` / `CISPO` remain compact implementations; if paper-faithful behavior is required, the repo should explicitly pick target formulations and close the remaining gap.
- The default smoke and benchmark datasets are still lightweight; accepted math/code/alignment suites are the next bar for serious comparison.

## Architectural Assessment

- This is currently a compact objective-and-smoke-testing library, not yet a production-ready LLM post-training stack.
- The right next move is not “throw bigger models at it immediately.”
  The right next move is:
  1. Fix trainer correctness.
  2. Establish accepted baselines with standard stacks.
  3. Then spend the Blackwell GPUs on real model and eval ladders.

## Latest Follow-Up Review

- 1. Batch rollout generation in `src/policy_optimization/hf.py`.
  The current `sample_group_rollouts(...)` loop calls `model.generate(...)` once per prompt group, which serializes the most expensive part of the trainer and leaves the GPU underfed.
  Add batched tokenization plus grouped unpacking, then cover it with a targeted `tests/test_hf.py`.
- 2. Replace fixed group-count minibatching with token-budget-aware optimization in `src/policy_optimization/trainers/rollout.py`, `src/policy_optimization/trainers/step.py`, and the CLI surfaces in `src/policy_optimization/scripts/smoke_train.py` and `src/policy_optimization/scripts/benchmark_matrix.py`.
  The current trainer only slices by group count, so one long completion can still blow up memory while short batches waste available VRAM.
  Add a `max-tokens-per-minibatch` path and gradient accumulation so the node can trade compute for stability and larger effective batch size.
- 3. Add real checkpoint/resume for individual training runs in `src/policy_optimization/scripts/smoke_train.py` with a small helper module such as `src/policy_optimization/checkpoints.py`, and forward it from `src/policy_optimization/scripts/benchmark_matrix.py`.
  The matrix runner can skip completed rows, but a single interrupted `po-smoke` run still loses model state, optimizer state, RNG state, and eval context.
  That is wasted GPU time on the exact kind of longer single-node jobs this repo is moving toward.

## Implementation Results

- Implemented:
  - GPU preflight utility at `src/policy_optimization/gpu.py`.
  - Current-logprob recomputation in the optimization step.
  - Fixed-eval events before and after training in `po-smoke`.
  - Fixed-eval aggregation in `po-bench`.
  - Group-aware minibatching plus repeated epochs over frozen rollout data.
  - Optional KL reward shaping with a separately loaded frozen reference model.
  - LoRA adapter support as a trainable scope for smoke and benchmark runs.
  - `MaxRL`, `DGPO`, `DAPO`, and `GRPO` correctness hardening.
- Verification completed:
  - `pytest`: `29 passed, 1 skipped in 5.09s`
  - GPU smoke run:
    - `po-smoke --model-id Qwen/Qwen2.5-0.5B --objective rloo --device cuda:0 ...`
    - confirmed `gpu_preflight` emission and `recomputed_logprobs = 1.0`
  - GPU smoke run with KL/reference model and grouped minibatching:
    - `po-smoke --model-id Qwen/Qwen2.5-0.5B --reference-model-id Qwen/Qwen2.5-0.5B --reference-device cuda:1 --kl-beta 0.01 --minibatch-groups 1 ...`
    - confirmed dual-GPU preflight, grouped optimizer steps, and reference-model loading
  - Benchmark smoke run:
    - `po-bench --model-id Qwen/Qwen2.5-0.5B --device cuda:0 --objectives rloo --seeds 11 ...`
    - completed with `status=ok` and valid fixed-eval summaries
  - Benchmark smoke run with KL/reference forwarding:
    - `po-bench --model-id Qwen/Qwen2.5-0.5B --reference-model-id Qwen/Qwen2.5-0.5B --reference-device cuda:1 --kl-beta 0.01 --minibatch-groups 1 ...`
    - wrote valid fixed-eval artifacts for the new forwarding path
  - Benchmark smoke run with LoRA:
    - `po-bench --model-id Qwen/Qwen2.5-0.5B --trainable-scope lora --lora-rank 8 --device cuda:0 --objectives rloo --seeds 13 ...`
    - completed with `status=ok` and valid fixed-eval summaries
  - Developer workflow hardening:
    - `pytest` now adds `src/` via `pyproject.toml`, so a fresh checkout no longer depends on a pre-existing editable install just to collect tests
    - LoRA imports are lazy, and LoRA-only tests skip cleanly when `peft` is unavailable
  - Publish branch:
    - `codex/harden-trainer-bench-flow`
- Still outstanding:
  - Add richer KL/reference-policy training support and checkpoint resume.
  - Decide whether `DAPO` / `GSPO` / `CISPO` should stay compact variants or be rewritten to match a specific canonical paper formulation.
  - Replace synthetic arithmetic as the primary headline benchmark with accepted math/code/alignment suites.

## External References Used For The Next Plan

- Hugging Face TRL quickstart documents `SFTTrainer`, `GRPOTrainer`, and `DPOTrainer` as supported post-training baselines.
- The `Qwen/Qwen3-8B` model card documents an 8.2B model with 32,768 native context and validated long-context extension.
- The `DeepSeek-R1-Distill-Qwen-32B` model card provides an official 32B reasoning target, recommends multi-run averaging, and shows local serving examples with `vLLM` and `SGLang` using tensor parallelism.
- Meta’s official Hugging Face org lists `Llama-3.3-70B-Instruct` and `Llama 4` family checkpoints as current reference models; `Llama-3.3-70B-Instruct` exposes a `128k` context window.
- vLLM’s stable docs describe data-parallel and tensor-parallel serving layouts, including combined DP and TP deployments.
- NVIDIA describes NeMo RL as a scalable post-training library for multimodal models and multi-GPU / multi-node RL workloads.
