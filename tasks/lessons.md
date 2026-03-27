# Lessons

## 2026-03-27

- Before running any experiment or benchmark, perform an explicit GPU preflight and record the result in logs or output artifacts.
- When a benchmark claims model improvement, use a fixed evaluation set; never compare training-step rewards across changing prompts and present that as progress.
- For online RL objectives that depend on old-vs-current policy ratios, always recompute current logprobs from model weights during the optimization step instead of reusing cached rollout logprobs.
