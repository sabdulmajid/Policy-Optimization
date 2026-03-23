# Task Plan

- [x] Inspect repository state and local environment.
- [x] Scaffold a clean Python package with packaging metadata and task tracking.
- [x] Implement precision-safe tensor utilities and rollout data structures.
- [x] Implement modular RLOO, DAPO, GSPO, CISPO, and MaxRL objectives.
- [x] Add Hugging Face helpers plus a real-model smoke-training CLI.
- [x] Add unit tests for losses, filtering, and numerical utilities.
- [x] Run local verification on CPU and GPU.
- [x] Commit the repo in clean, professional chunks.
- [ ] Push to GitHub.
- [x] Launch subagents for follow-on experiment runs.

# Review

- `pytest` passed locally: `13 passed`.
- Real GPU smoke succeeded on `Qwen/Qwen2.5-0.5B` with all five objectives.
- Larger-checkpoint GPU smoke succeeded on `Qwen/Qwen2.5-7B` with `rloo`.
- Moonlight remote-code preflight succeeded for `moonshotai/Moonlight-16B-A3B`.
- Hugging Face cache must be redirected to `/pub7` on this machine because `/pub3` is full.
- `git push` is currently blocked by missing GitHub credentials on this machine.
