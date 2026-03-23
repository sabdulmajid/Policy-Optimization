# Task Plan

- [x] Inspect repository state and local environment.
- [x] Scaffold a clean Python package with packaging metadata and task tracking.
- [x] Implement precision-safe tensor utilities and rollout data structures.
- [x] Implement modular RLOO, DAPO, GSPO, CISPO, and MaxRL objectives.
- [x] Add Hugging Face helpers plus a real-model smoke-training CLI.
- [x] Add unit tests for losses, filtering, and numerical utilities.
- [x] Run local verification on CPU and GPU.
- [ ] Commit the repo in clean, professional chunks.
- [ ] Push to GitHub.
- [ ] Launch subagents for follow-on experiment runs after the main implementation is pushed.

# Review

- `pytest` passed locally: `13 passed`.
- Real GPU smoke succeeded on `Qwen/Qwen2.5-0.5B` with all five objectives.
- Hugging Face cache must be redirected to `/pub7` on this machine because `/pub3` is full.
- A larger `Qwen/Qwen2.5-7B` smoke run is in progress.
