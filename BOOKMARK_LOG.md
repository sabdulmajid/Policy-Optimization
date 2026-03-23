# Bookmark Log

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
