from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from statistics import mean, pstdev

from policy_optimization.gpu import inspect_gpu_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-seed benchmark matrix for policy objectives.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default="/pub7/hf-cache")
    parser.add_argument("--trainable-scope", choices=["full", "lm_head", "lora"], default="lm_head")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dataset", choices=["synthetic", "gsm8k"], default="synthetic")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--objectives", nargs="+", default=["rloo", "dapo", "gspo", "cispo", "maxrl"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 24, 25])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--updates-per-rollout", type=int, default=4)
    parser.add_argument("--minibatch-groups", type=int, default=0)
    parser.add_argument("--prompts-per-step", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--eval-prompts", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=424242)
    parser.add_argument("--eval-max-new-tokens", type=int, default=0)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--reference-model-id", default="")
    parser.add_argument("--reference-device", default="")
    parser.add_argument("--output-prefix", default="qwen_0.5b_benchmark_long_horizon_2026-03-23")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _safe_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(mean(values)), float(pstdev(values) if len(values) > 1 else 0.0)


def _dedupe_latest_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    latest: dict[tuple[str, int], dict[str, object]] = {}
    passthrough: list[dict[str, object]] = []
    for row in rows:
        objective = row.get("objective")
        seed = row.get("seed")
        if isinstance(objective, str) and isinstance(seed, int):
            latest[(objective, seed)] = row
        else:
            passthrough.append(row)
    return list(latest.values()) + passthrough


def _valid_metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    valid: list[dict[str, object]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        first_step = row.get("first_step")
        last_step = row.get("last_step")
        baseline_eval = row.get("baseline_eval")
        final_eval = row.get("final_eval")
        if not isinstance(first_step, dict) or not isinstance(last_step, dict):
            continue
        if not isinstance(baseline_eval, dict) or not isinstance(final_eval, dict):
            continue
        valid.append(row)
    return valid


def _parse_log_file(log_path: Path, expected_steps: int, returncode: int) -> dict[str, object]:
    text = log_path.read_text(errors="replace")
    rows: list[dict[str, object]] = []
    parse_errors = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            parse_errors += 1
    train_steps = [row for row in rows if row.get("event") == "train_step"]
    skipped_steps = [row for row in rows if row.get("event") == "skip_step"]
    eval_events = [row for row in rows if row.get("event") == "eval"]
    baseline_eval = next((row for row in eval_events if row.get("stage") == "before_training"), None)
    final_eval = next((row for row in reversed(eval_events) if row.get("stage") == "after_training"), None)
    has_traceback = "Traceback (most recent call last)" in text
    completed_step_events = len(train_steps) + len(skipped_steps)
    has_required_eval = isinstance(baseline_eval, dict) and isinstance(final_eval, dict)
    status = "ok" if returncode == 0 and completed_step_events == expected_steps and not has_traceback and has_required_eval else "bad"
    return {
        "parse_errors": parse_errors,
        "has_traceback": has_traceback,
        "step_count": len(train_steps),
        "skip_step_count": len(skipped_steps),
        "completed_step_events": completed_step_events,
        "first_step": train_steps[0] if train_steps else None,
        "last_step": train_steps[-1] if train_steps else None,
        "baseline_eval": baseline_eval,
        "final_eval": final_eval,
        "status": status,
    }


def _render_markdown(
    *,
    model_id: str,
    device: str,
    trainable_scope: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    dataset: str,
    dataset_split: str,
    objectives: list[str],
    seeds: list[int],
    steps: int,
    updates_per_rollout: int,
    minibatch_groups: int,
    prompts_per_step: int,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    eval_prompts: int,
    eval_seed: int,
    eval_max_new_tokens: int,
    eval_temperature: float,
    eval_top_p: float,
    kl_beta: float,
    reference_model_id: str,
    reference_device: str,
    rows: list[dict[str, object]],
    raw_jsonl_path: Path,
) -> str:
    deduped_rows = _dedupe_latest_rows(rows)
    by_objective = {obj: [row for row in deduped_rows if row.get("objective") == obj] for obj in objectives}

    lines: list[str] = []
    lines.append("# Long-Horizon Benchmark Impact Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Model: `{model_id}`")
    lines.append(f"- Device: `{device}`")
    lines.append(f"- Trainable scope: `{trainable_scope}`")
    if trainable_scope == "lora":
        lines.append(f"- LoRA config: `rank={lora_rank}`, `alpha={lora_alpha}`, `dropout={lora_dropout}`")
    lines.append(f"- Dataset: `{dataset}`")
    lines.append(f"- Dataset split: `{dataset_split}`")
    lines.append(f"- Objectives: {', '.join(f'`{obj}`' for obj in objectives)}")
    lines.append(f"- Seeds: {', '.join(f'`{seed}`' for seed in seeds)}")
    lines.append(
        f"- Per-run config: `steps={steps}`, `updates-per-rollout={updates_per_rollout}`, `minibatch-groups={minibatch_groups}`, `prompts-per-step={prompts_per_step}`, `group-size={group_size}`, `max-new-tokens={max_new_tokens}`, `temperature={temperature}`"
    )
    lines.append(
        f"- Fixed eval config: `eval-prompts={eval_prompts}`, `eval-seed={eval_seed}`, `eval-max-new-tokens={eval_max_new_tokens}`, `eval-temperature={eval_temperature}`, `eval-top-p={eval_top_p}`"
    )
    lines.append(
        f"- KL config: `kl-beta={kl_beta}`, `reference-model-id={reference_model_id or 'none'}`, `reference-device={reference_device or 'n/a'}`"
    )
    lines.append(f"- Raw parsed artifact: `{raw_jsonl_path.as_posix()}`")
    lines.append("")

    lines.append("## Aggregate Results On Fixed Eval Set (Mean ± Std across seeds)")
    lines.append("")
    lines.append("| Objective | Valid runs | Eval reward before | Eval reward after | Eval reward Δ | Eval success before | Eval success after | Eval success Δ | GradNorm step0 | GradNorm stepN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for objective in objectives:
        objective_rows = by_objective[objective]
        metric_rows = _valid_metric_rows(objective_rows)

        first_rewards = [float(row["baseline_eval"]["eval_reward_mean"]) for row in metric_rows]
        last_rewards = [float(row["final_eval"]["eval_reward_mean"]) for row in metric_rows]
        reward_deltas = [float(row["final_eval"]["eval_reward_mean"]) - float(row["baseline_eval"]["eval_reward_mean"]) for row in metric_rows]

        first_success = [float(row["baseline_eval"]["eval_success_rate"]) for row in metric_rows]
        last_success = [float(row["final_eval"]["eval_success_rate"]) for row in metric_rows]
        success_deltas = [float(row["final_eval"]["eval_success_rate"]) - float(row["baseline_eval"]["eval_success_rate"]) for row in metric_rows]

        first_grad = [float(row["first_step"].get("grad_norm", float("nan"))) for row in metric_rows]
        last_grad = [float(row["last_step"].get("grad_norm", float("nan"))) for row in metric_rows]

        fr_m, fr_s = _safe_stats(first_rewards)
        lr_m, lr_s = _safe_stats(last_rewards)
        rd_m, rd_s = _safe_stats(reward_deltas)
        fs_m, fs_s = _safe_stats(first_success)
        ls_m, ls_s = _safe_stats(last_success)
        sd_m, sd_s = _safe_stats(success_deltas)
        fg_m, fg_s = _safe_stats(first_grad)
        lg_m, lg_s = _safe_stats(last_grad)
        ok_runs = len(metric_rows)

        lines.append(
            f"| `{objective}` | {ok_runs}/{len(seeds)} | "
            f"{fr_m:.4f} ± {fr_s:.4f} | "
            f"{lr_m:.4f} ± {lr_s:.4f} | "
            f"{rd_m:+.4f} ± {rd_s:.4f} | "
            f"{fs_m:.4f} ± {fs_s:.4f} | "
            f"{ls_m:.4f} ± {ls_s:.4f} | "
            f"{sd_m:+.4f} ± {sd_s:.4f} | "
            f"{fg_m:.3f} ± {fg_s:.3f} | "
            f"{lg_m:.3f} ± {lg_s:.3f} |"
        )

    lines.append("")
    lines.append("## Data Quality Checks")
    lines.append("")
    bad_rows = [row for row in deduped_rows if row.get("status") != "ok"]
    lines.append(f"- Parsed runs (deduped latest per objective/seed): `{len(deduped_rows)}`")
    lines.append(f"- Invalid runs: `{len(bad_rows)}`")
    lines.append(f"- Any traceback in logs: `{any(bool(row.get('has_traceback')) for row in deduped_rows)}`")
    lines.append(f"- Any JSON parse errors: `{sum(int(row.get('parse_errors', 0)) for row in deduped_rows)}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- These summaries use a fixed held-out eval prompt set, not changing training prompts, so before/after deltas are comparable.")
    lines.append("- This is still a compact-node benchmark, not a final SOTA claim; use longer runs, larger models, and accepted eval suites for that bar.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    eval_max_new_tokens = args.max_new_tokens if args.eval_max_new_tokens <= 0 else args.eval_max_new_tokens
    print(json.dumps({"event": "gpu_preflight", **inspect_gpu_environment(args.device)}), flush=True)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl_path = reports_dir / f"{args.output_prefix}.jsonl"
    summary_json_path = reports_dir / f"{args.output_prefix}.summary.json"
    markdown_path = reports_dir / f"{args.output_prefix}.md"
    errors_path = reports_dir / f"{args.output_prefix}.errors.jsonl"

    rows: list[dict[str, object]] = []
    objectives = [objective.lower() for objective in args.objectives]
    completed: dict[tuple[str, int], dict[str, object]] = {}

    if args.resume and raw_jsonl_path.exists():
        for raw_line in raw_jsonl_path.read_text(errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            objective = row.get("objective")
            seed = row.get("seed")
            status = row.get("status")
            if isinstance(objective, str) and isinstance(seed, int) and status == "ok":
                completed[(objective, seed)] = row
                rows.append(row)

    raw_mode = "a" if args.resume and raw_jsonl_path.exists() else "w"
    err_mode = "a" if args.resume and errors_path.exists() else "w"

    with raw_jsonl_path.open(raw_mode) as out_file, errors_path.open(err_mode) as err_file:
        for seed in args.seeds:
            for objective in objectives:
                if (objective, seed) in completed:
                    print(f"skip objective={objective} seed={seed} reason=resume-already-ok", flush=True)
                    continue
                log_path = reports_dir / f"{objective}_{args.output_prefix}_seed{seed}.log"
                command = [
                    "./.venv/bin/po-smoke",
                    "--model-id",
                    args.model_id,
                    "--objective",
                    objective,
                    "--device",
                    args.device,
                    "--cache-dir",
                    args.cache_dir,
                    "--trainable-scope",
                    args.trainable_scope,
                    "--lora-rank",
                    str(args.lora_rank),
                    "--lora-alpha",
                    str(args.lora_alpha),
                    "--lora-dropout",
                    str(args.lora_dropout),
                    "--dataset",
                    args.dataset,
                    "--dataset-split",
                    args.dataset_split,
                    "--steps",
                    str(args.steps),
                    "--updates-per-rollout",
                    str(args.updates_per_rollout),
                    "--minibatch-groups",
                    str(args.minibatch_groups),
                    "--prompts-per-step",
                    str(args.prompts_per_step),
                    "--group-size",
                    str(args.group_size),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--temperature",
                    str(args.temperature),
                    "--top-p",
                    str(args.top_p),
                    "--eval-prompts",
                    str(args.eval_prompts),
                    "--eval-seed",
                    str(args.eval_seed),
                    "--eval-max-new-tokens",
                    str(eval_max_new_tokens),
                    "--eval-temperature",
                    str(args.eval_temperature),
                    "--eval-top-p",
                    str(args.eval_top_p),
                    "--kl-beta",
                    str(args.kl_beta),
                    "--seed",
                    str(seed),
                ]
                if args.reference_model_id:
                    command.extend(["--reference-model-id", args.reference_model_id])
                if args.reference_device:
                    command.extend(["--reference-device", args.reference_device])

                env = os.environ.copy()
                env["HF_HOME"] = args.cache_dir
                env["HUGGINGFACE_HUB_CACHE"] = args.cache_dir
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

                start_time = time.time()
                with log_path.open("w") as log_file:
                    proc = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, env=env, text=True)
                elapsed_sec = int(time.time() - start_time)

                parsed = _parse_log_file(log_path, expected_steps=args.steps, returncode=proc.returncode)
                row = {
                    "objective": objective,
                    "seed": seed,
                    "returncode": proc.returncode,
                    "elapsed_sec": elapsed_sec,
                    "log_file": log_path.as_posix(),
                    **parsed,
                }
                rows.append(row)
                out_file.write(json.dumps(row) + "\n")
                out_file.flush()
                print(
                    f"recorded objective={objective} seed={seed} status={row['status']} rc={proc.returncode} steps={row['step_count']} elapsed={elapsed_sec}s",
                    flush=True,
                )

                if proc.returncode != 0 or row["status"] != "ok":
                    err_file.write(json.dumps(row) + "\n")
                    err_file.flush()

    deduped_rows = _dedupe_latest_rows(rows)
    by_objective = {objective: [row for row in deduped_rows if row.get("objective") == objective] for objective in objectives}
    summary: list[dict[str, object]] = []
    for objective in objectives:
        objective_rows = by_objective[objective]
        metric_rows = _valid_metric_rows(objective_rows)
        reward_delta = [float(row["final_eval"]["eval_reward_mean"]) - float(row["baseline_eval"]["eval_reward_mean"]) for row in metric_rows]
        success_delta = [float(row["final_eval"]["eval_success_rate"]) - float(row["baseline_eval"]["eval_success_rate"]) for row in metric_rows]
        rd_m, rd_s = _safe_stats(reward_delta)
        sd_m, sd_s = _safe_stats(success_delta)
        summary.append(
            {
                "objective": objective,
                "runs": len(objective_rows),
                "ok_runs": len(metric_rows),
                "reward_delta_mean": rd_m,
                "reward_delta_std": rd_s,
                "success_delta_mean": sd_m,
                "success_delta_std": sd_s,
            }
        )

    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n")
    markdown_path.write_text(
        _render_markdown(
            model_id=args.model_id,
            device=args.device,
            trainable_scope=args.trainable_scope,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            dataset=args.dataset,
            dataset_split=args.dataset_split,
            objectives=objectives,
            seeds=list(args.seeds),
            steps=args.steps,
            updates_per_rollout=args.updates_per_rollout,
            minibatch_groups=args.minibatch_groups,
            prompts_per_step=args.prompts_per_step,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            eval_prompts=args.eval_prompts,
            eval_seed=args.eval_seed,
            eval_max_new_tokens=eval_max_new_tokens,
            eval_temperature=args.eval_temperature,
            eval_top_p=args.eval_top_p,
            kl_beta=args.kl_beta,
            reference_model_id=args.reference_model_id,
            reference_device=args.reference_device,
            rows=rows,
            raw_jsonl_path=raw_jsonl_path,
        )
        + "\n"
    )

    invalid_runs = [row for row in deduped_rows if row.get("status") != "ok"]
    print(f"wrote {raw_jsonl_path.as_posix()}")
    print(f"wrote {summary_json_path.as_posix()}")
    print(f"wrote {markdown_path.as_posix()}")
    print(f"parsed_runs={len(deduped_rows)} invalid_runs={len(invalid_runs)}")


if __name__ == "__main__":
    main()
