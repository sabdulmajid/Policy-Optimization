from __future__ import annotations

import argparse
import json

from PIL import Image
import torch

from policy_optimization.driving.drivingvqa import load_drivingvqa_questions
from policy_optimization.driving.image_ops import mask_entities
from policy_optimization.losses import compute_objective
from policy_optimization.types import PreferenceBatch
from policy_optimization.vlm import build_vlm_prompt, load_vlm, score_vlm_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VLM on DrivingVQA and compute policy-objective diagnostics.")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--objective", choices=["dpo", "mdpo", "dgpo"], default="dgpo")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--report-path", default="")
    return parser.parse_args()


def evaluate_examples(model: torch.nn.Module, processor: object, limit: int, split: str) -> tuple[PreferenceBatch, dict[str, float], list[dict[str, object]]]:
    questions = load_drivingvqa_questions(split=split, limit=limit)
    chosen_logprobs = []
    rejected_logprobs = []
    context_chosen_logprobs = []
    context_rejected_logprobs = []
    rewards = []
    group_ids = []
    rows: list[dict[str, object]] = []

    for question in questions:
        image = Image.open(question.image_path).convert("RGB")
        masked_image = mask_entities(image, question.entity_boxes)
        prompt = build_vlm_prompt(question.question, question.options)
        candidate_texts = [f"{letter}. {text}" for letter, text in question.options.items()]
        real_scores = score_vlm_candidates(model, processor, image, prompt, candidate_texts)
        masked_scores = score_vlm_candidates(model, processor, masked_image, prompt, candidate_texts)
        real_tensor = torch.stack([score.sequence_logprob for score in real_scores])
        masked_tensor = torch.stack([score.sequence_logprob for score in masked_scores])

        option_letters = list(question.options)
        correct_index = option_letters.index(question.correct_letter)
        incorrect_indices = [index for index in range(len(option_letters)) if index != correct_index]
        rejected_index = incorrect_indices[int(real_tensor[incorrect_indices].argmax().item())]
        predicted_index = int(real_tensor.argmax().item())
        masked_predicted_index = int(masked_tensor.argmax().item())

        chosen_logprobs.append(real_tensor[correct_index])
        rejected_logprobs.append(real_tensor[rejected_index])
        context_chosen_logprobs.append(masked_tensor[correct_index])
        context_rejected_logprobs.append(masked_tensor[rejected_index])
        rewards.append(question.risk_score)
        group_ids.append(int(question.scene_id))

        rows.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "correct_option": candidate_texts[correct_index],
                "predicted_option": candidate_texts[predicted_index],
                "masked_predicted_option": candidate_texts[masked_predicted_index],
                "risk_score": question.risk_score,
                "grounding_gap": float(((real_tensor[correct_index] - real_tensor[rejected_index]) - (masked_tensor[correct_index] - masked_tensor[rejected_index])).item()),
                "is_correct": float(predicted_index == correct_index),
                "is_masked_correct": float(masked_predicted_index == correct_index),
            }
        )

    chosen_tensor = torch.stack(chosen_logprobs)
    rejected_tensor = torch.stack(rejected_logprobs)
    context_chosen_tensor = torch.stack(context_chosen_logprobs)
    context_rejected_tensor = torch.stack(context_rejected_logprobs)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=chosen_tensor.device)
    group_tensor = torch.tensor(group_ids, dtype=torch.long, device=chosen_tensor.device)
    batch = PreferenceBatch(
        chosen_logprobs=chosen_tensor,
        rejected_logprobs=rejected_tensor,
        ref_chosen_logprobs=chosen_tensor.detach(),
        ref_rejected_logprobs=rejected_tensor.detach(),
        context_chosen_logprobs=context_chosen_tensor,
        context_rejected_logprobs=context_rejected_tensor,
        rewards=reward_tensor,
        group_ids=group_tensor,
    )
    metrics = {
        "accuracy": sum(row["is_correct"] for row in rows) / len(rows),
        "masked_accuracy": sum(row["is_masked_correct"] for row in rows) / len(rows),
        "mean_grounding_gap": sum(row["grounding_gap"] for row in rows) / len(rows),
        "mean_risk_score": sum(row["risk_score"] for row in rows) / len(rows),
    }
    return batch, metrics, rows


def main() -> None:
    args = parse_args()
    model, processor = load_vlm(
        args.model_id,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    with torch.inference_mode():
        batch, eval_metrics, rows = evaluate_examples(model, processor, limit=args.limit, split=args.split)
        objective_output = compute_objective(args.objective, batch)

    payload = {
        "model_id": args.model_id,
        "split": args.split,
        "limit": args.limit,
        "objective": args.objective,
        **eval_metrics,
        **objective_output.metrics,
        "loss": float(objective_output.loss.item()),
    }
    print(json.dumps({"event": "drivingvqa_eval", **payload}))
    if args.report_path:
        with open(args.report_path, "w", encoding="utf-8") as handle:
            json.dump({"summary": payload, "rows": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
