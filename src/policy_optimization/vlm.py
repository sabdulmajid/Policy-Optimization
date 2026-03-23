from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, PreTrainedModel, ProcessorMixin

from policy_optimization.ops import gather_logprobs
from policy_optimization.precision import as_float32


@dataclass(slots=True)
class CandidateScore:
    candidate: str
    sequence_logprob: torch.Tensor


def _torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return getattr(torch, dtype_name)


def load_vlm(
    model_id: str,
    *,
    device: torch.device | str,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> tuple[PreTrainedModel, ProcessorMixin]:
    device = torch.device(device)
    torch_dtype = _torch_dtype(dtype, device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    model.to(device)
    model.config.use_cache = False
    return model, processor


def build_vlm_prompt(question: str, options: dict[str, str]) -> str:
    options_text = "\n".join(f"{letter}. {text}" for letter, text in options.items())
    return (
        "You are an autonomous-driving policy model. Use the image and answer the driving question.\n"
        "Return exactly one option in the form `LETTER. option text`.\n"
        f"Question: {question}\n"
        f"Options:\n{options_text}\n"
        "Answer:"
    )


def _chat_prompt(processor: ProcessorMixin, image: Image.Image, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def score_vlm_candidates(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    image: Image.Image,
    prompt: str,
    candidates: list[str],
) -> list[CandidateScore]:
    prompt_text = _chat_prompt(processor, image, prompt)
    prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    prompt_inputs = {key: value.to(model.device) for key, value in prompt_inputs.items()}
    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    scores: list[CandidateScore] = []

    for candidate in candidates:
        completion = candidate if candidate.startswith((" ", "\n")) else f" {candidate}"
        full_inputs = processor(text=[prompt_text + completion], images=[image], return_tensors="pt")
        full_inputs = {key: value.to(model.device) for key, value in full_inputs.items()}
        outputs = model(**full_inputs, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        target_ids = full_inputs["input_ids"][:, 1:]
        token_logprobs = gather_logprobs(logits, target_ids)
        answer_length = int(full_inputs["input_ids"].shape[1] - prompt_length)
        if answer_length <= 0:
            raise ValueError("Candidate produced zero answer tokens; cannot score response.")
        answer_mask = torch.zeros_like(token_logprobs, dtype=torch.bool)
        answer_mask[:, -answer_length:] = True
        sequence_logprob = (as_float32(token_logprobs) * answer_mask.float()).sum(dim=-1).squeeze(0)
        scores.append(CandidateScore(candidate=candidate, sequence_logprob=sequence_logprob))
    return scores
