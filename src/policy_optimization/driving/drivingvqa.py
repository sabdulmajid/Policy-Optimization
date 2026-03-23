from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import zipfile

from huggingface_hub import hf_hub_download

from policy_optimization.driving.rewards import risk_score_from_entities

DRIVINGVQA_REPO_ID = "EPFL-DrivingVQA/DrivingVQA"


@dataclass(slots=True)
class DrivingVQAQuestion:
    scene_id: str
    question_id: str
    image_path: Path
    question: str
    options: dict[str, str]
    correct_letter: str
    explanation: str
    entity_boxes: list[list[float]]
    entity_names: list[str]
    risk_score: float
    exam_type: str


def _download_drivingvqa_file(filename: str) -> Path:
    return Path(hf_hub_download(repo_id=DRIVINGVQA_REPO_ID, repo_type="dataset", filename=filename))


def ensure_drivingvqa_images_extracted() -> Path:
    zip_path = _download_drivingvqa_file("images.zip")
    extract_root = zip_path.parent
    images_dir = extract_root / "images"
    if images_dir.exists():
        return extract_root
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_root)
    return extract_root


def _option_groups(possible_answers: dict[str, str], question_count: int) -> list[list[str]]:
    letters = sorted(possible_answers)
    if question_count <= 1:
        return [letters]
    if len(letters) % question_count != 0:
        raise ValueError("Cannot evenly assign answer options across grouped questions.")
    group_size = len(letters) // question_count
    return [letters[index * group_size : (index + 1) * group_size] for index in range(question_count)]


def flatten_drivingvqa_record(scene_id: str, record: dict[str, object], extract_root: Path) -> list[DrivingVQAQuestion]:
    questions = list(record["questions"])
    possible_answers = dict(record["possible_answers"])
    true_answers = list(record["true_answers"])
    grouped_letters = _option_groups(possible_answers, len(questions))
    image_path = extract_root / str(record["img_filename"])
    relevant_entities = list(record.get("relevant_entities", []))
    entity_names = [next(iter(entity.keys())) for entity in relevant_entities]
    entity_boxes = [list(next(iter(entity.values()))) for entity in relevant_entities]
    output: list[DrivingVQAQuestion] = []
    for index, question in enumerate(questions):
        option_letters = grouped_letters[index]
        options = {letter: str(possible_answers[letter]) for letter in option_letters}
        correct_letter = str(true_answers[index])
        output.append(
            DrivingVQAQuestion(
                scene_id=scene_id,
                question_id=f"{scene_id}:{index}",
                image_path=image_path,
                question=str(question),
                options=options,
                correct_letter=correct_letter,
                explanation=str(record.get("explanation", "")),
                entity_boxes=entity_boxes,
                entity_names=entity_names,
                risk_score=risk_score_from_entities(str(question), entity_names),
                exam_type=str(record.get("exam_type", "unknown")),
            )
        )
    return output


def load_drivingvqa_questions(split: str = "train", limit: int | None = None) -> list[DrivingVQAQuestion]:
    json_path = _download_drivingvqa_file(f"{split}.json")
    extract_root = ensure_drivingvqa_images_extracted()
    with open(json_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    questions: list[DrivingVQAQuestion] = []
    for scene_id, record in records.items():
        questions.extend(flatten_drivingvqa_record(scene_id, record, extract_root))
        if limit is not None and len(questions) >= limit:
            return questions[:limit]
    return questions
