from pathlib import Path

from PIL import Image

from policy_optimization.driving.drivingvqa import flatten_drivingvqa_record
from policy_optimization.driving.image_ops import mask_entities


def test_flatten_drivingvqa_record_splits_grouped_questions(tmp_path: Path) -> None:
    record = {
        "img_filename": "images/0002.jpg",
        "questions": ["Do I proceed?", "Do I have right of way?"],
        "possible_answers": {"A": "Yes", "B": "No", "C": "Yes", "D": "No"},
        "true_answers": ["B", "C"],
        "explanation": "Traffic light is red.",
        "exam_type": "car",
        "relevant_entities": [{"traffic light": [0.0, 0.0, 10.0, 10.0]}],
    }
    questions = flatten_drivingvqa_record("1", record, tmp_path)
    assert len(questions) == 2
    assert questions[0].options == {"A": "Yes", "B": "No"}
    assert questions[1].options == {"C": "Yes", "D": "No"}
    assert questions[0].correct_letter == "B"
    assert questions[1].correct_letter == "C"


def test_mask_entities_blacks_out_entity_regions() -> None:
    image = Image.new("RGB", (20, 20), color=(255, 255, 255))
    masked = mask_entities(image, [[2, 2, 5, 5]])
    assert masked.getpixel((3, 3)) == (0, 0, 0)
    assert masked.getpixel((15, 15)) == (255, 255, 255)
