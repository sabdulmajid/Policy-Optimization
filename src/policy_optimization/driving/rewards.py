from __future__ import annotations

RISK_WEIGHTS = {
    "pedestrian": 3.0,
    "child": 3.0,
    "cyclist": 3.0,
    "traffic light": 2.5,
    "traffic sign": 2.0,
    "stop sign": 2.0,
    "priority sign": 1.75,
    "zebra": 2.5,
    "crosswalk": 2.5,
    "car": 1.0,
    "bus": 1.25,
    "truck": 1.25,
}

QUESTION_RISK_KEYWORDS = {
    "right of way": 0.75,
    "proceed": 0.5,
    "overtake": 1.0,
    "safety distance": 1.0,
    "red light": 1.0,
    "stop": 0.75,
}


def risk_score_from_entities(question: str, entity_names: list[str]) -> float:
    score = 1.0
    lowered = question.lower()
    for entity in entity_names:
        entity_lower = entity.lower()
        score += RISK_WEIGHTS.get(entity_lower, 0.5)
    for phrase, weight in QUESTION_RISK_KEYWORDS.items():
        if phrase in lowered:
            score += weight
    return score
