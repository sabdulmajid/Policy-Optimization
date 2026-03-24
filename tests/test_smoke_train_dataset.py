from policy_optimization.scripts.smoke_train import extract_last_number, normalize_number_text, parse_gsm8k_final_answer


def test_normalize_number_text_handles_commas_and_trailing_zeros() -> None:
    assert normalize_number_text("+1,234.500") == "1234.5"
    assert normalize_number_text("42.000") == "42"


def test_parse_gsm8k_final_answer_prefers_marker() -> None:
    answer = "First compute 10 + 5 = 15. #### 15"
    assert parse_gsm8k_final_answer(answer) == "15"


def test_extract_last_number_from_completion() -> None:
    completion = "Reasoning... Final answer: 2,048"
    assert extract_last_number(completion) == "2048"
