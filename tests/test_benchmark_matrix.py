from pathlib import Path

from policy_optimization.scripts.benchmark_matrix import _parse_log_file


def test_parse_log_file_marks_nonzero_returncode_bad(tmp_path: Path) -> None:
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        "\n".join(
            [
                '{"event":"eval","stage":"before_training","eval_reward_mean":0.1,"eval_success_rate":0.1}',
                '{"event":"train_step","step":0,"reward_mean":0.2,"success_rate":0.2,"loss":1.0,"grad_norm":1.0}',
                '{"event":"eval","stage":"after_training","eval_reward_mean":0.3,"eval_success_rate":0.3}',
            ]
        )
        + "\n"
    )
    parsed = _parse_log_file(log_path, expected_steps=1, returncode=1)
    assert parsed["status"] == "bad"


def test_parse_log_file_requires_fixed_eval_events(tmp_path: Path) -> None:
    log_path = tmp_path / "bench.log"
    log_path.write_text('{"event":"train_step","step":0,"reward_mean":0.2,"success_rate":0.2,"loss":1.0,"grad_norm":1.0}\n')
    parsed = _parse_log_file(log_path, expected_steps=1, returncode=0)
    assert parsed["status"] == "bad"
