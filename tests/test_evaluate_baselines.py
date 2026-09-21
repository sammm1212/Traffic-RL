import csv
from pathlib import Path

import pytest

from evaluate_baselines import (
    BaselineEvaluationResult,
    CSV_FIELDS,
    evaluate_baselines,
    paired_differences,
    summarize_metrics,
)
from evaluate_dqn import EVAL_SEEDS, EvaluationResult


def baseline(controller: str, seed: int) -> BaselineEvaluationResult:
    return BaselineEvaluationResult(controller, seed, 2.0 + seed, 3.0, 4, -5.0)


def write_dqn_results(path: Path, seeds: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "seed",
                "total_reward",
                "mean_waiting_time",
                "mean_queue_length",
                "throughput",
            ),
        )
        writer.writeheader()
        for seed in seeds:
            writer.writerow(
                {
                    "seed": seed,
                    "total_reward": -1.0,
                    "mean_waiting_time": 5.0 + seed,
                    "mean_queue_length": 5.0,
                    "throughput": 10,
                }
            )


def test_summary_uses_sample_standard_deviation() -> None:
    summary = summarize_metrics(
        [
            BaselineEvaluationResult("random", 100, 2.0, 1.0, 8, -1.0),
            BaselineEvaluationResult("random", 101, 4.0, 3.0, 12, -2.0),
        ]
    )

    assert summary["mean_waiting_time"].mean == 3.0
    assert summary["mean_waiting_time"].standard_deviation == pytest.approx(2**0.5)
    assert summary["throughput"].mean == 10.0
    assert summary["throughput"].standard_deviation == pytest.approx(8**0.5)


def test_paired_differences_match_on_seed() -> None:
    dqn = [
        EvaluationResult(100, -1.0, 5.0, 7.0, 12),
        EvaluationResult(101, -2.0, 8.0, 9.0, 14),
    ]
    fixed = [
        BaselineEvaluationResult("fixed_time", 101, 5.0, 4.0, 10, -3.0),
        BaselineEvaluationResult("fixed_time", 100, 4.0, 5.0, 11, -4.0),
    ]

    differences = paired_differences(dqn, fixed)

    assert differences["mean_waiting_time"].mean == 2.0
    assert differences["mean_queue_length"].mean == 3.5
    assert differences["throughput"].mean == 2.5


def test_evaluation_writes_required_rows_and_reuses_exact_seed_set(
    tmp_path: Path,
) -> None:
    dqn_path = tmp_path / "dqn.csv"
    output_path = tmp_path / "baseline.csv"
    write_dqn_results(dqn_path, EVAL_SEEDS)
    calls: list[tuple[str, int]] = []

    def fixed_runner(seed: int) -> BaselineEvaluationResult:
        calls.append(("fixed_time", seed))
        return baseline("fixed_time", seed)

    def random_runner(seed: int) -> BaselineEvaluationResult:
        calls.append(("random", seed))
        return baseline("random", seed)

    results, _ = evaluate_baselines(
        output_path=output_path,
        dqn_path=dqn_path,
        episode_runners={"fixed_time": fixed_runner, "random": random_runner},
    )

    assert calls == [
        (controller, seed)
        for controller in ("fixed_time", "random")
        for seed in EVAL_SEEDS
    ]
    assert len(results) == 20
    with output_path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
    assert reader.fieldnames == list(CSV_FIELDS)
    assert {int(row["seed"]) for row in rows} == set(EVAL_SEEDS)
    assert {row["controller"] for row in rows} == {"fixed_time", "random"}


def test_evaluation_rejects_any_other_seed_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires seeds"):
        evaluate_baselines(
            seeds=[100],
            output_path=tmp_path / "baseline.csv",
            dqn_path=tmp_path / "unused.csv",
            episode_runners={},
        )
