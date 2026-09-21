"""Statistical and output checks for the 30-seed paired evaluation."""

import csv
import math
from pathlib import Path

import pytest

from evaluate_paired import (
    EPISODE_FIELDS,
    PAIRED_FIELDS,
    SEEDS,
    SUMMARY_FIELDS,
    T_95_DF_29,
    EpisodeResult,
    evaluate,
    paired_statistics,
)


def result(controller: str, seed: int, *, wait: float, generated: int = 100) -> EpisodeResult:
    return EpisodeResult(seed, controller, 80, wait, 3.0, generated, 20, 8, 10.0)


def test_paired_statistics_use_matched_seeds_and_student_t_interval() -> None:
    results = []
    for index, seed in enumerate(SEEDS):
        results.extend((result("fixed_time", seed, wait=10.0), result("dqn", seed, wait=10.0 + index)))
    rows = paired_statistics(list(reversed(results)))
    waiting = next(row for row in rows if row["metric"] == "mean_waiting_time")
    half_width = T_95_DF_29 * math.sqrt(77.5) / math.sqrt(30)
    assert waiting["mean_difference"] == 14.5
    assert waiting["std_difference"] == pytest.approx(math.sqrt(77.5))
    assert waiting["ci_95_lower"] == pytest.approx(14.5 - half_width)
    assert waiting["ci_95_upper"] == pytest.approx(14.5 + half_width)
    assert waiting["relative_change_percentage"] == 145.0
    assert rows[2]["mean_difference"] == 0.0


def test_paired_statistics_reject_mismatched_seeds() -> None:
    results = [result("dqn", seed, wait=1) for seed in SEEDS]
    results += [result("fixed_time", seed, wait=1) for seed in SEEDS[:-1]]
    with pytest.raises(ValueError, match="matching seeds"):
        paired_statistics(results)


def test_evaluation_writes_all_outputs_and_checks_equal_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    agent = object()
    monkeypatch.setattr("evaluate_paired.load_agent", lambda path: agent)
    monkeypatch.setattr("evaluate_paired.set_deterministic_seed", lambda seed: None)
    calls = []

    def runner(controller: str, seed: int, passed_agent: object | None) -> EpisodeResult:
        calls.append((controller, seed, passed_agent))
        assert passed_agent is (agent if controller == "dqn" else None)
        return result(controller, seed, wait=2.0 if controller == "dqn" else 4.0)

    output_dir = tmp_path / "evaluation"
    evaluate(checkpoint=checkpoint, output_dir=output_dir, episode_runner=runner)
    assert len(calls) == 60
    with (output_dir / "episodes.csv").open(newline="") as source:
        reader = csv.DictReader(source)
        episodes = list(reader)
        assert reader.fieldnames == list(EPISODE_FIELDS)
    assert len(episodes) == 60
    assert {(row["controller"], int(row["seed"])) for row in episodes} == {
        (controller, seed) for controller in ("dqn", "fixed_time") for seed in SEEDS
    }
    with (output_dir / "summary.csv").open(newline="") as source:
        reader = csv.DictReader(source)
        summary = list(reader)
        assert reader.fieldnames == list(SUMMARY_FIELDS)
    assert len(summary) == 2
    assert all(row["number_of_seeds"] == "30" for row in summary)
    with (output_dir / "paired_differences.csv").open(newline="") as source:
        reader = csv.DictReader(source)
        differences = list(reader)
        assert reader.fieldnames == list(PAIRED_FIELDS)
    assert len(differences) == 3
    assert float(differences[0]["mean_difference"]) == -2.0
    assert float(differences[0]["relative_change_percentage"]) == -50.0


def test_evaluation_rejects_unequal_generated_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evaluate_paired.load_agent", lambda path: object())
    monkeypatch.setattr("evaluate_paired.set_deterministic_seed", lambda seed: None)

    def runner(controller: str, seed: int, agent: object | None) -> EpisodeResult:
        return result(controller, seed, wait=1.0, generated=100 if controller == "dqn" else 101)

    with pytest.raises(ValueError, match="demand differs"):
        evaluate(output_dir=tmp_path, episode_runner=runner)
    assert not (tmp_path / "summary.csv").exists()
