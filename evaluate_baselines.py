"""Evaluate fixed-time and random controllers on the DQN evaluation seeds."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from tempfile import NamedTemporaryFile
from typing import Callable, Mapping, Sequence

from evaluate_dqn import (
    DECISION_INTERVAL,
    EPISODE_SECONDS,
    EVAL_SEEDS,
    EvaluationResult as DQNEvaluationResult,
)
from run_baseline_experiments import _run_with_fresh_sumo
from src.agents.random_agent import RandomAgent
from src.environment.traffic_env import TrafficEnvironment
from src.simulation.metrics import MetricsAccumulator, SimulationSummary
from src.simulation.traffic import TrafficSimulation


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = ROOT / "results" / "evaluation" / "baseline_eval.csv"
DEFAULT_DQN_PATH = ROOT / "results" / "evaluation" / "dqn_100_episode_eval.csv"
CONTROLLERS = ("fixed_time", "random")
METRICS = ("mean_waiting_time", "mean_queue_length", "throughput")
CSV_FIELDS = (
    "controller",
    "seed",
    "mean_waiting_time",
    "mean_queue_length",
    "throughput",
    "total_reward",
)


@dataclass(frozen=True)
class BaselineEvaluationResult:
    """Comparable metrics from one baseline evaluation episode."""

    controller: str
    seed: int
    mean_waiting_time: float
    mean_queue_length: float
    throughput: int
    total_reward: float

    @classmethod
    def from_summary(
        cls,
        *,
        controller: str,
        seed: int,
        total_reward: float,
        traffic: SimulationSummary,
    ) -> "BaselineEvaluationResult":
        """Use the exact shared summary metrics selected by DQN evaluation."""
        return cls(
            controller=controller,
            seed=seed,
            mean_waiting_time=traffic.mean_queueing_delay_per_generated_vehicle,
            mean_queue_length=traffic.mean_queue_length,
            throughput=traffic.vehicles_completed,
            total_reward=total_reward,
        )


@dataclass(frozen=True)
class MetricStatistics:
    """Mean and sample standard deviation for one metric."""

    mean: float
    standard_deviation: float


def _mean_and_sample_sd(values: Sequence[float]) -> MetricStatistics:
    """Summarize values with the sample standard deviation used by DQN eval."""
    if not values:
        raise ValueError("at least one value is required")
    return MetricStatistics(
        mean=fmean(values),
        standard_deviation=stdev(values) if len(values) > 1 else 0.0,
    )


def summarize_metrics(
    results: Sequence[BaselineEvaluationResult | DQNEvaluationResult],
) -> dict[str, MetricStatistics]:
    """Compute comparable aggregate statistics for the three traffic metrics."""
    if not results:
        raise ValueError("at least one result is required")
    return {
        metric: _mean_and_sample_sd(
            [float(getattr(result, metric)) for result in results]
        )
        for metric in METRICS
    }


def _fixed_time_episode(
    seed: int, simulation: TrafficSimulation
) -> BaselineEvaluationResult:
    """Observe SUMO's unchanged fixed-time program for one evaluation episode."""
    environment = TrafficEnvironment(
        simulation=simulation,
        decision_interval=DECISION_INTERVAL,
        max_simulation_time=EPISODE_SECONDS,
    )
    environment.reset(seed=seed)
    accumulator = MetricsAccumulator()
    total_reward = 0.0

    for _ in range(EPISODE_SECONDS // DECISION_INTERVAL):
        for _ in range(DECISION_INTERVAL):
            accumulator.add(simulation.step())
        # Sampling reward after each decision interval matches environment.step().
        total_reward += environment._get_reward()

    summary = accumulator.summary()
    _validate_episode_time("fixed-time", summary)
    return BaselineEvaluationResult.from_summary(
        controller="fixed_time",
        seed=seed,
        total_reward=total_reward,
        traffic=summary,
    )


def run_fixed_time_episode(seed: int) -> BaselineEvaluationResult:
    """Run fixed-time control with the same SUMO command as DQN evaluation."""
    return _run_with_fresh_sumo(
        seed,
        lambda simulation: _fixed_time_episode(seed, simulation),
    )


def _random_episode(
    seed: int, simulation: TrafficSimulation
) -> BaselineEvaluationResult:
    """Run the existing uniformly random controller through the DQN environment."""
    environment = TrafficEnvironment(
        simulation=simulation,
        decision_interval=DECISION_INTERVAL,
        max_simulation_time=EPISODE_SECONDS,
    )
    agent = RandomAgent()
    environment.reset(seed=seed)
    environment.action_space.seed(seed)
    total_reward = 0.0

    while True:
        action = agent.select_action(environment)
        _, reward, terminated, truncated, _ = environment.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    summary = environment.episode_summary()
    _validate_episode_time("random", summary)
    return BaselineEvaluationResult.from_summary(
        controller="random",
        seed=seed,
        total_reward=total_reward,
        traffic=summary,
    )


def run_random_episode(seed: int) -> BaselineEvaluationResult:
    """Run random control with identical transitions and timing to DQN."""
    return _run_with_fresh_sumo(
        seed,
        lambda simulation: _random_episode(seed, simulation),
    )


def _validate_episode_time(controller: str, summary: SimulationSummary) -> None:
    if summary.simulation_time != EPISODE_SECONDS:
        raise RuntimeError(
            f"{controller} episode ended at {summary.simulation_time}, "
            f"expected {EPISODE_SECONDS}"
        )


def write_results(path: Path, results: Sequence[BaselineEvaluationResult]) -> None:
    """Atomically save completed baseline controller/seed results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_baseline_results(path: Path) -> dict[tuple[str, int], BaselineEvaluationResult]:
    """Load resumable baseline results, rejecting incompatible output schemas."""
    if not path.exists():
        return {}
    results: dict[tuple[str, int], BaselineEvaluationResult] = {}
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError(f"unexpected columns in {path}")
        for row in reader:
            result = BaselineEvaluationResult(
                controller=row["controller"],
                seed=int(row["seed"]),
                mean_waiting_time=float(row["mean_waiting_time"]),
                mean_queue_length=float(row["mean_queue_length"]),
                throughput=int(row["throughput"]),
                total_reward=float(row["total_reward"]),
            )
            if result.controller not in CONTROLLERS:
                raise ValueError(f"unknown baseline controller: {result.controller}")
            numeric_values = (
                result.mean_waiting_time,
                result.mean_queue_length,
                float(result.throughput),
                result.total_reward,
            )
            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError(f"non-finite result for {result.controller}/{result.seed}")
            key = (result.controller, result.seed)
            if key in results:
                raise ValueError(f"duplicate result for {result.controller}/{result.seed}")
            results[key] = result
    return results


def load_dqn_results(
    path: Path, *, seeds: Sequence[int]
) -> list[DQNEvaluationResult]:
    """Load DQN results and require exactly one row for every evaluation seed."""
    by_seed: dict[int, DQNEvaluationResult] = {}
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        expected_fields = [
            "seed",
            "total_reward",
            "mean_waiting_time",
            "mean_queue_length",
            "throughput",
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError(f"unexpected columns in {path}")
        for row in reader:
            result = DQNEvaluationResult(
                seed=int(row["seed"]),
                total_reward=float(row["total_reward"]),
                mean_waiting_time=float(row["mean_waiting_time"]),
                mean_queue_length=float(row["mean_queue_length"]),
                throughput=int(row["throughput"]),
            )
            if result.seed in by_seed:
                raise ValueError(f"duplicate DQN seed {result.seed} in {path}")
            by_seed[result.seed] = result

    expected = set(seeds)
    actual = set(by_seed)
    if actual != expected:
        raise ValueError(
            f"DQN seeds in {path} do not match evaluation seeds: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return [by_seed[seed] for seed in seeds]


def paired_differences(
    dqn_results: Sequence[DQNEvaluationResult],
    baseline_results: Sequence[BaselineEvaluationResult],
) -> dict[str, MetricStatistics]:
    """Summarize paired DQN-minus-baseline differences by matching seed."""
    dqn_by_seed = {result.seed: result for result in dqn_results}
    baseline_by_seed = {result.seed: result for result in baseline_results}
    if set(dqn_by_seed) != set(baseline_by_seed):
        raise ValueError("paired results must contain identical seeds")
    return {
        metric: _mean_and_sample_sd(
            [
                float(getattr(dqn_by_seed[seed], metric))
                - float(getattr(baseline_by_seed[seed], metric))
                for seed in sorted(dqn_by_seed)
            ]
        )
        for metric in METRICS
    }


def _format(statistics: MetricStatistics) -> str:
    return f"{statistics.mean:.3f} ± {statistics.standard_deviation:.3f}"


def print_comparison(
    baseline_results: Sequence[BaselineEvaluationResult],
    dqn_results: Sequence[DQNEvaluationResult],
) -> None:
    """Print aggregate and paired descriptive comparisons."""
    groups: list[tuple[str, Sequence[BaselineEvaluationResult | DQNEvaluationResult]]] = [
        (
            "Fixed-time",
            [item for item in baseline_results if item.controller == "fixed_time"],
        ),
        ("Random", [item for item in baseline_results if item.controller == "random"]),
        ("DQN", dqn_results),
    ]
    print("\nCombined comparison (mean ± sample standard deviation)")
    print(
        f"{'Controller':<12} | {'Mean waiting time':>24} | "
        f"{'Mean queue length':>24} | {'Mean throughput':>24}"
    )
    print("-" * 95)
    for name, results in groups:
        summary = summarize_metrics(results)
        print(
            f"{name:<12} | {_format(summary['mean_waiting_time']):>24} | "
            f"{_format(summary['mean_queue_length']):>24} | "
            f"{_format(summary['throughput']):>24}"
        )

    print("\nPaired differences (DQN minus baseline; mean ± sample SD)")
    for controller, label in (("fixed_time", "DQN vs fixed-time"), ("random", "DQN vs random")):
        controller_results = [
            item for item in baseline_results if item.controller == controller
        ]
        differences = paired_differences(dqn_results, controller_results)
        print(
            f"{label}: waiting={_format(differences['mean_waiting_time'])}, "
            f"queue={_format(differences['mean_queue_length'])}, "
            f"throughput={_format(differences['throughput'])}"
        )
    print("These are descriptive comparisons; means alone do not establish statistical superiority.")


EpisodeRunner = Callable[[int], BaselineEvaluationResult]


def evaluate_baselines(
    *,
    seeds: Sequence[int] = EVAL_SEEDS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    dqn_path: Path = DEFAULT_DQN_PATH,
    episode_runners: Mapping[str, EpisodeRunner] | None = None,
) -> tuple[list[BaselineEvaluationResult], list[DQNEvaluationResult]]:
    """Evaluate/resume both baselines and print their comparison with DQN."""
    if list(seeds) != EVAL_SEEDS:
        raise ValueError(f"controlled evaluation requires seeds {EVAL_SEEDS}")
    if EPISODE_SECONDS % DECISION_INTERVAL:
        raise ValueError("episode duration must be divisible by decision interval")

    # Validate the comparison data before spending time on SUMO runs.
    dqn_results = load_dqn_results(dqn_path, seeds=seeds)
    runners = episode_runners or {
        "fixed_time": run_fixed_time_episode,
        "random": run_random_episode,
    }
    completed = {
        key: result
        for key, result in load_baseline_results(output_path).items()
        if key[0] in CONTROLLERS and key[1] in seeds
    }

    for controller in CONTROLLERS:
        for seed in seeds:
            key = (controller, seed)
            if key in completed:
                print(f"Skipping {controller} seed {seed}: existing result found", flush=True)
                continue
            print(f"Running {controller} controller, seed {seed}...", flush=True)
            result = runners[controller](seed)
            if result.controller != controller or result.seed != seed:
                raise ValueError(
                    f"runner returned {result.controller}/{result.seed} "
                    f"for {controller}/{seed}"
                )
            completed[key] = result
            ordered = [
                completed[(name, item_seed)]
                for name in CONTROLLERS
                for item_seed in seeds
                if (name, item_seed) in completed
            ]
            write_results(output_path, ordered)
            print(
                f"Completed {controller} seed {seed}: "
                f"wait={result.mean_waiting_time:.3f}, "
                f"queue={result.mean_queue_length:.3f}, "
                f"throughput={result.throughput}",
                flush=True,
            )

    baseline_results = [
        completed[(controller, seed)]
        for controller in CONTROLLERS
        for seed in seeds
    ]
    write_results(output_path, baseline_results)
    print_comparison(baseline_results, dqn_results)
    print(f"\nBaseline results: {output_path}")
    print(f"Identical SUMO seeds used for all controllers: {list(seeds)}")
    return baseline_results, dqn_results


def main() -> None:
    """Run the controlled baseline evaluation."""
    evaluate_baselines()


if __name__ == "__main__":
    main()
