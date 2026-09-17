"""Run a reproducible multi-seed fixed-time baseline experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Iterable, Sequence

from src.simulation.metrics import SimulationSummary
from src.simulation.run import run_simulation

from .scenario import ROOT, write_demand_scenario


DEFAULT_EPISODES = 20
DEFAULT_START_SEED = 1
DEFAULT_GENERATION_SECONDS = 900
DEFAULT_CLEARANCE_SECONDS = 300
DEFAULT_OUTPUT_PATH = ROOT / "results" / "baseline" / "fixed_time_baseline.csv"


@dataclass(frozen=True)
class EpisodeResult:
    """CSV-ready metrics from one fixed-time episode."""

    episode: int
    seed: int
    vehicles_generated: int
    vehicles_completed: int
    vehicles_remaining: int
    completion_percentage: float
    mean_total_queue_length: float
    maximum_total_queue_length: int
    cumulative_queueing_delay_seconds: float
    mean_queueing_delay_per_generated_vehicle_seconds: float
    throughput_vehicles_per_hour: float

    @classmethod
    def from_summary(
        cls,
        *,
        episode: int,
        seed: int,
        summary: SimulationSummary,
    ) -> "EpisodeResult":
        """Create an episode record from shared simulation aggregates."""
        return cls(
            episode=episode,
            seed=seed,
            vehicles_generated=summary.vehicles_generated,
            vehicles_completed=summary.vehicles_completed,
            vehicles_remaining=summary.vehicles_remaining,
            completion_percentage=summary.completion_percentage,
            mean_total_queue_length=summary.mean_queue_length,
            maximum_total_queue_length=summary.maximum_queue_length,
            cumulative_queueing_delay_seconds=summary.total_waiting_time,
            mean_queueing_delay_per_generated_vehicle_seconds=(
                summary.mean_queueing_delay_per_generated_vehicle
            ),
            throughput_vehicles_per_hour=summary.throughput_vehicles_per_hour,
        )


@dataclass(frozen=True)
class AggregateStatistics:
    """Descriptive statistics for one metric across episodes."""

    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


PRIMARY_METRICS = (
    "vehicles_generated",
    "vehicles_completed",
    "vehicles_remaining",
    "completion_percentage",
    "mean_total_queue_length",
    "maximum_total_queue_length",
    "cumulative_queueing_delay_seconds",
    "mean_queueing_delay_per_generated_vehicle_seconds",
    "throughput_vehicles_per_hour",
)


def aggregate_results(
    results: Sequence[EpisodeResult],
) -> dict[str, AggregateStatistics]:
    """Calculate population statistics for each primary episode metric."""
    if not results:
        raise ValueError("at least one episode result is required")

    aggregates = {}
    for metric in PRIMARY_METRICS:
        values = [float(getattr(result, metric)) for result in results]
        aggregates[metric] = AggregateStatistics(
            mean=fmean(values),
            standard_deviation=pstdev(values),
            minimum=min(values),
            maximum=max(values),
        )
    return aggregates


def write_results(path: Path, results: Iterable[EpisodeResult]) -> None:
    """Write episode-level results to CSV, replacing an earlier run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EpisodeResult.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def run_baseline(
    *,
    episodes: int = DEFAULT_EPISODES,
    start_seed: int = DEFAULT_START_SEED,
    generation_seconds: int = DEFAULT_GENERATION_SECONDS,
    clearance_seconds: int = DEFAULT_CLEARANCE_SECONDS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[EpisodeResult]:
    """Run identical fixed-time scenarios over a consecutive seed sequence."""
    if episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if clearance_seconds < 0:
        raise ValueError("clearance_seconds must be non-negative")

    horizon = generation_seconds + clearance_seconds
    results: list[EpisodeResult] = []
    with TemporaryDirectory(prefix="traffic-baseline-") as directory:
        route_path = write_demand_scenario(
            Path(directory) / "demand.rou.xml",
            generation_seconds=generation_seconds,
        )
        for episode_index in range(episodes):
            seed = start_seed + episode_index
            summary = run_simulation(
                seed=seed,
                steps=horizon,
                route_path=route_path,
                stop_when_empty=False,
            )
            result = EpisodeResult.from_summary(
                episode=episode_index + 1,
                seed=seed,
                summary=summary,
            )
            results.append(result)
            print(
                f"Episode {episode_index + 1}/{episodes} seed={seed}: "
                f"completed={result.vehicles_completed}/"
                f"{result.vehicles_generated}"
            )

    write_results(output_path, results)
    return results


def print_aggregates(
    aggregates: dict[str, AggregateStatistics],
    *,
    execution_seconds: float,
    output_path: Path,
) -> None:
    """Print aggregate episode statistics and experiment metadata."""
    print("\nAggregate fixed-time baseline (population standard deviation)")
    print(f"{'metric':55} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}")
    for metric, values in aggregates.items():
        print(
            f"{metric:55} {values.mean:12.3f} "
            f"{values.standard_deviation:12.3f} {values.minimum:12.3f} "
            f"{values.maximum:12.3f}"
        )
    print(f"\nResults: {output_path}")
    print(f"Wall-clock execution time: {execution_seconds:.3f} seconds")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the baseline experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument(
        "--generation-seconds", type=int, default=DEFAULT_GENERATION_SECONDS
    )
    parser.add_argument(
        "--clearance-seconds", type=int, default=DEFAULT_CLEARANCE_SECONDS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    started = perf_counter()
    results = run_baseline(
        episodes=args.episodes,
        start_seed=args.start_seed,
        generation_seconds=args.generation_seconds,
        clearance_seconds=args.clearance_seconds,
        output_path=args.output,
    )
    elapsed = perf_counter() - started
    print_aggregates(
        aggregate_results(results),
        execution_seconds=elapsed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
