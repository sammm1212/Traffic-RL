import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.experiments.fixed_time_baseline import (
    EpisodeResult,
    aggregate_results,
    run_baseline,
)
from src.experiments.scenario import write_demand_scenario


def episode(seed: int, completed: int, mean_queue: float) -> EpisodeResult:
    return EpisodeResult(
        episode=seed,
        seed=seed,
        vehicles_generated=10,
        vehicles_completed=completed,
        vehicles_remaining=10 - completed,
        completion_percentage=completed * 10.0,
        mean_total_queue_length=mean_queue,
        maximum_total_queue_length=4,
        cumulative_queueing_delay_seconds=20.0,
        mean_queueing_delay_per_generated_vehicle_seconds=2.0,
        throughput_vehicles_per_hour=30.0,
    )


class ExperimentTests(unittest.TestCase):
    def test_aggregation_calculates_population_statistics(self) -> None:
        aggregates = aggregate_results([episode(1, 8, 1.0), episode(2, 10, 3.0)])

        completed = aggregates["vehicles_completed"]
        self.assertEqual(completed.mean, 9.0)
        self.assertEqual(completed.standard_deviation, 1.0)
        self.assertEqual(completed.minimum, 8.0)
        self.assertEqual(completed.maximum, 10.0)
        self.assertEqual(aggregates["mean_total_queue_length"].mean, 2.0)

    def test_scenario_changes_only_flow_end_times(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "scenario.rou.xml"
            write_demand_scenario(destination, generation_seconds=17)
            text = destination.read_text(encoding="utf-8")

        self.assertEqual(text.count('end="17"'), 4)
        self.assertEqual(text.count('probability="0.12"'), 4)

    def test_same_seed_produces_same_episode_result(self) -> None:
        with TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.csv"
            second_path = Path(directory) / "second.csv"
            first = run_baseline(
                episodes=1,
                start_seed=91,
                generation_seconds=30,
                clearance_seconds=30,
                output_path=first_path,
            )
            second = run_baseline(
                episodes=1,
                start_seed=91,
                generation_seconds=30,
                clearance_seconds=30,
                output_path=second_path,
            )

            with first_path.open(encoding="utf-8", newline="") as output:
                rows = list(csv.DictReader(output))

        self.assertEqual(first, second)
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["seed"]), 91)

    def test_rejects_empty_aggregation(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            aggregate_results([])
