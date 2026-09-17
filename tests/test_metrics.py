import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.simulation.metrics import (
    ApproachMetrics,
    MetricsAccumulator,
    TrafficMetrics,
)
from src.simulation.recording import MetricsRecorder, metric_record


def observation(
    time: float,
    queues: tuple[int, int, int, int],
    *,
    generated: int = 4,
    completed: int = 1,
) -> TrafficMetrics:
    approaches = {
        name: ApproachMetrics(queue, queue + 1, float(queue * 2))
        for name, queue in zip(("north", "south", "east", "west"), queues)
    }
    return TrafficMetrics(time, approaches, 3, 7, generated, completed)


class TrafficMetricsTests(unittest.TestCase):
    def test_aggregates_approaches_and_builds_state(self) -> None:
        metrics = observation(1.0, (1, 2, 3, 4))

        self.assertEqual(metrics.total_queue_length, 10)
        self.assertEqual(metrics.mean_queue_length, 2.5)
        self.assertEqual(metrics.maximum_queue_length, 4)
        self.assertEqual(metrics.total_waiting_time, 20.0)
        self.assertEqual(metrics.state_vector(), [1, 2, 3, 4, 3])

    def test_run_aggregation_uses_total_queue_and_elapsed_time(self) -> None:
        accumulator = MetricsAccumulator()
        accumulator.add(observation(1.0, (1, 0, 1, 0)))
        accumulator.add(observation(3.0, (2, 1, 1, 0), generated=5, completed=2))

        summary = accumulator.summary()

        self.assertEqual(summary.mean_queue_length, 3.0)
        self.assertEqual(summary.maximum_queue_length, 4)
        self.assertEqual(summary.total_waiting_time, 10.0)
        self.assertEqual(summary.mean_queueing_delay_per_generated_vehicle, 2.0)
        self.assertEqual(summary.vehicles_completed, 2)
        self.assertEqual(summary.vehicles_remaining, 3)
        self.assertEqual(summary.completion_percentage, 40.0)
        self.assertAlmostEqual(summary.throughput_vehicles_per_hour, 2400.0)

    def test_record_schema_contains_required_fields(self) -> None:
        record = metric_record(observation(5.0, (1, 2, 3, 4)))

        self.assertEqual(
            set(record),
            {
                "time",
                "north_queue",
                "south_queue",
                "east_queue",
                "west_queue",
                "light_phase",
                "throughput",
                "mean_waiting_time",
            },
        )

    def test_json_recorder_writes_valid_array(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "steps.json"
            with MetricsRecorder(path) as recorder:
                recorder.record(observation(1.0, (0, 1, 0, 1)))
                recorder.record(observation(2.0, (1, 0, 1, 0)))

            records = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["time"], 2.0)

    def test_recorder_rejects_unknown_extension(self) -> None:
        with self.assertRaisesRegex(ValueError, "csv or .json"):
            MetricsRecorder(Path("steps.txt"))
