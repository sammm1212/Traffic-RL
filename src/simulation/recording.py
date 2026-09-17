"""Optional per-step metric recording, separate from simulation control."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TextIO

from .metrics import APPROACHES, TrafficMetrics


FIELDS = (
    "time",
    "north_queue",
    "south_queue",
    "east_queue",
    "west_queue",
    "light_phase",
    "throughput",
    "mean_waiting_time",
)


def metric_record(metrics: TrafficMetrics) -> dict[str, int | float]:
    """Convert one observation into the stable presentation-data schema."""
    record: dict[str, int | float] = {"time": metrics.time}
    record.update(
        {
            f"{name}_queue": metrics.approaches[name].queue_length
            for name in APPROACHES
        }
    )
    record.update(
        light_phase=metrics.light_phase,
        throughput=metrics.throughput,
        mean_waiting_time=metrics.mean_waiting_time,
    )
    return record


class MetricsRecorder:
    """Write step observations as CSV or a JSON array based on file extension."""

    def __init__(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".json"}:
            raise ValueError("recording path must end in .csv or .json")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._format = suffix
        self._file: TextIO = path.open("w", encoding="utf-8", newline="")
        self._first_json_record = True
        self._writer: Any = None
        if suffix == ".csv":
            self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
            self._writer.writeheader()
        else:
            self._file.write("[\n")

    def record(self, metrics: TrafficMetrics) -> None:
        """Write one metric observation."""
        data = metric_record(metrics)
        if self._format == ".csv":
            self._writer.writerow(data)
        else:
            if not self._first_json_record:
                self._file.write(",\n")
            self._file.write("  " + json.dumps(data))
            self._first_json_record = False

    def close(self) -> None:
        """Finish and close the recording file."""
        if self._file.closed:
            return
        if self._format == ".json":
            self._file.write("\n]\n")
        self._file.close()

    def __enter__(self) -> "MetricsRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
