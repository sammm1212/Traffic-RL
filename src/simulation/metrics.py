"""SUMO-independent traffic observations and run-level aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean


APPROACHES = ("north", "south", "east", "west")


@dataclass(frozen=True)
class ApproachMetrics:
    """Measurements for one incoming approach at a single simulation step."""

    queue_length: int
    vehicle_count: int
    waiting_time: float


@dataclass(frozen=True)
class TrafficMetrics:
    """A reusable observation of the intersection at one simulation step."""

    time: float
    approaches: dict[str, ApproachMetrics]
    light_phase: int
    vehicles_in_simulation: int
    vehicles_generated: int
    vehicles_completed: int

    @property
    def total_queue_length(self) -> int:
        """Return the number of halted vehicles across all approaches."""
        return sum(item.queue_length for item in self.approaches.values())

    @property
    def mean_queue_length(self) -> float:
        """Return the arithmetic mean of the four approach queue lengths."""
        return fmean(item.queue_length for item in self.approaches.values())

    @property
    def maximum_queue_length(self) -> int:
        """Return the longest individual approach queue in this observation."""
        return max((item.queue_length for item in self.approaches.values()), default=0)

    @property
    def total_waiting_time(self) -> float:
        """Current continuous waiting time summed over incoming vehicles."""
        return sum(item.waiting_time for item in self.approaches.values())

    @property
    def mean_waiting_time(self) -> float:
        """Return current incoming-lane waiting time per incoming vehicle."""
        incoming = sum(item.vehicle_count for item in self.approaches.values())
        return self.total_waiting_time / incoming if incoming else 0.0

    @property
    def throughput(self) -> int:
        """Return the cumulative number of vehicles that completed their routes."""
        return self.vehicles_completed

    def state_vector(self) -> list[int]:
        """Return queues in compass order followed by the signal phase index."""
        return [
            *(self.approaches[name].queue_length for name in APPROACHES),
            self.light_phase,
        ]


@dataclass(frozen=True)
class SimulationSummary:
    """Aggregate measurements for a completed simulation run."""

    simulation_time: float
    vehicles_generated: int
    vehicles_completed: int
    mean_queue_length: float
    maximum_queue_length: int
    mean_queueing_delay_per_generated_vehicle: float
    total_waiting_time: float

    @property
    def vehicles_remaining(self) -> int:
        """Return generated vehicles that have not completed their route."""
        return self.vehicles_generated - self.vehicles_completed

    @property
    def completion_percentage(self) -> float:
        """Return the percentage of generated vehicles that completed."""
        if not self.vehicles_generated:
            return 0.0
        return 100.0 * self.vehicles_completed / self.vehicles_generated

    @property
    def throughput_vehicles_per_hour(self) -> float:
        """Return completed vehicles per simulated hour."""
        if not self.simulation_time:
            return 0.0
        return 3600.0 * self.vehicles_completed / self.simulation_time


class MetricsAccumulator:
    """Aggregate step observations without depending on TraCI or SUMO IDs."""

    def __init__(self) -> None:
        """Initialize empty queue, delay, and latest-observation aggregates."""
        self._samples = 0
        self._queue_sum = 0
        self._maximum_queue = 0
        self._waiting_vehicle_seconds = 0.0
        self._previous_time = 0.0
        self._latest: TrafficMetrics | None = None

    def add(self, metrics: TrafficMetrics) -> None:
        """Add a chronologically ordered step observation."""
        if metrics.time < self._previous_time:
            raise ValueError("metric observations must be chronological")
        elapsed = metrics.time - self._previous_time
        self._samples += 1
        self._queue_sum += metrics.total_queue_length
        self._maximum_queue = max(self._maximum_queue, metrics.total_queue_length)
        # A SUMO queue is made up of halted vehicles. Integrating that count
        # over time gives waiting vehicle-seconds without double-counting each
        # vehicle's cumulative waiting-time counter.
        self._waiting_vehicle_seconds += metrics.total_queue_length * elapsed
        self._previous_time = metrics.time
        self._latest = metrics

    def summary(self) -> SimulationSummary:
        """Return run aggregates from all observations received so far."""
        if self._latest is None:
            return SimulationSummary(0.0, 0, 0, 0.0, 0, 0.0, 0.0)
        generated = self._latest.vehicles_generated
        return SimulationSummary(
            simulation_time=self._latest.time,
            vehicles_generated=generated,
            vehicles_completed=self._latest.vehicles_completed,
            mean_queue_length=self._queue_sum / self._samples,
            maximum_queue_length=self._maximum_queue,
            mean_queueing_delay_per_generated_vehicle=(
                self._waiting_vehicle_seconds / generated if generated else 0.0
            ),
            total_waiting_time=self._waiting_vehicle_seconds,
        )
