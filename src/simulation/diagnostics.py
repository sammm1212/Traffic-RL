"""Detailed, read-only diagnostics for controller evaluation episodes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .metrics import APPROACHES, TrafficMetrics
from .traffic import INCOMING_LANES


ROUTE_APPROACHES = {
    "north_to_south": "north",
    "south_to_north": "south",
    "east_to_west": "east",
    "west_to_east": "west",
}
FLOW_APPROACHES = {
    "north_flow": "north",
    "south_flow": "south",
    "east_flow": "east",
    "west_flow": "west",
}


@dataclass(frozen=True)
class ApproachDiagnostics:
    """Counts and queue measurements for one incoming approach."""

    generated: int
    departed: int
    completed: int
    remaining: int
    final_incoming: int
    final_outgoing: int
    final_internal: int
    mean_queue_length: float
    final_queue_length: int
    final_waiting_vehicles: int


@dataclass(frozen=True)
class VehicleDiagnostics:
    """Episode-end vehicle inventory plus per-approach measurements."""

    vehicles_generated: int
    vehicles_departed: int
    vehicles_completed: int
    vehicles_remaining: int
    vehicles_on_incoming_lanes: int
    vehicles_on_outgoing_lanes: int
    vehicles_on_internal_lanes: int
    vehicles_pending_insertion: int
    vehicles_teleported: int
    final_queue_length: int
    final_waiting_vehicles: int
    ns_green_seconds: int
    ew_green_seconds: int
    transition_seconds: int
    signal_changes: int
    approaches: dict[str, ApproachDiagnostics]


class EvaluationDiagnostics:
    """Collect detailed counts without changing simulation or control behavior."""

    def __init__(self, connection: Any) -> None:
        """Initialize counters for an active TraCI connection."""
        self._traci = connection
        self._generated: Counter[str] = Counter()
        self._departed: Counter[str] = Counter()
        self._completed: Counter[str] = Counter()
        self._queue_sums: Counter[str] = Counter()
        self._vehicle_approaches: dict[str, str] = {}
        self._samples = 0
        self._teleported = 0
        self._phase_seconds: Counter[str] = Counter()
        self._last_green_direction: int | None = None
        self._signal_changes = 0
        self._latest: TrafficMetrics | None = None

    @staticmethod
    def _approach_from_id(vehicle_id: str) -> str | None:
        flow_id = vehicle_id.rsplit(".", 1)[0]
        return FLOW_APPROACHES.get(flow_id)

    def _approach_for_active_vehicle(self, vehicle_id: str) -> str | None:
        approach = self._vehicle_approaches.get(vehicle_id)
        if approach is not None:
            return approach
        try:
            route_id = self._traci.vehicle.getRouteID(vehicle_id)
        except Exception:
            return self._approach_from_id(vehicle_id)
        return ROUTE_APPROACHES.get(route_id) or self._approach_from_id(vehicle_id)

    def add(self, metrics: TrafficMetrics) -> None:
        """Record events and lane measurements after one simulation step."""
        for vehicle_id in self._traci.simulation.getLoadedIDList():
            approach = self._approach_from_id(vehicle_id)
            if approach is not None:
                self._generated[approach] += 1

        for vehicle_id in self._traci.simulation.getDepartedIDList():
            approach = self._approach_for_active_vehicle(vehicle_id)
            if approach is not None:
                self._vehicle_approaches[vehicle_id] = approach
                self._departed[approach] += 1

        for vehicle_id in self._traci.simulation.getArrivedIDList():
            approach = self._vehicle_approaches.pop(
                vehicle_id, self._approach_from_id(vehicle_id)
            )
            if approach is not None:
                self._completed[approach] += 1

        self._teleported += self._traci.simulation.getStartingTeleportNumber()
        for approach, values in metrics.approaches.items():
            self._queue_sums[approach] += values.queue_length
        self._samples += 1
        if metrics.light_phase == 0:
            self._phase_seconds["ns"] += 1
            green_direction = 0
        elif metrics.light_phase == 3:
            self._phase_seconds["ew"] += 1
            green_direction = 1
        else:
            self._phase_seconds["transition"] += 1
            green_direction = None
        if green_direction is not None:
            if (
                self._last_green_direction is not None
                and green_direction != self._last_green_direction
            ):
                self._signal_changes += 1
            self._last_green_direction = green_direction
        self._latest = metrics

    def summary(self) -> VehicleDiagnostics:
        """Return terminal inventory and per-approach diagnostic aggregates."""
        if self._latest is None:
            raise RuntimeError("cannot summarize diagnostics without observations")

        active_by_approach: Counter[str] = Counter()
        incoming_by_approach: Counter[str] = Counter()
        outgoing_by_approach: Counter[str] = Counter()
        internal_by_approach: Counter[str] = Counter()
        for vehicle_id in self._traci.vehicle.getIDList():
            approach = self._approach_for_active_vehicle(vehicle_id)
            if approach is not None:
                active_by_approach[approach] += 1
                road_id = self._traci.vehicle.getRoadID(vehicle_id)
                if road_id.endswith("_in"):
                    incoming_by_approach[approach] += 1
                elif road_id.endswith("_out"):
                    outgoing_by_approach[approach] += 1
                else:
                    internal_by_approach[approach] += 1

        waiting_by_approach: Counter[str] = Counter()
        for approach, lane_id in INCOMING_LANES.items():
            for vehicle_id in self._traci.lane.getLastStepVehicleIDs(lane_id):
                if self._traci.vehicle.getWaitingTime(vehicle_id) > 0:
                    waiting_by_approach[approach] += 1

        per_approach = {
            approach: ApproachDiagnostics(
                generated=self._generated[approach],
                departed=self._departed[approach],
                completed=self._completed[approach],
                remaining=active_by_approach[approach],
                final_incoming=incoming_by_approach[approach],
                final_outgoing=outgoing_by_approach[approach],
                final_internal=internal_by_approach[approach],
                mean_queue_length=self._queue_sums[approach] / self._samples,
                final_queue_length=self._latest.approaches[approach].queue_length,
                final_waiting_vehicles=waiting_by_approach[approach],
            )
            for approach in APPROACHES
        }
        return VehicleDiagnostics(
            vehicles_generated=sum(self._generated.values()),
            vehicles_departed=sum(self._departed.values()),
            vehicles_completed=sum(self._completed.values()),
            vehicles_remaining=self._traci.vehicle.getIDCount(),
            vehicles_on_incoming_lanes=sum(incoming_by_approach.values()),
            vehicles_on_outgoing_lanes=sum(outgoing_by_approach.values()),
            vehicles_on_internal_lanes=sum(internal_by_approach.values()),
            vehicles_pending_insertion=len(
                self._traci.simulation.getPendingVehicles()
            ),
            vehicles_teleported=self._teleported,
            final_queue_length=self._latest.total_queue_length,
            final_waiting_vehicles=sum(waiting_by_approach.values()),
            ns_green_seconds=self._phase_seconds["ns"],
            ew_green_seconds=self._phase_seconds["ew"],
            transition_seconds=self._phase_seconds["transition"],
            signal_changes=self._signal_changes,
            approaches=per_approach,
        )
