"""TraCI adapter for observing the fixed-time intersection."""

from __future__ import annotations

from typing import Any

from .metrics import ApproachMetrics, TrafficMetrics


INCOMING_LANES = {
    "north": "north_in_0",
    "south": "south_in_0",
    "east": "east_in_0",
    "west": "west_in_0",
}

TRAFFIC_LIGHT_ID = "center"


class TrafficSimulation:
    """Read intersection observations and advance an active TraCI connection."""

    def __init__(
        self,
        connection: Any,
        reload_args: list[str] | None = None,
    ) -> None:
        self._traci = connection
        self._reload_args = reload_args
        self._vehicles_generated = 0
        self._vehicles_completed = 0

    def step(self) -> TrafficMetrics:
        """Advance SUMO by one step and return the resulting observation."""
        self._traci.simulationStep()
        self._vehicles_generated += self._traci.simulation.getDepartedNumber()
        self._vehicles_completed += self._traci.simulation.getArrivedNumber()

        return self.observe()

    def observe(self) -> TrafficMetrics:
        """Read the current intersection state without advancing simulation."""

        approaches = {
            name: ApproachMetrics(
                queue_length=self._traci.lane.getLastStepHaltingNumber(lane_id),
                vehicle_count=self._traci.lane.getLastStepVehicleNumber(lane_id),
                waiting_time=self._traci.lane.getWaitingTime(lane_id),
            )
            for name, lane_id in INCOMING_LANES.items()
        }

        return TrafficMetrics(
            time=self._traci.simulation.getTime(),
            approaches=approaches,
            light_phase=self._traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID),
            vehicles_in_simulation=self._traci.vehicle.getIDCount(),
            vehicles_generated=self._vehicles_generated,
            vehicles_completed=self._vehicles_completed,
        )

    def set_traffic_light(self, action: int) -> None:
        if action == 0:
            phase = 0
        elif action == 1:
            phase = 3
        else:
            raise ValueError(f"Invalid traffic-light action: {action}")

        self._traci.trafficlight.setPhase(
            TRAFFIC_LIGHT_ID,
            phase,
        )

    def reset(self) -> TrafficMetrics:
        """Reload SUMO and reset episode counters."""

        if self._reload_args is None:
            raise RuntimeError("SUMO reload arguments were not configured")
        self._traci.load(self._reload_args)

        self._vehicles_generated = 0
        self._vehicles_completed = 0

        return self.observe()

    def apply_action(self, action: int, decision_interval: int) -> None:
        if action == 0:
            target_phase = 0
        elif action == 1:
            target_phase = 3
        else:
            raise ValueError(f"Invalid traffic-light action: {action}")

        current_phase = self._traci.trafficlight.getPhase(
            TRAFFIC_LIGHT_ID
        )

        if decision_interval < 4:
            raise ValueError(
                "decision_interval must be at least 4 seconds "
                "to allow yellow and all-red transitions"
            )

        # If we're already showing the requested green,
        # restart that phase so SUMO does not enter its fixed-time yellow
        # while the agent intends to hold the green for this interval.
        if current_phase == target_phase:
            self._traci.trafficlight.setPhase(
                TRAFFIC_LIGHT_ID,
                target_phase,
            )
            for _ in range(decision_interval):
                self.step()

            return

        # N/S green -> E/W green
        if current_phase == 0 and target_phase == 3:
            self._traci.trafficlight.setPhase(
                TRAFFIC_LIGHT_ID,
                1,
            )

            # 3 seconds N/S yellow
            for _ in range(3):
                self.step()

            self._traci.trafficlight.setPhase(
                TRAFFIC_LIGHT_ID,
                2,
            )

            # 1 second all red
            self.step()

            self._traci.trafficlight.setPhase(
                TRAFFIC_LIGHT_ID,
                3,
            )

            remaining_steps = decision_interval - 4

            for _ in range(remaining_steps):
                self.step()

            return

        # E/W green -> N/S green
        if current_phase == 3 and target_phase == 0:
            self._traci.trafficlight.setPhase(
                TRAFFIC_LIGHT_ID,
                4,
            )

            # 3 seconds E/W yellow
            for _ in range(3):
                self.step()

            self._traci.trafficlight.setPhase(
                TRAFFIC_LIGHT_ID,
                5,
            )

            # 1 second all red
            self.step()

            self._traci.trafficlight.setPhase(
                TRAFFIC_LIGHT_ID,
                0,
            )

            remaining_steps = decision_interval - 4

            for _ in range(remaining_steps):
                self.step()

            return
