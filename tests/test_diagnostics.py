from unittest.mock import MagicMock

from evaluate_diagnostics import _action_values
from src.simulation.diagnostics import EvaluationDiagnostics
from src.simulation.metrics import ApproachMetrics, TrafficMetrics


def test_action_diagnostics_count_initial_change_and_runs() -> None:
    values = _action_values([1, 1, 0, 1, 1, 1])

    assert values["action_0_count"] == 1
    assert values["action_1_count"] == 5
    assert values["signal_changes"] == 3
    assert values["longest_action_0_run_decisions"] == 1
    assert values["longest_action_1_run_decisions"] == 3


def test_vehicle_diagnostics_separate_loaded_and_departed_counts() -> None:
    connection = MagicMock()
    connection.simulation.getLoadedIDList.return_value = (
        "north_flow.0",
        "east_flow.0",
    )
    connection.simulation.getDepartedIDList.return_value = ("north_flow.0",)
    connection.simulation.getArrivedIDList.return_value = ()
    connection.simulation.getStartingTeleportNumber.return_value = 0
    connection.simulation.getPendingVehicles.return_value = ("east_flow.0",)
    connection.vehicle.getRouteID.return_value = "north_to_south"
    connection.vehicle.getIDList.return_value = ("north_flow.0",)
    connection.vehicle.getIDCount.return_value = 1
    connection.vehicle.getRoadID.return_value = "north_in"
    connection.vehicle.getWaitingTime.return_value = 1.0
    connection.lane.getLastStepVehicleIDs.side_effect = lambda lane: (
        ("north_flow.0",) if lane == "north_in_0" else ()
    )
    metrics = TrafficMetrics(
        time=1.0,
        approaches={
            "north": ApproachMetrics(1, 1, 1.0),
            "south": ApproachMetrics(0, 0, 0.0),
            "east": ApproachMetrics(0, 0, 0.0),
            "west": ApproachMetrics(0, 0, 0.0),
        },
        light_phase=0,
        vehicles_in_simulation=1,
        vehicles_generated=1,
        vehicles_completed=0,
    )

    collector = EvaluationDiagnostics(connection)
    collector.add(metrics)
    result = collector.summary()

    assert result.vehicles_generated == 2
    assert result.vehicles_departed == 1
    assert result.vehicles_completed == 0
    assert result.vehicles_remaining == 1
    assert result.vehicles_on_incoming_lanes == 1
    assert result.vehicles_on_outgoing_lanes == 0
    assert result.vehicles_pending_insertion == 1
    assert result.final_queue_length == 1
    assert result.final_waiting_vehicles == 1
    assert result.approaches["north"].mean_queue_length == 1.0


def test_vehicle_diagnostics_count_observed_green_direction_changes() -> None:
    connection = MagicMock()
    connection.simulation.getLoadedIDList.return_value = ()
    connection.simulation.getDepartedIDList.return_value = ()
    connection.simulation.getArrivedIDList.return_value = ()
    connection.simulation.getStartingTeleportNumber.return_value = 0
    connection.simulation.getPendingVehicles.return_value = ()
    connection.vehicle.getIDList.return_value = ()
    connection.vehicle.getIDCount.return_value = 0
    connection.lane.getLastStepVehicleIDs.return_value = ()
    approaches = {
        name: ApproachMetrics(0, 0, 0.0)
        for name in ("north", "south", "east", "west")
    }
    collector = EvaluationDiagnostics(connection)

    for time, phase in enumerate((0, 1, 2, 3, 3, 4, 5, 0), start=1):
        collector.add(TrafficMetrics(time, approaches, phase, 0, 0, 0))

    assert collector.summary().signal_changes == 2
