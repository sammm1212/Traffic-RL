import unittest
from unittest.mock import MagicMock, call

from src.simulation.traffic import (
    INCOMING_LANES,
    TRAFFIC_LIGHT_ID,
    TrafficSimulation,
)


class TrafficSimulationTests(unittest.TestCase):
    def test_observe_queries_canonical_lanes_and_traffic_light(self) -> None:
        connection = MagicMock()
        connection.simulation.getTime.return_value = 12.0
        connection.trafficlight.getPhase.return_value = 3
        connection.vehicle.getIDCount.return_value = 5
        simulation = TrafficSimulation(connection)

        metrics = simulation.observe()

        self.assertEqual(
            INCOMING_LANES,
            {
                "north": "north_in_0",
                "south": "south_in_0",
                "east": "east_in_0",
                "west": "west_in_0",
            },
        )
        self.assertEqual(TRAFFIC_LIGHT_ID, "center")
        expected_lane_calls = [call(lane_id) for lane_id in INCOMING_LANES.values()]
        self.assertEqual(
            connection.lane.getLastStepHaltingNumber.call_args_list,
            expected_lane_calls,
        )
        self.assertEqual(
            connection.lane.getLastStepVehicleNumber.call_args_list,
            expected_lane_calls,
        )
        self.assertEqual(
            connection.lane.getWaitingTime.call_args_list,
            expected_lane_calls,
        )
        connection.trafficlight.getPhase.assert_called_once_with("center")
        self.assertEqual(metrics.time, 12.0)
        self.assertEqual(metrics.light_phase, 3)

    def test_step_advances_and_counts_departures_and_arrivals(self) -> None:
        connection = MagicMock()
        connection.simulation.getDepartedNumber.return_value = 2
        connection.simulation.getArrivedNumber.return_value = 1
        simulation = TrafficSimulation(connection)

        metrics = simulation.step()

        connection.simulationStep.assert_called_once_with()
        self.assertEqual(metrics.vehicles_generated, 2)
        self.assertEqual(metrics.vehicles_completed, 1)

    def test_reselecting_current_green_holds_it_for_decision_interval(self) -> None:
        connection = MagicMock()
        connection.trafficlight.getPhase.return_value = 0
        simulation = TrafficSimulation(connection)
        simulation.step = MagicMock()

        simulation.apply_action(action=0, decision_interval=5)

        connection.trafficlight.setPhase.assert_called_once_with(
            TRAFFIC_LIGHT_ID,
            0,
        )
        self.assertEqual(simulation.step.call_count, 5)

    def test_action_reports_each_internal_step_to_metrics_callback(self) -> None:
        connection = MagicMock()
        connection.trafficlight.getPhase.return_value = 0
        simulation = TrafficSimulation(connection)
        observations = [object() for _ in range(5)]
        simulation.step = MagicMock(side_effect=observations)
        received = []

        simulation.apply_action(
            action=0,
            decision_interval=5,
            on_step=received.append,
        )

        self.assertEqual(received, observations)

    def test_valid_switch_preserves_yellow_all_red_green_sequence(self) -> None:
        connection = MagicMock()
        connection.trafficlight.getPhase.return_value = 0
        simulation = TrafficSimulation(connection)
        simulation.step = MagicMock()

        simulation.apply_action(action=1, decision_interval=5)

        phases = [call.args[1] for call in connection.trafficlight.setPhase.call_args_list]
        self.assertEqual(phases, [1, 2, 3])
        self.assertEqual(simulation.step.call_count, 5)


if __name__ == "__main__":
    unittest.main()
