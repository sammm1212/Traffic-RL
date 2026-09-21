from unittest.mock import MagicMock

from src.environment.traffic_env import TrafficEnvironment
from src.simulation.metrics import ApproachMetrics, TrafficMetrics


def _metrics(phase: int = 0, time: float = 0.0) -> TrafficMetrics:
    approaches = {
        name: ApproachMetrics(0, 0, 0.0)
        for name in ("north", "south", "east", "west")
    }
    return TrafficMetrics(time, approaches, phase, 0, 0, 0)


def _environment() -> tuple[TrafficEnvironment, MagicMock]:
    simulation = MagicMock()
    simulation.reset.return_value = _metrics()
    simulation.observe.return_value = _metrics()
    environment = TrafficEnvironment(
        simulation,
        decision_interval=5,
        minimum_green_duration=10,
    )
    environment.reset()
    return environment, simulation


def test_switch_before_minimum_green_is_ignored_without_transition() -> None:
    environment, simulation = _environment()

    _, _, _, _, first_info = environment.step(1)
    _, _, _, _, second_info = environment.step(1)

    assert [call.args[0] for call in simulation.apply_action.call_args_list] == [0, 0]
    assert first_info["switch_blocked"] is True
    assert second_info["switch_blocked"] is True
    assert second_info["effective_action"] == 0
    assert second_info["green_elapsed_seconds"] == 10


def test_switch_is_allowed_after_two_complete_hold_intervals() -> None:
    environment, simulation = _environment()
    environment.step(1)
    environment.step(1)

    _, _, _, _, info = environment.step(1)

    assert simulation.apply_action.call_args.args[0] == 1
    assert info["switch_blocked"] is False
    assert info["effective_action"] == 1
    assert info["green_elapsed_seconds"] == 0


def test_reset_restores_initial_direction_and_minimum_green_timer() -> None:
    environment, simulation = _environment()
    environment.step(0)
    environment.step(0)
    environment.step(1)
    simulation.apply_action.reset_mock()

    environment.reset()
    _, _, _, _, info = environment.step(1)

    assert simulation.apply_action.call_args.args[0] == 0
    assert info["switch_blocked"] is True
    assert info["green_elapsed_seconds"] == 5
