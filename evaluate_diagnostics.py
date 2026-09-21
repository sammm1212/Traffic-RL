"""Diagnose DQN and fixed-time behavior on the controlled evaluation seeds."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence

from evaluate_dqn import (
    DECISION_INTERVAL,
    DEFAULT_CHECKPOINT_PATH,
    EPISODE_SECONDS,
    EVAL_SEEDS,
    greedy_action,
    load_agent,
    set_deterministic_seed,
)
from src.environment.traffic_env import MINIMUM_GREEN_DURATION, TrafficEnvironment
from src.simulation.diagnostics import EvaluationDiagnostics, VehicleDiagnostics
from src.simulation.metrics import MetricsAccumulator
from src.simulation.recording import MetricsRecorder
from src.simulation.run import build_sumo_command
from src.simulation.traffic import TrafficSimulation


ROOT = Path(__file__).resolve().parent
DEFAULT_DQN_OUTPUT = ROOT / "results" / "evaluation" / "dqn_diagnostics.csv"
DEFAULT_FIXED_OUTPUT = ROOT / "results" / "evaluation" / "fixed_time_diagnostics.csv"


@dataclass(frozen=True)
class DiagnosticResult:
    """One flat, CSV-ready diagnostic record."""

    values: dict[str, object]


def _base_values(
    *,
    controller: str,
    seed: int,
    diagnostics: VehicleDiagnostics,
    mean_queue_length: float,
    mean_waiting_time: float,
) -> dict[str, object]:
    values: dict[str, object] = {
        "controller": controller,
        "seed": seed,
        "vehicles_generated": diagnostics.vehicles_generated,
        "vehicles_departed": diagnostics.vehicles_departed,
        "vehicles_completed": diagnostics.vehicles_completed,
        "vehicles_remaining": diagnostics.vehicles_remaining,
        "vehicles_on_incoming_lanes": diagnostics.vehicles_on_incoming_lanes,
        "vehicles_on_outgoing_lanes": diagnostics.vehicles_on_outgoing_lanes,
        "vehicles_on_internal_lanes": diagnostics.vehicles_on_internal_lanes,
        "vehicles_pending_insertion": diagnostics.vehicles_pending_insertion,
        "vehicles_teleported": diagnostics.vehicles_teleported,
        "final_queue_length": diagnostics.final_queue_length,
        "final_waiting_vehicles": diagnostics.final_waiting_vehicles,
        "mean_queue_length": mean_queue_length,
        "mean_waiting_time": mean_waiting_time,
    }
    for approach, item in diagnostics.approaches.items():
        values.update(
            {
                f"{approach}_generated": item.generated,
                f"{approach}_departed": item.departed,
                f"{approach}_completed": item.completed,
                f"{approach}_remaining": item.remaining,
                f"{approach}_final_incoming": item.final_incoming,
                f"{approach}_final_outgoing": item.final_outgoing,
                f"{approach}_final_internal": item.final_internal,
                f"{approach}_mean_queue_length": item.mean_queue_length,
                f"{approach}_final_queue_length": item.final_queue_length,
                f"{approach}_final_waiting_vehicles": item.final_waiting_vehicles,
            }
        )
    total_seconds = (
        diagnostics.ns_green_seconds
        + diagnostics.ew_green_seconds
        + diagnostics.transition_seconds
    )
    values.update(
        {
            "ns_green_seconds": diagnostics.ns_green_seconds,
            "ns_green_proportion": diagnostics.ns_green_seconds / total_seconds,
            "ew_green_seconds": diagnostics.ew_green_seconds,
            "ew_green_proportion": diagnostics.ew_green_seconds / total_seconds,
            "transition_seconds": diagnostics.transition_seconds,
            "transition_proportion": diagnostics.transition_seconds / total_seconds,
        }
    )
    return values


def _action_values(actions: Sequence[int]) -> dict[str, object]:
    counts = Counter(actions)
    decisions = len(actions)
    changes = sum(current != previous for previous, current in zip(actions, actions[1:]))
    # The signal starts in north-south green, so selecting east-west first is a
    # real direction change even though there is no preceding selected action.
    if actions and actions[0] != 0:
        changes += 1

    longest_run = {0: 0, 1: 0}
    if actions:
        current_action = actions[0]
        run_length = 0
        for action in actions:
            if action == current_action:
                run_length += 1
            else:
                longest_run[current_action] = max(
                    longest_run[current_action], run_length
                )
                current_action = action
                run_length = 1
        longest_run[current_action] = max(longest_run[current_action], run_length)

    return {
        "decisions": decisions,
        "action_0_count": counts[0],
        "action_0_proportion": counts[0] / decisions if decisions else 0.0,
        "action_1_count": counts[1],
        "action_1_proportion": counts[1] / decisions if decisions else 0.0,
        "signal_changes": changes,
        "longest_action_0_run_decisions": longest_run[0],
        "longest_action_1_run_decisions": longest_run[1],
    }


def _run_episode(
    *,
    controller: str,
    seed: int,
    agent: Any | None,
    episode_seconds: int,
    route_path: Path | None = None,
    record_path: Path | None = None,
) -> DiagnosticResult:
    import traci

    command = build_sumo_command(
        gui=False, seed=seed, steps=episode_seconds, route_path=route_path
    )
    started = False
    recorder: MetricsRecorder | None = None
    try:
        traci.start(command)
        started = True
        recorder = MetricsRecorder(record_path) if record_path is not None else None
        simulation = TrafficSimulation(traci, reload_args=command[1:])
        collector = EvaluationDiagnostics(traci)
        accumulator = MetricsAccumulator()
        actions: list[int] = []

        def record_step(metrics: Any) -> None:
            accumulator.add(metrics)
            collector.add(metrics)
            if recorder is not None:
                recorder.record(metrics)

        environment = TrafficEnvironment(
            simulation=simulation,
            decision_interval=DECISION_INTERVAL,
            max_simulation_time=episode_seconds,
            minimum_green_duration=(
                MINIMUM_GREEN_DURATION if controller == "dqn" else None
            ),
            step_observer=record_step,
        )
        environment.reset(seed=seed)

        if controller == "dqn":
            state = environment.get_state()
            while simulation.observe().time < episode_seconds:
                requested_action = greedy_action(agent, state)
                state, _, _, _, info = environment.step(requested_action)
                actions.append(info["effective_action"])
        elif controller == "fixed_time":
            for _ in range(episode_seconds):
                record_step(simulation.step())
        else:
            raise ValueError(f"unknown controller: {controller}")

        traffic = accumulator.summary()
        details = collector.summary()
        values = _base_values(
            controller=controller,
            seed=seed,
            diagnostics=details,
            mean_queue_length=traffic.mean_queue_length,
            mean_waiting_time=traffic.mean_queueing_delay_per_generated_vehicle,
        )
        values.update(_action_values(actions))
        values["total_waiting_vehicle_seconds"] = traffic.total_waiting_time
        # Count observed changes of principal green, including the autonomous
        # fixed-time program, rather than relying on controller requests.
        values["signal_changes"] = details.signal_changes
        return DiagnosticResult(values)
    finally:
        if recorder is not None:
            recorder.close()
        if started:
            episode_failed = sys.exc_info()[0] is not None
            try:
                traci.close()
            except Exception:
                if not episode_failed:
                    raise


def write_results(path: Path, results: Sequence[DiagnosticResult]) -> None:
    """Atomically write diagnostic records using their shared flat schema."""
    if not results:
        raise ValueError("at least one diagnostic result is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].values)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result.values for result in results)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def evaluate_diagnostics(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    seeds: Sequence[int] = EVAL_SEEDS,
    episode_seconds: int = EPISODE_SECONDS,
    dqn_output: Path = DEFAULT_DQN_OUTPUT,
    fixed_output: Path = DEFAULT_FIXED_OUTPUT,
) -> tuple[list[DiagnosticResult], list[DiagnosticResult]]:
    """Run detailed DQN and fixed-time diagnostics on identical seeds."""
    if not seeds:
        raise ValueError("at least one seed is required")
    if episode_seconds <= 0 or episode_seconds % DECISION_INTERVAL:
        raise ValueError("episode duration must be positive and divisible by 5")

    set_deterministic_seed(0)
    agent = load_agent(checkpoint_path)
    dqn_results: list[DiagnosticResult] = []
    fixed_results: list[DiagnosticResult] = []
    for seed in seeds:
        set_deterministic_seed(seed)
        print(f"Running DQN diagnostics for seed {seed}...", flush=True)
        result = _run_episode(
            controller="dqn",
            seed=seed,
            agent=agent,
            episode_seconds=episode_seconds,
        )
        dqn_results.append(result)
        write_results(dqn_output, dqn_results)
        print(
            f"DQN {seed}: departed={result.values['vehicles_departed']} "
            f"completed={result.values['vehicles_completed']} "
            f"remaining={result.values['vehicles_remaining']} "
            f"changes={result.values['signal_changes']}",
            flush=True,
        )

        print(f"Running fixed-time diagnostics for seed {seed}...", flush=True)
        result = _run_episode(
            controller="fixed_time",
            seed=seed,
            agent=None,
            episode_seconds=episode_seconds,
        )
        fixed_results.append(result)
        write_results(fixed_output, fixed_results)
        print(
            f"Fixed {seed}: departed={result.values['vehicles_departed']} "
            f"completed={result.values['vehicles_completed']} "
            f"remaining={result.values['vehicles_remaining']}",
            flush=True,
        )
    return dqn_results, fixed_results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse diagnostic evaluation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    parser.add_argument("--episode-seconds", type=int, default=EPISODE_SECONDS)
    parser.add_argument("--dqn-output", type=Path, default=DEFAULT_DQN_OUTPUT)
    parser.add_argument("--fixed-output", type=Path, default=DEFAULT_FIXED_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run command-line diagnostics."""
    args = parse_args(argv)
    evaluate_diagnostics(
        checkpoint_path=args.checkpoint,
        seeds=args.seeds,
        episode_seconds=args.episode_seconds,
        dqn_output=args.dqn_output,
        fixed_output=args.fixed_output,
    )


if __name__ == "__main__":
    main()
