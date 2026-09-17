"""Run the fixed-time intersection simulation through TraCI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import traci
from sumolib import checkBinary

from .metrics import MetricsAccumulator, SimulationSummary
from .recording import MetricsRecorder
from .traffic import TrafficSimulation


DEFAULT_STEPS = 600
DEFAULT_SEED = 42
CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "simulation"
    / "config"
    / "intersection.sumocfg"
)


def build_sumo_command(
    *,
    gui: bool,
    seed: int,
    steps: int,
    route_path: Path | None = None,
) -> list[str]:
    """Build the SUMO command used by TraCI."""
    binary = checkBinary("sumo-gui" if gui else "sumo")
    command = [
        binary,
        "-c",
        str(CONFIG_PATH),
        "--seed",
        str(seed),
        "--end",
        str(steps),
        "--quit-on-end",
    ]
    if route_path is not None:
        command.extend(("--route-files", str(route_path.resolve())))
    return command


def run_simulation(
    *,
    gui: bool = False,
    seed: int = DEFAULT_SEED,
    steps: int = DEFAULT_STEPS,
    record_path: Path | None = None,
    route_path: Path | None = None,
    stop_when_empty: bool = True,
) -> SimulationSummary:
    """Run SUMO and return aggregate metrics for the fixed-time baseline."""
    if steps <= 0:
        raise ValueError("steps must be a positive integer")

    recorder = MetricsRecorder(record_path) if record_path else None
    try:
        traci.start(
            build_sumo_command(
                gui=gui,
                seed=seed,
                steps=steps,
                route_path=route_path,
            )
        )
    except BaseException:
        if recorder:
            recorder.close()
        raise
    simulation = TrafficSimulation(traci)
    accumulator = MetricsAccumulator()
    try:
        while traci.simulation.getTime() < steps:
            if stop_when_empty and traci.simulation.getMinExpectedNumber() <= 0:
                break
            metrics = simulation.step()
            accumulator.add(metrics)
            if recorder:
                recorder.record(metrics)
    finally:
        if recorder:
            recorder.close()
        traci.close()
    return accumulator.summary()


def print_summary(summary: SimulationSummary, seed: int) -> None:
    """Print the required fixed-time baseline report."""
    print("Simulation complete")
    print(f"Seed: {seed}")
    print(f"Simulation time: {summary.simulation_time:.1f}")
    print(f"Vehicles generated: {summary.vehicles_generated}")
    print(f"Vehicles completed: {summary.vehicles_completed}")
    print(f"Mean queue length: {summary.mean_queue_length:.2f}")
    print(f"Maximum queue length: {summary.maximum_queue_length}")
    print(
        "Mean queueing delay per generated vehicle: "
        f"{summary.mean_queueing_delay_per_generated_vehicle:.2f}"
    )
    print(f"Total waiting time: {summary.total_waiting_time:.2f}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="run with sumo-gui")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="SUMO random seed")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="simulation duration in seconds",
    )
    parser.add_argument(
        "--record",
        type=Path,
        help="optional per-step output path ending in .csv or .json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    summary = run_simulation(
        gui=args.gui,
        seed=args.seed,
        steps=args.steps,
        record_path=args.record,
    )
    print_summary(summary, args.seed)


if __name__ == "__main__":
    main()
