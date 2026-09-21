"""Trace greedy checkpoint decisions without changing validation outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from evaluate_dqn import greedy_action
from src.environment.traffic_env import MINIMUM_GREEN_DURATION, TrafficEnvironment
from src.simulation.connection import sumo_connection
from src.simulation.run import build_sumo_command
from src.simulation.traffic import TrafficSimulation
from validate_extended import load_policy


FIELDS = (
    "checkpoint", "seed", "time", "north_queue", "south_queue", "east_queue",
    "west_queue", "observed_phase", "q_north_south", "q_east_west",
    "requested_action", "effective_action", "switch_blocked", "phase_after",
)


def trace(checkpoints: list[Path], seed: int, output: Path) -> None:
    """Write one row per decision for each checkpoint on the same traffic seed."""
    agents = [(path, *load_policy(path)) for path in checkpoints]
    seconds = agents[0][2]["configuration"]["episode_seconds"]
    if any(state["configuration"]["episode_seconds"] != seconds for _, _, state in agents):
        raise ValueError("checkpoint episode durations differ")

    command = build_sumo_command(gui=False, seed=seed, steps=seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    with sumo_connection(command) as connection, output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for path, agent, _ in agents:
            simulation = TrafficSimulation(connection, reload_args=command[1:])
            environment = TrafficEnvironment(
                simulation, decision_interval=5, max_simulation_time=seconds,
                minimum_green_duration=MINIMUM_GREEN_DURATION,
            )
            observation, _ = environment.reset(seed=seed)
            while True:
                with torch.inference_mode():
                    q_values = agent.policy_network(
                        torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                    )[0].tolist()
                action = greedy_action(agent, observation)
                next_observation, _, terminated, truncated, info = environment.step(action)
                writer.writerow({
                    "checkpoint": path.name, "seed": seed,
                    "time": int(simulation.observe().time - environment.decision_interval),
                    "north_queue": int(observation[0]), "south_queue": int(observation[1]),
                    "east_queue": int(observation[2]), "west_queue": int(observation[3]),
                    "observed_phase": int(observation[4]),
                    "q_north_south": q_values[0], "q_east_west": q_values[1],
                    "requested_action": action,
                    "effective_action": info["effective_action"],
                    "switch_blocked": info["switch_blocked"],
                    "phase_after": int(next_observation[4]),
                })
                observation = next_observation
                if terminated or truncated:
                    break
            summary = environment.episode_summary()
            print(
                f"{path.name}: wait={summary.mean_queueing_delay_per_generated_vehicle}, "
                f"queue={summary.mean_queue_length}, "
                f"throughput={summary.vehicles_completed}, "
                f"remaining={summary.vehicles_remaining}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    args = parser.parse_args()
    trace(args.checkpoints, args.seed, args.output)
