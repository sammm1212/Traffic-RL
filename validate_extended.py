"""Evaluate extended checkpoints on fixed, held-out traffic seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import fmean
from typing import Sequence

import torch

from evaluate_dqn import greedy_action
from src.agents.dqn_agent import DQNAgent
from src.environment.traffic_env import MINIMUM_GREEN_DURATION, TrafficEnvironment
from src.simulation.connection import sumo_connection
from src.simulation.run import build_sumo_command
from src.simulation.traffic import TrafficSimulation
from train_extended import DEFAULT_OUTPUT_DIR, TRAIN_SEED_BASE, training_seed


VALIDATION_SEEDS = tuple(range(2000, 2010))
PREVIOUS_TEST_SEEDS = frozenset(range(1000, 1030))
PREVIOUS_EVAL_SEEDS = frozenset(range(100, 110))
FIELDS = (
    "checkpoint", "episode", "seed", "mean_waiting_time",
    "mean_queue_length", "throughput", "vehicles_remaining",
)


def validate_seed_sets(episodes: int, seed_base: int) -> None:
    """Reject any overlap among training, validation, and prior evaluation seeds."""
    training = {training_seed(episode, seed_base) for episode in range(1, episodes + 1)}
    validation = set(VALIDATION_SEEDS)
    if training & (validation | PREVIOUS_TEST_SEEDS | PREVIOUS_EVAL_SEEDS):
        raise ValueError("training and evaluation seeds overlap")
    if validation & (PREVIOUS_TEST_SEEDS | PREVIOUS_EVAL_SEEDS):
        raise ValueError("validation and previous test seeds overlap")


def load_policy(path: Path) -> tuple[DQNAgent, dict]:
    """Load a local training checkpoint for greedy inference only."""
    # Extended checkpoints contain NumPy replay entries and RNG state.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint: {path}")
    agent = DQNAgent()
    agent.policy_network.load_state_dict(state["policy_net_state_dict"])
    agent.target_network.load_state_dict(state["target_net_state_dict"])
    agent.epsilon = 0.0
    agent.policy_network.eval()
    agent.target_network.eval()
    return agent, state


def evaluate_one(agent: DQNAgent, seed: int, seconds: int, *, connection=None) -> dict:
    """Run one greedy, minimum-green SUMO episode and collect traffic metrics."""
    command = build_sumo_command(gui=False, seed=seed, steps=seconds)
    if connection is None:
        with sumo_connection(command) as active_connection:
            return evaluate_one(agent, seed, seconds, connection=active_connection)
    simulation = TrafficSimulation(connection, reload_args=command[1:])
    environment = TrafficEnvironment(
        simulation, decision_interval=5, max_simulation_time=seconds,
        minimum_green_duration=MINIMUM_GREEN_DURATION,
    )
    state, _ = environment.reset(seed=seed)
    while True:
        state, _, terminated, truncated, _ = environment.step(greedy_action(agent, state))
        if terminated or truncated:
            break
    summary = environment.episode_summary()
    return {
        "seed": seed,
        "mean_waiting_time": summary.mean_queueing_delay_per_generated_vehicle,
        "mean_queue_length": summary.mean_queue_length,
        "throughput": summary.vehicles_completed,
        "vehicles_remaining": summary.vehicles_remaining,
    }


def summarize_checkpoints(rows: Sequence[dict]) -> list[dict]:
    """Aggregate held-out traffic metrics for every complete checkpoint."""
    checkpoints = sorted({row["checkpoint"] for row in rows})
    if not checkpoints:
        raise ValueError("no validation results")
    scores = []
    for checkpoint in checkpoints:
        samples = [row for row in rows if row["checkpoint"] == checkpoint]
        if {row["seed"] for row in samples} != set(VALIDATION_SEEDS):
            raise ValueError(f"incomplete validation seeds for {checkpoint}")
        scores.append({
            "checkpoint": checkpoint,
            "episode": samples[0]["episode"],
            "mean_waiting_time": fmean(row["mean_waiting_time"] for row in samples),
            "mean_queue_length": fmean(row["mean_queue_length"] for row in samples),
            "throughput": fmean(row["throughput"] for row in samples),
            "vehicles_remaining": fmean(row["vehicles_remaining"] for row in samples),
        })
    return scores


def choose_best(rows: Sequence[dict]) -> dict:
    """Rank checkpoints by validation wait, then queue, throughput, and remaining."""
    return min(summarize_checkpoints(rows), key=lambda row: (
        row["mean_waiting_time"], row["mean_queue_length"],
        -row["throughput"], row["vehicles_remaining"], row["episode"],
    ))


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    """Write a complete CSV in the isolated extended output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    """Evaluate each saved checkpoint and choose the best by held-out traffic."""
    checkpoints = sorted((output_dir / "checkpoints").glob("episode_*.pt"))
    if not checkpoints:
        raise FileNotFoundError("no extended-training checkpoints found")
    rows = []
    first_agent, first_state = load_policy(checkpoints[0])
    expected_config = first_state["configuration"]
    validate_seed_sets(expected_config["episodes"], expected_config["training_seed_base"])
    command = build_sumo_command(
        gui=False, seed=VALIDATION_SEEDS[0], steps=expected_config["episode_seconds"],
    )
    with sumo_connection(command) as connection:
        for index, checkpoint in enumerate(checkpoints):
            agent, state = (first_agent, first_state) if index == 0 else load_policy(checkpoint)
            config = state["configuration"]
            validate_seed_sets(config["episodes"], config["training_seed_base"])
            if config != expected_config:
                raise ValueError("checkpoint configurations differ")
            for seed in VALIDATION_SEEDS:
                result = evaluate_one(agent, seed, config["episode_seconds"], connection=connection)
                rows.append({"checkpoint": checkpoint.name, "episode": state["episode"], **result})
            write_csv(output_dir / "validation" / "per_seed.csv", FIELDS, rows)
    scores = summarize_checkpoints(rows)
    write_csv(output_dir / "validation" / "checkpoint_summary.csv", tuple(scores[0]), scores)
    best = choose_best(rows)
    write_csv(output_dir / "validation" / "best_checkpoint.csv", tuple(best), [best])
    print(f"Best checkpoint: {best['checkpoint']} | mean wait: {best['mean_waiting_time']:.3f}s")
    return best


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the extended-run output directory to validate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    validate(parse_args().output_dir)
