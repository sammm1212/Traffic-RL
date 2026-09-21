"""Run an isolated, resumable minimum-green DQN training experiment."""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

import numpy as np
import torch

from src.agents.dqn_agent import (
    ACTION_SIZE, EPSILON_DECAY, EPSILON_END, EPSILON_START, GAMMA,
    HIDDEN_SIZE, LEARNING_RATE, STATE_SIZE, DQNAgent,
)
from src.agents.replay_buffer import ReplayBuffer
from src.environment.traffic_env import MINIMUM_GREEN_DURATION, TrafficEnvironment
from src.simulation.run import build_sumo_command
from src.simulation.traffic import TrafficSimulation
from train import (
    BATCH_SIZE, EPISODE_SECONDS, REPLAY_CAPACITY, TARGET_UPDATE_FREQUENCY,
    EpisodeMetrics, TrainingMetricsWriter, print_episode,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "results" / "extended_training"
DEFAULT_EPISODES = 700
TRAIN_SEED_BASE = 20_000
INITIAL_SEED = 42
CHECKPOINT_INTERVAL = 25
DECISION_INTERVAL = 5
CSV_FIELDS = ("traffic_seed", *TrainingMetricsWriter.fieldnames)


def training_seed(episode: int, base: int = TRAIN_SEED_BASE) -> int:
    """Return a unique SUMO seed for a one-based training episode."""
    if episode < 1 or base < 0 or base + episode >= 2**31:
        raise ValueError("invalid training episode or seed base")
    seed = base + episode
    if seed in range(1000, 1030) or seed in range(2000, 2010) or seed in range(100, 110):
        raise ValueError("training seed overlaps an evaluation seed")
    return seed


def configuration(*, episodes: int, episode_seconds: int, seed_base: int) -> dict:
    """Record all experiment settings that must match on resume."""
    return {
        "episodes": episodes, "episode_seconds": episode_seconds,
        "initial_seed": INITIAL_SEED, "training_seed_base": seed_base,
        "decision_interval": DECISION_INTERVAL,
        "minimum_green_duration": MINIMUM_GREEN_DURATION,
        "batch_size": BATCH_SIZE, "replay_capacity": REPLAY_CAPACITY,
        "target_update_frequency": TARGET_UPDATE_FREQUENCY,
        "state_size": STATE_SIZE, "action_size": ACTION_SIZE,
        "hidden_size": HIDDEN_SIZE, "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END, "epsilon_decay": EPSILON_DECAY,
        "gamma": GAMMA, "learning_rate": LEARNING_RATE,
        "reward": "negative_total_halted_queue",
    }


def seed_learning(seed: int) -> None:
    """Seed the three learning RNGs and enable deterministic Torch operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def save_training_checkpoint(
    path: Path, agent: DQNAgent, replay: ReplayBuffer, *,
    episode: int, training_step: int, config: dict,
) -> None:
    """Atomically persist complete state at a finished episode boundary."""
    state = {
        "format_version": 1, "configuration": config,
        "episode": episode, "training_step": training_step,
        "policy_net_state_dict": agent.policy_network.state_dict(),
        "target_net_state_dict": agent.target_network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "replay_buffer": list(replay.buffer),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
            temporary_path = Path(output.name)
            torch.save(state, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def restore_training_checkpoint(
    path: Path, agent: DQNAgent, replay: ReplayBuffer, config: dict,
) -> tuple[int, int]:
    """Restore a locally created checkpoint and all learning RNG states."""
    # This file contains NumPy arrays; only load checkpoints created by this project.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1 or state.get("configuration") != config:
        raise ValueError("checkpoint configuration does not match this run")
    if not 0 <= state["episode"] <= config["episodes"]:
        raise ValueError("invalid checkpoint episode")
    if len(state["replay_buffer"]) > replay.buffer.maxlen:
        raise ValueError("checkpoint replay buffer exceeds configured capacity")
    agent.policy_network.load_state_dict(state["policy_net_state_dict"])
    agent.target_network.load_state_dict(state["target_net_state_dict"])
    agent.optimizer.load_state_dict(state["optimizer_state_dict"])
    agent.epsilon = state["epsilon"]
    replay.buffer.extend(state["replay_buffer"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    return state["episode"], state["training_step"]


def prepare_metrics(path: Path, completed_episode: int, *, resume: bool) -> None:
    """Initialize the log or trim rows newer than the restored checkpoint."""
    if not resume:
        if path.exists():
            raise FileExistsError(f"training metrics already exist: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output:
            csv.DictWriter(output, fieldnames=CSV_FIELDS).writeheader()
        return
    if not path.is_file():
        raise FileNotFoundError(f"training metrics missing: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("training metrics schema does not match")
        rows = [row for row in reader if int(row["episode"]) <= completed_episode]
    if [int(row["episode"]) for row in rows] != list(range(1, completed_episode + 1)):
        raise ValueError("training metrics do not match checkpoint episode")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_metrics(path: Path, seed: int, metrics: EpisodeMetrics) -> None:
    """Append and flush one episode's traffic and learning metrics."""
    with path.open("a", newline="", encoding="utf-8") as output:
        csv.DictWriter(output, fieldnames=CSV_FIELDS).writerow(
            {"traffic_seed": seed, **asdict(metrics)}
        )
        output.flush()
        os.fsync(output.fileno())


def train_extended(
    *, episodes: int = DEFAULT_EPISODES, episode_seconds: int = EPISODE_SECONDS,
    seed_base: int = TRAIN_SEED_BASE, output_dir: Path = DEFAULT_OUTPUT_DIR,
    resume: Path | None = None,
) -> None:
    """Train from scratch or resume from a saved extended-run checkpoint."""
    import traci

    if episodes <= 0 or episode_seconds <= 0:
        raise ValueError("episodes and episode_seconds must be positive")
    for episode in range(1, episodes + 1):
        training_seed(episode, seed_base)
    output_dir = output_dir.resolve()
    checkpoints = output_dir / "checkpoints"
    metrics_path = output_dir / "training_metrics.csv"
    if resume is None and (output_dir.exists() and any(output_dir.iterdir())):
        raise FileExistsError(f"extended output directory is not empty: {output_dir}")
    if resume is not None and resume.resolve().parent != checkpoints:
        raise ValueError("resume checkpoint must be in this output directory")

    config = configuration(episodes=episodes, episode_seconds=episode_seconds, seed_base=seed_base)
    seed_learning(INITIAL_SEED)
    agent = DQNAgent()
    replay = ReplayBuffer(REPLAY_CAPACITY)
    completed, training_step = (0, 0)
    if resume is not None:
        completed, training_step = restore_training_checkpoint(resume, agent, replay, config)
    prepare_metrics(metrics_path, completed, resume=resume is not None)
    learning_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())

    started = False
    try:
        first_seed = training_seed(completed + 1, seed_base) if completed < episodes else None
        if first_seed is None:
            return
        command = build_sumo_command(gui=False, seed=first_seed, steps=episode_seconds)
        traci.start(command)
        started = True
        random.setstate(learning_rng[0])
        np.random.set_state(learning_rng[1])
        torch.set_rng_state(learning_rng[2])
        simulation = TrafficSimulation(traci, reload_args=command[1:])
        environment = TrafficEnvironment(
            simulation, decision_interval=DECISION_INTERVAL,
            max_simulation_time=episode_seconds,
            minimum_green_duration=MINIMUM_GREEN_DURATION,
        )
        for episode in range(completed + 1, episodes + 1):
            traffic_seed = training_seed(episode, seed_base)
            command = build_sumo_command(gui=False, seed=traffic_seed, steps=episode_seconds)
            simulation.set_reload_args(command[1:])
            state, _ = environment.reset()
            total_reward = 0.0
            losses: list[float] = []
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = environment.step(action)
                replay.push(state, action, reward, next_state, terminated or truncated)
                state = next_state
                total_reward += reward
                if len(replay) >= BATCH_SIZE:
                    losses.append(agent.train_step(replay.sample(BATCH_SIZE)))
                    training_step += 1
                    if training_step % TARGET_UPDATE_FREQUENCY == 0:
                        agent.update_target_network()
            agent.decay_epsilon()
            metrics = EpisodeMetrics.from_episode(
                episode=episode, total_reward=total_reward, losses=losses,
                epsilon=agent.epsilon, traffic=environment.episode_summary(),
            )
            append_metrics(metrics_path, traffic_seed, metrics)
            print_episode(metrics, episodes)
            if episode % CHECKPOINT_INTERVAL == 0 or episode == episodes:
                save_training_checkpoint(
                    checkpoints / f"episode_{episode:04d}.pt", agent, replay,
                    episode=episode, training_step=training_step, config=config,
                )
    finally:
        if started:
            traci.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse extended-run settings without changing the original train CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--episode-seconds", type=int, default=EPISODE_SECONDS)
    parser.add_argument("--seed-base", type=int, default=TRAIN_SEED_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train_extended(episodes=args.episodes, episode_seconds=args.episode_seconds,
                   seed_base=args.seed_base, output_dir=args.output_dir, resume=args.resume)
