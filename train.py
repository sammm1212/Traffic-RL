"""Train the prototype DQN controller and record episode-level metrics."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile
from typing import Sequence

import torch

from src.simulation.metrics import SimulationSummary


NUM_EPISODES = 100
BATCH_SIZE = 64
REPLAY_CAPACITY = 10_000
TARGET_UPDATE_FREQUENCY = 500

EPISODE_SECONDS = 300
SEED = 0
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "results" / "training" / "training_metrics.csv"
)
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "results" / "training" / "dqn_100_episode.pt"
)


@dataclass(frozen=True)
class EpisodeMetrics:
    """One CSV-ready record of learning and traffic performance."""

    episode: int
    total_reward: float
    mean_loss: float
    epsilon: float
    mean_waiting_time: float
    mean_queue_length: float
    throughput: int

    @classmethod
    def from_episode(
        cls,
        *,
        episode: int,
        total_reward: float,
        losses: Sequence[float],
        epsilon: float,
        traffic: SimulationSummary,
    ) -> "EpisodeMetrics":
        """Combine DQN values with the shared baseline traffic summary."""
        mean_loss = fmean(losses) if losses else 0.0
        numeric_values = (
            total_reward,
            mean_loss,
            epsilon,
            traffic.mean_queueing_delay_per_generated_vehicle,
            traffic.mean_queue_length,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("episode metrics must contain only finite values")
        return cls(
            episode=episode,
            total_reward=total_reward,
            mean_loss=mean_loss,
            epsilon=epsilon,
            # This is the baseline's integrated queueing delay divided by all
            # generated vehicles, retained under the requested CSV column name.
            mean_waiting_time=traffic.mean_queueing_delay_per_generated_vehicle,
            mean_queue_length=traffic.mean_queue_length,
            throughput=traffic.vehicles_completed,
        )


class TrainingMetricsWriter:
    """Write and durably flush one training-metrics row at a time."""

    fieldnames = tuple(EpisodeMetrics.__dataclass_fields__)

    def __init__(self, path: Path) -> None:
        """Create a fresh metrics CSV and write its header."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._output = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._output, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._flush()

    def write(self, metrics: EpisodeMetrics) -> None:
        """Append and flush one completed episode."""
        self._writer.writerow(asdict(metrics))
        self._flush()

    def _flush(self) -> None:
        """Push buffered CSV data to disk for interruption resilience."""
        self._output.flush()
        os.fsync(self._output.fileno())

    def close(self) -> None:
        """Close the output CSV."""
        self._output.close()

    def __enter__(self) -> "TrainingMetricsWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def save_checkpoint(path: Path, agent: object, episode: int) -> None:
    """Atomically save the trained DQN state in the evaluator's format."""
    from src.agents.dqn_agent import ACTION_SIZE, HIDDEN_SIZE, STATE_SIZE

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "policy_net_state_dict": agent.policy_network.state_dict(),
        "target_net_state_dict": agent.target_network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "episode": episode,
        "state_size": STATE_SIZE,
        "action_size": ACTION_SIZE,
        "hidden_size": HIDDEN_SIZE,
    }
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            torch.save(checkpoint, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def print_episode(metrics: EpisodeMetrics, total_episodes: int) -> None:
    """Print a compact learning and traffic summary."""
    print(
        f"Episode {metrics.episode}/{total_episodes} | "
        f"Reward: {metrics.total_reward:.2f} | "
        f"Loss: {metrics.mean_loss:.4f} | "
        f"Epsilon: {metrics.epsilon:.3f}"
    )
    print(
        f"Mean queue: {metrics.mean_queue_length:.2f} | "
        f"Mean wait: {metrics.mean_waiting_time:.2f}s | "
        f"Throughput: {metrics.throughput}",
        flush=True,
    )


def train(
    *,
    num_episodes: int = NUM_EPISODES,
    episode_seconds: int = EPISODE_SECONDS,
    seed: int = SEED,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> None:
    """Run DQN training, record metrics, and save the completed model."""
    import traci

    from src.agents.dqn_agent import DQNAgent
    from src.agents.replay_buffer import ReplayBuffer
    from src.environment.traffic_env import MINIMUM_GREEN_DURATION, TrafficEnvironment
    from src.simulation.run import build_sumo_command
    from src.simulation.traffic import TrafficSimulation

    if num_episodes <= 0:
        raise ValueError("num_episodes must be a positive integer")
    if episode_seconds <= 0:
        raise ValueError("episode_seconds must be a positive integer")

    command = build_sumo_command(gui=False, seed=seed, steps=episode_seconds)
    traci.start(command)

    simulation = TrafficSimulation(traci, reload_args=command[1:])
    environment = TrafficEnvironment(
        simulation=simulation,
        decision_interval=5,
        max_simulation_time=episode_seconds,
        minimum_green_duration=MINIMUM_GREEN_DURATION,
    )
    agent = DQNAgent()
    replay_buffer = ReplayBuffer(capacity=REPLAY_CAPACITY)
    training_step = 0

    try:
        with TrainingMetricsWriter(output_path) as metrics_writer:
            for episode_index in range(num_episodes):
                state, _ = environment.reset()
                terminated = False
                truncated = False
                total_reward = 0.0
                episode_losses: list[float] = []

                while not (terminated or truncated):
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, _ = environment.step(
                        action
                    )
                    replay_buffer.push(
                        state,
                        action,
                        reward,
                        next_state,
                        terminated or truncated,
                    )
                    state = next_state
                    total_reward += reward

                    if len(replay_buffer) >= BATCH_SIZE:
                        loss = agent.train_step(replay_buffer.sample(BATCH_SIZE))
                        episode_losses.append(loss)
                        training_step += 1
                        if training_step % TARGET_UPDATE_FREQUENCY == 0:
                            agent.update_target_network()

                # Epsilon is decayed once per completed episode, so this is the
                # current value at the episode boundary requested for logging.
                agent.decay_epsilon()
                metrics = EpisodeMetrics.from_episode(
                    episode=episode_index + 1,
                    total_reward=total_reward,
                    losses=episode_losses,
                    epsilon=agent.epsilon,
                    traffic=environment.episode_summary(),
                )
                metrics_writer.write(metrics)
                print_episode(metrics, num_episodes)
        save_checkpoint(checkpoint_path, agent, num_episodes)
    finally:
        # Ensure the external SUMO process is not left running after errors.
        traci.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional training and smoke-test controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--episode-seconds", type=int, default=EPISODE_SECONDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    train(
        num_episodes=args.episodes,
        episode_seconds=args.episode_seconds,
        seed=args.seed,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
