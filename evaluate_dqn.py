"""Evaluate a trained DQN greedily on unseen, reproducible SUMO seeds."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from tempfile import NamedTemporaryFile
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent
from src.simulation.metrics import SimulationSummary


ROOT = Path(__file__).resolve().parent
EVAL_SEEDS = list(range(100, 110))
EPISODE_SECONDS = 300
DECISION_INTERVAL = 5
DEFAULT_CHECKPOINT_PATH = ROOT / "results" / "training" / "dqn_100_episode.pt"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "evaluation" / "dqn_100_episode_eval.csv"
CSV_FIELDS = (
    "seed",
    "total_reward",
    "mean_waiting_time",
    "mean_queue_length",
    "throughput",
)


@dataclass(frozen=True)
class EvaluationResult:
    """Traffic and reward metrics for one greedy evaluation episode."""

    seed: int
    total_reward: float
    mean_waiting_time: float
    mean_queue_length: float
    throughput: int

    @classmethod
    def from_summary(
        cls,
        *,
        seed: int,
        total_reward: float,
        traffic: SimulationSummary,
    ) -> "EvaluationResult":
        """Use the same shared metric definitions as training and baselines."""
        return cls(
            seed=seed,
            total_reward=total_reward,
            # Training stores this shared summary value under the same name.
            mean_waiting_time=traffic.mean_queueing_delay_per_generated_vehicle,
            mean_queue_length=traffic.mean_queue_length,
            throughput=traffic.vehicles_completed,
        )


@dataclass(frozen=True)
class AggregateStatistics:
    """Aggregate evaluation metrics across all completed seeds."""

    number_of_seeds: int
    mean_total_reward: float
    std_total_reward: float
    mean_waiting_time: float
    std_waiting_time: float
    mean_queue_length: float
    std_queue_length: float
    mean_throughput: float
    std_throughput: float


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic Torch behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _policy_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Extract a policy state dict from common local checkpoint layouts."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a PyTorch state dictionary")

    for key in (
        "policy_net_state_dict",
        "policy_network_state_dict",
        "policy_state_dict",
        "model_state_dict",
        "state_dict",
    ):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            return candidate

    if checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint
    raise ValueError("checkpoint does not contain a recognized policy state dictionary")


def load_agent(checkpoint_path: Path) -> DQNAgent:
    """Load trained policy weights and lock the agent into greedy inference mode."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DQN checkpoint not found: {checkpoint_path}")

    agent = DQNAgent()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = dict(_policy_state_dict(checkpoint))
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    agent.policy_network.load_state_dict(state_dict, strict=True)
    agent.target_network.load_state_dict(agent.policy_network.state_dict())
    agent.epsilon = 0.0
    agent.policy_network.eval()
    agent.target_network.eval()
    return agent


def greedy_action(agent: DQNAgent, state: np.ndarray) -> int:
    """Return the highest-value action without consulting exploration randomness."""
    if agent.epsilon != 0.0:
        raise RuntimeError("greedy evaluation requires epsilon = 0")
    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        return int(agent.policy_network(state_tensor).argmax(dim=1).item())


def run_dqn_episode(
    seed: int,
    agent: DQNAgent,
    *,
    episode_seconds: int = EPISODE_SECONDS,
) -> EvaluationResult:
    """Run one fresh SUMO episode using only greedy policy inference."""
    import traci

    from src.environment.traffic_env import MINIMUM_GREEN_DURATION, TrafficEnvironment
    from src.simulation.run import build_sumo_command
    from src.simulation.traffic import TrafficSimulation

    command = build_sumo_command(gui=False, seed=seed, steps=episode_seconds)
    started = False
    try:
        traci.start(command)
        started = True
        simulation = TrafficSimulation(traci, reload_args=command[1:])
        environment = TrafficEnvironment(
            simulation=simulation,
            decision_interval=DECISION_INTERVAL,
            max_simulation_time=episode_seconds,
            minimum_green_duration=MINIMUM_GREEN_DURATION,
        )
        state, _ = environment.reset(seed=seed)
        total_reward = 0.0
        while True:
            action = greedy_action(agent, state)
            state, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return EvaluationResult.from_summary(
            seed=seed,
            total_reward=total_reward,
            traffic=environment.episode_summary(),
        )
    finally:
        if started:
            episode_failed = sys.exc_info()[0] is not None
            try:
                traci.close()
            except Exception:
                if not episode_failed:
                    raise


def write_results(path: Path, results: Sequence[EvaluationResult]) -> None:
    """Atomically write completed per-seed results with the required schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def summarize(results: Sequence[EvaluationResult]) -> AggregateStatistics:
    """Compute means and sample standard deviations across evaluation seeds."""
    if not results:
        raise ValueError("at least one evaluation result is required")

    def mean_and_std(values: Sequence[float]) -> tuple[float, float]:
        return fmean(values), stdev(values) if len(values) > 1 else 0.0

    reward_mean, reward_std = mean_and_std([item.total_reward for item in results])
    wait_mean, wait_std = mean_and_std([item.mean_waiting_time for item in results])
    queue_mean, queue_std = mean_and_std([item.mean_queue_length for item in results])
    throughput_mean, throughput_std = mean_and_std(
        [float(item.throughput) for item in results]
    )
    return AggregateStatistics(
        number_of_seeds=len(results),
        mean_total_reward=reward_mean,
        std_total_reward=reward_std,
        mean_waiting_time=wait_mean,
        std_waiting_time=wait_std,
        mean_queue_length=queue_mean,
        std_queue_length=queue_std,
        mean_throughput=throughput_mean,
        std_throughput=throughput_std,
    )


def print_result(result: EvaluationResult) -> None:
    """Print one concise per-seed evaluation line."""
    print(
        f"Seed {result.seed} | Reward: {result.total_reward:.2f} | "
        f"Mean queue: {result.mean_queue_length:.2f} | "
        f"Mean wait: {result.mean_waiting_time:.2f}s | "
        f"Throughput: {result.throughput}",
        flush=True,
    )


def print_summary(summary: AggregateStatistics) -> None:
    """Print aggregate statistics in a compact, readable block."""
    print("\nAggregate summary (sample standard deviation)")
    print(f"Seeds: {summary.number_of_seeds}")
    print(
        f"Total reward: mean={summary.mean_total_reward:.3f}, "
        f"std={summary.std_total_reward:.3f}"
    )
    print(
        f"Waiting time: mean={summary.mean_waiting_time:.3f}s, "
        f"std={summary.std_waiting_time:.3f}s"
    )
    print(
        f"Queue length: mean={summary.mean_queue_length:.3f}, "
        f"std={summary.std_queue_length:.3f}"
    )
    print(
        f"Throughput: mean={summary.mean_throughput:.3f}, "
        f"std={summary.std_throughput:.3f}"
    )


EpisodeRunner = Callable[[int, DQNAgent], EvaluationResult]


def evaluate(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    seeds: Sequence[int] = EVAL_SEEDS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    episode_seconds: int = EPISODE_SECONDS,
    episode_runner: EpisodeRunner | None = None,
) -> tuple[list[EvaluationResult], AggregateStatistics]:
    """Load one policy and evaluate it without any learning or exploration."""
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    if episode_seconds <= 0:
        raise ValueError("episode_seconds must be a positive integer")

    set_deterministic_seed(0)
    agent = load_agent(checkpoint_path)
    runner = episode_runner
    results: list[EvaluationResult] = []
    for seed in seeds:
        set_deterministic_seed(seed)
        if runner is None:
            result = run_dqn_episode(seed, agent, episode_seconds=episode_seconds)
        else:
            result = runner(seed, agent)
        if result.seed != seed:
            raise ValueError(f"episode runner returned seed {result.seed} for seed {seed}")
        results.append(result)
        # Preserve all completed episodes if a later SUMO run is interrupted.
        write_results(output_path, results)
        print_result(result)

    aggregate = summarize(results)
    print_summary(aggregate)
    print(f"Results: {output_path}")
    return results, aggregate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse checkpoint, seed, duration, and output overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    parser.add_argument("--episode-seconds", type=int, default=EPISODE_SECONDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    evaluate(
        checkpoint_path=args.checkpoint,
        seeds=args.seeds,
        episode_seconds=args.episode_seconds,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
