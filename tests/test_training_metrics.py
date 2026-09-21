import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from evaluate_dqn import load_agent
from src.agents.dqn_agent import ACTION_SIZE, HIDDEN_SIZE, STATE_SIZE, DQNAgent
from src.simulation.metrics import SimulationSummary
from train import EpisodeMetrics, TrainingMetricsWriter, save_checkpoint


def summary() -> SimulationSummary:
    return SimulationSummary(
        simulation_time=300.0,
        vehicles_generated=120,
        vehicles_completed=117,
        mean_queue_length=3.7,
        maximum_queue_length=9,
        mean_queueing_delay_per_generated_vehicle=21.4,
        total_waiting_time=2568.0,
    )


def test_episode_metrics_reuse_baseline_summary_and_zero_empty_loss() -> None:
    metrics = EpisodeMetrics.from_episode(
        episode=1,
        total_reward=-281.0,
        losses=[],
        epsilon=0.995,
        traffic=summary(),
    )

    assert metrics.mean_loss == 0.0
    assert metrics.mean_waiting_time == 21.4
    assert metrics.mean_queue_length == 3.7
    assert metrics.throughput == 117


def test_episode_metrics_reject_non_finite_loss() -> None:
    with pytest.raises(ValueError, match="finite"):
        EpisodeMetrics.from_episode(
            episode=1,
            total_reward=-281.0,
            losses=[math.nan],
            epsilon=0.995,
            traffic=summary(),
        )


def test_training_metrics_writer_writes_requested_schema_incrementally() -> None:
    metrics = EpisodeMetrics.from_episode(
        episode=1,
        total_reward=-281.0,
        losses=[0.8, 0.88],
        epsilon=0.995,
        traffic=summary(),
    )

    with TemporaryDirectory() as directory:
        path = Path(directory) / "training" / "training_metrics.csv"
        with TrainingMetricsWriter(path) as writer:
            writer.write(metrics)
            with path.open(encoding="utf-8", newline="") as input_file:
                rows = list(csv.DictReader(input_file))

    assert list(rows[0]) == [
        "episode",
        "total_reward",
        "mean_loss",
        "epsilon",
        "mean_waiting_time",
        "mean_queue_length",
        "throughput",
    ]
    assert rows[0]["mean_loss"] == "0.8400000000000001"
    assert rows[0]["throughput"] == "117"


def test_checkpoint_has_training_state_and_loads_in_evaluator(tmp_path: Path) -> None:
    agent = DQNAgent()
    agent.epsilon = 0.75
    checkpoint_path = tmp_path / "training" / "dqn.pt"

    save_checkpoint(checkpoint_path, agent, episode=3)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert set(checkpoint) == {
        "policy_net_state_dict",
        "target_net_state_dict",
        "optimizer_state_dict",
        "epsilon",
        "episode",
        "state_size",
        "action_size",
        "hidden_size",
    }
    assert checkpoint["episode"] == 3
    assert checkpoint["epsilon"] == 0.75
    assert checkpoint["state_size"] == STATE_SIZE
    assert checkpoint["action_size"] == ACTION_SIZE
    assert checkpoint["hidden_size"] == HIDDEN_SIZE

    loaded = load_agent(checkpoint_path)
    for expected, actual in zip(
        agent.policy_network.parameters(), loaded.policy_network.parameters()
    ):
        assert torch.equal(expected, actual)
