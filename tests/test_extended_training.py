"""Focused checks for extended-run isolation and exact resume state."""

import csv
import random

import numpy as np
import pytest
import torch

from src.agents.dqn_agent import DQNAgent
from src.agents.replay_buffer import ReplayBuffer
from train import DEFAULT_CHECKPOINT_PATH, DEFAULT_OUTPUT_PATH, parse_args as original_args
from train_extended import (
    DEFAULT_OUTPUT_DIR, configuration, prepare_metrics, restore_training_checkpoint,
    save_training_checkpoint, training_seed,
)
from validate_extended import VALIDATION_SEEDS, choose_best, validate_seed_sets


def test_seed_ranges_and_original_defaults() -> None:
    seeds = [training_seed(episode) for episode in range(1, 701)]
    assert seeds == list(range(20_001, 20_701))
    assert not set(seeds) & set(VALIDATION_SEEDS)
    assert not set(seeds) & set(range(1000, 1030))
    validate_seed_sets(700, 20_000)
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_sets(700, 999)
    args = original_args([])
    assert args.episodes == 100
    assert args.output == DEFAULT_OUTPUT_PATH
    assert args.checkpoint == DEFAULT_CHECKPOINT_PATH
    assert DEFAULT_OUTPUT_DIR != DEFAULT_OUTPUT_PATH.parent


def test_complete_checkpoint_restores_learning_state(tmp_path) -> None:
    config = configuration(episodes=700, episode_seconds=300, seed_base=20_000)
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    agent = DQNAgent()
    replay = ReplayBuffer(10_000)
    for _ in range(64):
        replay.push(np.array([1, 2, 0, 0, 0], dtype=np.float32), 1, -3.0,
                    np.array([2, 3, 0, 0, 0], dtype=np.float32), False)
    agent.train_step(replay.sample(64))
    agent.epsilon = 0.8
    expected_random = random.getstate()
    expected_numpy = np.random.get_state()
    expected_torch = torch.get_rng_state().clone()
    path = tmp_path / "episode_0025.pt"
    save_training_checkpoint(path, agent, replay, episode=25, training_step=41, config=config)
    expected_weights = [weight.clone() for weight in agent.policy_network.parameters()]

    random.random()
    np.random.random()
    torch.rand(1)
    restored = DQNAgent()
    restored_replay = ReplayBuffer(10_000)
    assert restore_training_checkpoint(path, restored, restored_replay, config) == (25, 41)
    assert restored.epsilon == 0.8
    assert len(restored_replay) == 64
    assert restored.optimizer.state_dict()["state"]
    assert restored.optimizer.state_dict()["state"].keys() == agent.optimizer.state_dict()["state"].keys()
    assert all(torch.equal(a, b) for a, b in zip(expected_weights, restored.policy_network.parameters()))
    assert random.getstate() == expected_random
    assert np.array_equal(np.random.get_state()[1], expected_numpy[1])
    assert torch.equal(torch.get_rng_state(), expected_torch)
    with pytest.raises(ValueError, match="configuration"):
        restore_training_checkpoint(path, DQNAgent(), ReplayBuffer(10_000),
                                    configuration(episodes=700, episode_seconds=60, seed_base=20_000))


def test_resume_trims_unsaved_metrics(tmp_path) -> None:
    path = tmp_path / "training_metrics.csv"
    prepare_metrics(path, 0, resume=False)
    with path.open("a", newline="") as output:
        writer = csv.writer(output)
        for episode in range(1, 28):
            writer.writerow([20_000 + episode, episode, -1, 0, 0.9, 1, 2, 3])
    prepare_metrics(path, 25, resume=True)
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 25
    assert rows[-1]["episode"] == "25"
    with pytest.raises(FileExistsError):
        prepare_metrics(path, 0, resume=False)


def test_checkpoint_selection_uses_validation_wait_not_training_reward() -> None:
    rows = []
    for seed in VALIDATION_SEEDS:
        rows.extend((
            dict(checkpoint="episode_0025.pt", episode=25, seed=seed,
                 mean_waiting_time=8.0, mean_queue_length=2.0,
                 throughput=15, vehicles_remaining=3),
            dict(checkpoint="episode_0050.pt", episode=50, seed=seed,
                 mean_waiting_time=5.0, mean_queue_length=3.0,
                 throughput=10, vehicles_remaining=8),
        ))
    assert choose_best(rows)["checkpoint"] == "episode_0050.pt"
    with pytest.raises(ValueError, match="incomplete"):
        choose_best(rows[:-1])
