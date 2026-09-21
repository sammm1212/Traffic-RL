import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluate_dqn import (
    CSV_FIELDS,
    EvaluationResult,
    evaluate,
    greedy_action,
    load_agent,
    write_results,
)
from src.agents.dqn_agent import DQNAgent
from src.simulation.metrics import SimulationSummary


def save_checkpoint(path: Path) -> DQNAgent:
    agent = DQNAgent()
    torch.save({"policy_network_state_dict": agent.policy_network.state_dict()}, path)
    return agent


def test_checkpoint_loading_sets_greedy_eval_mode(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "dqn.pt"
    original = save_checkpoint(checkpoint_path)

    loaded = load_agent(checkpoint_path)

    assert loaded.epsilon == 0.0
    assert not loaded.policy_network.training
    assert not loaded.target_network.training
    for expected, actual in zip(
        original.policy_network.parameters(), loaded.policy_network.parameters()
    ):
        assert torch.equal(expected, actual)


def test_result_reuses_training_and_baseline_summary_metrics() -> None:
    traffic = SimulationSummary(
        simulation_time=300.0,
        vehicles_generated=120,
        vehicles_completed=117,
        mean_queue_length=3.7,
        maximum_queue_length=9,
        mean_queueing_delay_per_generated_vehicle=21.4,
        total_waiting_time=2568.0,
    )

    result = EvaluationResult.from_summary(
        seed=100,
        total_reward=-281.0,
        traffic=traffic,
    )

    assert result.mean_waiting_time == 21.4
    assert result.mean_queue_length == 3.7
    assert result.throughput == 117


def test_greedy_action_uses_argmax_without_exploration(monkeypatch) -> None:
    agent = DQNAgent()
    agent.epsilon = 0.0
    agent.policy_network.eval()
    with torch.no_grad():
        for parameter in agent.policy_network.parameters():
            parameter.zero_()
        agent.policy_network.network[-1].bias.copy_(torch.tensor([-1.0, 2.0]))

    monkeypatch.setattr(
        "random.random",
        lambda: pytest.fail("exploration randomness must not be consulted"),
    )
    assert greedy_action(agent, np.zeros(5, dtype=np.float32)) == 1


def test_evaluate_never_trains_or_decays_epsilon(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "dqn.pt"
    save_checkpoint(checkpoint_path)
    output_path = tmp_path / "evaluation" / "results.csv"

    monkeypatch.setattr(
        DQNAgent,
        "train_step",
        lambda *args, **kwargs: pytest.fail("evaluation must not train"),
    )
    monkeypatch.setattr(
        DQNAgent,
        "decay_epsilon",
        lambda *args, **kwargs: pytest.fail("evaluation must not decay epsilon"),
    )
    monkeypatch.setattr(
        torch.optim.Adam,
        "step",
        lambda *args, **kwargs: pytest.fail("evaluation must not step an optimizer"),
    )

    def fake_episode(seed: int, agent: DQNAgent) -> EvaluationResult:
        assert agent.epsilon == 0.0
        assert not agent.policy_network.training
        return EvaluationResult(seed, -10.0 - seed, 2.0, 3.0, 4)

    results, aggregate = evaluate(
        checkpoint_path=checkpoint_path,
        seeds=[100, 101],
        output_path=output_path,
        episode_runner=fake_episode,
    )

    assert [result.seed for result in results] == [100, 101]
    assert aggregate.number_of_seeds == 2


def test_output_csv_has_required_schema(tmp_path: Path) -> None:
    output_path = tmp_path / "evaluation" / "dqn_100_episode_eval.csv"
    write_results(
        output_path,
        [EvaluationResult(100, -245.0, 9.34, 4.21, 111)],
    )

    with output_path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)

    assert reader.fieldnames == list(CSV_FIELDS)
    assert rows == [
        {
            "seed": "100",
            "total_reward": "-245.0",
            "mean_waiting_time": "9.34",
            "mean_queue_length": "4.21",
            "throughput": "111",
        }
    ]
