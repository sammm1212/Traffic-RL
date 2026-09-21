"""Demand, pairing, and output checks without launching SUMO."""

import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate_generalisation as experiment
from src.experiments.generalisation_demand import (
    SCENARIOS, TOTAL_ARRIVAL_RATE, demand_bytes, prepare_demand, scheduled_counts,
)
from src.simulation.metrics import APPROACHES


def test_scenario_rates_and_changing_boundaries() -> None:
    expected = {
        "balanced": (0.12, 0.12, 0.12, 0.12),
        "ns_heavy": (0.168, 0.168, 0.072, 0.072),
        "ew_heavy": (0.072, 0.072, 0.168, 0.168),
    }
    for name, rates in expected.items():
        actual = SCENARIOS[name].probabilities_at(0, 300)
        assert tuple(actual.values()) == pytest.approx(rates)
        assert sum(actual.values()) == pytest.approx(TOTAL_ARRIVAL_RATE)
    changing = SCENARIOS["changing"]
    for second, name in ((0, "balanced"), (99, "balanced"),
                         (100, "ns_heavy"), (199, "ns_heavy"),
                         (200, "ew_heavy"), (299, "ew_heavy")):
        assert changing.probabilities_at(second, 300) == SCENARIOS[name].probabilities_at(0, 300)


def test_demand_is_deterministic_and_existing_files_are_verified(tmp_path: Path) -> None:
    scenario = SCENARIOS["changing"]
    path = tmp_path / "seed_3000.rou.xml"
    counts = prepare_demand(path, scenario, 3000, 300)
    original = path.read_bytes()
    assert original == demand_bytes(scenario, 3000, 300)
    assert counts == scheduled_counts(path)
    assert prepare_demand(path, scenario, 3000, 300) == counts
    assert path.read_bytes() == original
    assert demand_bytes(scenario, 3001, 300) != original
    with pytest.raises(ValueError, match="existing demand differs"):
        prepare_demand(path, scenario, 3001, 300)


def test_paired_difference_and_t_interval() -> None:
    rows = []
    for seed, difference in ((3000, 1), (3001, 2), (3002, 3)):
        for controller, completed in (("fixed_time", 10), ("dqn", 10 + difference)):
            rows.append({"seed": seed, "controller": controller,
                         "demand_sha256": str(seed), "vehicles_completed": completed,
                         "mean_waiting_time": 5, "mean_queue_length": 3,
                         "vehicles_remaining": 20 - completed})
    paired = experiment._paired_rows(rows)
    differences = [row["dqn_minus_fixed"] for row in paired if row["metric"] == "vehicles_completed"]
    assert differences == [1, 2, 3]
    mean, sd, low, high = experiment.paired_interval(differences)
    assert mean == 2 and sd == 1
    assert low == pytest.approx(2 - experiment.T_CRITICAL[2] / 3**0.5)
    assert high == pytest.approx(2 + experiment.T_CRITICAL[2] / 3**0.5)
    assert experiment.paired_interval([1]) is None
    rows[-1]["demand_sha256"] = "different"
    with pytest.raises(ValueError, match="paired demand mismatch"):
        experiment._paired_rows(rows)


def test_evaluator_uses_same_route_and_greedy_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen")
    agent = SimpleNamespace(epsilon=0.0, policy_network=SimpleNamespace(training=False))
    loaded = []
    calls = []
    monkeypatch.setattr(experiment, "load_agent", lambda path: loaded.append(path) or agent)
    monkeypatch.setattr(experiment, "set_deterministic_seed", lambda seed: None)

    def fake_run(*, controller, seed, agent, episode_seconds, route_path):
        calls.append((controller, seed, agent, route_path, hashlib.sha256(route_path.read_bytes()).hexdigest()))
        scheduled = scheduled_counts(route_path)
        total = sum(scheduled.values())
        values = {"vehicles_departed": total, "vehicles_completed": total - 2,
                  "vehicles_generated": total, "mean_waiting_time": 4.0,
                  "mean_queue_length": 2.0, "total_waiting_vehicle_seconds": 4 * total,
                  "signal_changes": 8, "transition_seconds": 32,
                  "ns_green_seconds": 148, "ew_green_seconds": 120}
        for approach in APPROACHES:
            values[f"{approach}_departed"] = scheduled[approach]
            values[f"{approach}_completed"] = scheduled[approach]
        return SimpleNamespace(values=values)

    monkeypatch.setattr(experiment, "_run_episode", fake_run)
    out = tmp_path / "results"
    experiment.evaluate(scenarios=["balanced", "ns_heavy"], seed_start=3000,
                        seed_count=2, output_dir=out, checkpoint=checkpoint)
    assert loaded == [checkpoint]
    assert len(calls) == 8
    for left, right in zip(calls[::2], calls[1::2]):
        assert left[0] == "dqn" and left[2] is agent
        assert right[0] == "fixed_time" and right[2] is None
        assert left[3:] == right[3:]
    assert (out / "balanced/demand/seed_3000.rou.xml").exists()
    assert (out / "ns_heavy/demand/seed_3000.rou.xml").exists()
    with (out / "consolidated_summary.csv").open(newline="") as source:
        assert {row["scenario"] for row in csv.DictReader(source)} == {"balanced", "ns_heavy"}
    before = len(calls)
    experiment.evaluate(scenarios=["balanced", "ns_heavy"], seed_start=3000,
                        seed_count=2, output_dir=out, checkpoint=checkpoint)
    assert len(calls) == before
    agent.epsilon = 0.1
    with pytest.raises(RuntimeError, match="greedy"):
        experiment.evaluate(scenarios=["balanced", "ns_heavy"], seed_start=3000,
                            seed_count=2, output_dir=out, checkpoint=checkpoint)


def test_existing_evaluator_defaults_stay_intact() -> None:
    from evaluate_paired import CHECKPOINT, SEEDS
    assert experiment.CHECKPOINT == CHECKPOINT
    assert list(SEEDS) == list(range(1000, 1030))
    assert experiment.EPISODE_SECONDS == 300
    assert experiment.SEED_START == 3000
