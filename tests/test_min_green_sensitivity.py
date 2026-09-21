"""Focused checks for the frozen-policy sensitivity runner."""

from pathlib import Path
from unittest.mock import patch

import pytest

from evaluate_min_green_sensitivity import (
    comparisons, load_or_run, parse_args, plot_results, validate_green_runs,
)
from evaluate_signal_allocation import interval
from src.environment.traffic_env import MINIMUM_GREEN_DURATION, TrafficEnvironment
from src.simulation.metrics import ApproachMetrics, TrafficMetrics


def metrics(phase=0, time=0):
    approaches = {name: ApproachMetrics(0, 0, 0.0) for name in ('north', 'south', 'east', 'west')}
    return TrafficMetrics(time, approaches, phase, 0, 0, 0)


class FakeSimulation:
    def __init__(self):
        self.time = 0
        self.phase = 0
        self.actions = []

    def reset(self):
        self.time = 0
        self.phase = 0

    def observe(self):
        return metrics(self.phase, self.time)

    def apply_action(self, action, interval, on_step=None):
        self.actions.append(action)
        self.time += interval
        self.phase = 0 if action == 0 else 3


@pytest.mark.parametrize('minimum,first_switch_decision', [(10, 3), (15, 4), (20, 5), (30, 7)])
def test_guard_applies_selected_duration_at_five_second_decisions(minimum, first_switch_decision):
    simulation = FakeSimulation()
    env = TrafficEnvironment(simulation, minimum_green_duration=minimum)
    env.reset()
    for decision in range(1, first_switch_decision + 1):
        _, _, _, _, info = env.step(1)
        assert info['effective_action'] == (1 if decision == first_switch_decision else 0)
        assert info['switch_blocked'] == (decision < first_switch_decision)
    assert simulation.actions[-1] == 1
    assert env._green_elapsed_seconds == 0
    for _ in range(minimum // 5):
        _, _, _, _, info = env.step(0)
        assert info['switch_blocked']
    _, _, _, _, info = env.step(0)
    assert not info['switch_blocked']
    assert info['effective_action'] == 0


def test_default_constant_and_cli():
    assert MINIMUM_GREEN_DURATION == 10
    args = parse_args(['--min-green-values', '15', '30', '--scenario', 'ew_heavy', '--seed-count', '1'])
    assert args.min_green_values == [15, 30]
    assert args.scenario == ['ew_heavy']
    assert args.seed_count == 1
    assert parse_args([]).min_green_values == [10, 15, 20, 30]


def test_complete_green_runs_only_and_no_transition_counting():
    seconds = ([{'active_green': 'NS'}] * 10 + [{'active_green': ''}] * 4 +
               [{'active_green': 'EW'}] * 11 + [{'active_green': ''}] * 4 +
               [{'active_green': 'NS'}] * 3)
    validate_green_runs(seconds, 10)
    with pytest.raises(ValueError, match='shorter'):
        validate_green_runs(seconds, 11)
    # The final three-second green is episode-truncated and exempt.
    assert seconds[-1]['active_green'] == 'NS'


def test_pairing_uses_matching_seed_and_route():
    base = {'scenario': 'balanced', 'seed': 3000, 'demand_sha256': 'same'}
    metrics = {'vehicles_completed': 10, 'mean_waiting_time': 2, 'mean_queue_length': 3,
               'vehicles_remaining': 4, 'transition_seconds': 20,
               'executed_direction_changes': 5, 'ns_green_seconds': 140, 'ew_green_seconds': 140}
    fixed = {**base, **metrics, 'controller': 'fixed_time', 'minimum_green_seconds': ''}
    original = {**base, **metrics, 'controller': 'dqn', 'minimum_green_seconds': 10}
    alternative = {**base, **metrics, 'vehicles_completed': 12, 'controller': 'dqn', 'minimum_green_seconds': 15}
    vs_fixed, direct, fixed_summary, direct_summary = comparisons([fixed, original, alternative], [10, 15])
    assert next(r for r in direct if r['metric'] == 'vehicles_completed')['alternative_minus_original'] == 2
    assert next(r for r in fixed_summary if r['minimum_green_seconds'] == 15 and r['metric'] == 'vehicles_completed')['n'] == 1
    assert next(r for r in direct_summary if r['metric'] == 'vehicles_completed')['mean'] == 2
    assert len(vs_fixed) == 16
    with pytest.raises(ValueError, match='route mismatch'):
        comparisons([fixed, original, {**alternative, 'demand_sha256': 'other'}], [10, 15])


def test_paired_t_interval():
    result = interval([1.0, 3.0])
    assert result['mean'] == 2.0
    assert result['sample_sd'] == pytest.approx(2**0.5)
    assert result['ci_95_lower'] < 0 < result['ci_95_upper']


def test_incompatible_cache_is_rejected(tmp_path: Path):
    seconds = tmp_path / 'records/balanced/seed_3000_min_15_seconds.csv'
    decisions = seconds.with_name('seed_3000_min_15_decisions.csv')
    manifest = seconds.with_name('seed_3000_min_15_manifest.json')
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    seconds.write_text('corrupt')
    decisions.write_text('corrupt')
    with patch('evaluate_min_green_sensitivity.digest', return_value='expected'):
        with pytest.raises(ValueError, match='different settings'):
            load_or_run(tmp_path, 'balanced', 3000, 15, tmp_path / 'route.xml', 'routehash', 'modelhash', None)


def test_single_seed_plots_without_confidence_interval(tmp_path: Path):
    values = {'scenario': 'balanced', 'seed': 3000, 'vehicles_completed': 10,
              'mean_waiting_time': 2.0, 'mean_queue_length': 3.0,
              'executed_direction_changes': 8, 'transition_seconds': 32,
              'transition_total_time_share': 32 / 300,
              'ns_green_seconds': 144, 'ew_green_seconds': 124,
              'ew_green_share': 124 / 268}
    rows = [{**values, 'controller': 'dqn', 'minimum_green_seconds': 10},
            {**values, 'controller': 'fixed_time', 'minimum_green_seconds': ''}]
    plot_results(tmp_path, rows, [])
    assert (tmp_path / 'plots/performance.png').is_file()
