"""Evaluate frozen DQN minimum-green constraints on saved paired SUMO demand."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

from evaluate_dqn import DECISION_INTERVAL, EPISODE_SECONDS, load_agent, set_deterministic_seed
from evaluate_generalisation import _read_csv
from evaluate_paired import CHECKPOINT
from evaluate_signal_allocation import (
    EXPECTED_HASH, SOURCE, _read, _runs, _save, _validate_reproduction,
    _windows, interval, run_episode, summarize_seconds, validate_seconds,
)
from src.experiments.generalisation_demand import SCENARIOS, scheduled_counts
from src.simulation.run import CONFIG_PATH

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results/min_green_sensitivity'
ORIGINAL = ROOT / 'results/signal_allocation/records'
VALUES = (10, 15, 20, 30)
METRICS = ('vehicles_completed', 'mean_waiting_time', 'mean_queue_length',
           'vehicles_remaining', 'transition_seconds', 'executed_direction_changes',
           'ns_green_seconds', 'ew_green_seconds')
DIRECT_METRICS = tuple(m for m in METRICS if m != 'vehicles_remaining')
NETWORK = ROOT / 'simulation/network/intersection.net.xml'
PHASES = ROOT / 'simulation/network/intersection.tll.xml'


def digest(path: Path) -> str:
    """Return the SHA-256 digest of an immutable experiment input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_green_runs(seconds: list[dict], minimum: int) -> None:
    """Require complete actual green runs to meet the selected lower bound."""
    for direction in ('NS', 'EW'):
        runs = _runs(seconds, direction)
        if seconds[-1]['active_green'] == direction:
            runs = runs[:-1]  # The episode can truncate its last green.
        if any(length < minimum for length in runs):
            raise ValueError(f'{direction} green shorter than {minimum}: {runs}')


def prior_row(scenario: str, seed: int, controller: str, route_hash: str) -> dict:
    """Find one verified prior result for a controller and saved route."""
    rows = _read_csv(SOURCE / scenario / 'episodes.csv')
    matches = [r for r in rows if int(r['seed']) == seed and r['controller'] == controller]
    if len(matches) != 1 or matches[0]['demand_sha256'] != route_hash:
        raise ValueError(f'prior result or route mismatch: {scenario}/{seed}/{controller}')
    return matches[0]


def record_paths(output: Path, scenario: str, seed: int, minimum: int) -> tuple[Path, Path, Path]:
    """Locate per-second, per-decision and provenance files for one DQN run."""
    base = output / 'records' / scenario / f'seed_{seed}_min_{minimum}'
    return (base.with_name(base.name + '_seconds.csv'),
            base.with_name(base.name + '_decisions.csv'),
            base.with_name(base.name + '_manifest.json'))


def load_or_run(output: Path, scenario: str, seed: int, minimum: int,
                route: Path, route_hash: str, checkpoint_hash: str, agent,
                *, fresh: bool = False) -> tuple[list[dict], list[dict], str]:
    """Reuse only records with matching provenance, otherwise run SUMO."""
    if minimum == 10 and not fresh:
        base = ORIGINAL / scenario
        seconds = _read(base / f'seed_{seed}_dqn_seconds.csv')
        decisions = _read(base / f'seed_{seed}_dqn_decisions.csv')
        return seconds, decisions, 'signal_allocation'
    seconds_path, decisions_path, manifest_path = record_paths(output, scenario, seed, minimum)
    provenance = {'scenario': scenario, 'seed': seed, 'minimum_green_seconds': minimum,
                  'checkpoint_sha256': checkpoint_hash, 'demand_sha256': route_hash,
                  'episode_seconds': EPISODE_SECONDS, 'decision_interval_seconds': DECISION_INTERVAL,
                  'sumo_config_sha256': digest(CONFIG_PATH), 'network_sha256': digest(NETWORK),
                  'phase_sha256': digest(PHASES)}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(key) != value for key, value in provenance.items()):
            raise ValueError(f'cached run has different settings: {manifest_path}')
        if not seconds_path.exists() or not decisions_path.exists():
            raise ValueError(f'incomplete cached run: {manifest_path}')
        if manifest.get('seconds_sha256') != digest(seconds_path) or manifest.get('decisions_sha256') != digest(decisions_path):
            raise ValueError(f'cached records changed: {manifest_path}')
        return _read(seconds_path), _read(decisions_path), 'local_cache'
    if seconds_path.exists() or decisions_path.exists():
        raise ValueError(f'unverified existing records: {seconds_path}')
    set_deterministic_seed(seed)
    seconds, decisions = run_episode(scenario, seed, 'dqn', route, agent, minimum)
    validate_seconds(seconds)
    validate_green_runs(seconds, minimum)
    _save(seconds_path, seconds)
    _save(decisions_path, decisions)
    manifest_path.write_text(json.dumps({**provenance, 'seconds_sha256': digest(seconds_path),
                                         'decisions_sha256': digest(decisions_path)}, indent=2) + '\n')
    return seconds, decisions, 'new_run'


def fixed_records(scenario: str, seed: int) -> list[dict]:
    """Read the unchanged fixed-time seconds from the prior diagnostic."""
    return _read(ORIGINAL / scenario / f'seed_{seed}_fixed_time_seconds.csv')


def episode_row(seconds: list[dict], decisions: list[dict] | None, old: dict,
                scenario: str, seed: int, controller: str, minimum: int | str,
                route: Path, route_hash: str, source: str) -> dict:
    """Combine prior metrics with governing-phase diagnostics without changing definitions."""
    validate_seconds(seconds)
    signal = summarize_seconds(seconds, decisions)
    if minimum == 10 or controller == 'fixed_time':
        _validate_reproduction(signal, old)
    inserted = signal['ns_inserted'] + signal['ew_inserted']
    completed = signal['vehicles_completed']
    scheduled = sum(scheduled_counts(route).values())
    if not 0 <= completed <= inserted <= scheduled:
        raise ValueError('vehicle inventory is inconsistent')
    queue_seconds = signal['ns_queue_seconds'] + signal['ew_queue_seconds']
    if controller == 'fixed_time' or minimum == 10:
        if abs(queue_seconds / EPISODE_SECONDS - float(old['mean_queue_length'])) > 1e-9:
            raise ValueError('queue metric differs from prior evaluation')
    return {**signal, 'scenario': scenario, 'seed': seed, 'controller': controller,
            'minimum_green_seconds': minimum, 'source': source,
            'demand_file': str(route.resolve()), 'demand_sha256': route_hash,
            'scheduled_vehicles': scheduled, 'inserted_vehicles': inserted,
            'vehicles_remaining': scheduled - completed,
            'vehicles_not_inserted': scheduled - inserted,
            'completion_percentage': 100 * completed / scheduled if scheduled else '',
            'mean_waiting_time': queue_seconds / inserted if inserted else 0.0,
            'mean_queue_length': queue_seconds / EPISODE_SECONDS,
            'total_waiting_vehicle_seconds': queue_seconds}


def comparisons(episodes: list[dict], values: list[int]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Pair controller outcomes by scenario, seed, and metric."""
    lookup = {(r['scenario'], int(r['seed']), r['controller'], str(r['minimum_green_seconds'])): r
              for r in episodes}
    vs_fixed, vs_original = [], []
    for (scenario, seed, controller, minimum), row in lookup.items():
        if controller != 'dqn':
            continue
        fixed = lookup[scenario, seed, 'fixed_time', '']
        if row['demand_sha256'] != fixed['demand_sha256']:
            raise ValueError('paired route mismatch')
        for metric in METRICS:
            left, right = float(row[metric]), float(fixed[metric])
            vs_fixed.append({'scenario': scenario, 'seed': seed, 'minimum_green_seconds': int(minimum),
                             'metric': metric, 'dqn': left, 'fixed_time': right,
                             'dqn_minus_fixed': left - right})
        if int(minimum) != 10 and (scenario, seed, 'dqn', '10') in lookup:
            original = lookup[scenario, seed, 'dqn', '10']
            if row['demand_sha256'] != original['demand_sha256']:
                raise ValueError('original DQN route mismatch')
            for metric in DIRECT_METRICS:
                left, right = float(row[metric]), float(original[metric])
                vs_original.append({'scenario': scenario, 'seed': seed,
                                    'minimum_green_seconds': int(minimum), 'reference_minimum_green_seconds': 10,
                                    'metric': metric, 'alternative_dqn': left, 'original_dqn': right,
                                    'alternative_minus_original': left - right})
    def summaries(pairs: list[dict], delta_key: str, left_key: str, right_key: str) -> list[dict]:
        keys = sorted({(r['scenario'], r['minimum_green_seconds'], r['metric']) for r in pairs})
        result = []
        for scenario, minimum, metric in keys:
            group = [r for r in pairs if (r['scenario'], r['minimum_green_seconds'], r['metric']) == (scenario, minimum, metric)]
            result.append({'scenario': scenario, 'minimum_green_seconds': minimum, 'metric': metric,
                           'dqn_mean': fmean(r[left_key] for r in group),
                           'reference_mean': fmean(r[right_key] for r in group),
                           **interval([r[delta_key] for r in group])})
        return result
    return (vs_fixed, vs_original,
            summaries(vs_fixed, 'dqn_minus_fixed', 'dqn', 'fixed_time'),
            summaries(vs_original, 'alternative_minus_original', 'alternative_dqn', 'original_dqn'))


def save(path: Path, rows: list[dict]) -> None:
    """Write a table only when rows exist."""
    if rows:
        _save(path, rows)


def plot_results(output: Path, episodes: list[dict], windows: list[dict]) -> None:
    """Render presentation figures with separate axes for different units."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = output / 'plots'
    plot_dir.mkdir(exist_ok=True)
    scenarios = list(dict.fromkeys(r['scenario'] for r in episodes))
    values = sorted({int(r['minimum_green_seconds']) for r in episodes if r['controller'] == 'dqn'})
    def mean(scenario: str, minimum: int, metric: str) -> float:
        group = [float(r[metric]) for r in episodes if r['scenario'] == scenario and r['controller'] == 'dqn' and int(r['minimum_green_seconds']) == minimum]
        return fmean(group)
    def panels(filename: str, metrics: tuple[str, ...], labels: tuple[str, ...], title: str,
               fixed: bool = False) -> None:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.7 * len(metrics)), squeeze=False)
        for ax, metric, label in zip(axes[:, 0], metrics, labels):
            for scenario in scenarios:
                scenario_values = [v for v in values if any(r['scenario'] == scenario and r['controller'] == 'dqn'
                                    and int(r['minimum_green_seconds']) == v for r in episodes)]
                series = [[float(r[metric]) for r in episodes if r['scenario'] == scenario
                           and r['controller'] == 'dqn' and int(r['minimum_green_seconds']) == v]
                          for v in scenario_values]
                means = [fmean(items) for items in series]
                if all(len(items) > 1 for items in series):
                    estimates = [interval(items) for items in series]
                    errors = [[center - estimate['ci_95_lower'] for center, estimate in zip(means, estimates)],
                              [estimate['ci_95_upper'] - center for center, estimate in zip(means, estimates)]]
                    line = ax.errorbar(scenario_values, means, yerr=errors, marker='o', capsize=3, label=scenario)
                    color = line[0].get_color()
                else:
                    color = ax.plot(scenario_values, means, marker='o', label=scenario)[0].get_color()
                if fixed:
                    reference = [float(r[metric]) for r in episodes if r['scenario'] == scenario and r['controller'] == 'fixed_time']
                    ax.axhline(fmean(reference), linestyle=':', alpha=.6, color=color)
            ax.set(xlabel='Minimum green (seconds)', ylabel=label)
            ax.set_xticks(values)
            ax.legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)
    panels('performance.png', ('vehicles_completed',), ('Completed vehicles',), 'Throughput versus minimum green; dotted lines are fixed time', True)
    panels('waiting_and_queue.png', ('mean_waiting_time', 'mean_queue_length'),
           ('Mean waiting measure (seconds per inserted vehicle)', 'Mean total queue (vehicles)'), 'Traffic delay versus minimum green')
    panels('switching.png', ('executed_direction_changes', 'transition_seconds', 'transition_total_time_share'),
           ('Executed changes', 'Transition seconds', 'Transition share of 300 seconds'), 'Switching and lost green time')
    panels('allocation.png', ('ns_green_seconds', 'ew_green_seconds', 'ew_green_share'),
           ('NS green seconds', 'EW green seconds', 'EW share of available green'), 'Directional green allocation')
    if windows:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, metric, label in zip(axes.flat, ('ns_green_seconds', 'ew_green_seconds', 'mean_ns_queue', 'mean_ew_queue'),
                                     ('NS green seconds', 'EW green seconds', 'Mean NS queue (vehicles)', 'Mean EW queue (vehicles)')):
            for minimum in values:
                group = [r for r in windows if int(r['minimum_green_seconds']) == minimum]
                if group:
                    ax.plot((1, 2, 3), [fmean(float(r[metric]) for r in group if int(r['window']) == w) for w in (1, 2, 3)],
                            marker='o', label=f'{minimum} s')
            ax.set(xlabel='Demand window (100 seconds)', ylabel=label, xticks=(1, 2, 3))
            ax.legend(fontsize=8)
        fig.suptitle('Changing demand: allocation and observed queues')
        fig.tight_layout()
        fig.savefig(plot_dir / 'changing_response.png', dpi=180)
        plt.close(fig)


def report(output: Path, episodes: list[dict], fixed_summary: list[dict], direct_summary: list[dict], windows: list[dict]) -> None:
    """Write a concise empirical report from completed paired runs."""
    settings = json.loads((output / 'config.json').read_text())
    scenarios = list(dict.fromkeys(r['scenario'] for r in episodes))
    values = sorted({int(r['minimum_green_seconds']) for r in episodes if r['controller'] == 'dqn'})
    lines = ['# Frozen DQN minimum-green sensitivity', '',
             'Exploratory controller-development experiment on saved seeds. Each point uses a 300-second episode and matched route file. The policy checkpoint was trained under the original 10-second constraint.', '',
             f'Checkpoint: `{settings["checkpoint"]}`. Verified SHA-256: `{EXPECTED_HASH}`. Selected development seeds: {min(settings["seeds"])}–{max(settings["seeds"])} ({len(settings["seeds"])} seeds).', '',
             'The environment guard counts completed five-second green-hold decisions and resets after a switch decision. The switch decision contains 3 yellow seconds, 1 all-red second, and 1 new green second. That first green second is omitted from the guard, so actual complete green runs may exceed the requested minimum. Yellow and all-red do not count as green.', '',
             'Waiting time is total per-second halted-vehicle counts divided by inserted vehicles; mean queue is the mean per-second total halted count. Completion percentage uses scheduled vehicles as denominator. These are the previous evaluation definitions.', '',
             '## Paired results versus fixed time', '',
             '| Scenario | Minimum (s) | DQN completed | Fixed completed | Difference (95% CI) | DQN transition (s) | Fixed transition (s) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def find(rows, scenario, minimum, metric):
        return next((r for r in rows if r['scenario'] == scenario and int(r['minimum_green_seconds']) == minimum and r['metric'] == metric), None)
    def ci(row):
        return f"{row['mean']:+.2f} [{row['ci_95_lower']:+.2f}, {row['ci_95_upper']:+.2f}]" if row['n'] > 1 else f"{row['mean']:+.2f} (n=1)"
    for scenario in scenarios:
        for minimum in values:
            completed = find(fixed_summary, scenario, minimum, 'vehicles_completed')
            transition = find(fixed_summary, scenario, minimum, 'transition_seconds')
            if completed is None or transition is None:
                continue
            lines.append(f"| {scenario} | {minimum} | {completed['dqn_mean']:.2f} | {completed['reference_mean']:.2f} | {ci(completed)} | {transition['dqn_mean']:.2f} | {transition['reference_mean']:.2f} |")
    lines += ['', '## Direct effect versus 10-second DQN', '',
              '| Scenario | Minimum (s) | Completed difference (95% CI) | Transition difference (95% CI) | Changes difference (95% CI) |',
              '|---|---:|---:|---:|---:|']
    for scenario in scenarios:
        for minimum in values:
            if minimum == 10:
                continue
            group = [find(direct_summary, scenario, minimum, metric) for metric in ('vehicles_completed', 'transition_seconds', 'executed_direction_changes')]
            if all(group):
                lines.append(f'| {scenario} | {minimum} | {ci(group[0])} | {ci(group[1])} | {ci(group[2])} |')
    lines += ['', '## Waiting, queues, and direction', '',
              '| Scenario | Minimum (s) | DQN wait (s/inserted) | DQN queue (vehicles) | NS green (s) | EW green (s) | EW green share |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for scenario in scenarios:
        for minimum in values:
            group = [r for r in episodes if r['scenario'] == scenario and r['controller'] == 'dqn'
                     and int(r['minimum_green_seconds']) == minimum]
            if not group:
                continue
            avg = lambda key: fmean(float(r[key]) for r in group)
            lines.append(f'| {scenario} | {minimum} | {avg("mean_waiting_time"):.2f} | '
                         f'{avg("mean_queue_length"):.2f} | {avg("ns_green_seconds"):.2f} | '
                         f'{avg("ew_green_seconds"):.2f} | {avg("ew_green_share"):.3f} |')
    if windows:
        lines += ['', '## Changing-demand response', '',
                  '| Minimum (s) | Window 2 EW green share | Window 3 EW green share | Within-seed share change | Window 3 NS queue | Window 3 EW queue |',
                  '|---:|---:|---:|---:|---:|---:|']
        for minimum in values:
            second = [r for r in windows if int(r['minimum_green_seconds']) == minimum and int(r['window']) == 2]
            third = [r for r in windows if int(r['minimum_green_seconds']) == minimum and int(r['window']) == 3]
            if second and third:
                avg = lambda group, key: fmean(float(r[key]) for r in group)
                lines.append(f'| {minimum} | {avg(second, "ew_green_share"):.3f} | '
                             f'{avg(third, "ew_green_share"):.3f} | '
                             f'{avg(third, "ew_share_change_from_window_2"):+.3f} | '
                             f'{avg(third, "mean_ns_queue"):.2f} | {avg(third, "mean_ew_queue"):.2f} |')
    if 10 in values and len(values) > 1:
        lines += ['', '## Changes from the original constraint', '',
                  'The table gives paired mean differences for the longest tested duration versus 10 seconds. Positive completion means more vehicles finished; negative transition time means less time was spent in yellow or all-red.', '',
                  '| Scenario | Completed Δ | Waiting Δ (s/inserted) | Queue Δ (vehicles) | Changes Δ | Transition Δ (s) | EW green Δ (s) |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        longest = max(values)
        for scenario in scenarios:
            differences = [find(direct_summary, scenario, longest, metric) for metric in
                           ('vehicles_completed', 'mean_waiting_time', 'mean_queue_length',
                            'executed_direction_changes', 'transition_seconds', 'ew_green_seconds')]
            if all(differences):
                lines.append(f'| {scenario} | ' + ' | '.join(f'{r["mean"]:+.2f}' for r in differences) + ' |')
        lines += ['', 'A lower switch count usually releases some transition seconds for green time, but the agent may distribute those seconds differently between NS and EW. The directional changes above and the observed queue states should be used to interpret any throughput gain or loss. A throughput change cannot be attributed to transition reduction alone without a separate intervention that isolates it.', '',
                  'Differences across scenarios and any non-monotonic outcomes remain unexplained by the minimum-green constraint alone. The frozen policy receives no elapsed-green feature, and its action choices respond to the queues and phase states produced by earlier decisions. Longer holds can reduce switching while delaying a response to a queue. The changing-demand window table describes that trade-off; it does not establish an optimal response at second 200 because NS queues may persist from the preceding window.']
    if (30 in values and set(scenarios) == set(SCENARIOS)
            and all(find(fixed_summary, scenario, 30, 'vehicles_completed') is not None
                    and find(fixed_summary, scenario, 10, 'vehicles_completed') is not None
                    for scenario in scenarios)):
        lines += ['', 'At 30 seconds, the DQN averages about eight executed changes and 32 transition seconds in each scenario, close to fixed time. Its mean allocation is approximately 144 NS and 124 EW green seconds across scenarios. This indicates that the longer guard strongly constrains when this frozen policy can act, even though a switch is never forced when the minimum expires. Throughput is not uniformly better: compared with the 10-second DQN it rises in EW-heavy and changing demand but falls in balanced and NS-heavy demand. At 30 seconds, mean waiting and queue length are higher than at 10 seconds in all four scenarios.', '',
                  'In changing demand, the 30-second configuration has a mean within-seed EW green-share change of -0.186 from the NS-heavy to EW-heavy window, while the 10-second configuration has +0.034. Window-3 NS and EW queues are also higher at 30 seconds. This is consistent with reduced adaptation to the demand shift, but starting queues and phase timing matter; the window averages alone cannot identify the optimal response.']
    lines += ['', '## Interpretation', '',
              'This intervention changes the minimum-green guard for a frozen policy while preserving routes, observations, weights, decisions and transition phases. Paired differences estimate its effect under these simulation conditions. Changes in transition time alone do not isolate a causal mechanism for throughput changes. Inspect waiting, queue and directional allocation summaries alongside completion.', '',
              'The 3000–3029 seeds are a development set. The changing-demand windows are balanced (0–99), NS-heavy (100–199), and EW-heavy (200–299); accumulated queues can make immediate EW allocation inappropriate. The window file records starting and ending directional queues and within-seed EW share change from window 2 to 3.', '',
              'A new minimum-green setting selected from these exploratory comparisons needs evaluation on previously unused seeds before final performance claims. The frozen policy has not learned a new policy under longer constraints.', '',
              f'Completed DQN episodes: {sum(r["controller"] == "dqn" for r in episodes)}; reused fixed-time episodes: {sum(r["controller"] == "fixed_time" for r in episodes)}.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


def evaluate(scenarios: list[str], seeds: list[int], values: list[int], output: Path,
             checkpoint: Path, *, fresh_reference: bool = False) -> None:
    """Verify immutable inputs, run missing episodes, and rebuild analyses."""
    if not scenarios or any(s not in SCENARIOS for s in scenarios) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('select known scenarios and distinct seeds')
    if not values or any(v <= 0 for v in values) or len(set(values)) != len(values):
        raise ValueError('select distinct positive minimum-green durations')
    if len(seeds) > 30:
        raise ValueError('paired t utility supports at most 30 seeds')
    checkpoint_hash = digest(checkpoint)
    if checkpoint_hash != EXPECTED_HASH:
        raise ValueError(f'checkpoint SHA-256 mismatch: {checkpoint_hash}')
    baseline_config = json.loads((SOURCE / 'config.json').read_text())
    if baseline_config['checkpoint_sha256'] != checkpoint_hash or baseline_config['episode_seconds'] != EPISODE_SECONDS or baseline_config['decision_interval_seconds'] != DECISION_INTERVAL:
        raise ValueError('prior experiment settings differ')
    diagnostic_config = json.loads((ROOT / 'results/signal_allocation/config.json').read_text())
    if diagnostic_config['checkpoint_sha256'] != checkpoint_hash or diagnostic_config['sumo_config_sha256'] != digest(CONFIG_PATH):
        raise ValueError('prior diagnostic settings differ')
    demand = {}
    for scenario in scenarios:
        for seed in seeds:
            route = SOURCE / scenario / 'demand' / f'seed_{seed}.rou.xml'
            route_hash = digest(route)
            prior_row(scenario, seed, 'dqn', route_hash)
            prior_row(scenario, seed, 'fixed_time', route_hash)
            if diagnostic_config['demand'][f'{scenario}/{seed}']['sha256'] != route_hash:
                raise ValueError(f'prior diagnostic route differs: {route}')
            demand[f'{scenario}/{seed}'] = {'path': str(route.resolve()), 'sha256': route_hash}
    common = {'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': checkpoint_hash,
              'episode_seconds': EPISODE_SECONDS, 'decision_interval_seconds': DECISION_INTERVAL,
              'sumo_config_sha256': digest(CONFIG_PATH), 'network_sha256': digest(NETWORK),
              'phase_sha256': digest(PHASES), 'source': str(SOURCE.resolve())}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if any(old.get(k) != v for k, v in common.items()):
            raise ValueError('output directory has incompatible experiment settings')
    else:
        old = {}
    combined_scenarios = list(dict.fromkeys(old.get('scenarios', []) + scenarios))
    combined_seeds = sorted(set(old.get('seeds', []) + seeds))
    combined_values = sorted(set(old.get('minimum_green_values', []) + values))
    config = {**common, 'scenarios': combined_scenarios, 'seeds': combined_seeds,
              'minimum_green_values': combined_values,
              'scenario_shares': {s: [list(w) for w in SCENARIOS[s].windows] for s in combined_scenarios},
              'demand': {**old.get('demand', {}), **demand},
              'fixed_time_source': 'generalisation and signal_allocation records',
              'metric_definitions': baseline_config['metric_definitions'],
              'record_timing': 'phase before SUMO step; queue and vehicle events after step'}
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    set_deterministic_seed(0)
    agent = load_agent(checkpoint)
    if agent.epsilon != 0 or agent.policy_network.training:
        raise RuntimeError('checkpoint is not in greedy inference mode')
    episodes, windows = [], []
    for scenario in combined_scenarios:
        for seed in combined_seeds:
            key = f'{scenario}/{seed}'
            if key not in config['demand']:
                continue
            route = Path(config['demand'][key]['path'])
            route_hash = digest(route)
            if route_hash != config['demand'][key]['sha256']:
                raise ValueError(f'demand changed: {route}')
            old_fixed = prior_row(scenario, seed, 'fixed_time', route_hash)
            fixed = episode_row(fixed_records(scenario, seed), None, old_fixed, scenario, seed,
                                'fixed_time', '', route, route_hash, 'signal_allocation')
            episodes.append(fixed)
            for minimum in combined_values:
                seconds_path, decisions_path, manifest = record_paths(output, scenario, seed, minimum)
                if minimum != 10 and not manifest.exists() and (scenario not in scenarios or seed not in seeds or minimum not in values):
                    continue
                seconds, decisions, source = load_or_run(output, scenario, seed, minimum, route,
                                                         route_hash, checkpoint_hash, agent,
                                                         fresh=fresh_reference and minimum == 10 and scenario in scenarios and seed in seeds)
                validate_seconds(seconds)
                validate_green_runs(seconds, minimum)
                if len(decisions) != EPISODE_SECONDS // DECISION_INTERVAL:
                    raise ValueError('missing decisions')
                for d in decisions:
                    if int(d['switch_eligible']) != int(int(d['environment_green_elapsed_seconds']) >= minimum):
                        raise ValueError('recorded minimum-green eligibility differs')
                    if int(d['switch_blocked']) and int(d['executed_switch']):
                        raise ValueError('blocked request executed')
                old_dqn = prior_row(scenario, seed, 'dqn', route_hash)
                row = episode_row(seconds, decisions, old_dqn, scenario, seed, 'dqn', minimum,
                                  route, route_hash, source)
                episodes.append(row)
                if scenario == 'changing':
                    for w in _windows(seconds):
                        w.update(minimum_green_seconds=minimum,
                                 ew_share_change_from_window_2='')
                        windows.append(w)
                print(f'Verified {scenario}/{seed}/min={minimum} ({source})', flush=True)
    # Window 2-to-3 response is a within-seed difference, repeated on the third-window row.
    by_window = {(int(w['seed']), int(w['minimum_green_seconds']), int(w['window'])): w for w in windows}
    for (seed, minimum, index), row in by_window.items():
        if index == 3:
            row['ew_share_change_from_window_2'] = float(row['ew_green_share']) - float(by_window[seed, minimum, 2]['ew_green_share'])
    fixed_pairs, direct_pairs, fixed_summary, direct_summary = comparisons(episodes, combined_values)
    save(output / 'episodes.csv', episodes)
    save(output / 'paired_vs_fixed.csv', fixed_pairs)
    save(output / 'paired_vs_original_dqn.csv', direct_pairs)
    save(output / 'summary_vs_fixed.csv', fixed_summary)
    save(output / 'summary_vs_original_dqn.csv', direct_summary)
    save(output / 'changing_window_summary.csv', windows)
    plot_results(output, episodes, windows)
    report(output, episodes, fixed_summary, direct_summary, windows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse sensitivity experiment selections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-green-values', nargs='+', type=int, default=list(VALUES))
    parser.add_argument('--scenario', action='append', choices=tuple(SCENARIOS))
    parser.add_argument('--seed-start', type=int, default=3000)
    parser.add_argument('--seed-count', type=int, default=30)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--fresh-reference', action='store_true', help='replay 10-second reference for validation')
    args = parser.parse_args(argv)
    if args.seed_count < 1:
        parser.error('seed-count must be positive')
    return args


def main() -> None:
    """Run the selected frozen-policy sensitivity experiment."""
    args = parse_args()
    evaluate(args.scenario or list(SCENARIOS), list(range(args.seed_start, args.seed_start + args.seed_count)),
             args.min_green_values, args.output_dir, args.checkpoint, fresh_reference=args.fresh_reference)


if __name__ == '__main__':
    main()
