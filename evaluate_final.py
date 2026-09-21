"""Run and audit the preregistered independent frozen-policy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

from evaluate_dqn import DECISION_INTERVAL, EPISODE_SECONDS, load_agent, set_deterministic_seed
from evaluate_min_green_sensitivity import digest, validate_green_runs
from evaluate_paired import CHECKPOINT
from evaluate_signal_allocation import _read, _save, interval, run_episode, summarize_seconds, validate_seconds
from src.experiments.generalisation_demand import SCENARIOS, prepare_demand, scheduled_counts

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'results/final_evaluation'
EXPECTED_HASH = 'e820fb9912eab609f7e10e295d01375e944faf655f4bd9c660d7e8cbcb22c4d0'
MINIMA = (10, 15, 20, 30)
CONTROLLERS = ('fixed_time', 'dqn_10', 'dqn_15', 'dqn_20', 'dqn_30')
PRIMARY = ('vehicles_completed', 'mean_waiting_time', 'mean_queue_length')
SUPPORTING = PRIMARY + ('executed_direction_changes', 'transition_seconds', 'ns_green_seconds', 'ew_green_seconds')


def protocol() -> dict:
    """Load the registered protocol and verify its frozen checkpoint."""
    value = json.loads((DEFAULT_OUTPUT / 'protocol.json').read_text())
    if value['primary_candidate']['minimum_green_seconds'] != 15 or value['seeds'] != list(range(4000, 4030)):
        raise ValueError('registered primary or seed range changed')
    if value['checkpoint_sha256'] != EXPECTED_HASH or digest(CHECKPOINT) != EXPECTED_HASH:
        raise ValueError('frozen checkpoint SHA-256 mismatch')
    for name, old_hash in value['source_sha256'].items():
        if digest(ROOT / name) != old_hash:
            raise ValueError(f'preregistered source changed: {name}')
    return value


def selected_seeds(start: int, count: int, smoke: bool) -> list[int]:
    """Keep final seeds reserved and smoke runs separate."""
    seeds = list(range(start, start + count))
    if count < 1 or (smoke and set(seeds) & set(range(4000, 4030))):
        raise ValueError('invalid smoke seed selection')
    if not smoke and not set(seeds) <= set(range(4000, 4030)):
        raise ValueError('final evaluation seeds must be within 4000-4029')
    return seeds


def paths(output: Path, scenario: str, seed: int, controller: str) -> tuple[Path, Path, Path]:
    """Return provenance and record paths for one episode."""
    base = output / 'records' / scenario / f'seed_{seed}_{controller}'
    return (base.with_name(base.name + '_seconds.csv'),
            base.with_name(base.name + '_decisions.csv'),
            base.with_name(base.name + '_manifest.json'))


def checked_episode(output: Path, scenario: str, seed: int, controller: str,
                    route: Path, agent: object) -> dict:
    """Run once or validate and reuse an exactly matching completed record."""
    minimum = int(controller.split('_')[1]) if controller != 'fixed_time' else None
    seconds_path, decisions_path, manifest_path = paths(output, scenario, seed, controller)
    provenance = {'scenario': scenario, 'seed': seed, 'controller': controller,
                  'minimum_green_seconds': minimum, 'checkpoint_sha256': EXPECTED_HASH,
                  'demand_sha256': digest(route), 'episode_seconds': EPISODE_SECONDS,
                  'decision_interval_seconds': DECISION_INTERVAL}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(k) != v for k, v in provenance.items()):
            raise ValueError(f'cached episode configuration differs: {manifest_path}')
        if not seconds_path.exists() or manifest['seconds_sha256'] != digest(seconds_path):
            raise ValueError(f'cached seconds changed: {seconds_path}')
        if minimum is not None and (not decisions_path.exists() or manifest['decisions_sha256'] != digest(decisions_path)):
            raise ValueError(f'cached decisions changed: {decisions_path}')
        seconds = _read(seconds_path)
        decisions = _read(decisions_path) if minimum is not None else None
    else:
        if seconds_path.exists() or decisions_path.exists():
            raise ValueError(f'unverified partial record: {seconds_path}')
        set_deterministic_seed(seed)
        seconds, decisions = run_episode(scenario, seed, 'dqn' if minimum else 'fixed_time',
                                         route, agent if minimum else None, minimum or 10)
        validate_seconds(seconds)
        if minimum is not None:
            validate_green_runs(seconds, minimum)
        _save(seconds_path, seconds)
        if minimum is not None:
            _save(decisions_path, decisions)
        manifest_path.write_text(json.dumps({**provenance, 'seconds_sha256': digest(seconds_path),
                           'decisions_sha256': digest(decisions_path) if minimum is not None else None}, indent=2) + '\n')
    validate_seconds(seconds)
    if minimum is not None:
        validate_green_runs(seconds, minimum)
        if len(decisions) != EPISODE_SECONDS // DECISION_INTERVAL:
            raise ValueError('missing DQN decisions')
        for d in decisions:
            if int(d['switch_eligible']) != int(int(d['environment_green_elapsed_seconds']) >= minimum):
                raise ValueError('minimum-green eligibility differs')
            if int(d['switch_blocked']) and int(d['executed_switch']):
                raise ValueError('blocked switch executed')
    summary = summarize_seconds(seconds, decisions)
    scheduled = scheduled_counts(route)
    total = sum(scheduled.values())
    inserted = summary['ns_inserted'] + summary['ew_inserted']
    completed = summary['vehicles_completed']
    if summary['ns_scheduled'] + summary['ew_scheduled'] != total or not 0 <= completed <= inserted <= total:
        raise ValueError('vehicle inventory does not reconcile')
    if sum(summary[f'{k}_seconds'] for k in ('ns_green', 'ew_green', 'yellow', 'all_red')) != 300:
        raise ValueError('phase durations do not sum to 300')
    queue_seconds = summary['ns_queue_seconds'] + summary['ew_queue_seconds']
    return {**summary, 'scenario': scenario, 'seed': seed, 'controller': controller,
            'minimum_green_seconds': minimum or '', 'demand_file': str(route.resolve()),
            'demand_sha256': digest(route), 'scheduled_vehicles': total, 'inserted_vehicles': inserted,
            'vehicles_remaining': total - completed, 'vehicles_not_inserted': total - inserted,
            'inserted_unfinished': inserted - completed,
            'completion_percentage': 100 * completed / total if total else 0,
            'mean_waiting_time': queue_seconds / inserted if inserted else 0,
            'mean_queue_length': queue_seconds / EPISODE_SECONDS,
            'total_waiting_vehicle_seconds': queue_seconds, 'episode_seconds': EPISODE_SECONDS,
            'seconds_sha256': digest(seconds_path),
            'decisions_sha256': digest(decisions_path) if minimum is not None else ''}


def audit(rows: list[dict], scenarios: list[str], seeds: list[int], complete: bool) -> dict:
    """Check exact episode coverage, matched demand and per-episode accounting."""
    keys = [(r['scenario'], int(r['seed']), r['controller']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate episode')
    expected = {(s, seed, c) for s in scenarios for seed in seeds for c in CONTROLLERS}
    if complete and set(keys) != expected:
        raise ValueError(f'episode coverage incomplete: {len(set(keys))}/{len(expected)}')
    for (scenario, seed), group in _groups(rows).items():
        if {r['demand_sha256'] for r in group} != {digest(Path(group[0]['demand_file']))}:
            raise ValueError(f'route mismatch: {scenario}/{seed}')
        if complete and {r['controller'] for r in group} != set(CONTROLLERS):
            raise ValueError(f'incomplete group: {scenario}/{seed}')
        for row in group:
            scheduled, inserted, completed = (int(row[k]) for k in ('scheduled_vehicles', 'inserted_vehicles', 'vehicles_completed'))
            if not 0 <= completed <= inserted <= scheduled or int(row['vehicles_remaining']) != scheduled - completed:
                raise ValueError('vehicle accounting mismatch')
            if int(row['episode_seconds']) != 300 or sum(int(row[f'{k}_seconds']) for k in ('ns_green','ew_green','yellow','all_red')) != 300:
                raise ValueError('episode duration mismatch')
    return {'status': 'complete' if complete else 'partial', 'completed_episodes': len(rows),
            'expected_episodes': len(expected), 'unique_episode_keys': len(set(keys)),
            'scenario_seed_groups': len(_groups(rows)),
            'seeds_per_scenario': {s: len({seed for scenario, seed in _groups(rows) if scenario == s}) for s in scenarios},
            'controllers_per_complete_group': len(CONTROLLERS),
            'matched_route_hashes': True, 'route_hash_mismatches': 0,
            'checkpoint_sha256': digest(CHECKPOINT), 'episode_seconds': EPISODE_SECONDS,
            'total_audited_phase_seconds': len(rows) * EPISODE_SECONDS,
            'minimum_green_violations': 0, 'duplicate_episode_keys': 0,
            'vehicle_accounting_mismatches': 0, 'all_accounting_checks_passed': True}


def _groups(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row['scenario'], int(row['seed'])), []).append(row)
    return groups


def paired(rows: list[dict], reference: str, choices: tuple[str, ...], metrics: tuple[str, ...], label: str) -> tuple[list[dict], list[dict]]:
    """Calculate seed-matched differences and Student t intervals."""
    pairs = []
    for (scenario, seed), group in _groups(rows).items():
        lookup = {r['controller']: r for r in group}
        for choice in choices:
            if choice not in lookup or reference not in lookup:
                continue
            left, right = lookup[choice], lookup[reference]
            if left['demand_sha256'] != right['demand_sha256']:
                raise ValueError('paired route mismatch')
            for metric in metrics:
                a, b = float(left[metric]), float(right[metric])
                pairs.append({'analysis': label, 'scenario': scenario, 'seed': seed, 'controller': choice,
                              'reference': reference, 'metric': metric, 'controller_value': a,
                              'reference_value': b, 'paired_difference': a-b})
    summaries = []
    for key in sorted({(r['scenario'], r['controller'], r['reference'], r['metric']) for r in pairs}):
        group = [r for r in pairs if (r['scenario'],r['controller'],r['reference'],r['metric']) == key]
        if len({r['seed'] for r in group}) != len(group):
            raise ValueError('duplicate paired seed')
        stats = interval([r['paired_difference'] for r in group])
        summaries.append({'analysis': label, 'scenario': key[0], 'controller': key[1],
                          'reference': key[2], 'metric': key[3],
                          'controller_mean': fmean(r['controller_value'] for r in group),
                          'reference_mean': fmean(r['reference_value'] for r in group), **stats})
    return pairs, summaries


def save_outputs(output: Path, rows: list[dict], scenarios: list[str], seeds: list[int], full: bool,
                 *, report_only: bool = False) -> None:
    """Produce analyses only for a complete full protocol run."""
    full_complete = full and len(rows) == 600
    validation = audit(rows, scenarios, seeds, full_complete)
    (output / 'validation_report.json').write_text(json.dumps(validation, indent=2) + '\n')
    if rows and not report_only:
        _save(output / 'episodes.csv', rows)
    if not full_complete:
        (output / 'report.md').write_text('# Incomplete evaluation\n\nThis subset is not the final independent evaluation. '
            f'{len(rows)} of 600 required controller episodes are available. Resume with the same output directory.\n')
        return
    primary_pairs, primary_summary = paired(rows, 'fixed_time', ('dqn_15',), PRIMARY, 'prespecified_primary')
    support_fixed, support_summary_fixed = paired(rows, 'fixed_time', ('dqn_10','dqn_20','dqn_30'), SUPPORTING, 'supporting_vs_fixed')
    support_15, support_summary_15 = paired(rows, 'dqn_15', ('dqn_10','dqn_20','dqn_30'), SUPPORTING, 'supporting_vs_primary')
    if any(r['n'] != 30 for r in primary_summary + support_summary_fixed + support_summary_15):
        raise ValueError('comparison has fewer than 30 matched seeds')
    for filename, data in (('paired_primary.csv',primary_pairs),('primary_summary.csv',primary_summary),
                           ('paired_supporting.csv',support_fixed+support_15),
                           ('supporting_summary.csv',support_summary_fixed+support_summary_15)):
        _save(output / filename, data)
    make_plots(output, rows, primary_summary, support_summary_fixed)
    make_report(output, rows, primary_summary, support_summary_fixed, support_summary_15)


def make_plots(output: Path, rows: list[dict], primary: list[dict], support: list[dict]) -> None:
    """Render the five requested presentation figures from audited episodes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = output / 'plots'
    plot_dir.mkdir(exist_ok=True)
    scenarios = list(SCENARIOS)
    labels = ['Balanced','NS heavy','EW heavy','Changing']
    def value(s, c, metric):
        return fmean(float(r[metric]) for r in rows if r['scenario']==s and r['controller']==c)
    def save(fig, name):
        fig.tight_layout(); fig.savefig(plot_dir / name, dpi=180); plt.close(fig)
    for metric, name, title in [('vehicles_completed','primary_completed.png','Completed vehicles: DQN 15 s minus fixed time'),
                                ('mean_waiting_time','primary_waiting.png','Waiting measure: DQN 15 s minus fixed time'),
                                ('mean_queue_length','primary_queue.png','Mean queue: DQN 15 s minus fixed time')]:
        fig, ax = plt.subplots(figsize=(9,4.5))
        group = [next(r for r in primary if r['scenario']==s and r['metric']==metric) for s in scenarios]
        means = [r['mean'] for r in group]
        ax.errorbar(range(4), means, yerr=[[m-r['ci_95_lower'] for m,r in zip(means,group)],
                                               [r['ci_95_upper']-m for m,r in zip(means,group)]], fmt='o', capsize=5)
        ax.axhline(0,color='black',linewidth=.8); ax.set_xticks(range(4),labels)
        ax.set_ylabel('Paired difference' + (' (vehicles)' if metric=='vehicles_completed' else ' (seconds)' if metric=='mean_waiting_time' else ' (vehicles)'))
        ax.set_title(title); save(fig,name)
    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    for ax,metric,title in zip(axes,PRIMARY,['Completed vehicles','Waiting (s)','Mean queue']):
        for s,label in zip(scenarios,labels):
            ax.plot([10,15,20,30],[value(s,f'dqn_{m}',metric) for m in MINIMA],marker='o',label=label)
            ax.axhline(value(s,'fixed_time',metric),linestyle=':',alpha=.3)
        ax.set_title(title);ax.set_xlabel('Minimum green (s)');ax.set_xticks(MINIMA)
    axes[0].legend(fontsize=8);fig.suptitle('Supporting sensitivity; dotted lines show fixed time');save(fig,'minimum_green_sensitivity.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,metric,title in zip(axes,['executed_direction_changes','transition_seconds'],['Signal changes','Transition seconds']):
        for i,c in enumerate(CONTROLLERS):
            ax.plot(range(4),[value(s,c,metric) for s in scenarios],marker='o',label=c)
        ax.set_xticks(range(4),labels);ax.set_title(title)
    axes[0].legend(fontsize=8);save(fig,'switching_transition.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,c in zip(axes,['fixed_time','dqn_15']):
        x=range(4);ns=[value(s,c,'ns_green_seconds') for s in scenarios];ew=[value(s,c,'ew_green_seconds') for s in scenarios]
        ax.bar(x,ns,label='NS green');ax.bar(x,ew,bottom=ns,label='EW green')
        ax.set_xticks(range(4),labels);ax.set_title(c);ax.set_ylabel('Seconds')
    axes[0].legend();save(fig,'directional_green_allocation.png')


def make_report(output: Path, rows: list[dict], primary: list[dict], supporting_fixed: list[dict], supporting_15: list[dict]) -> None:
    """Write a report that keeps confirmatory and exploratory comparisons distinct."""
    def fmt(x): return f'{float(x):+.2f}'
    lines=['# Final independent evaluation','','Prespecified primary: frozen DQN with 15-second minimum green versus unchanged fixed time. Independent seeds 4000–4029, 30 matched seeds per scenario, 300-second episodes. All 600 episodes passed the audit.','','Checkpoint SHA-256: `'+EXPECTED_HASH+'`. The same checkpoint was used for every DQN configuration.','','## Prespecified primary comparison','','Differences are DQN 15 s minus fixed time. Intervals are two-sided paired Student t 95% intervals over 30 seed-level differences.','','| Scenario | Metric | Fixed mean | DQN 15 mean | Paired difference (95% CI) | SD of differences | n |','|---|---|---:|---:|---:|---:|---:|']
    for s in SCENARIOS:
        for metric in PRIMARY:
            r=next(x for x in primary if x['scenario']==s and x['metric']==metric)
            lines.append(f"| {s} | {metric} | {float(r['reference_mean']):.2f} | {float(r['controller_mean']):.2f} | {fmt(r['mean'])} [{fmt(r['ci_95_lower'])}, {fmt(r['ci_95_upper'])}] | {float(r['sample_sd']):.2f} | {r['n']} |")
    lines += ['','## Supporting minimum-green sensitivity','','These comparisons are exploratory and do not change the primary candidate. Intervals are unadjusted descriptive intervals, not a joint confirmatory test.','','| Scenario | Configuration | Reference | Metric | Difference (95% CI) |','|---|---|---|---|---:|']
    for r in supporting_fixed + supporting_15:
        lines.append(f"| {r['scenario']} | {r['controller']} | {r['reference']} | {r['metric']} | {fmt(r['mean'])} [{fmt(r['ci_95_lower'])}, {fmt(r['ci_95_upper'])}] |")
    lines += ['','## Interpretation','','The primary table reports throughput, waiting and queue together. Positive completed-vehicle differences favour DQN; negative waiting and queue differences favour DQN. The supporting signal-change, transition-time and directional-green comparisons describe associations and cannot establish a causal mechanism. Results should be interpreted separately for each of the four specified demand profiles.','','The development seeds 3000–3029 selected the 15-second candidate; the independent seeds 4000–4029 were not used to change it. Compare the primary table with `results/min_green_sensitivity/report.md` for the development results.','','Mean waiting time here is integrated halted-vehicle seconds divided by actual inserted vehicles, including vehicles still active at 300 seconds but excluding vehicles blocked from insertion. Completion percentage uses scheduled vehicles as its denominator. A 300-second horizon has no clearance period. The simulation represents one junction with straight-through traffic and four specified stochastic demand profiles; it does not establish performance under arbitrary real-world traffic.','','## Reproduce','','Run `PYTHONPATH=. .venv/bin/python evaluate_final.py` from the repository root. Use `--report-only` to verify stored records and rebuild the report without SUMO.']
    interpretation = output / 'interpretation.md'
    if interpretation.exists():
        lines.extend(['', interpretation.read_text().strip()])
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def evaluate(args: argparse.Namespace) -> None:
    """Verify the protocol, safely resume episodes, and audit results."""
    p=protocol()
    output=args.output_dir.resolve()
    scenarios=args.scenario or list(SCENARIOS)
    seeds=selected_seeds(args.seed_start,args.seed_count,args.smoke)
    full=not args.smoke and scenarios==list(SCENARIOS) and seeds==p['seeds']
    if output==DEFAULT_OUTPUT.resolve() and args.smoke:
        raise ValueError('smoke runs require a separate output directory')
    output.mkdir(parents=True,exist_ok=True)
    config={'protocol_sha256':digest(DEFAULT_OUTPUT/'protocol.json'),'mode':'smoke' if args.smoke else 'final',
            'scenarios':scenarios,'seeds':seeds,'checkpoint_sha256':EXPECTED_HASH,
            'runner_sha256':digest(Path(__file__)),'full_protocol_selection':full}
    config_path=output/'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if any(previous.get(k) != v for k, v in config.items()):
            raise ValueError('output configuration differs; use a new directory')
    if not config_path.exists(): config_path.write_text(json.dumps(config,indent=2)+'\n')
    agent=None
    if not args.report_only:
        set_deterministic_seed(0)
        agent=load_agent(CHECKPOINT)
        if agent.epsilon!=0 or agent.policy_network.training or agent.target_network.training:
            raise ValueError('DQN is not in greedy inference mode')
    rows=[]
    for scenario in scenarios:
        for seed in seeds:
            route=output/'demand'/scenario/f'seed_{seed}.rou.xml'
            if args.report_only:
                if not route.exists(): raise ValueError(f'missing demand: {route}')
                if route.read_bytes()!=__import__('src.experiments.generalisation_demand',fromlist=['demand_bytes']).demand_bytes(SCENARIOS[scenario],seed,300):
                    raise ValueError('saved demand differs from deterministic generation')
            else:
                prepare_demand(route,SCENARIOS[scenario],seed,300)
            for controller in CONTROLLERS:
                if args.report_only and not paths(output,scenario,seed,controller)[2].exists():
                    continue
                try:
                    row=checked_episode(output,scenario,seed,controller,route,agent)
                except Exception as error:
                    failure=output/'failures.jsonl'
                    with failure.open('a') as f: f.write(json.dumps({'scenario':scenario,'seed':seed,'controller':controller,'error':str(error)})+'\n')
                    raise
                rows.append(row)
                print(f'{scenario}/{seed}/{controller}: completed={row["vehicles_completed"]}',flush=True)
    save_outputs(output,rows,scenarios,seeds,full,report_only=args.report_only)


def main() -> None:
    """Parse command-line selections for full, subset, smoke or report runs."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',action='append',choices=list(SCENARIOS))
    parser.add_argument('--seed-start',type=int,default=4000)
    parser.add_argument('--seed-count',type=int,default=30)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    evaluate(args)


if __name__=='__main__': main()
