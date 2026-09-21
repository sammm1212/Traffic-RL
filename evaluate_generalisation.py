"""Evaluate the frozen minimum-green DQN against fixed time on paired demand."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
from typing import Sequence
import xml.etree.ElementTree as ET

from evaluate_diagnostics import _run_episode
from evaluate_dqn import DECISION_INTERVAL, EPISODE_SECONDS, load_agent, set_deterministic_seed
from evaluate_paired import CHECKPOINT, T_95_DF_29, _write_csv
from src.environment.traffic_env import MINIMUM_GREEN_DURATION
from src.experiments.generalisation_demand import SCENARIOS, TOTAL_ARRIVAL_RATE, prepare_demand
from src.simulation.metrics import APPROACHES
from src.visualization.replay import load_recording


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results/generalisation"
SEED_START = 3000  # 1000–1029 already informed the earlier minimum-green comparison.
SEED_COUNT = 30
PAIRED_METRICS = ("vehicles_completed", "mean_waiting_time", "mean_queue_length", "vehicles_remaining")
REPORT_METRICS = ("vehicles_completed", "mean_waiting_time", "mean_queue_length")
T_CRITICAL = (0, 12.706204736, 4.30265273, 3.182446305, 2.776445105,
              2.570581836, 2.446911851, 2.364624252, 2.306004135,
              2.262157163, 2.228138852, 2.20098516, 2.17881283,
              2.160368656, 2.144786688, 2.131449546, 2.119905299,
              2.109815578, 2.10092204, 2.093024054, 2.085963447,
              2.079613845, 2.073873068, 2.06865761, 2.063898562,
              2.059538553, 2.055529439, 2.051830516, 2.048407142,
              T_95_DF_29)


def paired_interval(differences: Sequence[float]) -> tuple[float, float, float, float] | None:
    """Return mean, sample SD and 95% t limits for seed-level differences."""
    count = len(differences)
    if count < 2:
        return None
    if count - 1 >= len(T_CRITICAL):
        raise ValueError("paired t intervals support at most 30 seeds")
    mean = fmean(differences)
    sd = stdev(differences)
    half_width = T_CRITICAL[count - 1] * sd / math.sqrt(count)
    return mean, sd, mean - half_width, mean + half_width


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _config(checkpoint: Path, scenarios: Sequence[str], seeds: Sequence[int]) -> dict:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "scenarios": list(scenarios), "seeds": list(seeds),
        "episode_seconds": EPISODE_SECONDS,
        "generation_seconds": EPISODE_SECONDS,
        "decision_interval_seconds": DECISION_INTERVAL,
        "minimum_green_seconds": MINIMUM_GREEN_DURATION,
        "total_expected_arrivals_per_second": TOTAL_ARRIVAL_RATE,
        "scenario_shares": {name: [list(window) for window in SCENARIOS[name].windows] for name in scenarios},
        "scenario_approach_probabilities_per_second": {
            name: [[TOTAL_ARRIVAL_RATE * share for share in window]
                   for window in SCENARIOS[name].windows] for name in scenarios
        },
        "approach_order": list(APPROACHES),
        "demand_method": "independent seeded Bernoulli trials; explicit immutable vehicles reused by both controllers",
        "metric_definitions": {
            "mean_waiting_time": "sum of per-second halted vehicles / inserted vehicles (seconds); includes time for active vehicles",
            "mean_queue_length": "mean per-second total halted vehicles across four incoming lanes",
            "vehicles_remaining": "scheduled vehicles minus completed; includes active and not inserted",
            "completion_percentage": "100 * completed / scheduled vehicles",
        },
    }


def _record(controller: str, scenario: str, seed: int, route_path: Path, scheduled: dict[str, int], agent: object | None) -> dict:
    result = _run_episode(
        controller=controller, seed=seed, agent=agent,
        episode_seconds=EPISODE_SECONDS, route_path=route_path,
    ).values
    total = sum(scheduled.values())
    inserted = int(result["vehicles_departed"])
    completed = int(result["vehicles_completed"])
    if not 0 <= completed <= inserted <= total:
        raise ValueError(f"invalid vehicle inventory for {scenario}/{seed}/{controller}")
    if int(result["vehicles_generated"]) > total:
        raise ValueError("SUMO loaded more vehicles than scheduled")
    row = {
        "scenario": scenario, "seed": seed, "controller": controller,
        "demand_file": str(route_path.resolve()),
        "demand_sha256": hashlib.sha256(route_path.read_bytes()).hexdigest(),
        "scheduled_vehicles": total,
        "inserted_vehicles": inserted,
        "vehicles_completed": completed,
        "vehicles_remaining": total - completed,
        "vehicles_not_inserted": total - inserted,
        "completion_percentage": 100 * completed / total if total else "",
        "mean_waiting_time": result["mean_waiting_time"],
        "mean_queue_length": result["mean_queue_length"],
        "total_waiting_vehicle_seconds": result["total_waiting_vehicle_seconds"],
        "signal_changes": result["signal_changes"],
        "transition_seconds": result["transition_seconds"],
        "ns_green_seconds": result["ns_green_seconds"],
        "ew_green_seconds": result["ew_green_seconds"],
    }
    for approach in APPROACHES:
        row[f"{approach}_scheduled"] = scheduled[approach]
        row[f"{approach}_inserted"] = result[f"{approach}_departed"]
        row[f"{approach}_completed"] = result[f"{approach}_completed"]
    return row


def _paired_rows(rows: Sequence[dict]) -> list[dict]:
    pairs: dict[int, dict[str, dict]] = {}
    for row in rows:
        pairs.setdefault(int(row["seed"]), {})[row["controller"]] = row
    output = []
    for seed, pair in sorted(pairs.items()):
        if set(pair) != {"dqn", "fixed_time"}:
            continue
        if pair["dqn"]["demand_sha256"] != pair["fixed_time"]["demand_sha256"]:
            raise ValueError(f"paired demand mismatch for seed {seed}")
        for metric in PAIRED_METRICS:
            left, right = float(pair["dqn"][metric]), float(pair["fixed_time"][metric])
            output.append({"seed": seed, "metric": metric, "dqn": left,
                           "fixed_time": right, "dqn_minus_fixed": left - right})
    return output


def _summary_rows(rows: Sequence[dict], paired: Sequence[dict]) -> list[dict]:
    output = []
    for metric in PAIRED_METRICS:
        differences = [float(row["dqn_minus_fixed"]) for row in paired if row["metric"] == metric]
        interval = paired_interval(differences)
        for controller in ("fixed_time", "dqn"):
            values = [float(row[metric]) for row in rows if row["controller"] == controller]
            output.append({
                "metric": metric, "controller": controller, "number_of_episodes": len(values),
                "controller_mean": fmean(values) if values else "",
                "controller_sample_sd": stdev(values) if len(values) > 1 else "",
                "number_of_valid_pairs": len(differences),
                "mean_dqn_minus_fixed": interval[0] if interval else (differences[0] if differences else ""),
                "paired_sample_sd": interval[1] if interval else "",
                "paired_ci_95_lower": interval[2] if interval else "",
                "paired_ci_95_upper": interval[3] if interval else "",
            })
    return output


def _write_scenario(path: Path, rows: Sequence[dict]) -> list[dict]:
    paired = _paired_rows(rows)
    summary = _summary_rows(rows, paired)
    if rows:
        _write_csv(path / "episodes.csv", tuple(rows[0]), rows)
    if paired:
        _write_csv(path / "paired_differences.csv", tuple(paired[0]), paired)
    if summary:
        _write_csv(path / "summary.csv", tuple(summary[0]), summary)
    return summary


def _report(output_dir: Path, scenarios: Sequence[str]) -> None:
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    lines = ["# Generalisation experiment", "", "Frozen 100-episode, 10-second minimum-green DQN versus the original 30/3/1/30/3/1-second fixed-time signal. Paired differences are DQN minus fixed time.", "", f"Checkpoint: `{config['checkpoint']}` (SHA-256 `{config['checkpoint_sha256']}`). Seeds: {config['seeds'][0]}–{config['seeds'][-1]}. Each episode lasts {EPISODE_SECONDS} seconds; generation also lasts {EPISODE_SECONDS} seconds, with no separate clearance period.", "", "## Paired outcomes", "", "| Scenario | Fixed completed | DQN completed | Δ completed (95% CI) | Fixed wait (s) | DQN wait (s) | Δ wait (95% CI) | Fixed queue | DQN queue | Δ queue (95% CI) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in scenarios:
        summary = _read_csv(output_dir / name / "summary.csv")
        if not summary:
            continue
        lookup = {(row["metric"], row["controller"]): row for row in summary}
        cells = []
        for metric in REPORT_METRICS:
            fixed, dqn = lookup[metric, "fixed_time"], lookup[metric, "dqn"]
            cells.extend((f"{float(fixed['controller_mean']):.2f}", f"{float(dqn['controller_mean']):.2f}"))
            if fixed["paired_ci_95_lower"]:
                cells.append(f"{float(fixed['mean_dqn_minus_fixed']):+.2f} [{float(fixed['paired_ci_95_lower']):+.2f}, {float(fixed['paired_ci_95_upper']):+.2f}]")
            else:
                cells.append("CI unavailable")
        lines.append("| " + " | ".join((name, *cells)) + " |")
    lines.extend(("", "## Demand and signal behavior", "", "All scenarios have 0.48 expected arrivals per second (144 over 300 seconds). Per-approach probabilities, in north/south/east/west order: balanced 0.12/0.12/0.12/0.12; NS-heavy 0.168/0.168/0.072/0.072; EW-heavy 0.072/0.072/0.168/0.168. Changing demand uses those three profiles in consecutive 100-second windows.", "", "| Scenario | Scheduled total | Realised N/S/E/W scheduled | Inserted DQN/fixed | Completed DQN/fixed | NS green DQN/fixed (s) | EW green DQN/fixed (s) | Changes DQN/fixed | Transition DQN/fixed (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|") )
    for name in scenarios:
        rows = _read_csv(output_dir / name / "episodes.csv")
        if not rows:
            continue
        dqn = [row for row in rows if row["controller"] == "dqn"]
        fixed = [row for row in rows if row["controller"] == "fixed_time"]
        if not dqn or not fixed:
            continue
        totals = "/".join(str(sum(int(row[f"{approach}_scheduled"]) for row in dqn)) for approach in APPROACHES)
        def total(group: Sequence[dict], key: str) -> int:
            return sum(int(float(row[key])) for row in group)
        def mean(group: Sequence[dict], key: str) -> str:
            return f"{fmean(float(row[key]) for row in group):.1f}"
        cells = (name, str(total(dqn, "scheduled_vehicles")), totals,
                 f"{total(dqn, 'inserted_vehicles')}/{total(fixed, 'inserted_vehicles')}",
                 f"{total(dqn, 'vehicles_completed')}/{total(fixed, 'vehicles_completed')}",
                 f"{mean(dqn, 'ns_green_seconds')}/{mean(fixed, 'ns_green_seconds')}",
                 f"{mean(dqn, 'ew_green_seconds')}/{mean(fixed, 'ew_green_seconds')}",
                 f"{mean(dqn, 'signal_changes')}/{mean(fixed, 'signal_changes')}",
                 f"{mean(dqn, 'transition_seconds')}/{mean(fixed, 'transition_seconds')}")
        lines.append("| " + " | ".join(cells) + " |")
    changing_demands = list((output_dir / "changing" / "demand").glob("*.rou.xml"))
    if changing_demands:
        lines.extend(("", "Changing-demand scheduled arrivals by 100-second window across all saved seeds:", "", "| Seconds | Scheduled total | N/S/E/W scheduled | Realised N/S/E/W share |", "|---|---:|---:|---:|"))
        windows = [Counter() for _ in range(3)]
        for route_path in changing_demands:
            for vehicle in ET.parse(route_path).getroot().findall("vehicle"):
                window = int(float(vehicle.attrib["depart"])) // 100
                approach = vehicle.attrib["id"].split("_flow.", 1)[0]
                windows[window][approach] += 1
        for index, counts in enumerate(windows):
            total_count = sum(counts.values())
            scheduled = "/".join(str(counts[approach]) for approach in APPROACHES)
            shares = "/".join(f"{100 * counts[approach] / total_count:.1f}%" for approach in APPROACHES)
            lines.append(f"| {index * 100}–{(index + 1) * 100} | {total_count} | {scheduled} | {shares} |")
    lines.extend(("", "## Interpretation and limits", "", "Balanced demand: the DQN lowers mean queue and waiting time, while the paired completed-vehicle interval includes zero. Under north-south-heavy demand, it also completes more vehicles on average. Under east-west-heavy demand, it completes fewer vehicles and does not allocate more east-west green time than fixed time. Under changing demand, queue and waiting measures improve but completed journeys fall. The DQN uses roughly twice as many signal changes as fixed time, increasing transition time in all four scenarios; this is a plausible contributor to lost throughput, not a proven cause.", "", "The changing scenario's aggregate green times do not establish whether the policy responds to each 100-second demand shift. A representative replay shows timing for one seed only. The 30 seeds are repeated draws from each specified scenario, not 30 distinct traffic environments. No inference here establishes performance on every demand distribution.", "", "The current train.py uses one SUMO seed for its 100 episode reloads, so its traffic generator would repeat the same seeded arrival realisation. The checkpoint stores no per-episode demand provenance. Seeds 1000–1029 were already used in the earlier minimum-green comparison and may have informed controller design; this experiment uses fresh seeds 3000–3029.", "", "Mean wait is integrated halted vehicle-seconds divided by inserted vehicles; it includes vehicles still in the network, but excludes vehicles blocked from insertion. Scheduled demand can exceed insertions under congestion. Completion percentage uses scheduled demand as denominator. Each CI uses seed-level paired differences; a one-seed smoke run has no CI.", "", "The route files are explicit scheduled departures; each DQN/fixed pair uses the same file. The original SUMO flow file is overridden for these episodes. See config.json for scenario shares and metric definitions, and each scenario's episodes.csv for per-seed counts. Representative replay CSVs, when generated, are in each scenario's replays/ directory.", "", "## Reproduce", "", "From the repository root, run `.venv/bin/python evaluate_generalisation.py --output-dir results/generalisation_repeat --replay-seed 3000` for an independent repeat. Add `--scenario ns_heavy --seed-start 3000 --seed-count 2 --output-dir results/generalisation_ns_smoke` for a small subset. Reusing this report's output directory verifies the stored demand and resumes incomplete pairs.", "", "Next experiment: record green allocation by 100-second window across all 30 changing-demand seeds, then test whether the policy's allocation follows each demand shift. Keep the current checkpoint and controllers frozen for that diagnostic comparison.", ""))
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _replays(path: Path, seed: int, route_path: Path, agent: object, by_key: dict) -> None:
    """Record one reproducible pair in the existing visualiser's CSV format."""
    for controller in ("dqn", "fixed_time"):
        destination = path / "replays" / f"seed_{seed}_{controller}.csv"
        if destination.exists():
            if len(load_recording(destination).frames) != EPISODE_SECONDS:
                raise ValueError(f"incomplete replay: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.stem}.tmp.csv")
        set_deterministic_seed(seed)
        try:
            result = _run_episode(
                controller=controller, seed=seed,
                agent=agent if controller == "dqn" else None,
                episode_seconds=EPISODE_SECONDS, route_path=route_path,
                record_path=temporary,
            ).values
            if int(result["vehicles_completed"]) != int(by_key[seed, controller]["vehicles_completed"]):
                raise ValueError(f"replay result differs for {path.name}/{seed}/{controller}")
            if len(load_recording(temporary).frames) != EPISODE_SECONDS:
                raise ValueError("replay did not record every simulation second")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def evaluate(*, scenarios: Sequence[str], seed_start: int = SEED_START,
             seed_count: int = SEED_COUNT, output_dir: Path = OUTPUT_DIR,
             checkpoint: Path = CHECKPOINT, replay_seed: int | None = None) -> None:
    """Run or resume each paired scenario using frozen, explicit route files."""
    if seed_count < 1 or seed_count > 30 or seed_start < 0:
        raise ValueError("choose 1–30 nonnegative evaluation seeds")
    if not scenarios or len(set(scenarios)) != len(scenarios) or any(name not in SCENARIOS for name in scenarios):
        raise ValueError("select distinct known scenarios")
    seeds = list(range(seed_start, seed_start + seed_count))
    if replay_seed is not None and replay_seed not in seeds:
        raise ValueError("replay seed must be part of the evaluation seed range")
    if set(seeds) & (set(range(0, 110)) | set(range(1000, 1030)) | set(range(2000, 2010)) | set(range(20001, 20701))):
        raise ValueError("evaluation seeds overlap known training, validation or prior evaluation seeds")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = _config(checkpoint, scenarios, seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("output directory contains a different experiment configuration")
    elif any(output_dir.iterdir()):
        raise ValueError("output directory is not empty and has no matching config.json")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    set_deterministic_seed(0)
    agent = load_agent(checkpoint)
    if agent.epsilon != 0 or agent.policy_network.training:
        raise RuntimeError("evaluation requires a greedy inference-mode checkpoint")
    consolidated = []
    for name in scenarios:
        path = output_dir / name
        existing = _read_csv(path / "episodes.csv")
        by_key = {(int(row["seed"]), row["controller"]): row for row in existing}
        if len(by_key) != len(existing):
            raise ValueError(f"duplicate episode in {path}")
        for seed in seeds:
            route_path = path / "demand" / f"seed_{seed}.rou.xml"
            scheduled = prepare_demand(route_path, SCENARIOS[name], seed, EPISODE_SECONDS)
            demand_hash = hashlib.sha256(route_path.read_bytes()).hexdigest()
            for controller in ("dqn", "fixed_time"):
                key = (seed, controller)
                if key in by_key:
                    if by_key[key]["demand_sha256"] != demand_hash:
                        raise ValueError(f"stored episode demand differs for {name}/{seed}")
                    continue
                set_deterministic_seed(seed)
                row = _record(controller, name, seed, route_path, scheduled,
                              agent if controller == "dqn" else None)
                by_key[key] = row
                _write_scenario(path, list(by_key.values()))
            pair = (by_key[seed, "dqn"], by_key[seed, "fixed_time"])
            if pair[0]["scheduled_vehicles"] != pair[1]["scheduled_vehicles"]:
                raise ValueError("paired scheduled counts differ")
            print(f"{name} seed {seed}: scheduled={pair[0]['scheduled_vehicles']} "
                  f"inserted DQN/fixed={pair[0]['inserted_vehicles']}/{pair[1]['inserted_vehicles']} "
                  f"completed={pair[0]['vehicles_completed']}/{pair[1]['vehicles_completed']}", flush=True)
        rows = list(by_key.values())
        if replay_seed is not None:
            _replays(path, replay_seed, path / "demand" / f"seed_{replay_seed}.rou.xml", agent, by_key)
        for item in _write_scenario(path, rows):
            consolidated.append({"scenario": name, **item})
        _write_csv(output_dir / "consolidated_summary.csv", tuple(consolidated[0]), consolidated)
        _report(output_dir, scenarios)


def main(argv: Sequence[str] | None = None) -> None:
    """Run all scenarios, or a named subset, with a selected fresh seed range."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", choices=SCENARIOS, dest="scenarios")
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--replay-seed", type=int, help="record one paired seed per scenario for the existing replay visualiser")
    args = parser.parse_args(argv)
    evaluate(scenarios=args.scenarios or list(SCENARIOS), seed_start=args.seed_start,
             seed_count=args.seed_count, output_dir=args.output_dir,
             checkpoint=args.checkpoint, replay_seed=args.replay_seed)


if __name__ == "__main__":
    main()
