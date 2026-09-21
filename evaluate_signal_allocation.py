"""Replay frozen paired demand and diagnose signal allocation without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from statistics import fmean, stdev

from evaluate_dqn import DECISION_INTERVAL, EPISODE_SECONDS, greedy_action, load_agent, set_deterministic_seed
from evaluate_generalisation import T_CRITICAL
from evaluate_paired import CHECKPOINT, _write_csv
from src.environment.traffic_env import MINIMUM_GREEN_DURATION, TrafficEnvironment
from src.experiments.generalisation_demand import SCENARIOS, TOTAL_ARRIVAL_RATE
from src.simulation.metrics import APPROACHES
from src.simulation.run import CONFIG_PATH, build_sumo_command
from src.simulation.traffic import TRAFFIC_LIGHT_ID, TrafficSimulation


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/generalisation"
DEFAULT_OUTPUT = ROOT / "results/signal_allocation"
EXPECTED_HASH = "e820fb9912eab609f7e10e295d01375e944faf655f4bd9c660d7e8cbcb22c4d0"
PHASES = {0: ("ns_green", "NS"), 1: ("yellow", None), 2: ("all_red", None),
          3: ("ew_green", "EW"), 4: ("yellow", None), 5: ("all_red", None)}
PAIRED_METRICS = ("ew_green_seconds", "transition_seconds", "executed_direction_changes",
                  "ns_queue_seconds", "ew_queue_seconds", "vehicles_completed")


def phase_kind(phase: int) -> tuple[str, str | None]:
    """Classify the verified six-phase SUMO signal program."""
    if phase not in PHASES:
        raise ValueError(f"unknown traffic-light phase {phase}")
    return PHASES[phase]


def queue_relation(ns: int, ew: int) -> str:
    """Classify observed directional queues, including equal zero queues."""
    return "NS_greater" if ns > ew else "EW_greater" if ew > ns else "equal"


def window_index(second: int) -> int:
    """Return the changing-demand window containing a zero-based second."""
    if not 0 <= second < EPISODE_SECONDS:
        raise ValueError("second is outside the episode")
    return second // 100


def interval(values: list[float]) -> dict:
    """Summarise independent seed-level differences with a paired t interval."""
    if not values:
        return {"n": 0, "mean": "", "sample_sd": "", "ci_95_lower": "", "ci_95_upper": ""}
    mean = fmean(values)
    if len(values) < 2:
        return {"n": 1, "mean": mean, "sample_sd": "", "ci_95_lower": "", "ci_95_upper": ""}
    if len(values) > 30:
        raise ValueError("paired t table supports at most 30 seeds")
    sd = stdev(values)
    half = T_CRITICAL[len(values) - 1] * sd / math.sqrt(len(values))
    return {"n": len(values), "mean": mean, "sample_sd": sd,
            "ci_95_lower": mean - half, "ci_95_upper": mean + half}


def scheduled_by_second(path: Path) -> dict[int, Counter]:
    """Read immutable scheduled vehicles by departure second and approach."""
    output: dict[int, Counter] = {}
    for vehicle in ET.parse(path).getroot().findall("vehicle"):
        second = int(float(vehicle.attrib["depart"]))
        approach = vehicle.attrib["id"].split("_flow.", 1)[0]
        if second not in range(EPISODE_SECONDS) or approach not in APPROACHES:
            raise ValueError(f"invalid scheduled vehicle in {path}")
        output.setdefault(second, Counter())[approach] += 1
    return output


class ObservedSimulation(TrafficSimulation):
    """Capture the phase before each ordinary SUMO step for interval accounting."""

    def __init__(self, connection, reload_args):
        super().__init__(connection, reload_args)
        self.governing_phase = 0

    def step(self):
        self.governing_phase = self._traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
        return super().step()


def _direction_counts(connection, method: str, known: dict[str, str]) -> Counter:
    counts: Counter = Counter()
    for vehicle_id in getattr(connection.simulation, method)():
        approach = known.get(vehicle_id) or vehicle_id.split("_flow.", 1)[0]
        if approach in APPROACHES:
            counts[approach] += 1
            if method == "getDepartedIDList":
                known[vehicle_id] = approach
            elif method == "getArrivedIDList":
                known.pop(vehicle_id, None)
    return counts


def run_episode(scenario: str, seed: int, controller: str, route: Path, agent,
                minimum_green_duration: int = MINIMUM_GREEN_DURATION) -> tuple[list[dict], list[dict]]:
    """Run the unchanged controller and collect one record per governed second."""
    import traci

    command = build_sumo_command(gui=False, seed=seed, steps=EPISODE_SECONDS, route_path=route)
    scheduled = scheduled_by_second(route)
    seconds: list[dict] = []
    decisions: list[dict] = []
    known: dict[str, str] = {}
    started = False
    try:
        traci.start(command)
        started = True
        simulation = ObservedSimulation(traci, command[1:])
        previous_green = "NS"
        green_elapsed = 0

        def observe_step(metrics):
            nonlocal previous_green, green_elapsed
            second = len(seconds)
            kind, direction = phase_kind(simulation.governing_phase)
            if direction is not None:
                changed = int(direction != previous_green)
                green_elapsed = 1 if changed or second == 0 or seconds[-1]["active_green"] != direction else green_elapsed + 1
                previous_green = direction
            else:
                changed = 0
                green_elapsed = 0
            departed = _direction_counts(traci, "getDepartedIDList", known)
            arrived = _direction_counts(traci, "getArrivedIDList", known)
            queues = {name: metrics.approaches[name].queue_length for name in APPROACHES}
            row = {"scenario": scenario, "seed": seed, "controller": controller,
                   "second": second, "time_start": second, "time_end": int(metrics.time),
                   "phase": simulation.governing_phase, "phase_kind": kind,
                   "active_green": direction or "", "is_yellow": int(kind == "yellow"),
                   "is_all_red": int(kind == "all_red"), "executed_direction_change": changed,
                   "green_elapsed_seconds": green_elapsed if direction else "",
                   "ns_queue": queues["north"] + queues["south"],
                   "ew_queue": queues["east"] + queues["west"],
                   "throughput_cumulative": metrics.vehicles_completed,
                   "requested_action": "", "effective_action": "", "requested_switch": "",
                   "executed_switch": "", "switch_eligible": "", "switch_blocked": ""}
            for approach in APPROACHES:
                row[f"{approach}_queue"] = queues[approach]
                row[f"{approach}_scheduled"] = scheduled.get(second, Counter())[approach]
                row[f"{approach}_inserted"] = departed[approach]
                row[f"{approach}_completed"] = arrived[approach]
            seconds.append(row)

        environment = TrafficEnvironment(
            simulation=simulation, decision_interval=DECISION_INTERVAL,
            max_simulation_time=EPISODE_SECONDS,
            minimum_green_duration=minimum_green_duration if controller == "dqn" else None,
            step_observer=observe_step)
        if controller == "dqn":
            environment.reset(seed=seed)
            while len(seconds) < EPISODE_SECONDS:
                before = simulation.observe()
                current = "NS" if environment._active_green_direction == 0 else "EW"
                ns = sum(before.approaches[a].queue_length for a in ("north", "south"))
                ew = sum(before.approaches[a].queue_length for a in ("east", "west"))
                requested = greedy_action(agent, environment.get_state())
                elapsed = environment._green_elapsed_seconds
                eligible = elapsed >= minimum_green_duration
                _, _, _, _, info = environment.step(requested)
                decision = {"scenario": scenario, "seed": seed, "second": int(before.time),
                                  "current_green": current, "requested_action": "NS" if requested == 0 else "EW",
                                  "effective_action": "NS" if info["effective_action"] == 0 else "EW",
                                  "requested_switch": int(requested != environment_action(current)),
                                  "executed_switch": int(info["effective_action"] != environment_action(current)),
                                  "switch_eligible": int(eligible), "switch_blocked": int(info["switch_blocked"]),
                                  "environment_green_elapsed_seconds": elapsed,
                                  "ns_queue": ns, "ew_queue": ew, "queue_difference_ns_minus_ew": ns - ew,
                                  "queue_relation": queue_relation(ns, ew)}
                decisions.append(decision)
                for key in ("requested_action", "effective_action", "requested_switch", "executed_switch", "switch_eligible", "switch_blocked"):
                    seconds[int(before.time)][key] = decision[key]
        elif controller == "fixed_time":
            for _ in range(EPISODE_SECONDS):
                observe_step(simulation.step())
        else:
            raise ValueError(controller)
    finally:
        if started:
            failed = sys.exc_info()[0] is not None
            try:
                traci.close()
            except Exception:
                if not failed:
                    raise
    if len(seconds) != EPISODE_SECONDS or [r["second"] for r in seconds] != list(range(EPISODE_SECONDS)):
        raise ValueError("episode did not record 300 unique seconds")
    return seconds, decisions


def environment_action(direction: str) -> int:
    """Map a directional label to the existing two-action interface."""
    return 0 if direction == "NS" else 1


def _runs(rows: list[dict], direction: str) -> list[int]:
    lengths: list[int] = []
    current = 0
    for row in rows:
        if row["active_green"] == direction:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def summarize_seconds(rows: list[dict], decisions: list[dict] | None = None) -> dict:
    """Calculate duration, queues, arrivals, and switching from interval records."""
    n = len(rows)
    counts = Counter(r["phase_kind"] for r in rows)
    if sum(counts.values()) != n:
        raise ValueError("phase accounting mismatch")
    greens = counts["ns_green"] + counts["ew_green"]
    result = {f"{key}_seconds": counts[key] for key in ("ns_green", "ew_green", "yellow", "all_red")}
    result.update(transition_seconds=counts["yellow"] + counts["all_red"],
                  executed_direction_changes=sum(int(r["executed_direction_change"]) for r in rows),
                  ns_green_total_time_share=counts["ns_green"] / n,
                  ew_green_total_time_share=counts["ew_green"] / n,
                  transition_total_time_share=(counts["yellow"] + counts["all_red"]) / n,
                  ns_green_share=counts["ns_green"] / greens if greens else "",
                  ew_green_share=counts["ew_green"] / greens if greens else "",
                  mean_ns_green_run_seconds=fmean(_runs(rows, "NS")) if _runs(rows, "NS") else "",
                  mean_ew_green_run_seconds=fmean(_runs(rows, "EW")) if _runs(rows, "EW") else "",
                  ns_queue_seconds=sum(int(r["ns_queue"]) for r in rows),
                  ew_queue_seconds=sum(int(r["ew_queue"]) for r in rows),
                  mean_ns_queue=fmean(int(r["ns_queue"]) for r in rows),
                  mean_ew_queue=fmean(int(r["ew_queue"]) for r in rows),
                  max_ns_queue=max(int(r["ns_queue"]) for r in rows),
                  max_ew_queue=max(int(r["ew_queue"]) for r in rows),
                  ns_queue_greater_share=sum(int(r["ns_queue"]) > int(r["ew_queue"]) for r in rows) / n,
                  ew_queue_greater_share=sum(int(r["ew_queue"]) > int(r["ns_queue"]) for r in rows) / n,
                  equal_queue_share=sum(int(r["ew_queue"]) == int(r["ns_queue"]) for r in rows) / n)
    for direction, approaches in (("ns", ("north", "south")), ("ew", ("east", "west"))):
        for event in ("scheduled", "inserted", "completed"):
            result[f"{direction}_{event}"] = sum(int(r[f"{a}_{event}"]) for r in rows for a in approaches)
    result["vehicles_completed"] = result["ns_completed"] + result["ew_completed"]
    if decisions is not None:
        result["requested_direction_changes"] = sum(int(d["requested_switch"]) for d in decisions)
        result["blocked_switch_requests"] = sum(int(d["switch_blocked"]) for d in decisions)
    else:
        result["requested_direction_changes"] = ""
        result["blocked_switch_requests"] = ""
    return result


def validate_seconds(rows: list[dict]) -> None:
    """Reject gaps, impossible signal jumps, and inconsistent action records."""
    if len(rows) != EPISODE_SECONDS:
        raise ValueError("episode must contain 300 seconds")
    for second, row in enumerate(rows):
        if int(row["second"]) != second or int(row["time_end"]) != second + 1:
            raise ValueError("duplicate or missing simulation second")
        phase = int(row["phase"])
        if row["phase_kind"] != phase_kind(phase)[0]:
            raise ValueError("phase classification disagrees with SUMO phase")
        if second and phase not in (int(rows[second - 1]["phase"]), (int(rows[second - 1]["phase"]) + 1) % 6):
            raise ValueError("impossible signal phase jump")
        if row["controller"] == "fixed_time" and row["requested_action"] != "":
            raise ValueError("fixed time has a DQN action")
        if row["requested_action"] != "":
            if int(row["switch_blocked"]) and (not int(row["requested_switch"]) or int(row["executed_switch"])):
                raise ValueError("blocked request was recorded as executed")


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _save(path: Path, rows: list[dict]) -> None:
    if rows:
        _write_csv(path, tuple(rows[0]), rows)


def _expected(scenario: str, seed: int, controller: str) -> dict:
    rows = _read(SOURCE / scenario / "episodes.csv")
    matches = [r for r in rows if int(r["seed"]) == seed and r["controller"] == controller]
    if len(matches) != 1:
        raise ValueError(f"missing previous generalisation episode: {scenario}/{seed}/{controller}")
    return matches[0]


def _validate_reproduction(summary: dict, old: dict) -> None:
    for key, previous in (("vehicles_completed", "vehicles_completed"),
                          ("ns_green_seconds", "ns_green_seconds"),
                          ("ew_green_seconds", "ew_green_seconds"),
                          ("transition_seconds", "transition_seconds"),
                          ("executed_direction_changes", "signal_changes")):
        if int(summary[key]) != int(float(old[previous])):
            raise ValueError(f"replay differs from generalisation for {key}: {summary[key]} versus {old[previous]}")
    total_queue = summary["ns_queue_seconds"] + summary["ew_queue_seconds"]
    if abs(total_queue / EPISODE_SECONDS - float(old["mean_queue_length"])) > 1e-9:
        raise ValueError("replay mean queue differs from generalisation")
    inserted = summary["ns_inserted"] + summary["ew_inserted"]
    if inserted != int(old["inserted_vehicles"]):
        raise ValueError("replay inserted vehicles differ from generalisation")
    waiting = total_queue / inserted if inserted else 0.0
    if abs(waiting - float(old["mean_waiting_time"])) > 1e-9:
        raise ValueError("replay waiting measure differs from generalisation")


def _paired(episodes: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = {(r["scenario"], int(r["seed"]), r["controller"]): r for r in episodes}
    pairs = []
    for scenario, seed, controller in grouped:
        if controller != "dqn":
            continue
        dqn = grouped[scenario, seed, "dqn"]
        fixed = grouped[scenario, seed, "fixed_time"]
        if dqn["demand_sha256"] != fixed["demand_sha256"]:
            raise ValueError("paired demand hash mismatch")
        for metric in PAIRED_METRICS:
            pairs.append({"scenario": scenario, "seed": seed, "metric": metric,
                          "dqn": float(dqn[metric]), "fixed_time": float(fixed[metric]),
                          "dqn_minus_fixed": float(dqn[metric]) - float(fixed[metric])})
    summaries = []
    for scenario in sorted({r["scenario"] for r in episodes}):
        for metric in PAIRED_METRICS:
            relevant = [r for r in pairs if r["scenario"] == scenario and r["metric"] == metric]
            stats = interval([r["dqn_minus_fixed"] for r in relevant])
            summaries.append({"scenario": scenario, "metric": metric,
                              "dqn_mean": fmean(r["dqn"] for r in relevant),
                              "fixed_mean": fmean(r["fixed_time"] for r in relevant), **stats})
    return pairs, summaries


def _windows(seconds: list[dict]) -> list[dict]:
    output = []
    for index in range(3):
        subset = [r for r in seconds if window_index(int(r["second"])) == index]
        if len(subset) != 100:
            raise ValueError("each demand window must contain 100 seconds")
        metrics = summarize_seconds(subset)
        # End-of-step queue at t=100 or t=200 is the start-of-window state.
        # At t=0 the freshly loaded network has no queued vehicles.
        start = seconds[index * 100 - 1] if index else None
        metrics.update(scenario="changing", seed=seconds[0]["seed"], controller=seconds[0]["controller"],
                       window=index + 1, start_second=index * 100, end_second=index * 100 + 99,
                       ns_queue_at_start=start["ns_queue"] if start else 0,
                       ew_queue_at_start=start["ew_queue"] if start else 0,
                       ns_queue_at_end=subset[-1]["ns_queue"], ew_queue_at_end=subset[-1]["ew_queue"])
        output.append(metrics)
    return output


def _decision_summary(decisions: list[dict]) -> list[dict]:
    output = []
    for scenario in sorted({d["scenario"] for d in decisions}):
        for relation in ("NS_greater", "EW_greater", "equal"):
            for eligible in (0, 1):
                group = [d for d in decisions if d["scenario"] == scenario and d["queue_relation"] == relation
                         and int(d["switch_eligible"]) == eligible]
                output.append({"scenario": scenario, "queue_relation": relation,
                               "switch_eligible": eligible, "decisions": len(group),
                               "requested_ns": sum(d["requested_action"] == "NS" for d in group),
                               "requested_ew": sum(d["requested_action"] == "EW" for d in group),
                               "blocked_switches": sum(int(d["switch_blocked"]) for d in group),
                               "requested_ns_share": sum(d["requested_action"] == "NS" for d in group) / len(group) if group else ""})
    return output


def _window_comparisons(windows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return seed-paired EW share changes and controller window means."""
    by_key = {(int(r["seed"]), r["controller"], int(r["window"])): r for r in windows}
    seeds = sorted({seed for seed, controller, index in by_key if controller == "dqn" and index == 1})
    changes = []
    for seed in seeds:
        for first, last in ((1, 2), (2, 3)):
            before = float(by_key[seed, "dqn", first]["ew_green_share"])
            after = float(by_key[seed, "dqn", last]["ew_green_share"])
            changes.append({"seed": seed, "first_window": first, "last_window": last,
                            "ew_share_first": before, "ew_share_last": after,
                            "ew_share_change": after - before})
    means = []
    for controller in ("dqn", "fixed_time"):
        for index in (1, 2, 3):
            group = [r for r in windows if r["controller"] == controller and int(r["window"]) == index]
            if not group:
                continue
            metrics = ("ns_green_seconds", "ew_green_seconds", "yellow_seconds", "all_red_seconds",
                       "executed_direction_changes", "ns_green_share", "ew_green_share",
                       "mean_ns_queue", "mean_ew_queue", "ns_scheduled", "ew_scheduled",
                       "ns_inserted", "ew_inserted", "ns_completed", "ew_completed",
                       "ns_queue_at_start", "ew_queue_at_start", "ns_queue_at_end", "ew_queue_at_end")
            means.append({"controller": controller, "window": index, "seeds": len(group),
                          **{key: fmean(float(r[key]) for r in group) for key in metrics}})
    return changes, means


def _plots(output: Path, episodes: list[dict], seconds: list[dict], windows: list[dict], decisions: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    scenarios = [s for s in SCENARIOS if any(r["scenario"] == s for r in episodes)]
    fig, (ax, share_ax) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                      gridspec_kw={"height_ratios": (2, 1)})
    positions = []
    labels = []
    for i, scenario in enumerate(scenarios):
        for j, controller in enumerate(("dqn", "fixed_time")):
            group = [r for r in episodes if r["scenario"] == scenario and r["controller"] == controller]
            x = i * 3 + j
            bottom = 0
            for metric, color, name in (("ns_green_seconds", "#3977a8", "NS green"),
                                         ("ew_green_seconds", "#d98342", "EW green"),
                                         ("transition_seconds", "#777777", "transition")):
                value = fmean(float(r[metric]) for r in group)
                ax.bar(x, value, bottom=bottom, color=color, label=name if i == j == 0 else None)
                bottom += value
            positions.append(x)
            labels.append(f"{scenario}\n{controller}")
            share_ax.bar(x, fmean(float(r["ew_green_share"]) for r in group), color="#d98342")
    ax.set(xticks=positions, xticklabels=labels, ylabel="Mean seconds per 300-second episode", title="Absolute signal time by scenario")
    ax.legend()
    share_ax.set(xticks=positions, xticklabels=labels,
                 ylabel="EW share of available green", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(plot_dir / "allocation_by_scenario.png", dpi=180)
    plt.close(fig)

    ew = [r for r in episodes if r["scenario"] == "ew_heavy"]
    if ew:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for ax, metric, title in zip(axes.flat, ("ew_green_seconds", "transition_seconds", "ew_queue_seconds", "vehicles_completed"),
                                     ("EW green (s)", "Transitions (s)", "EW queue-seconds", "Completed vehicles")):
            ax.bar(("DQN", "Fixed"), [fmean(float(r[metric]) for r in ew if r["controller"] == c) for c in ("dqn", "fixed_time")], color=("#3977a8", "#d98342"))
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(plot_dir / "ew_heavy_diagnostic.png", dpi=180)
        plt.close(fig)
    if windows:
        fig, ax = plt.subplots(figsize=(9, 5))
        for controller, linestyle in (("dqn", "-"), ("fixed_time", "--")):
            for direction, color in (("ns", "#3977a8"), ("ew", "#d98342")):
                y = [fmean(float(r[f"{direction}_green_seconds"]) for r in windows if r["controller"] == controller and int(r["window"]) == w) for w in (1, 2, 3)]
                ax.plot((1, 2, 3), y, marker="o", color=color, linestyle=linestyle, label=f"{controller} {direction.upper()}")
        ax.set(xticks=(1, 2, 3), xticklabels=("Balanced", "NS heavy", "EW heavy"), ylabel="Mean green seconds per 100-second window", title="Response to changing scheduled demand")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "changing_windows.png", dpi=180)
        plt.close(fig)
    changing = [r for r in seconds if r["scenario"] == "changing" and r["controller"] == "dqn"]
    if changing:
        import numpy as np
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        bins = range(0, EPISODE_SECONDS, 10)
        seed_groups = sorted({int(r["seed"]) for r in changing})
        for direction, color in (("ns", "#3977a8"), ("ew", "#d98342")):
            queue_samples = np.array([[fmean(float(r[f"{direction}_queue"]) for r in changing if int(r["seed"]) == seed and b <= int(r["second"]) < b + 10) for b in bins] for seed in seed_groups])
            green_samples = np.array([[fmean(r["active_green"] == direction.upper() for r in changing if int(r["seed"]) == seed and b <= int(r["second"]) < b + 10) for b in bins] for seed in seed_groups])
            for axis, samples in ((axes[0], queue_samples), (axes[1], green_samples)):
                axis.plot(list(bins), samples.mean(axis=0), color=color, label=direction.upper())
                if len(seed_groups) > 1:
                    axis.fill_between(list(bins), np.percentile(samples, 10, axis=0),
                                      np.percentile(samples, 90, axis=0), color=color, alpha=0.12)
        for ax in axes:
            ax.axvline(100, color="black", linestyle=":")
            ax.axvline(200, color="black", linestyle=":")
            ax.legend()
        axes[0].set_ylabel("Mean directional queue")
        axes[1].set(xlabel="Simulation second (10-second bins; bands show 10th–90th seed percentiles)", ylabel="Fraction of seconds green", ylim=(0, 1))
        fig.tight_layout()
        fig.savefig(plot_dir / "changing_time_series.png", dpi=180)
        plt.close(fig)
    if decisions:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = ("NS larger", "EW larger", "Equal")
        groups = ("NS_greater", "EW_greater", "equal")
        eligible = [d for d in decisions if int(d["switch_eligible"])]
        ns = [sum(d["requested_action"] == "NS" for d in eligible if d["queue_relation"] == g) for g in groups]
        ew = [sum(d["requested_action"] == "EW" for d in eligible if d["queue_relation"] == g) for g in groups]
        ax.bar(labels, ns, label="Request NS", color="#3977a8")
        ax.bar(labels, ew, bottom=ns, label="Request EW", color="#d98342")
        ax.set(ylabel="Eligible decisions across seed episodes", title="Descriptive DQN requests by observed queue imbalance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "actions_vs_queue.png", dpi=180)
        plt.close(fig)


def _report(output: Path, episodes: list[dict], paired: list[dict], windows: list[dict], decisions: list[dict]) -> None:
    lookup = {(r["scenario"], r["metric"]): r for r in paired}
    lines = ["# Frozen DQN signal-allocation diagnostic", "", "All values are descriptive means over independent seed episodes. Paired differences are DQN minus fixed time; intervals use seed-level Student t estimates. The five designated diagnostics are EW green seconds, transition seconds, executed changes, directional queue-seconds, and the DQN EW share change from changing-demand window 2 to 3.", "", "## Scenario summary", "", "| Scenario | DQN / fixed NS green (s) | DQN / fixed EW green (s) | DQN / fixed transitions (s) | DQN / fixed changes | DQN / fixed EW queue-seconds | DQN / fixed completions |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scenario in SCENARIOS:
        group = [r for r in episodes if r["scenario"] == scenario]
        if not group:
            continue
        values = []
        for metric in ("ns_green_seconds", "ew_green_seconds", "transition_seconds", "executed_direction_changes", "ew_queue_seconds", "vehicles_completed"):
            values.append(" / ".join(f"{fmean(float(r[metric]) for r in group if r['controller'] == c):.1f}" for c in ("dqn", "fixed_time")))
        lines.append("| " + " | ".join((scenario, *values)) + " |")
    lines += ["", "## Main paired differences (DQN minus fixed)", "", "| Scenario | Metric | Mean difference | Sample SD | 95% CI | Pairs |", "|---|---|---:|---:|---:|---:|"]
    for row in paired:
        if row["metric"] not in PAIRED_METRICS:
            continue
        ci = f"[{row['ci_95_lower']:.2f}, {row['ci_95_upper']:.2f}]" if row["ci_95_lower"] != "" else "unavailable"
        sd = "" if row["sample_sd"] == "" else f"{row['sample_sd']:.2f}"
        lines.append(f"| {row['scenario']} | {row['metric']} | {row['mean']:+.2f} | {sd} | {ci} | {row['n']} |")
    if windows:
        lines += ["", "## Changing-demand windows", "", "| Window | Controller | NS / EW green (s) | NS / EW mean queue | NS / EW inserted |", "|---|---|---:|---:|---:|"]
        for w in (1, 2, 3):
            for c in ("dqn", "fixed_time"):
                group = [r for r in windows if int(r["window"]) == w and r["controller"] == c]
                if group:
                    mean = lambda key: fmean(float(r[key]) for r in group)
                    lines.append(f"| {w} | {c} | {mean('ns_green_seconds'):.1f} / {mean('ew_green_seconds'):.1f} | {mean('mean_ns_queue'):.1f} / {mean('mean_ew_queue'):.1f} | {mean('ns_inserted'):.1f} / {mean('ew_inserted'):.1f} |")
        by_seed = {(int(r["seed"]), int(r["window"])): r for r in windows if r["controller"] == "dqn"}
        for a, b in ((1, 2), (2, 3)):
            differences = [float(by_seed[seed, b]["ew_green_share"]) - float(by_seed[seed, a]["ew_green_share"]) for seed in sorted({s for s, w in by_seed if w == 1})]
            stats = interval(differences)
            lines.append(f"\nDQN EW available-green share change, window {a} to {b}: {stats['mean']:+.3f}; 95% CI [{stats['ci_95_lower']:.3f}, {stats['ci_95_upper']:.3f}], n={stats['n']}." if stats["n"] > 1 else f"\nDQN EW share change, window {a} to {b}: {stats['mean']:+.3f}; CI unavailable (n=1).")
    eligible = [d for d in decisions if int(d["switch_eligible"])]
    lines += ["", "## Eligible DQN decisions", "", "| Observed queues | NS requests | EW requests | NS request share |", "|---|---:|---:|---:|"]
    for relation in ("NS_greater", "EW_greater", "equal"):
        group = [d for d in eligible if d["queue_relation"] == relation]
        ns = sum(d["requested_action"] == "NS" for d in group)
        lines.append(f"| {relation} | {ns} | {len(group)-ns} | {ns/len(group):.3f} |" if group else f"| {relation} | 0 | 0 | n/a |")
    if len(episodes) == 240:
        ew = [r for r in episodes if r["scenario"] == "ew_heavy"]
        dqn = [r for r in ew if r["controller"] == "dqn"]
        fixed = [r for r in ew if r["controller"] == "fixed_time"]
        mean = lambda group, key: fmean(float(r[key]) for r in group)
        ew_share = mean(dqn, "ew_green_share")
        ns_share = mean(dqn, "ns_green_share")
        ew_queue = lookup["ew_heavy", "ew_queue_seconds"]
        lines += ["", "## Evidence-based reading", "",
                  f"In EW-heavy traffic, DQN available-green shares are NS {ns_share:.3f} and EW {ew_share:.3f}; its absolute EW green is {mean(dqn, 'ew_green_seconds'):.1f} s versus {mean(fixed, 'ew_green_seconds'):.1f} s for fixed time. The paired EW green difference is {lookup['ew_heavy', 'ew_green_seconds']['mean']:+.1f} s. Directional share and absolute seconds answer different questions because transitions remove available green time.", "",
                  f"The DQN spends {mean(dqn, 'transition_seconds'):.1f} s transitioning in EW-heavy traffic versus {mean(fixed, 'transition_seconds'):.1f} s for fixed time. The EW queue-seconds difference is {ew_queue['mean']:+.1f}, with a seed-level 95% interval [{ew_queue['ci_95_lower']:+.1f}, {ew_queue['ci_95_upper']:+.1f}]. It completes {lookup['ew_heavy', 'vehicles_completed']['mean']:+.1f} vehicles relative to fixed time. These co-occurring differences do not identify which mechanism caused the completion deficit.", ""]
        if windows:
            changes, means = _window_comparisons(windows)
            w2 = next(r for r in means if r["controller"] == "dqn" and r["window"] == 2)
            w3 = next(r for r in means if r["controller"] == "dqn" and r["window"] == 3)
            shift = interval([r["ew_share_change"] for r in changes if r["first_window"] == 2])
            lines += [f"After scheduled demand shifts from NS-heavy to EW-heavy at second 200, DQN EW green changes from {w2['ew_green_seconds']:.1f} to {w3['ew_green_seconds']:.1f} s per window. The within-seed change in EW share is {shift['mean']:+.3f} [{shift['ci_95_lower']:+.3f}, {shift['ci_95_upper']:+.3f}]. At the start of window 3 the mean queues are NS {w3['ns_queue_at_start']:.1f} and EW {w3['ew_queue_at_start']:.1f}; the scheduled-demand shift therefore should not be treated as an instantaneous queue reversal.", ""]
        eligible_ew = [d for d in eligible if d["queue_relation"] == "EW_greater"]
        if eligible_ew:
            requested_ew = sum(d["requested_action"] == "EW" for d in eligible_ew)
            lines += [f"When EW queues exceed NS queues and switching is eligible, the DQN requests EW in {requested_ew}/{len(eligible_ew)} decisions across all scenarios. This conditional count should be read with current phase and queue history, not as an isolated preference estimate.", ""]
    lines += ["", "## Interpretation and limits", "", "Per-second phases are sampled immediately before each SUMO step; queues and insertion/completion events are sampled immediately after it. An executed change is counted when the next principal green first governs a second. The environment's minimum-green clock counts completed five-second hold intervals after a transition, so its eligibility can differ from the visible green seconds. The per-decision file records both clocks and blocked requests.", "", "Scheduled departures represent demand; inserted vehicles and current queues can differ at any window boundary. Associations between transition time and throughput do not identify a causal mechanism. Eligible action counts describe the frozen policy under observed states; they do not establish optimality or a learned directional bias independently of state and signal history. The 30 seeds sample each specified demand profile and do not establish generalisation to all traffic patterns.", "", "## Files", "", "Per-second and per-decision CSVs are under `records/`; seed summaries are `episodes.csv`, `paired_differences.csv`, `windows.csv`, `window_share_changes.csv`, `window_summary.csv`, and `decision_summary.csv`. Scenario estimates are `scenario_summary.csv`; plots are under `plots/`. `config.json` records checkpoint and demand hashes.", "", "Next experiment: test the same frozen policy under a controlled intervention that varies switching opportunity or demand history while preserving paired routes; only then consider changes to training or state representation."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(scenarios: list[str], seeds: list[int], output: Path, checkpoint: Path) -> None:
    """Verify frozen inputs, run missing records, and regenerate all analyses."""
    if not scenarios or any(s not in SCENARIOS for s in scenarios) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("select known scenarios and distinct seeds")
    if len(seeds) > 30:
        raise ValueError("at most 30 seeds are supported")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != EXPECTED_HASH:
        raise ValueError(f"checkpoint SHA-256 mismatch: {digest}")
    demand = {}
    for scenario in scenarios:
        for seed in seeds:
            route = SOURCE / scenario / "demand" / f"seed_{seed}.rou.xml"
            old = _expected(scenario, seed, "dqn")
            _expected(scenario, seed, "fixed_time")
            route_hash = hashlib.sha256(route.read_bytes()).hexdigest()
            if route_hash != old["demand_sha256"]:
                raise ValueError(f"saved demand hash mismatch: {route}")
            demand[f"{scenario}/{seed}"] = {"path": str(route.resolve()), "sha256": route_hash}
    config = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest,
              "scenarios": scenarios, "seeds": seeds, "episode_seconds": EPISODE_SECONDS,
              "decision_interval_seconds": DECISION_INTERVAL, "minimum_green_seconds": MINIMUM_GREEN_DURATION,
              "expected_arrivals_per_second": TOTAL_ARRIVAL_RATE,
              "scenario_shares": {s: [list(window) for window in SCENARIOS[s].windows] for s in scenarios},
              "sumo_config": str(CONFIG_PATH), "sumo_config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
              "demand": demand, "record_timing": "phase before step; queues and events after step; second is zero-based interval start",
              "missing_value": "empty CSV cell; fixed-time has no DQN actions"}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("output directory has a different experiment configuration")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    set_deterministic_seed(0)
    agent = load_agent(checkpoint)
    if agent.epsilon != 0 or agent.policy_network.training:
        raise RuntimeError("checkpoint is not in greedy inference mode")
    episodes, all_seconds, all_decisions, windows = [], [], [], []
    for scenario in scenarios:
        for seed in seeds:
            for controller in ("dqn", "fixed_time"):
                path = output / "records" / scenario / f"seed_{seed}_{controller}_seconds.csv"
                decision_path = output / "records" / scenario / f"seed_{seed}_dqn_decisions.csv"
                if path.exists() and (controller == "fixed_time" or decision_path.exists()):
                    seconds = _read(path)
                    decisions = _read(decision_path) if controller == "dqn" else []
                else:
                    set_deterministic_seed(seed)
                    seconds, decisions = run_episode(scenario, seed, controller, Path(demand[f"{scenario}/{seed}"]["path"]), agent if controller == "dqn" else None)
                    _save(path, seconds)
                    if controller == "dqn":
                        _save(decision_path, decisions)
                validate_seconds(seconds)
                summary = summarize_seconds(seconds, decisions if controller == "dqn" else None)
                _validate_reproduction(summary, _expected(scenario, seed, controller))
                summary.update(scenario=scenario, seed=seed, controller=controller,
                               demand_file=demand[f"{scenario}/{seed}"]["path"],
                               demand_sha256=demand[f"{scenario}/{seed}"]["sha256"])
                episodes.append(summary)
                all_seconds.extend(seconds)
                all_decisions.extend(decisions)
                if scenario == "changing":
                    parts = _windows(seconds)
                    for metric in ("ns_green_seconds", "ew_green_seconds", "yellow_seconds", "all_red_seconds", "executed_direction_changes"):
                        if sum(part[metric] for part in parts) != summary[metric]:
                            raise ValueError(f"window {metric} does not reconcile with episode")
                    windows.extend(parts)
                print(f"Verified {scenario}/{seed}/{controller}", flush=True)
    pairs, summary = _paired(episodes)
    _save(output / "episodes.csv", episodes)
    _save(output / "paired_differences.csv", pairs)
    _save(output / "scenario_summary.csv", summary)
    _save(output / "windows.csv", windows)
    window_changes, window_means = _window_comparisons(windows)
    _save(output / "window_share_changes.csv", window_changes)
    _save(output / "window_summary.csv", window_means)
    _save(output / "decision_summary.csv", _decision_summary(all_decisions))
    _plots(output, episodes, all_seconds, windows, all_decisions)
    _report(output, episodes, summary, windows, all_decisions)


def main() -> None:
    """Run all scenarios or a selected saved-demand subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), action="append", help="repeat for multiple scenarios; default is all")
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.seed_count < 1:
        parser.error("seed-count must be positive")
    evaluate(args.scenario or list(SCENARIOS), list(range(args.seed_start, args.seed_start + args.seed_count)), args.output_dir, args.checkpoint)


if __name__ == "__main__":
    main()
