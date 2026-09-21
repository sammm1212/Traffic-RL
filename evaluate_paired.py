"""Compare the 10-second minimum-green DQN with fixed time on 30 new seeds."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from tempfile import NamedTemporaryFile
from typing import Callable, Sequence

from evaluate_diagnostics import _run_episode
from evaluate_dqn import (
    DECISION_INTERVAL,
    EPISODE_SECONDS,
    load_agent,
    set_deterministic_seed,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "results/training/dqn_100_episode_min_green.pt"
OUTPUT_DIR = ROOT / "results/evaluation/paired_30_min_green"
SEEDS = tuple(range(1000, 1030))
METRICS = (
    "throughput",
    "mean_waiting_time",
    "mean_queue_length",
    "vehicles_generated",
    "vehicles_remaining",
    "signal_changes",
    "transition_time_percentage",
)
EPISODE_FIELDS = ("seed", "controller", *METRICS)
SUMMARY_FIELDS = ("controller", "number_of_seeds", *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")))
PAIRED_FIELDS = ("metric", "number_of_pairs", "mean_difference", "std_difference", "ci_95_lower", "ci_95_upper", "relative_change_percentage")
# Two-sided 95% Student-t critical value for 29 degrees of freedom.
T_95_DF_29 = 2.045229642132703


@dataclass(frozen=True)
class EpisodeResult:
    """One controller's measurements from one 300-second episode."""

    seed: int
    controller: str
    throughput: int
    mean_waiting_time: float
    mean_queue_length: float
    vehicles_generated: int
    vehicles_remaining: int
    signal_changes: int
    transition_time_percentage: float


def run_episode(controller: str, seed: int, agent: object | None) -> EpisodeResult:
    """Reuse the existing diagnostic runner and shared traffic metrics."""
    values = _run_episode(
        controller=controller,
        seed=seed,
        agent=agent,
        episode_seconds=EPISODE_SECONDS,
    ).values
    generated = int(values["vehicles_generated"])  # SUMO loaded vehicles
    completed = int(values["vehicles_completed"])
    return EpisodeResult(
        seed=seed,
        controller=controller,
        throughput=completed,
        mean_waiting_time=float(values["mean_waiting_time"]),
        mean_queue_length=float(values["mean_queue_length"]),
        vehicles_generated=generated,
        vehicles_remaining=generated - completed,  # active plus pending insertion
        signal_changes=int(values["signal_changes"]),
        transition_time_percentage=100.0 * float(values["transition_proportion"]),
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    """Replace a CSV only after its full contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def summarize(results: Sequence[EpisodeResult]) -> list[dict]:
    """Return controller means and sample standard deviations for every metric."""
    rows = []
    for controller in ("dqn", "fixed_time"):
        group = [item for item in results if item.controller == controller]
        if not group:
            raise ValueError(f"missing results for {controller}")
        row: dict[str, str | int | float] = {
            "controller": controller,
            "number_of_seeds": len(group),
        }
        for metric in METRICS:
            values = [float(getattr(item, metric)) for item in group]
            row[f"{metric}_mean"] = fmean(values)
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def paired_statistics(results: Sequence[EpisodeResult]) -> list[dict]:
    """Compute DQN-minus-fixed paired differences and two-sided t intervals."""
    by_key = {(item.controller, item.seed): item for item in results}
    if len(by_key) != len(results):
        raise ValueError("duplicate controller/seed result")
    dqn_seeds = {seed for controller, seed in by_key if controller == "dqn"}
    fixed_seeds = {seed for controller, seed in by_key if controller == "fixed_time"}
    if dqn_seeds != fixed_seeds or len(dqn_seeds) != 30:
        raise ValueError("paired comparison requires exactly 30 matching seeds")
    rows = []
    for metric in ("mean_waiting_time", "mean_queue_length", "throughput"):
        dqn = [float(getattr(by_key["dqn", seed], metric)) for seed in sorted(dqn_seeds)]
        fixed = [float(getattr(by_key["fixed_time", seed], metric)) for seed in sorted(dqn_seeds)]
        differences = [a - b for a, b in zip(dqn, fixed)]
        mean = fmean(differences)
        sd = stdev(differences)
        half_width = T_95_DF_29 * sd / math.sqrt(30)
        fixed_mean = fmean(fixed)
        rows.append({
            "metric": metric,
            "number_of_pairs": 30,
            "mean_difference": mean,
            "std_difference": sd,
            "ci_95_lower": mean - half_width,
            "ci_95_upper": mean + half_width,
            "relative_change_percentage": 100.0 * mean / fixed_mean if fixed_mean else "",
        })
    return rows


EpisodeRunner = Callable[[str, int, object | None], EpisodeResult]


def evaluate(
    *,
    checkpoint: Path = CHECKPOINT,
    output_dir: Path = OUTPUT_DIR,
    seeds: Sequence[int] = SEEDS,
    episode_runner: EpisodeRunner = run_episode,
) -> list[EpisodeResult]:
    """Run paired episodes, verify demand, and save episode and summary CSVs."""
    if len(seeds) != 30 or len(set(seeds)) != 30:
        raise ValueError("evaluation requires 30 distinct seeds")
    if set(seeds) & (set(range(10)) | set(range(100, 110))):
        raise ValueError("seeds overlap known training or prior evaluation seeds")
    if EPISODE_SECONDS % DECISION_INTERVAL:
        raise ValueError("episode duration must divide evenly into decisions")
    set_deterministic_seed(0)
    agent = load_agent(checkpoint)
    results: list[EpisodeResult] = []
    for seed in seeds:
        pair = []
        for controller in ("dqn", "fixed_time"):
            set_deterministic_seed(seed)
            result = episode_runner(controller, seed, agent if controller == "dqn" else None)
            if result.seed != seed or result.controller != controller:
                raise ValueError(f"incorrect result for {controller}/{seed}")
            pair.append(result)
        if pair[0].vehicles_generated != pair[1].vehicles_generated:
            raise ValueError(f"demand differs for seed {seed}: {pair[0].vehicles_generated} vs {pair[1].vehicles_generated}")
        results.extend(pair)
        _write_csv(output_dir / "episodes.csv", EPISODE_FIELDS, [asdict(item) for item in results])
        print(f"Completed seed {seed}: generated={pair[0].vehicles_generated}", flush=True)
    _write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summarize(results))
    _write_csv(output_dir / "paired_differences.csv", PAIRED_FIELDS, paired_statistics(results))
    return results


def main() -> None:
    """Run the default 30-seed comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    evaluate(checkpoint=args.checkpoint, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
