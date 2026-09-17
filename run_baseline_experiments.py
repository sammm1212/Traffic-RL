"""Compare the random and fixed-time controllers over identical SUMO seeds."""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from tempfile import NamedTemporaryFile
from typing import Callable, Mapping, Sequence

import traci

from src.agents.random_agent import RandomAgent
from src.environment.traffic_env import TrafficEnvironment
from src.simulation.run import build_sumo_command
from src.simulation.traffic import TrafficSimulation


SEEDS = list(range(10))
EPISODE_SECONDS = 300
DECISION_INTERVAL = 5
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PER_SEED_PATH = RESULTS_DIR / "baseline_per_seed.csv"
SUMMARY_PATH = RESULTS_DIR / "baseline_summary.csv"

PER_SEED_FIELDS = (
    "seed",
    "random_reward",
    "random_generated",
    "random_completed",
    "fixed_reward",
    "fixed_generated",
    "fixed_completed",
)
SUMMARY_FIELDS = (
    "controller",
    "reward_mean",
    "reward_sd",
    "generated_mean",
    "generated_sd",
    "completed_mean",
    "completed_sd",
)


@dataclass(frozen=True)
class EpisodeResult:
    """Metrics needed to compare one controller episode."""

    seed: int
    total_reward: float
    vehicles_generated: int
    vehicles_completed: int


@dataclass(frozen=True)
class SummaryStatistics:
    """Sample statistics across a controller's episodes."""

    reward_mean: float
    reward_sd: float
    generated_mean: float
    generated_sd: float
    completed_mean: float
    completed_sd: float


def _run_with_fresh_sumo(
    seed: int,
    episode: Callable[[TrafficSimulation], EpisodeResult],
) -> EpisodeResult:
    """Start a fresh seeded SUMO process, run one episode, and always close it."""
    command = build_sumo_command(
        gui=False,
        seed=seed,
        steps=EPISODE_SECONDS,
    )
    started = False
    try:
        traci.start(command)
        started = True
        simulation = TrafficSimulation(traci, reload_args=command[1:])
        return episode(simulation)
    finally:
        if started:
            # Preserve an in-flight KeyboardInterrupt or episode exception if
            # SUMO has already closed its socket and TraCI close also fails.
            episode_failed = sys.exc_info()[0] is not None
            try:
                traci.close()
            except Exception:
                if not episode_failed:
                    raise


def run_random_episode(seed: int) -> EpisodeResult:
    """Run one reproducible random-controller episode."""

    def episode(simulation: TrafficSimulation) -> EpisodeResult:
        environment = TrafficEnvironment(
            simulation=simulation,
            decision_interval=DECISION_INTERVAL,
        )
        agent = RandomAgent()

        # Gymnasium does not seed spaces when Env.reset(seed=...) is called.
        # Seed both so future environment randomness and action sampling are stable.
        environment.reset(seed=seed)
        environment.action_space.seed(seed)

        total_reward = 0.0
        while True:
            previous_time = simulation.observe().time
            action = agent.select_action(environment)
            _, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward
            current_time = simulation.observe().time
            if current_time != previous_time + DECISION_INTERVAL:
                raise RuntimeError(
                    "random controller did not advance by exactly one "
                    f"decision interval: {previous_time} -> {current_time}"
                )
            if terminated or truncated:
                break

        metrics = simulation.observe()
        if metrics.time != EPISODE_SECONDS:
            raise RuntimeError(
                f"random episode ended at {metrics.time}, expected {EPISODE_SECONDS}"
            )
        return EpisodeResult(
            seed=seed,
            total_reward=total_reward,
            vehicles_generated=metrics.vehicles_generated,
            vehicles_completed=metrics.vehicles_completed,
        )

    return _run_with_fresh_sumo(seed, episode)


def run_fixed_time_episode(seed: int) -> EpisodeResult:
    """Run one episode under SUMO's unchanged fixed-time signal program."""

    def episode(simulation: TrafficSimulation) -> EpisodeResult:
        # This environment is used only to reuse its reward definition. Calling
        # simulation.step() directly leaves SUMO's fixed-time program in control.
        environment = TrafficEnvironment(
            simulation=simulation,
            decision_interval=DECISION_INTERVAL,
        )
        total_reward = 0.0

        for _ in range(EPISODE_SECONDS // DECISION_INTERVAL):
            for _ in range(DECISION_INTERVAL):
                simulation.step()
            total_reward += environment._get_reward()

        metrics = simulation.observe()
        if metrics.time != EPISODE_SECONDS:
            raise RuntimeError(
                f"fixed-time episode ended at {metrics.time}, expected {EPISODE_SECONDS}"
            )
        return EpisodeResult(
            seed=seed,
            total_reward=total_reward,
            vehicles_generated=metrics.vehicles_generated,
            vehicles_completed=metrics.vehicles_completed,
        )

    return _run_with_fresh_sumo(seed, episode)


def summarize(results: Sequence[EpisodeResult]) -> SummaryStatistics:
    """Calculate means and sample standard deviations for episode results."""
    if len(results) < 2:
        raise ValueError("at least two episode results are required")

    rewards = [result.total_reward for result in results]
    generated = [result.vehicles_generated for result in results]
    completed = [result.vehicles_completed for result in results]
    return SummaryStatistics(
        reward_mean=fmean(rewards),
        reward_sd=stdev(rewards),
        generated_mean=fmean(generated),
        generated_sd=stdev(generated),
        completed_mean=fmean(completed),
        completed_sd=stdev(completed),
    )


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Replace a CSV atomically so interruption cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parse_result(
    row: Mapping[str, str],
    controller: str,
) -> EpisodeResult | None:
    """Parse one controller result, treating blank columns as incomplete."""
    values = [
        row.get(f"{controller}_reward", "").strip(),
        row.get(f"{controller}_generated", "").strip(),
        row.get(f"{controller}_completed", "").strip(),
    ]
    if not any(values):
        return None
    if not all(values):
        raise ValueError(f"incomplete {controller} result")

    seed = int(row["seed"])
    reward = float(values[0])
    generated = int(values[1])
    completed = int(values[2])
    if not math.isfinite(reward) or generated < 0 or completed < 0:
        raise ValueError(f"invalid {controller} result")
    return EpisodeResult(seed, reward, generated, completed)


def load_per_seed_results(
    path: Path = PER_SEED_PATH,
) -> dict[tuple[str, int], EpisodeResult]:
    """Load valid completed controller/seed pairs from an existing CSV."""
    if not path.exists():
        return {}

    results: dict[tuple[str, int], EpisodeResult] = {}
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != list(PER_SEED_FIELDS):
            raise ValueError(f"unexpected columns in {path}")
        for line_number, row in enumerate(reader, start=2):
            try:
                seed = int(row["seed"])
                for controller in ("random", "fixed"):
                    result = _parse_result(row, controller)
                    if result is not None:
                        results[(controller, seed)] = result
            except (KeyError, TypeError, ValueError) as error:
                print(
                    f"Ignoring invalid row {line_number} in {path}: {error}",
                    flush=True,
                )
    return results


def save_per_seed_results(
    results: Mapping[tuple[str, int], EpisodeResult],
    *,
    seeds: Sequence[int] = SEEDS,
    path: Path = PER_SEED_PATH,
) -> None:
    """Atomically save all complete and partial per-seed results."""
    rows: list[dict[str, object]] = []
    for seed in seeds:
        row: dict[str, object] = {field: "" for field in PER_SEED_FIELDS}
        row["seed"] = seed
        for controller in ("random", "fixed"):
            result = results.get((controller, seed))
            if result is not None:
                row[f"{controller}_reward"] = result.total_reward
                row[f"{controller}_generated"] = result.vehicles_generated
                row[f"{controller}_completed"] = result.vehicles_completed
        rows.append(row)
    _atomic_write_csv(path, PER_SEED_FIELDS, rows)


def save_summary_results(
    results: Mapping[tuple[str, int], EpisodeResult],
    *,
    seeds: Sequence[int] = SEEDS,
    path: Path = SUMMARY_PATH,
) -> bool:
    """Save valid controller summaries, returning whether any were available."""
    rows: list[dict[str, object]] = []
    for controller in ("random", "fixed"):
        controller_results = [
            results[(controller, seed)]
            for seed in seeds
            if (controller, seed) in results
        ]
        if len(controller_results) < 2:
            continue
        summary = summarize(controller_results)
        rows.append(
            {
                "controller": "fixed_time" if controller == "fixed" else controller,
                "reward_mean": summary.reward_mean,
                "reward_sd": summary.reward_sd,
                "generated_mean": summary.generated_mean,
                "generated_sd": summary.generated_sd,
                "completed_mean": summary.completed_mean,
                "completed_sd": summary.completed_sd,
            }
        )

    if not rows:
        return False
    _atomic_write_csv(path, SUMMARY_FIELDS, rows)
    return True


def print_results(name: str, results: Sequence[EpisodeResult]) -> None:
    """Print a per-seed table and aggregate statistics for one controller."""
    print(f"\n{name.title()} controller\n")
    print(f"{'Seed':>4} | {'Reward':>8} | {'Generated':>9} | {'Completed':>9}")
    print("-----|----------|-----------|----------")
    for result in results:
        print(
            f"{result.seed:>4} | {result.total_reward:>8.1f} | "
            f"{result.vehicles_generated:>9} | {result.vehicles_completed:>9}"
        )

    summary = summarize(results)
    heading = f"{name.upper()} CONTROLLER"
    print(f"\n{heading}\n{'-' * len(heading)}")
    print(f"Total reward:\nMean: {summary.reward_mean:.3f}\nSD: {summary.reward_sd:.3f}")
    print(
        "\nVehicles generated:\n"
        f"Mean: {summary.generated_mean:.3f}\nSD: {summary.generated_sd:.3f}"
    )
    print(
        "\nVehicles completed:\n"
        f"Mean: {summary.completed_mean:.3f}\nSD: {summary.completed_sd:.3f}"
    )


def run_experiment(
    seeds: Sequence[int] = SEEDS,
    *,
    per_seed_path: Path = PER_SEED_PATH,
    summary_path: Path = SUMMARY_PATH,
    episode_runners: Mapping[str, Callable[[int], EpisodeResult]] | None = None,
) -> tuple[list[EpisodeResult], list[EpisodeResult]]:
    """Resume and run both controllers over the same ordered seed sequence."""
    runners = episode_runners or {
        "random": run_random_episode,
        "fixed": run_fixed_time_episode,
    }
    results = load_per_seed_results(per_seed_path)

    for controller in ("random", "fixed"):
        display_name = "fixed-time" if controller == "fixed" else controller
        for seed in seeds:
            key = (controller, seed)
            if key in results:
                print(
                    f"Skipping {display_name} controller seed {seed}: "
                    "existing result found",
                    flush=True,
                )
                continue

            print(f"Running {display_name} controller, seed {seed}...", flush=True)
            result = runners[controller](seed)
            if result.seed != seed:
                raise ValueError(
                    f"{display_name} runner returned seed {result.seed} for seed {seed}"
                )
            results[key] = result
            save_per_seed_results(results, seeds=seeds, path=per_seed_path)
            save_summary_results(results, seeds=seeds, path=summary_path)
            print(
                f"Completed {display_name} seed {seed}: "
                f"reward={result.total_reward:.1f}, "
                f"generated={result.vehicles_generated}, "
                f"completed={result.vehicles_completed}",
                flush=True,
            )

    random_results = [results[("random", seed)] for seed in seeds]
    fixed_results = [results[("fixed", seed)] for seed in seeds]
    # Rebuild a missing or stale summary even when every episode was skipped.
    save_summary_results(results, seeds=seeds, path=summary_path)
    return random_results, fixed_results


def main() -> None:
    """Run, resume, and report the ten-seed baseline comparison."""
    try:
        random_results, fixed_time_results = run_experiment()
    except KeyboardInterrupt:
        print(
            "\nExperiment interrupted. All completed results were saved to "
            f"{PER_SEED_PATH}",
            flush=True,
        )
        if SUMMARY_PATH.exists():
            print(f"Available summary statistics: {SUMMARY_PATH}", flush=True)
        return

    print_results("Random", random_results)
    print_results("Fixed-time", fixed_time_results)
    print(f"\nPer-seed results: {PER_SEED_PATH}")
    print(f"Summary results: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
