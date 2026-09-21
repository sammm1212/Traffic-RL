"""Create presentation-ready plots from DQN training metrics."""

from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPOSITORY_ROOT / "results" / "training" / "training_metrics.csv"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "training" / "plots"
MOVING_AVERAGE_WINDOW = 10
REQUIRED_COLUMNS = {
    "episode",
    "total_reward",
    "mean_loss",
    "epsilon",
    "mean_waiting_time",
    "mean_queue_length",
    "throughput",
}

PLOT_SPECS = (
    ("total_reward", "DQN Training Reward", "Total Reward", "reward.png"),
    (
        "mean_queue_length",
        "Mean Queue Length During Training",
        "Mean Queue Length (vehicles)",
        "mean_queue_length.png",
    ),
    (
        "mean_waiting_time",
        "Mean Waiting Time During Training",
        "Mean Waiting Time (seconds)",
        "mean_waiting_time.png",
    ),
    ("throughput", "Throughput During Training", "Throughput (vehicles)", "throughput.png"),
)


def load_metrics(csv_path: Path) -> dict[str, list[float]]:
    """Load and validate the numeric columns needed for all plots."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Training metrics CSV not found: {csv_path}\n"
            "Run this script from a Traffic-RL checkout containing the training results."
        )

    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Training metrics CSV has no header: {csv_path}")

        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Training metrics CSV is missing required columns: {missing}")

        metrics = {column: [] for column in REQUIRED_COLUMNS}
        for row_number, row in enumerate(reader, start=2):
            for column in REQUIRED_COLUMNS:
                raw_value = row.get(column, "").strip()
                if not raw_value:
                    raise ValueError(
                        f"Missing value in column '{column}' at CSV row {row_number}."
                    )
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value {raw_value!r} in column '{column}' "
                        f"at CSV row {row_number}."
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value {raw_value!r} in column '{column}' "
                        f"at CSV row {row_number}."
                    )
                metrics[column].append(value)

    if not metrics["episode"]:
        raise ValueError(f"Training metrics CSV contains no data rows: {csv_path}")

    return metrics


def moving_average(values: list[float], window: int = MOVING_AVERAGE_WINDOW) -> list[float]:
    """Return a rolling average, using all available values at the start."""
    rolling_values: deque[float] = deque()
    rolling_sum = 0.0
    averages: list[float] = []

    for value in values:
        rolling_values.append(value)
        rolling_sum += value
        if len(rolling_values) > window:
            rolling_sum -= rolling_values.popleft()
        averages.append(rolling_sum / len(rolling_values))

    return averages


def style_axis(axis: plt.Axes, title: str, y_label: str) -> None:
    """Apply consistent, restrained styling to a plot axis."""
    axis.set_title(title, fontsize=14, pad=12)
    axis.set_xlabel("Episode")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.25, linewidth=0.8)
    axis.legend(frameon=False)


def plot_metric(
    episodes: list[float],
    values: list[float],
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Save one raw metric and its moving average as a high-resolution PNG."""
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(episodes, values, color="#4C78A8", alpha=0.3, linewidth=1.0, label="Raw")
    axis.plot(
        episodes,
        moving_average(values),
        color="#1F4E79",
        linewidth=2.4,
        label=f"{MOVING_AVERAGE_WINDOW}-episode moving average",
    )
    style_axis(axis, title, y_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_diagnostics(metrics: dict[str, list[float]], output_path: Path) -> None:
    """Save loss and epsilon in separate panels so their scales stay readable."""
    episodes = metrics["episode"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    diagnostic_specs = (
        (axes[0], "mean_loss", "Mean Loss During Training", "Mean Loss", "#E45756", "#9C2F2E"),
        (axes[1], "epsilon", "Epsilon During Training", "Epsilon", "#72B7B2", "#287D78"),
    )
    for axis, column, title, y_label, raw_color, average_color in diagnostic_specs:
        values = metrics[column]
        axis.plot(episodes, values, color=raw_color, alpha=0.3, linewidth=1.0, label="Raw")
        axis.plot(
            episodes,
            moving_average(values),
            color=average_color,
            linewidth=2.4,
            label=f"{MOVING_AVERAGE_WINDOW}-episode moving average",
        )
        style_axis(axis, title, y_label)

    axes[0].set_xlabel("")
    figure.suptitle("DQN Training Diagnostics", fontsize=16, y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Load the training CSV and generate all requested plots."""
    metrics = load_metrics(METRICS_PATH)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for column, title, y_label, filename in PLOT_SPECS:
        plot_metric(
            metrics["episode"],
            metrics[column],
            title,
            y_label,
            OUTPUT_DIRECTORY / filename,
        )

    plot_diagnostics(metrics, OUTPUT_DIRECTORY / "training_diagnostics.png")
    print(f"Created 5 plots in {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
