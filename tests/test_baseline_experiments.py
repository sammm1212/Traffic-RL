import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from run_baseline_experiments import (
    DECISION_INTERVAL,
    EPISODE_SECONDS,
    EpisodeResult,
    _run_with_fresh_sumo,
    load_per_seed_results,
    main,
    run_experiment,
    summarize,
)


class BaselineExperimentTests(unittest.TestCase):
    def test_summary_uses_sample_standard_deviation(self) -> None:
        results = [
            EpisodeResult(0, -10.0, 10, 8),
            EpisodeResult(1, -20.0, 14, 12),
        ]

        summary = summarize(results)

        self.assertEqual(summary.reward_mean, -15.0)
        self.assertAlmostEqual(summary.reward_sd, 7.0710678118654755)
        self.assertEqual(summary.generated_mean, 12.0)
        self.assertAlmostEqual(summary.generated_sd, 2.8284271247461903)
        self.assertEqual(summary.completed_mean, 10.0)
        self.assertAlmostEqual(summary.completed_sd, 2.8284271247461903)

    def test_episode_length_is_divisible_by_decision_interval(self) -> None:
        self.assertEqual(EPISODE_SECONDS % DECISION_INTERVAL, 0)
        self.assertEqual(EPISODE_SECONDS // DECISION_INTERVAL, 60)

    @patch("run_baseline_experiments.TrafficSimulation")
    @patch("run_baseline_experiments.build_sumo_command")
    @patch("run_baseline_experiments.traci")
    def test_fresh_sumo_is_closed_after_episode_error(
        self,
        mock_traci: MagicMock,
        mock_build_command: MagicMock,
        mock_simulation: MagicMock,
    ) -> None:
        command = ["sumo", "--seed", "4"]
        mock_build_command.return_value = command

        def fail(_simulation):
            raise RuntimeError("episode failed")

        with self.assertRaisesRegex(RuntimeError, "episode failed"):
            _run_with_fresh_sumo(4, fail)

        mock_traci.start.assert_called_once_with(command)
        mock_simulation.assert_called_once_with(mock_traci, reload_args=command[1:])
        mock_traci.close.assert_called_once_with()

    def test_partial_results_are_saved_and_resume_skips_them(self) -> None:
        with TemporaryDirectory() as directory:
            per_seed_path = Path(directory) / "baseline_per_seed.csv"
            summary_path = Path(directory) / "baseline_summary.csv"
            first_calls = []

            def interrupted_random(seed: int) -> EpisodeResult:
                first_calls.append(seed)
                if seed == 1:
                    raise KeyboardInterrupt
                return EpisodeResult(seed, -10.0, 10, 8)

            with self.assertRaises(KeyboardInterrupt):
                run_experiment(
                    seeds=[0, 1],
                    per_seed_path=per_seed_path,
                    summary_path=summary_path,
                    episode_runners={
                        "random": interrupted_random,
                        "fixed": MagicMock(),
                    },
                )

            self.assertEqual(first_calls, [0, 1])
            self.assertFalse(summary_path.exists())
            self.assertEqual(
                load_per_seed_results(per_seed_path),
                {("random", 0): EpisodeResult(0, -10.0, 10, 8)},
            )

            resumed_calls = []

            def resumed_random(seed: int) -> EpisodeResult:
                resumed_calls.append(("random", seed))
                return EpisodeResult(seed, -20.0, 12, 9)

            def resumed_fixed(seed: int) -> EpisodeResult:
                resumed_calls.append(("fixed", seed))
                return EpisodeResult(seed, -5.0 - seed, 11, 10)

            output = StringIO()
            with redirect_stdout(output):
                random_results, fixed_results = run_experiment(
                    seeds=[0, 1],
                    per_seed_path=per_seed_path,
                    summary_path=summary_path,
                    episode_runners={
                        "random": resumed_random,
                        "fixed": resumed_fixed,
                    },
                )

            self.assertEqual(
                resumed_calls,
                [("random", 1), ("fixed", 0), ("fixed", 1)],
            )
            self.assertIn(
                "Skipping random controller seed 0: existing result found",
                output.getvalue(),
            )
            self.assertEqual([result.seed for result in random_results], [0, 1])
            self.assertEqual([result.seed for result in fixed_results], [0, 1])

            with per_seed_path.open(encoding="utf-8", newline="") as input_file:
                per_seed_rows = list(csv.DictReader(input_file))
            with summary_path.open(encoding="utf-8", newline="") as input_file:
                summary_rows = list(csv.DictReader(input_file))

            self.assertEqual(len(per_seed_rows), 2)
            self.assertEqual(per_seed_rows[0]["random_reward"], "-10.0")
            self.assertEqual(per_seed_rows[0]["fixed_reward"], "-5.0")
            self.assertEqual(
                [row["controller"] for row in summary_rows],
                ["random", "fixed_time"],
            )

    @patch("run_baseline_experiments.run_experiment", side_effect=KeyboardInterrupt)
    def test_main_reports_clean_interruption(self, _mock_run: MagicMock) -> None:
        output = StringIO()
        with redirect_stdout(output):
            main()

        self.assertIn("Experiment interrupted", output.getvalue())
        self.assertIn("baseline_per_seed.csv", output.getvalue())


if __name__ == "__main__":
    unittest.main()
