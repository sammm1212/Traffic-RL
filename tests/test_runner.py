from pathlib import Path
import unittest
from unittest.mock import patch

from src.simulation.run import build_sumo_command, parse_args, run_simulation


class RunnerTests(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        args = parse_args([])

        self.assertFalse(args.gui)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.steps, 600)
        self.assertIsNone(args.record)

    @patch("src.simulation.run.checkBinary", return_value="/tools/sumo-gui")
    def test_command_contains_runtime_options(self, mock_check_binary) -> None:
        command = build_sumo_command(
            gui=True,
            seed=7,
            steps=120,
            route_path=Path("scenario.rou.xml"),
        )

        mock_check_binary.assert_called_once_with("sumo-gui")
        self.assertEqual(command[0], "/tools/sumo-gui")
        self.assertEqual(Path(command[command.index("-c") + 1]).name, "intersection.sumocfg")
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(command[command.index("--end") + 1], "120")
        self.assertTrue(command[command.index("--route-files") + 1].endswith("scenario.rou.xml"))

    def test_rejects_non_positive_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            run_simulation(steps=0)
