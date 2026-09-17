import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.visualization.replay import (
    EpisodeRecording,
    ReplayFrame,
    ReplayPlayer,
    load_recording,
)


ROWS = [
    {
        "time": 1.0,
        "north_queue": 1,
        "south_queue": 2,
        "east_queue": 3,
        "west_queue": 4,
        "light_phase": 0,
        "throughput": 0,
        "mean_waiting_time": 1.5,
    },
    {
        "time": 3.0,
        "north_queue": 2,
        "south_queue": 1,
        "east_queue": 0,
        "west_queue": 1,
        "light_phase": 1,
        "throughput": 3,
        "mean_waiting_time": 2.5,
    },
]


class RecordingLoaderTests(unittest.TestCase):
    def test_loads_existing_csv_schema(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episode.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=ROWS[0])
                writer.writeheader()
                writer.writerows(ROWS)

            recording = load_recording(path)

        self.assertEqual(len(recording.frames), 2)
        self.assertEqual(recording.frames[0].total_queue, 10)
        self.assertEqual(recording.frames[1].throughput, 3)

    def test_loads_existing_json_schema(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "episode.json"
            path.write_text(json.dumps(ROWS), encoding="utf-8")

            recording = load_recording(path)

        self.assertEqual(recording.frames[1].light_phase, 1)
        self.assertEqual(recording.frames[0].mean_waiting_time, 1.5)

    def test_rejects_aggregate_baseline_csv(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.csv"
            path.write_text("episode,seed,vehicles_completed\n1,1,10\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_recording(path)


class ReplayPlayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recording = EpisodeRecording(
            Path("episode.csv"),
            (
                ReplayFrame(1.0, dict.fromkeys(("north", "south", "east", "west"), 0), 0, 0),
                ReplayFrame(2.0, dict.fromkeys(("north", "south", "east", "west"), 1), 0, 1),
                ReplayFrame(4.0, dict.fromkeys(("north", "south", "east", "west"), 2), 3, 2),
            ),
        )

    def test_advances_using_recorded_timestamps_and_speed(self) -> None:
        player = ReplayPlayer(self.recording)
        player.set_speed(2.0)

        player.advance(0.6)

        self.assertEqual(player.frame.time, 2.0)

    def test_pause_and_restart(self) -> None:
        player = ReplayPlayer(self.recording)
        player.toggle_playback()
        player.advance(10.0)
        self.assertEqual(player.frame_index, 0)

        player.toggle_playback()
        player.advance(10.0)
        self.assertFalse(player.playing)
        self.assertEqual(player.frame_index, 2)

        player.restart()
        self.assertTrue(player.playing)
        self.assertEqual(player.frame_index, 0)

