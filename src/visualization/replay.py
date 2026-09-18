"""Replay a recorded traffic episode in a lightweight Pygame interface."""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


APPROACHES = ("north", "south", "east", "west")
REQUIRED_FIELDS = {
    "time",
    *(f"{approach}_queue" for approach in APPROACHES),
    "light_phase",
    "throughput",
}
PLAYBACK_SPEEDS = (1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class ReplayFrame:
    """One immutable frame read from an episode recording."""

    time: float
    queues: dict[str, int]
    light_phase: int
    throughput: int
    mean_waiting_time: float | None = None

    @property
    def total_queue(self) -> int:
        """Return the queue across all four approaches."""
        return sum(self.queues.values())


@dataclass(frozen=True)
class EpisodeRecording:
    """Chronological frames loaded from one CSV or JSON recording."""

    source: Path
    frames: tuple[ReplayFrame, ...]

    @property
    def times(self) -> tuple[float, ...]:
        """Return frame timestamps for playback lookup."""
        return tuple(frame.time for frame in self.frames)


def _number(row: Mapping[str, Any], field: str, row_number: int) -> float:
    """Read one required recording field as a number with row-aware errors."""
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"row {row_number}: missing value for {field}")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: {field} must be numeric") from error


def _parse_frame(row: Mapping[str, Any], row_number: int) -> ReplayFrame:
    """Validate and convert one serialized metric row into a replay frame."""
    missing = REQUIRED_FIELDS.difference(row)
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"row {row_number}: missing required fields: {fields}")

    queues = {
        approach: int(_number(row, f"{approach}_queue", row_number))
        for approach in APPROACHES
    }
    if any(queue < 0 for queue in queues.values()):
        raise ValueError(f"row {row_number}: queue lengths must be non-negative")

    waiting = row.get("mean_waiting_time")
    return ReplayFrame(
        time=_number(row, "time", row_number),
        queues=queues,
        light_phase=int(_number(row, "light_phase", row_number)),
        throughput=int(_number(row, "throughput", row_number)),
        mean_waiting_time=(
            None if waiting is None or waiting == "" else _number(row, "mean_waiting_time", row_number)
        ),
    )


def load_recording(path: Path) -> EpisodeRecording:
    """Load the existing per-step recording schema from CSV or JSON."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as input_file:
            rows: list[Mapping[str, Any]] = list(csv.DictReader(input_file))
    elif suffix == ".json":
        with path.open(encoding="utf-8") as input_file:
            data = json.load(input_file)
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise ValueError("JSON recording must be an array of objects")
        rows = data
    else:
        raise ValueError("recording path must end in .csv or .json")

    if not rows:
        raise ValueError("recording contains no frames")
    frames = tuple(_parse_frame(row, index) for index, row in enumerate(rows, start=1))
    if any(current.time < previous.time for previous, current in zip(frames, frames[1:])):
        raise ValueError("recording frames must be in chronological order")
    return EpisodeRecording(path, frames)


class ReplayPlayer:
    """Advance through recorded timestamps without generating traffic state."""

    def __init__(self, recording: EpisodeRecording) -> None:
        """Start playback at the recording's first frame and normal speed."""
        self.recording = recording
        self.frame_index = 0
        self.playing = True
        self.speed = PLAYBACK_SPEEDS[0]
        self._playback_time = recording.frames[0].time

    @property
    def frame(self) -> ReplayFrame:
        """Return the currently displayed recorded frame."""
        return self.recording.frames[self.frame_index]

    def toggle_playback(self) -> None:
        """Toggle between playing and paused states."""
        self.playing = not self.playing

    def restart(self) -> None:
        """Return to the first frame and resume playback."""
        self.frame_index = 0
        self._playback_time = self.recording.frames[0].time
        self.playing = True

    def set_speed(self, speed: float) -> None:
        """Select one of the supported playback rates."""
        if speed not in PLAYBACK_SPEEDS:
            raise ValueError(f"unsupported playback speed: {speed}")
        self.speed = speed

    def advance(self, elapsed_seconds: float) -> None:
        """Advance using wall-clock time and recorded frame timestamps."""
        if not self.playing or elapsed_seconds <= 0:
            return
        self._playback_time += elapsed_seconds * self.speed
        last_index = len(self.recording.frames) - 1
        self.frame_index = min(
            bisect_right(self.recording.times, self._playback_time) - 1,
            last_index,
        )
        if self.frame_index == last_index:
            self._playback_time = self.recording.frames[last_index].time
            self.playing = False


PHASE_LABELS = {
    0: "North–south green",
    1: "North–south yellow",
    2: "All red",
    3: "East–west green",
    4: "East–west yellow",
    5: "All red",
}


class JunctionPanel:
    """Draw one recording; a future comparison can instantiate this twice."""

    BACKGROUND = (22, 29, 38)
    ROAD = (52, 60, 70)
    ROAD_LINE = (189, 197, 208)
    TEXT = (236, 241, 247)
    MUTED = (160, 171, 185)
    QUEUE = (68, 155, 230)
    GREEN = (55, 201, 117)
    YELLOW = (246, 196, 68)
    RED = (235, 87, 87)

    def __init__(self, pygame_module: Any) -> None:
        """Create fonts and retain the injected Pygame module used for drawing."""
        self.pg = pygame_module
        self.title_font = self.pg.font.SysFont("arial", 24, bold=True)
        self.body_font = self.pg.font.SysFont("arial", 18)
        self.small_font = self.pg.font.SysFont("arial", 15)

    def draw(self, surface: Any, bounds: Any, frame: ReplayFrame, title: str) -> None:
        """Render a single replay frame inside the supplied rectangle."""
        pg = self.pg
        pg.draw.rect(surface, self.BACKGROUND, bounds, border_radius=12)
        surface.blit(self.title_font.render(title, True, self.TEXT), (bounds.x + 22, bounds.y + 18))

        size = min(bounds.width - 300, bounds.height - 100)
        map_rect = pg.Rect(bounds.x + 20, bounds.y + 58, size, size)
        centre = map_rect.center
        road_width = max(92, size // 4)
        vertical = pg.Rect(centre[0] - road_width // 2, map_rect.y, road_width, size)
        horizontal = pg.Rect(map_rect.x, centre[1] - road_width // 2, size, road_width)
        pg.draw.rect(surface, self.ROAD, vertical)
        pg.draw.rect(surface, self.ROAD, horizontal)
        self._draw_lane_markings(surface, map_rect, road_width)
        self._draw_queues(surface, map_rect, road_width, frame)
        self._draw_signals(surface, map_rect, road_width, frame.light_phase)

        metrics_x = map_rect.right + 25
        self._draw_metrics(surface, metrics_x, bounds.y + 68, frame)

    def _draw_lane_markings(self, surface: Any, area: Any, road_width: int) -> None:
        pg = self.pg
        cx, cy = area.center
        gap = road_width // 2 + 8
        for y1, y2 in ((area.top, cy - gap), (cy + gap, area.bottom)):
            for y in range(y1, y2, 24):
                pg.draw.line(surface, self.ROAD_LINE, (cx, y), (cx, min(y + 12, y2)), 2)
        for x1, x2 in ((area.left, cx - gap), (cx + gap, area.right)):
            for x in range(x1, x2, 24):
                pg.draw.line(surface, self.ROAD_LINE, (x, cy), (min(x + 12, x2), cy), 2)

    def _draw_queues(self, surface: Any, area: Any, road_width: int, frame: ReplayFrame) -> None:
        pg = self.pg
        cx, cy = area.center
        queue_space = (area.width - road_width) // 2 - 28
        maximum = max(max(frame.queues.values()), 1)
        lengths = {
            name: round(queue_space * value / maximum)
            for name, value in frame.queues.items()
        }
        thickness = 18
        bars = {
            "north": pg.Rect(cx + 8, cy - road_width // 2 - lengths["north"] - 14, thickness, lengths["north"]),
            "south": pg.Rect(cx - 26, cy + road_width // 2 + 14, thickness, lengths["south"]),
            "east": pg.Rect(cx + road_width // 2 + 14, cy + 8, lengths["east"], thickness),
            "west": pg.Rect(cx - road_width // 2 - lengths["west"] - 14, cy - 26, lengths["west"], thickness),
        }
        for approach, bar in bars.items():
            if frame.queues[approach]:
                pg.draw.rect(surface, self.QUEUE, bar, border_radius=4)
        labels = {
            "north": (cx + road_width // 2 + 10, area.top + 8),
            "south": (cx + road_width // 2 + 10, area.bottom - 27),
            "east": (area.right - 78, cy - road_width // 2 - 28),
            "west": (area.left + 6, cy + road_width // 2 + 8),
        }
        for name, position in labels.items():
            label = f"{name[0].upper()}  {frame.queues[name]}"
            surface.blit(self.body_font.render(label, True, self.TEXT), position)

    def _draw_signals(self, surface: Any, area: Any, road_width: int, phase: int) -> None:
        pg = self.pg
        cx, cy = area.center
        ns_colour = self._signal_colour(phase, movement="ns")
        ew_colour = self._signal_colour(phase, movement="ew")
        offset = road_width // 2 + 11
        for point in ((cx - offset, cy - offset), (cx + offset, cy + offset)):
            pg.draw.circle(surface, ns_colour, point, 9)
        for point in ((cx + offset, cy - offset), (cx - offset, cy + offset)):
            pg.draw.circle(surface, ew_colour, point, 9)

    def _signal_colour(self, phase: int, *, movement: str) -> tuple[int, int, int]:
        green_phase = 0 if movement == "ns" else 3
        yellow_phase = 1 if movement == "ns" else 4
        if phase == green_phase:
            return self.GREEN
        if phase == yellow_phase:
            return self.YELLOW
        return self.RED

    def _draw_metrics(self, surface: Any, x: int, y: int, frame: ReplayFrame) -> None:
        lines = (
            ("Simulation time", f"{frame.time:.1f} s"),
            ("Total queue", str(frame.total_queue)),
            ("Throughput", str(frame.throughput)),
            ("Signal phase", PHASE_LABELS.get(frame.light_phase, f"Phase {frame.light_phase}")),
            (
                # This is the recording's live incoming-vehicle mean, not the
                # experiment-level queueing delay per generated vehicle.
                "Current mean vehicle wait",
                "not recorded" if frame.mean_waiting_time is None else f"{frame.mean_waiting_time:.1f} s",
            ),
        )
        for index, (label, value) in enumerate(lines):
            top = y + index * 70
            surface.blit(self.small_font.render(label.upper(), True, self.MUTED), (x, top))
            surface.blit(self.body_font.render(value, True, self.TEXT), (x, top + 24))


class ReplayApp:
    """Pygame application for one recorded episode."""

    WIDTH = 1024
    HEIGHT = 700

    def __init__(self, recording: EpisodeRecording) -> None:
        """Prepare replay state while deferring Pygame window creation to run()."""
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError(
                "Pygame is required for the replay UI; install project requirements"
            ) from error
        self.pg = pygame
        self.recording = recording
        self.player = ReplayPlayer(recording)
        self.panel: JunctionPanel | None = None
        self.buttons: list[tuple[Any, str, float | None]] = []

    def run(self) -> None:
        """Open the replay window and process controls until it is closed."""
        pg = self.pg
        pg.init()
        try:
            screen = pg.display.set_mode((self.WIDTH, self.HEIGHT))
            pg.display.set_caption(f"Traffic episode replay — {self.recording.source.name}")
            clock = pg.time.Clock()
            self.panel = JunctionPanel(pg)
            running = True
            while running:
                elapsed = clock.tick(60) / 1000.0
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        running = False
                    elif event.type == pg.KEYDOWN:
                        running = self._handle_key(event.key)
                    elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos)
                self.player.advance(elapsed)
                self._draw(screen)
                pg.display.flip()
        finally:
            pg.quit()

    def _handle_key(self, key: int) -> bool:
        pg = self.pg
        if key in (pg.K_ESCAPE, pg.K_q):
            return False
        if key == pg.K_SPACE:
            self.player.toggle_playback()
        elif key == pg.K_r:
            self.player.restart()
        elif key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
            self.player.set_speed(PLAYBACK_SPEEDS[key - pg.K_1])
        return True

    def _handle_click(self, position: tuple[int, int]) -> None:
        for bounds, action, value in self.buttons:
            if bounds.collidepoint(position):
                if action == "play":
                    self.player.toggle_playback()
                elif action == "restart":
                    self.player.restart()
                elif action == "speed" and value is not None:
                    self.player.set_speed(value)

    def _draw(self, screen: Any) -> None:
        pg = self.pg
        screen.fill((12, 17, 24))
        panel_bounds = pg.Rect(20, 20, self.WIDTH - 40, self.HEIGHT - 105)
        assert self.panel is not None
        self.panel.draw(screen, panel_bounds, self.player.frame, self.recording.source.stem)
        self._draw_controls(screen)

    def _draw_controls(self, screen: Any) -> None:
        pg = self.pg
        font = pg.font.SysFont("arial", 16, bold=True)
        controls: list[tuple[str, str, float | None]] = [
            ("Pause" if self.player.playing else "Play", "play", None),
            ("Restart", "restart", None),
            *((f"{speed:g}×", "speed", speed) for speed in PLAYBACK_SPEEDS),
        ]
        self.buttons = []
        x = 26
        for label, action, value in controls:
            width = 88 if action != "speed" else 58
            bounds = pg.Rect(x, self.HEIGHT - 65, width, 40)
            selected = action == "speed" and value == self.player.speed
            pg.draw.rect(screen, (49, 120, 198) if selected else (38, 48, 62), bounds, border_radius=7)
            text = font.render(label, True, (244, 247, 251))
            screen.blit(text, text.get_rect(center=bounds.center))
            self.buttons.append((bounds, action, value))
            x += width + 10
        hint = pg.font.SysFont("arial", 14).render(
            "Space play/pause  •  R restart  •  1–4 speed  •  Q quit",
            True,
            (150, 162, 177),
        )
        screen.blit(hint, (self.WIDTH - hint.get_width() - 26, self.HEIGHT - 52))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse replay command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="per-step .csv or .json recording")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load a recording and launch its replay window."""
    args = parse_args(argv)
    ReplayApp(load_recording(args.recording)).run()


if __name__ == "__main__":
    main()
