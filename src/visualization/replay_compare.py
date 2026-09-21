"""Side-by-side replay of the frozen final-evaluation second records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .replay import APPROACHES, JunctionPanel, PHASE_LABELS, ReplayFrame


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "results/final_evaluation"
CHECKPOINT = ROOT / "results/training/dqn_100_episode_min_green.pt"
CHECKPOINT_SHA256 = "e820fb9912eab609f7e10e295d01375e944faf655f4bd9c660d7e8cbcb22c4d0"
SCENARIOS = ("balanced", "ns_heavy", "ew_heavy", "changing")
SCENARIO_NAMES = {"balanced": "Balanced", "ns_heavy": "NS-heavy", "ew_heavy": "EW-heavy", "changing": "Changing demand"}
PHASE_KINDS = {0: "ns_green", 1: "yellow", 2: "all_red", 3: "ew_green", 4: "yellow", 5: "all_red"}
SPEEDS = (1, 2, 5)


def digest(path: Path) -> str:
    """Return the SHA-256 of an immutable evaluation input or record."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def demand_period(scenario: str, second: int) -> str:
    """Describe the scheduled arrivals for the recorded second."""
    if scenario != "changing":
        return SCENARIO_NAMES[scenario]
    return ("Balanced · 0–99 s", "NS-heavy · 100–199 s", "EW-heavy · 200–299 s")[min(second // 100, 2)]


@dataclass(frozen=True)
class DemoFrame:
    """One governed second with evaluator-compatible cumulative measurements."""

    second: int
    replay: ReplayFrame
    inserted: int
    mean_queue: float
    waiting_per_inserted: float
    changes: int
    transition_seconds: int

    @property
    def summary_values(self) -> tuple[float, float, float, float, float]:
        """Values in the order shown in the episode summary."""
        return (self.replay.throughput, self.mean_queue, self.waiting_per_inserted,
                self.changes, self.transition_seconds)


@dataclass(frozen=True)
class DemoPair:
    """Matched final-evaluation episodes sharing one demand and timeline."""

    scenario: str
    seed: int
    demand_sha256: str
    fixed: tuple[DemoFrame, ...]
    dqn: tuple[DemoFrame, ...]
    queue_scale: int

    @property
    def duration(self) -> int:
        """Return the number of recorded simulation seconds."""
        return len(self.fixed)


def _integer(row: dict[str, str], key: str, location: str) -> int:
    try:
        raw = row[key]
        value = int(raw)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{location}: invalid {key}") from error
    if value < 0:
        raise ValueError(f"{location}: negative {key}")
    return value


def _load_episode(data: Path, scenario: str, seed: int, controller: str,
                  expected: dict[str, str]) -> tuple[tuple[DemoFrame, ...], str]:
    base = data / "records" / scenario / f"seed_{seed}_{controller}"
    record = base.with_name(base.name + "_seconds.csv")
    manifest_path = base.with_name(base.name + "_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        with record.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"missing or malformed replay: {base}") from error
    for key, value in (("scenario", scenario), ("seed", seed), ("controller", controller),
                       ("episode_seconds", 300), ("decision_interval_seconds", 5),
                       ("checkpoint_sha256", CHECKPOINT_SHA256),
                       ("minimum_green_seconds", 15 if controller == "dqn_15" else None)):
        if manifest.get(key) != value:
            raise ValueError(f"{manifest_path}: invalid {key}")
    if digest(record) != manifest.get("seconds_sha256"):
        raise ValueError(f"record hash mismatch: {record}")
    if len(rows) != 300:
        raise ValueError(f"{record}: expected 300 contiguous seconds")
    queue_seconds = inserted = changes = transitions = 0
    frames: list[DemoFrame] = []
    previous_phase: int | None = None
    previous_throughput = 0
    for second, row in enumerate(rows):
        location = f"{record}: row {second + 2}"
        if row.get("scenario") != scenario or row.get("controller") != ("dqn" if controller == "dqn_15" else controller):
            raise ValueError(f"{location}: scenario or controller mismatch")
        if (_integer(row, "seed", location) != seed or
                _integer(row, "second", location) != second or
                _integer(row, "time_start", location) != second or
                _integer(row, "time_end", location) != second + 1):
            raise ValueError(f"{location}: timeline mismatch")
        phase = _integer(row, "phase", location)
        if phase not in PHASE_KINDS or row.get("phase_kind") != PHASE_KINDS[phase]:
            raise ValueError(f"{location}: invalid signal phase")
        if previous_phase is not None and phase not in (previous_phase, (previous_phase + 1) % 6):
            raise ValueError(f"{location}: impossible signal phase jump")
        previous_phase = phase
        queues = {name: _integer(row, f"{name}_queue", location) for name in APPROACHES}
        if sum(queues[name] for name in ("north", "south")) != _integer(row, "ns_queue", location) or sum(queues[name] for name in ("east", "west")) != _integer(row, "ew_queue", location):
            raise ValueError(f"{location}: directional queues disagree")
        completed = _integer(row, "throughput_cumulative", location)
        if completed < previous_throughput:
            raise ValueError(f"{location}: throughput decreased")
        previous_throughput = completed
        inserted += sum(_integer(row, f"{name}_inserted", location) for name in APPROACHES)
        queue_seconds += sum(queues.values())
        changes += _integer(row, "executed_direction_change", location)
        transitions += phase in (1, 2, 4, 5)
        frames.append(DemoFrame(second, ReplayFrame(second + 1, queues, phase, completed),
                                inserted, queue_seconds / (second + 1),
                                queue_seconds / inserted if inserted else 0.0,
                                changes, transitions))
    final = frames[-1]
    checks = {"vehicles_completed": final.replay.throughput, "inserted_vehicles": final.inserted,
              "mean_queue_length": final.mean_queue, "mean_waiting_time": final.waiting_per_inserted,
              "executed_direction_changes": final.changes,
              "transition_seconds": final.transition_seconds}
    for key, actual in checks.items():
        try:
            saved = float(expected[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"saved episode is missing {key}") from error
        if abs(actual - saved) > 1e-8:
            raise ValueError(f"{record}: final {key} differs from saved evaluation")
    if expected.get("seconds_sha256") != manifest["seconds_sha256"] or expected.get("demand_sha256") != manifest.get("demand_sha256"):
        raise ValueError(f"{record}: saved episode provenance mismatch")
    return tuple(frames), manifest["demand_sha256"]


def load_pair(scenario: str = "balanced", seed: int = 4000, data: Path = DEFAULT_DATA,
              checkpoint: Path = CHECKPOINT) -> DemoPair:
    """Validate and load the matched immutable final-evaluation recordings."""
    if scenario not in SCENARIOS or seed not in range(4000, 4030):
        raise ValueError("select one final-evaluation scenario and seed 4000–4029")
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("frozen checkpoint SHA-256 mismatch")
    route = data / "demand" / scenario / f"seed_{seed}.rou.xml"
    try:
        with (data / "episodes.csv").open(newline="", encoding="utf-8") as source:
            selected = [row for row in csv.DictReader(source)
                        if row.get("scenario") == scenario and row.get("seed") == str(seed)
                        and row.get("controller") in ("fixed_time", "dqn_15")]
    except OSError as error:
        raise ValueError(f"missing evaluation episodes: {data / 'episodes.csv'}") from error
    expected = {row["controller"]: row for row in selected}
    if len(selected) != 2 or len(expected) != 2:
        raise ValueError(f"missing or duplicate saved episode: {scenario}/{seed}")
    try:
        demand_hash = digest(route)
    except OSError as error:
        raise ValueError(f"missing saved demand file: {route}") from error
    fixed, fixed_hash = _load_episode(data, scenario, seed, "fixed_time", expected["fixed_time"])
    dqn, dqn_hash = _load_episode(data, scenario, seed, "dqn_15", expected["dqn_15"])
    if fixed_hash != dqn_hash or fixed_hash != demand_hash:
        raise ValueError("paired demand hash mismatch")
    maximum = max(frame.replay.queues[name] for recording in (fixed, dqn)
                  for frame in recording for name in APPROACHES)
    return DemoPair(scenario, seed, demand_hash, fixed, dqn, max(maximum, 1))


class PairPlayer:
    """Drive both recordings from a single integer simulation clock."""

    def __init__(self, pair: DemoPair) -> None:
        self.pair = pair
        self.second = 0
        self.playing = True
        self.speed = 1
        self._fraction = 0.0

    def advance(self, elapsed: float) -> None:
        """Advance both panels by recorded seconds at the selected speed."""
        if not self.playing or elapsed <= 0:
            return
        self._fraction += elapsed * self.speed
        steps = int(self._fraction)
        self._fraction -= steps
        if steps:
            self.second = min(self.second + steps, self.pair.duration - 1)
            if self.second == self.pair.duration - 1:
                self._fraction = 0.0
                self.playing = False

    def seek(self, second: int, *, preserve_play: bool = False) -> None:
        """Move the shared cursor to a recorded second."""
        self.second = max(0, min(int(second), self.pair.duration - 1))
        self._fraction = 0.0
        if self.second == self.pair.duration - 1:
            self.playing = False
        elif not preserve_play:
            self.playing = False

    def restart(self) -> None:
        """Restart both recordings without loading simulation software."""
        self.second = 0
        self._fraction = 0.0
        self.playing = True

    def set_speed(self, speed: int) -> None:
        """Set one of the documented shared playback speeds."""
        if speed not in SPEEDS:
            raise ValueError("speed must be 1, 2, or 5")
        self.speed = speed

    def toggle(self) -> None:
        """Pause or play both recordings together."""
        if self.second == self.pair.duration - 1 and not self.playing:
            self.restart()
        else:
            self.playing = not self.playing


def summary_rows(pair: DemoPair) -> tuple[tuple[str, str, str, str], ...]:
    """Return episode values and DQN-minus-fixed differences for display."""
    labels = ("Completed vehicles", "Mean queue · vehicles", "Queue s / inserted", "Signal changes", "Transition seconds")
    fixed = pair.fixed[-1].summary_values
    dqn = pair.dqn[-1].summary_values
    output = []
    for index, label in enumerate(labels):
        fmt = ".2f" if index in (1, 2) else ".0f"
        output.append((label, format(fixed[index], fmt), format(dqn[index], fmt),
                       format(dqn[index] - fixed[index], f"+{fmt}")))
    return tuple(output)


class CompareApp:
    """Pygame comparison UI using the existing schematic junction renderer."""

    WIDTH, HEIGHT = 1280, 720
    BG, CARD, TEXT, MUTED, ACCENT = ((11, 17, 27), (22, 31, 44), (236, 244, 250),
                                     (156, 172, 190), (64, 169, 245))

    def __init__(self, pair: DemoPair) -> None:
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("Pygame is required; install project requirements") from error
        self.pg = pygame
        self.pair = pair
        self.player = PairPlayer(pair)
        self.panel: JunctionPanel | None = None
        self.buttons: list[tuple[Any, str, int | None]] = []
        self.progress: Any = None

    def run(self) -> None:
        """Run the shared playback loop until the user exits."""
        pg = self.pg
        pg.init()
        try:
            screen = pg.display.set_mode((self.WIDTH, self.HEIGHT))
            pg.display.set_caption("Traffic RL · final episode comparison")
            self.panel = JunctionPanel(pg)
            clock = pg.time.Clock()
            running = True
            while running:
                elapsed = clock.tick(60) / 1000
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        running = False
                    elif event.type == pg.KEYDOWN:
                        running = self.handle_key(event.key)
                    elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                        self.handle_click(event.pos)
                self.player.advance(elapsed)
                self.draw(screen)
                pg.display.flip()
        finally:
            pg.quit()

    def handle_key(self, key: int) -> bool:
        """Apply a keyboard command to the shared player."""
        pg = self.pg
        if key in (pg.K_ESCAPE, pg.K_q):
            return False
        if key == pg.K_SPACE:
            self.player.toggle()
        elif key == pg.K_r:
            self.player.restart()
        elif key in (pg.K_1, pg.K_2, pg.K_5):
            self.player.set_speed({pg.K_1: 1, pg.K_2: 2, pg.K_5: 5}[key])
        elif key == pg.K_LEFT:
            self.player.seek(self.player.second - 5)
        elif key == pg.K_RIGHT:
            self.player.seek(self.player.second + 5)
        return True

    def handle_click(self, position: tuple[int, int]) -> None:
        """Handle a playback button or a common progress-bar seek."""
        if self.progress is not None and self.progress.collidepoint(position):
            fraction = (position[0] - self.progress.x) / self.progress.width
            self.player.seek(round(fraction * (self.pair.duration - 1)))
            return
        for bounds, action, value in self.buttons:
            if bounds.collidepoint(position):
                if action == "play":
                    self.player.toggle()
                elif action == "restart":
                    self.player.restart()
                elif action == "speed" and value is not None:
                    self.player.set_speed(value)
                return

    def _text(self, screen: Any, value: str, x: int, y: int, size: int = 20,
              colour: tuple[int, int, int] | None = None, bold: bool = False) -> None:
        font = self.pg.font.SysFont("arial", size, bold=bold)
        screen.blit(font.render(value, True, colour or self.TEXT), (x, y))

    def _panel(self, screen: Any, bounds: Any, title: str, frame: DemoFrame) -> None:
        pg = self.pg
        assert self.panel is not None
        pg.draw.rect(screen, self.CARD, bounds, border_radius=12)
        self._text(screen, title, bounds.x + 20, bounds.y + 16, 25, bold=True)
        map_rect = pg.Rect(bounds.x + 18, bounds.y + 67, 310, 310)
        cx, cy = map_rect.center
        road_width = 90
        pg.draw.rect(screen, self.panel.ROAD, pg.Rect(cx - 45, map_rect.y, road_width, 310))
        pg.draw.rect(screen, self.panel.ROAD, pg.Rect(map_rect.x, cy - 45, 310, road_width))
        self.panel._draw_lane_markings(screen, map_rect, road_width)
        self.panel._draw_queues(screen, map_rect, road_width, frame.replay, self.pair.queue_scale)
        self.panel._draw_signals(screen, map_rect, road_width, frame.replay.light_phase)
        x = bounds.x + 345
        self._text(screen, "SIGNAL DURING SECOND", x, bounds.y + 70, 14, self.MUTED, True)
        self._text(screen, PHASE_LABELS[frame.replay.light_phase], x, bounds.y + 94, 18)
        values = (("Completed", str(frame.replay.throughput)),
                  ("Current queue", str(frame.replay.total_queue)),
                  ("Mean queue to now", f"{frame.mean_queue:.2f}"),
                  ("Queue s / inserted", f"{frame.waiting_per_inserted:.2f} s"),
                  ("Signal changes", str(frame.changes)),
                  ("Transition time", f"{frame.transition_seconds} s"))
        for index, (label, value) in enumerate(values):
            y = bounds.y + 132 + index * 52
            self._text(screen, label.upper(), x, y, 13, self.MUTED, True)
            self._text(screen, value, x, y + 17, 21)
        self._text(screen, "Schematic bars · shared scale", bounds.x + 22, bounds.bottom - 35, 15, self.MUTED)

    def draw(self, screen: Any) -> None:
        """Draw both panels at one recorded second and the common controls."""
        pg = self.pg
        p = self.player
        screen.fill(self.BG)
        self._text(screen, "FINAL EVALUATION  /  SINGLE SEED REPLAY", 24, 16, 15, self.ACCENT, True)
        self._text(screen, f"{SCENARIO_NAMES[self.pair.scenario]}  ·  seed {self.pair.seed}", 24, 39, 27, bold=True)
        self._text(screen, f"t = {p.second + 1:03d} / {self.pair.duration} s", 1010, 37, 26, bold=True)
        self._text(screen, f"Recorded second {p.second} · scheduled demand: {demand_period(self.pair.scenario, p.second)}", 25, 80, 17, self.MUTED)
        self._panel(screen, pg.Rect(20, 113, 610, 463), "FIXED TIME  ·  30 s green", self.pair.fixed[p.second])
        self._panel(screen, pg.Rect(650, 113, 610, 463), "DQN  ·  15 s minimum green", self.pair.dqn[p.second])
        if self.pair.scenario == "changing":
            for boundary in (100, 200):
                x = 24 + round((boundary - 1) / (p.pair.duration - 1) * 1232)
                pg.draw.line(screen, self.MUTED, (x, 592), (x, 608), 2)
        self.progress = pg.Rect(24, 591, 1232, 18)
        pg.draw.rect(screen, (40, 54, 71), self.progress, border_radius=8)
        fill = pg.Rect(24, 591, max(8, round((p.second + 1) / p.pair.duration * 1232)), 18)
        pg.draw.rect(screen, self.ACCENT, fill, border_radius=8)
        self.buttons = []
        controls = (("Pause" if p.playing else "Play", "play", None), ("Restart", "restart", None),
                    *((f"{speed}x", "speed", speed) for speed in SPEEDS))
        x = 24
        for label, action, value in controls:
            width = 90 if action != "speed" else 60
            bounds = pg.Rect(x, 631, width, 38)
            pg.draw.rect(screen, self.ACCENT if value == p.speed else (39, 52, 69), bounds, border_radius=7)
            self._text(screen, label, x + 12, 638, 17)
            self.buttons.append((bounds, action, value))
            x += width + 9
        self._text(screen, "Space play/pause · R restart · 1/2/5 speed · arrows ±5 s · Esc exit", 526, 641, 15, self.MUTED)
        if not p.playing and p.second == p.pair.duration - 1:
            self._end_summary(screen)

    def _end_summary(self, screen: Any) -> None:
        pg = self.pg
        rect = pg.Rect(220, 145, 840, 400)
        pg.draw.rect(screen, (29, 43, 59), rect, border_radius=14)
        pg.draw.rect(screen, self.ACCENT, rect, width=2, border_radius=14)
        self._text(screen, "EPISODE COMPLETE · SELECTED SEED", 245, 166, 23, bold=True)
        self._text(screen, "Metric", 245, 213, 16, self.MUTED, True)
        self._text(screen, "Fixed", 575, 213, 16, self.MUTED, True)
        self._text(screen, "DQN", 715, 213, 16, self.MUTED, True)
        self._text(screen, "DQN − fixed", 845, 213, 16, self.MUTED, True)
        for index, (label, fixed, dqn, difference) in enumerate(summary_rows(self.pair)):
            y = 249 + index * 45
            for value, x in ((label, 245), (fixed, 575), (dqn, 715), (difference, 845)):
                self._text(screen, value, x, y, 18)
        self._text(screen, "Queue s / inserted = integrated halted-vehicle seconds / inserted vehicles", 245, 489, 14, self.MUTED)
        self._text(screen, "R to replay · click timeline to inspect", 245, 513, 14, self.MUTED)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a scenario and seed from the completed independent evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="balanced")
    parser.add_argument("--seed", type=int, choices=range(4000, 4030), default=4000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Verify the saved pair and launch the comparison window."""
    args = parse_args(argv)
    CompareApp(load_pair(args.scenario, args.seed)).run()


if __name__ == "__main__":
    main()
